"""Datasets and dataloader factories for State Farm."""
from pathlib import Path
from typing import List, Optional, Tuple, Callable

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold

from .config import (
    SEED, N_FOLDS, PRETRAIN_IMG_SIZE, PRETRAIN_CACHE_RES,
    PRETRAIN_CACHE_PATH, PRETRAIN_CACHE_INDEX, get_data_root,
)
from .seed_utils import make_generator, worker_init_fn


def _fast_jpeg_open(path: str, draft_size: int) -> Image.Image:
    """JPEG decode at reduced DCT scale via libjpeg draft mode. 2-4x faster
    than full decode when target res << source res. Must be called before
    any pixel access (load/convert)."""
    img = Image.open(path)
    img.draft("RGB", (draft_size, draft_size))
    return img.convert("RGB")


def load_driver_table(data_root: Optional[Path] = None) -> pd.DataFrame:
    """Load driver_imgs_list.csv with columns: subject, classname, img."""
    data_root = data_root or get_data_root()
    csv_path = data_root / "driver_imgs_list.csv"
    df = pd.read_csv(csv_path)
    df["classname"] = df["classname"].astype(str)
    df["label"] = df["classname"].str[1].astype(int)  # 'c3' -> 3
    df["img_path"] = df.apply(
        lambda r: str(data_root / "imgs" / "train" / r["classname"] / r["img"]),
        axis=1,
    )
    return df


def list_test_images(data_root: Optional[Path] = None) -> List[str]:
    data_root = data_root or get_data_root()
    test_dir = data_root / "imgs" / "test"
    return sorted(str(p) for p in test_dir.glob("*.jpg"))


def list_train_images(data_root: Optional[Path] = None) -> List[str]:
    data_root = data_root or get_data_root()
    train_dir = data_root / "imgs" / "train"
    return sorted(str(p) for p in train_dir.rglob("*.jpg"))


def build_group_kfold(
    df: pd.DataFrame, n_splits: int = N_FOLDS
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """GroupKFold by subject. Deterministic — GroupKFold doesn't take random_state,
    but order is determined by group ordering, so seed-set numpy beforehand."""
    gkf = GroupKFold(n_splits=n_splits)
    indices = np.arange(len(df))
    return list(gkf.split(indices, df["label"].values, groups=df["subject"].values))


class UnlabeledImageDataset(Dataset):
    """For SimCLR pretrain: returns (view1, view2) per image."""

    def __init__(self, image_paths: List[str], view_generator: Callable):
        self.paths = image_paths
        self.view_generator = view_generator

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        # 2x draft target: RandomResizedCrop scale_min=0.5 -> need 2x crop size
        img = _fast_jpeg_open(self.paths[idx], PRETRAIN_IMG_SIZE * 2)
        return self.view_generator(img)


class MemmapUnlabeledDataset(Dataset):
    """SimCLR pretrain from a pre-decoded uint8 memmap of shape (N, R, R, 3).

    Eliminates JPEG decode from the hot loop. Slice -> torch tensor HWC uint8
    -> permute to CHW -> hand to v2 transforms (which accept uint8 tensors).

    Memmap is opened LAZILY per worker (in __getitem__) so that
    DDP spawn pickling does not carry a live mmap handle. Carrying the handle
    across `mp.spawn` triggers SIGKILL on Kaggle (handle invalid in child).
    """

    def __init__(
        self,
        memmap_path: str = PRETRAIN_CACHE_PATH,
        index_path: str = PRETRAIN_CACHE_INDEX,
        view_generator: Optional[Callable] = None,
        cache_res: int = PRETRAIN_CACHE_RES,
    ):
        import json
        with open(index_path) as f:
            meta = json.load(f)
        self.n = int(meta["n"])
        self.res = int(meta.get("res", cache_res))
        self.memmap_path = memmap_path
        self.view_generator = view_generator
        self._mm = None  # lazy; opened on first __getitem__ in each worker

    def _ensure_mm(self):
        if self._mm is None:
            self._mm = np.memmap(
                self.memmap_path, dtype=np.uint8, mode="r",
                shape=(self.n, self.res, self.res, 3),
            )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        self._ensure_mm()
        arr = np.array(self._mm[idx], copy=True)  # decouple worker from mmap
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW uint8
        if self.view_generator is None:
            return t
        return self.view_generator(t)

    def __getstate__(self):
        # Drop memmap from pickle so DDP spawn / worker fork doesn't carry handle.
        state = self.__dict__.copy()
        state["_mm"] = None
        return state


class LabeledImageDataset(Dataset):
    """For fine-tune: returns (image, label)."""

    def __init__(self, image_paths: List[str], labels: List[int], transform: Callable):
        self.paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), int(self.labels[idx])


class TestImageDataset(Dataset):
    """For test submission: returns (image_tensor, filename)."""

    def __init__(self, image_paths: List[str], transform: Callable):
        self.paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), Path(path).name


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    drop_last: bool = False,
    sampler=None,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
) -> DataLoader:
    """pin_memory default False — XLA does not benefit from CUDA pinning.
    Caller (CUDA fine-tune) can opt in via pin_memory=True."""
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        generator=make_generator(SEED),
        sampler=sampler,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)
