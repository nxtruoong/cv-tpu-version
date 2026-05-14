"""Test-time inference: TTA + K-fold ensemble + Kaggle submission file."""
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import torch_xla
import torch_xla.core.xla_model as xm

from .augmentation import build_tta_transforms
from .config import NUM_CLASSES, get_working_dir
from .data import list_test_images
from .model import ClassifierModel
from .seed_utils import set_seed


def load_bundle(bundle_path: str, device: torch.device) -> ClassifierModel:
    state = torch.load(bundle_path, map_location=device)
    pretrained = state["condition"] == "C_imagenet"
    model = ClassifierModel(pretrained_backbone=pretrained)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_with_tta(
    model: ClassifierModel,
    image_paths: List[str],
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Returns (N, 10) softmax probabilities averaged across TTA crops.
    Runs on XLA (bf16 native via XLA_USE_BF16)."""
    tta_transforms = build_tta_transforms()
    n = len(image_paths)
    all_probs = np.zeros((n, NUM_CLASSES), dtype=np.float64)

    for tf in tta_transforms:
        for start in tqdm(range(0, n, batch_size), desc="TTA pass"):
            batch_paths = image_paths[start:start + batch_size]
            imgs = torch.stack([
                tf(Image.open(p).convert("RGB")) for p in batch_paths
            ]).to(device, non_blocking=True)

            logits = model(imgs)
            probs = F.softmax(logits.float(), dim=1)
            xm.mark_step()
            all_probs[start:start + len(batch_paths)] += probs.cpu().numpy()

    all_probs /= len(tta_transforms)
    return all_probs


def ensemble_bundles(
    bundle_paths: List[str], image_paths: List[str], device: torch.device,
) -> np.ndarray:
    """Average TTA probabilities across multiple model checkpoints (K folds)."""
    probs = np.zeros((len(image_paths), NUM_CLASSES), dtype=np.float64)
    for bp in bundle_paths:
        model = load_bundle(bp, device)
        probs += predict_with_tta(model, image_paths, device)
        del model
    return probs / len(bundle_paths)


def write_submission(
    probs: np.ndarray, image_paths: List[str], out_path: Path,
) -> None:
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    filenames = [Path(p).name for p in image_paths]
    df = pd.DataFrame({"img": filenames})
    for i in range(NUM_CLASSES):
        df[f"c{i}"] = probs[:, i]
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")


def run_submit(args) -> None:
    set_seed()
    device = torch_xla.device()

    test_paths = list_test_images()
    bundles = args.bundles

    probs = ensemble_bundles(bundles, test_paths, device)

    out_dir = Path(args.output_dir) if args.output_dir else get_working_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "test_probs.npy", probs)
    write_submission(probs, test_paths, out_dir / args.output_csv)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundles", nargs="+", required=True,
                   help="Paths to demo_bundle_*.pth files to ensemble.")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--output-csv", type=str, default="submission.csv")
    return p.parse_args()


if __name__ == "__main__":
    run_submit(parse_args())
