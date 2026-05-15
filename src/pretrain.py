"""SimCLR pretraining loop. TPU SPMD (single-process, all chips auto-sharded).

v5e-8 on Kaggle exposes 8 chips to a single Python process. SPMD lets us
write data-parallel training as if single-device — XLA's SPMD compiler
inserts cross-chip collectives (all_reduce on grads, all_gather inside
NTXent matmul) automatically based on sharding annotations.

Key SPMD calls:
- xr.use_spmd() — enable SPMD mode at startup.
- xs.Mesh + xs.mark_sharding — declare that input batch dim is sharded
  across the 'data' mesh axis.
- pl.MpDeviceLoader(..., input_sharding=...) — applies mark_sharding to
  every batch as it lands on device.

No xmp.spawn, no DistributedSampler, no xm.optimizer_step — SPMD compiler
syncs grads via the implicit all-reduce on parameter shardings.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs

from .augmentation import build_pretrain_transform, ContrastiveViewGenerator, \
    build_pretrain_eval_transform
from .config import (
    PRETRAIN_BATCH_SIZE, PRETRAIN_EPOCHS, PRETRAIN_WARMUP_EPOCHS,
    PRETRAIN_LR, PRETRAIN_WEIGHT_DECAY, PRETRAIN_CACHE_PATH,
    PRETRAIN_CACHE_INDEX, get_working_dir,
)
from .data import (
    UnlabeledImageDataset, MemmapUnlabeledDataset, LabeledImageDataset,
    list_train_images, list_test_images, load_driver_table,
    build_group_kfold, make_loader,
)
from .diagnostics import linear_probe, sample_alignment_uniformity
from .loss import NTXentLoss
from .model import SimCLRModel
from .seed_utils import set_seed


def _unwrap(m: nn.Module) -> nn.Module:
    return getattr(m, "_orig_mod", m)


def _build_mesh():
    """1-D mesh over all chips, single 'data' axis for batch-dim sharding."""
    n = xr.global_runtime_device_count()
    return xs.Mesh(np.arange(n), (n,), ("data",))


def save_checkpoint(model, optimizer, scheduler, epoch, history, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": _unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "history": history,
    }
    path = out_dir / f"simclr_resnet18_ep{epoch:03d}.pth"
    xm.save(state, str(path))
    latest = out_dir / "simclr_resnet18_latest.pth"
    xm.save(state, str(latest))


def build_probe_loaders(batch_size: int, num_workers: int) -> tuple:
    df = load_driver_table()
    folds = build_group_kfold(df)
    train_idx, val_idx = folds[0]
    eval_tf = build_pretrain_eval_transform()

    train_ds = LabeledImageDataset(
        df.iloc[train_idx]["img_path"].tolist(),
        df.iloc[train_idx]["label"].tolist(),
        eval_tf,
    )
    val_ds = LabeledImageDataset(
        df.iloc[val_idx]["img_path"].tolist(),
        df.iloc[val_idx]["label"].tolist(),
        eval_tf,
    )
    train_loader = make_loader(train_ds, batch_size, shuffle=False,
                               num_workers=num_workers)
    val_loader = make_loader(val_ds, batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader


def run_pretrain(args) -> None:
    set_seed()
    xr.use_spmd()
    device = torch_xla.device()
    mesh = _build_mesh()
    n_chips = xr.global_runtime_device_count()
    print(f"SPMD enabled. chips={n_chips}, mesh={mesh.shape()}")

    view_gen = ContrastiveViewGenerator(build_pretrain_transform())
    use_cache = (Path(PRETRAIN_CACHE_PATH).exists()
                 and Path(PRETRAIN_CACHE_INDEX).exists())
    if use_cache:
        dataset = MemmapUnlabeledDataset(view_generator=view_gen)
        print(f"Pretrain dataset: {len(dataset)} images "
              f"(memmap cache @ {PRETRAIN_CACHE_PATH})")
    else:
        pretrain_paths = list_train_images()
        test_paths = list_test_images()
        all_paths = pretrain_paths + test_paths
        dataset = UnlabeledImageDataset(all_paths, view_gen)
        print(f"Pretrain dataset: {len(all_paths)} images "
              f"({len(pretrain_paths)} train + {len(test_paths)} test) "
              f"[JPEG path — slow; run `python -m src.pretrain_cache` first]")

    # Global batch sees all 8 chips at once in SPMD; no per-rank slicing.
    loader = make_loader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    # Shard each (B, C, H, W) input along dim 0 across the 'data' mesh axis.
    # partition_spec values are mesh axis names (or None for replicated),
    # ordered by tensor dim.
    input_sharding = xs.ShardingSpec(mesh, ("data", None, None, None))
    device_loader = pl.MpDeviceLoader(loader, device, input_sharding=input_sharding)

    model = SimCLRModel(pretrained_backbone=False).to(device)
    # SPMD without gather_distributed: compiler emits all_gather inside the
    # z @ z.t() matmul automatically when z is sharded on dim 0.
    loss_fn = NTXentLoss(gather_distributed=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=PRETRAIN_WEIGHT_DECAY,
    )
    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epochs, warmup_t=PRETRAIN_WARMUP_EPOCHS,
        warmup_lr_init=1e-6, lr_min=0.0,
    )

    start_epoch = 0
    history = []
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        _unwrap(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    out_dir = Path(args.output_dir) if args.output_dir else get_working_dir() / "simclr"

    for epoch in range(start_epoch, args.epochs):
        model.train()
        scheduler.step(epoch)

        loss_accum = torch.zeros((), device=device)
        DIAG_EVERY = 20
        n_batches = 0
        diag_count = 0
        t_data_sum = 0.0
        t_compute_sum = 0.0
        pos_sum = 0.0
        neg_sum = 0.0
        std_sum = 0.0
        images_seen = 0
        epoch_t0 = time.time()
        pbar = tqdm(device_loader, desc=f"epoch {epoch}")
        t_iter = time.time()
        for v1, v2 in pbar:
            t_data = time.time() - t_iter

            v = torch.cat([v1, v2], dim=0)
            z = model(v)
            z1, z2 = z.chunk(2, dim=0)
            loss = loss_fn(z1, z2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            xm.mark_step()

            loss_accum = loss_accum + loss.detach()
            n_batches += 1
            images_seen += v.size(0)
            t_compute = time.time() - t_iter - t_data
            t_data_sum += t_data
            t_compute_sum += t_compute

            if n_batches % DIAG_EVERY == 0:
                with torch.no_grad():
                    pos_t = (z1 * z2).sum(dim=1).mean()
                    B = z1.size(0)
                    sim_mat = z1 @ z2.t()
                    neg_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
                    neg_t = sim_mat[neg_mask].mean()
                    std_t = torch.cat([z1, z2], dim=0).std(dim=0).mean()
                pos_sum += pos_t.item()
                neg_sum += neg_t.item()
                std_sum += std_t.item()
                diag_count += 1
                pbar.set_postfix(
                    loss=(loss_accum / n_batches).item(),
                    pos=pos_sum / diag_count,
                    neg=neg_sum / diag_count,
                    std=std_sum / diag_count,
                )
            t_iter = time.time()

        xm.mark_step()
        epoch_sec = time.time() - epoch_t0
        throughput = images_seen / max(epoch_sec, 1e-6)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"[ep {epoch}] avg data={t_data_sum/max(n_batches,1):.3f}s "
              f"compute={t_compute_sum/max(n_batches,1):.3f}s "
              f"-> {'IO-bound' if t_data_sum > t_compute_sum else 'compute-bound'} "
              f"| {throughput:.0f} img/s | {epoch_sec/60:.1f} min "
              f"| lr={current_lr:.2e}")

        epoch_loss = (loss_accum / max(n_batches, 1)).item()
        log_entry = {
            "epoch": epoch,
            "loss": epoch_loss,
            "pos_sim": pos_sum / max(diag_count, 1),
            "neg_sim": neg_sum / max(diag_count, 1),
            "embed_std": std_sum / max(diag_count, 1),
            "lr": current_lr,
            "throughput_img_s": throughput,
            "epoch_sec": epoch_sec,
            "t_data_avg": t_data_sum / max(n_batches, 1),
            "t_compute_avg": t_compute_sum / max(n_batches, 1),
        }

        if (epoch + 1) % args.diagnostic_every == 0:
            base = _unwrap(model)
            base.eval()
            au = sample_alignment_uniformity(base, loader, device, max_batches=3)
            probe_train, probe_val = build_probe_loaders(
                batch_size=256, num_workers=args.num_workers,
            )
            probe = linear_probe(base.backbone, probe_train, probe_val, device)
            log_entry.update(au)
            log_entry.update(probe)
            print(f"[ep {epoch}] loss={epoch_loss:.4f} "
                  f"align={au['alignment']:.4f} uniform={au['uniformity']:.4f} "
                  f"probe_acc={probe['linear_probe_acc']:.4f} "
                  f"probe_ll={probe['linear_probe_log_loss']:.4f}")

        history.append(log_entry)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, optimizer, scheduler, epoch, history, out_dir)
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=PRETRAIN_EPOCHS)
    p.add_argument("--batch-size", type=int, default=PRETRAIN_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=PRETRAIN_LR)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--diagnostic-every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run_pretrain(parse_args())
