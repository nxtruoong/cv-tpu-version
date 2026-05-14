"""Fine-tune loop. Stage 1: freeze backbone, train classifier.
Stage 2: unfreeze, discriminative LR. Used for all 3 conditions.

Runs on XLA single device (one TPU core is plenty for this small dataset).
bf16 native via XLA_USE_BF16=1 — no torch.amp, no GradScaler.
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm

import torch_xla.core.xla_model as xm

from .augmentation import build_finetune_train_transform, build_eval_transform
from .config import (
    FINETUNE_BATCH_SIZE, FINETUNE_STAGE1_EPOCHS, FINETUNE_STAGE2_EPOCHS,
    FINETUNE_STAGE2_WARMUP, FINETUNE_LABEL_SMOOTHING,
    FINETUNE_EARLY_STOP_PATIENCE, get_working_dir, CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD,
)
from .data import (
    LabeledImageDataset, load_driver_table, build_group_kfold, make_loader,
)
from .model import (
    build_classifier_for_condition, freeze_backbone, unfreeze_backbone,
    discriminative_param_groups,
)
from .seed_utils import set_seed


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, loss_fn) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    ll_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * x.size(0)
        probs = F.softmax(logits, dim=1).clamp(1e-15, 1 - 1e-15)
        ll_sum += F.nll_loss(probs.log(), y, reduction="sum").item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        xm.mark_step()
    return {
        "val_loss": total_loss / total,
        "val_log_loss": ll_sum / total,
        "val_acc": correct / total,
    }


def train_one_epoch(model, loader, optim, loss_fn, device) -> float:
    model.train()
    running = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        xm.optimizer_step(optim, barrier=True)
        running += loss.item() * x.size(0)
        n += x.size(0)
    return running / n


def run_finetune(args) -> dict:
    set_seed()
    device = xm.xla_device()

    df = load_driver_table()
    folds = build_group_kfold(df)
    train_idx, val_idx = folds[args.fold]

    train_paths = df.iloc[train_idx]["img_path"].tolist()
    train_labels = df.iloc[train_idx]["label"].tolist()
    val_paths = df.iloc[val_idx]["img_path"].tolist()
    val_labels = df.iloc[val_idx]["label"].tolist()

    train_tf = build_finetune_train_transform()
    eval_tf = build_eval_transform()

    train_ds = LabeledImageDataset(train_paths, train_labels, train_tf)
    val_ds = LabeledImageDataset(val_paths, val_labels, eval_tf)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = build_classifier_for_condition(args.condition, args.simclr_ckpt).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=FINETUNE_LABEL_SMOOTHING)

    # ---- Stage 1: freeze backbone, train classifier ----
    freeze_backbone(model)
    optim1 = torch.optim.AdamW(
        model.classifier.parameters(), lr=1e-3, weight_decay=1e-4,
    )

    print(f"[{args.condition}] Stage 1: freeze backbone, train classifier "
          f"({FINETUNE_STAGE1_EPOCHS} epochs)")
    history = []
    for epoch in range(FINETUNE_STAGE1_EPOCHS):
        tl = train_one_epoch(model, train_loader, optim1, loss_fn, device)
        m = evaluate(model, val_loader, device, loss_fn)
        m.update({"stage": 1, "epoch": epoch, "train_loss": tl})
        history.append(m)
        print(f"  ep{epoch} train={tl:.4f} val_ll={m['val_log_loss']:.4f} "
              f"val_acc={m['val_acc']:.4f}")

    # ---- Stage 2: unfreeze, discriminative LR, cosine ----
    unfreeze_backbone(model)
    optim2 = torch.optim.AdamW(
        discriminative_param_groups(model), weight_decay=1e-4,
    )
    scheduler2 = CosineLRScheduler(
        optim2, t_initial=FINETUNE_STAGE2_EPOCHS, warmup_t=FINETUNE_STAGE2_WARMUP,
        warmup_lr_init=1e-7, lr_min=1e-6,
    )

    print(f"[{args.condition}] Stage 2: unfreeze all, discriminative LR "
          f"({FINETUNE_STAGE2_EPOCHS} epochs)")
    best_ll = float("inf")
    best_state = None
    patience = 0
    for epoch in range(FINETUNE_STAGE2_EPOCHS):
        scheduler2.step(epoch)
        tl = train_one_epoch(model, train_loader, optim2, loss_fn, device)
        m = evaluate(model, val_loader, device, loss_fn)
        m.update({"stage": 2, "epoch": epoch, "train_loss": tl})
        history.append(m)
        print(f"  ep{epoch} train={tl:.4f} val_ll={m['val_log_loss']:.4f} "
              f"val_acc={m['val_acc']:.4f}")

        if m["val_log_loss"] < best_ll:
            best_ll = m["val_log_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= FINETUNE_EARLY_STOP_PATIENCE:
                print(f"  early stop at ep{epoch} (patience {patience})")
                break

    out_dir = Path(args.output_dir) if args.output_dir else get_working_dir() / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / f"demo_bundle_{args.condition}_fold{args.fold}.pth"

    if best_state is not None:
        model.load_state_dict(best_state)

    xm.save({
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "class_names": CLASS_NAMES,
        "preprocessing": {"resize": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "condition": args.condition,
        "fold": args.fold,
        "best_val_log_loss": best_ll,
        "history": history,
    }, str(bundle_path))
    print(f"Saved: {bundle_path} (best val log loss: {best_ll:.4f})")

    with open(out_dir / f"history_{args.condition}_fold{args.fold}.json", "w") as f:
        json.dump(history, f, indent=2)

    return {"best_val_log_loss": best_ll, "bundle_path": str(bundle_path)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True,
                   choices=["A_scratch", "B_simclr", "C_imagenet"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--simclr-ckpt", type=str, default=None,
                   help="Required for B_simclr.")
    p.add_argument("--batch-size", type=int, default=FINETUNE_BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run_finetune(parse_args())
