"""SimCLR pretraining on Kaggle TPU VM v5e-8 (SPMD).

Paste this into a Kaggle notebook with Accelerator = TPU VM v5e-8.

Strategy:
- SPMD: single Python process owns all 8 chips. XLA compiler shards inputs
  along the batch axis and emits cross-chip collectives automatically.
- No xmp.spawn, no DistributedSampler.
- bf16 native via XLA_USE_BF16=1.
- 100 epochs total, global batch 768, LR sqrt-scaled.
- REQUIRED: run `python -m src.pretrain_cache` first to build the uint8
  memmap (eliminates JPEG decode bottleneck).

After session ends, download `simclr_resnet18_latest.pth` and upload as a
Kaggle Dataset so fine-tune notebook can attach it as input.
"""
import sys
import os
import argparse

REPO_ROOT = "/kaggle/working/CV"
sys.path.insert(0, REPO_ROOT)
os.environ["PYTHONPATH"] = REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")

from src.pretrain import run_pretrain
from src.config import PRETRAIN_BATCH_SIZE, PRETRAIN_EPOCHS, PRETRAIN_LR


# Set RESUME to checkpoint path from previous session (or None for fresh start)
RESUME = None  # e.g. "/kaggle/input/simclr-ckpt/simclr_resnet18_latest.pth"


def make_args(resume=None):
    return argparse.Namespace(
        epochs=PRETRAIN_EPOCHS,
        batch_size=PRETRAIN_BATCH_SIZE,
        lr=PRETRAIN_LR,
        num_workers=8,
        save_every=10,
        diagnostic_every=10,
        resume=resume,
        output_dir="/kaggle/working/simclr",
    )


if __name__ == "__main__":
    run_pretrain(make_args(resume=RESUME))
