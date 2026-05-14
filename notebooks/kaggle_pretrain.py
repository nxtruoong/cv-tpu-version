"""SimCLR pretraining on Kaggle TPU VM (v3-8 / v5e-8).

Paste this into a Kaggle notebook with Accelerator = TPU.

Strategy:
- PyTorch XLA via xmp.spawn (8 TPU cores).
- bf16 native — no AMP, no GradScaler, no torch.compile.
- Cross-replica gradient sync handled by xm.optimizer_step.
- 100 epochs total, global batch 768 (96 per core), LR sqrt-scaled.
- REQUIRED: run `python -m src.pretrain_cache` first to build the uint8
  memmap. pretrain.py auto-detects the cache.
- num_workers=4 per core (32 total) — Kaggle TPU VM has plenty of CPUs.
- Save every 10. Diagnostic every 10 (linear probe + align/uniform), rank0 only.
- Resume from previous checkpoint by setting RESUME below (optional).

After session ends, download `simclr_resnet18_latest.pth` and upload as a
Kaggle Dataset so fine-tune notebook can attach it as input.

NOTE: _mp_fn MUST be defined in src.pretrain (importable module), not in this
notebook script. xmp.spawn pickles by qualified name and child processes
re-import — functions defined inside exec() in a notebook are not picklable.
"""
import sys
import os
import argparse

REPO_ROOT = "/kaggle/working/CV"
sys.path.insert(0, REPO_ROOT)
os.environ["PYTHONPATH"] = REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

# XLA runtime selection. PJRT is the modern XLA runtime — required for v4/v5e.
os.environ.setdefault("PJRT_DEVICE", "TPU")
# bf16 by default on TPU. XLA_USE_BF16 downcasts all float32 ops to bf16 on
# device (parameters stay fp32 on host). Safe for SimCLR + AdamW.
os.environ.setdefault("XLA_USE_BF16", "1")

import torch_xla.distributed.xla_multiprocessing as xmp

from src.pretrain import run_pretrain, _mp_fn
from src.config import PRETRAIN_BATCH_SIZE, PRETRAIN_EPOCHS, PRETRAIN_LR


# Set RESUME to checkpoint path from previous session (or None for fresh start)
RESUME = None  # e.g. "/kaggle/input/simclr-ckpt/simclr_resnet18_latest.pth"


def make_args(resume=None):
    return argparse.Namespace(
        epochs=PRETRAIN_EPOCHS,
        batch_size=PRETRAIN_BATCH_SIZE,
        lr=PRETRAIN_LR,
        num_workers=4,  # per-core; 8 cores * 4 = 32 worker procs
        save_every=10,
        diagnostic_every=10,
        resume=resume,
        output_dir="/kaggle/working/simclr",
    )


if __name__ == "__main__":
    args = make_args(resume=RESUME)
    # nprocs=None auto-detects all available TPU cores (8 on v3-8 / v5e-8).
    xmp.spawn(_mp_fn, args=(args,), nprocs=None, start_method="fork")
