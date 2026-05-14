"""Day 4 PM: 5-fold fine-tune of winner condition for ensemble submission.

Set WINNER_CONDITION based on kaggle_finetune.py headline results.
Outputs 5 bundles + averaged-probability submission.
"""
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, "/kaggle/working/CV")
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")

from src.finetune import run_finetune
from src.submit import run_submit

WINNER_CONDITION = "C_imagenet"  # set after viewing headline results
SIMCLR_CKPT = "/kaggle/input/simclr-pretrain-final/simclr_resnet18_ep099.pth"
OUTPUT_DIR = "/kaggle/working/finetune_kfold"


def make_args(fold: int):
    return argparse.Namespace(
        condition=WINNER_CONDITION,
        fold=fold,
        simclr_ckpt=SIMCLR_CKPT if WINNER_CONDITION == "B_simclr" else None,
        batch_size=128,
        num_workers=4,
        output_dir=OUTPUT_DIR,
    )


bundle_paths = []
for fold in range(5):
    print(f"\n=========== {WINNER_CONDITION} fold {fold} ===========")
    result = run_finetune(make_args(fold))
    bundle_paths.append(result["bundle_path"])

print("\nAll folds done. Running TTA + ensemble submission...")
submit_args = argparse.Namespace(
    bundles=bundle_paths,
    output_dir="/kaggle/working",
    output_csv=f"submission_{WINNER_CONDITION}_5fold.csv",
)
run_submit(submit_args)
print("Done.")
