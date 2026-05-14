"""Day 4: Fine-tune all 3 conditions (1 fold each) for headline comparison.

Paste into Kaggle notebook. Requires:
- Repo at /kaggle/working/CV
- SimCLR checkpoint at SIMCLR_CKPT (attach as Kaggle Dataset input)
- State Farm competition data attached

Outputs three bundles under /kaggle/working/finetune/:
- demo_bundle_A_scratch_fold0.pth
- demo_bundle_B_simclr_fold0.pth
- demo_bundle_C_imagenet_fold0.pth

Then pick winner (lowest val log loss) and run kaggle_finetune_kfold.py
to train 5 folds for ensemble submission.
"""
import sys
import os
import argparse

sys.path.insert(0, "/kaggle/working/CV")
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")

from src.finetune import run_finetune

SIMCLR_CKPT = "/kaggle/input/simclr-pretrain-final/simclr_resnet18_ep099.pth"
FOLD = 0
OUTPUT_DIR = "/kaggle/working/finetune"


def make_args(condition: str):
    return argparse.Namespace(
        condition=condition,
        fold=FOLD,
        simclr_ckpt=SIMCLR_CKPT if condition == "B_simclr" else None,
        batch_size=128,
        num_workers=4,
        output_dir=OUTPUT_DIR,
    )


results = {}
for condition in ["A_scratch", "B_simclr", "C_imagenet"]:
    print(f"\n=========== {condition} ===========")
    result = run_finetune(make_args(condition))
    results[condition] = result

print("\n=========== Headline ===========")
for k, v in results.items():
    print(f"  {k}: val_log_loss = {v['best_val_log_loss']:.4f}")

best = min(results, key=lambda k: results[k]["best_val_log_loss"])
print(f"\nWinner: {best} (val_log_loss={results[best]['best_val_log_loss']:.4f})")
print(f"Next: run 5-fold ensemble on {best}")
