"""
Post-hoc temperature scaling (Guo et al., "On Calibration of Modern Neural
Networks", 2017).

A high-accuracy classifier isn't automatically a trustworthy one for a
clinical-facing tool -- a 92%-confident prediction should be right about
92% of the time, not 99%. Deep nets are typically over-confident out of
the box. This fits a single scalar temperature T on the held-out
validation split (never the test split -- see src/train.py's val_split.csv)
by minimizing NLL, then reports Expected Calibration Error before/after so
the improvement is measurable, not just asserted.

Usage:
    python -m src.calibrate --checkpoint runs/xai_skinnet_v1/best.pt --data-root data/raw

Writes `temperature.json` next to the checkpoint; src/evaluate.py picks it
up automatically (or via --temperature-file) and divides logits by T
before computing metrics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.ham10000 import HAM10000Dataset, get_transforms
from src.models.xai_skinnet import XAISkinNet
from src.utils import expected_calibration_error, get_device, load_config


@torch.no_grad()
def collect_logits(model, loader, device):
    all_logits, all_labels = [], []
    for images, labels, metadata in tqdm(loader, desc="Collecting logits"):
        images, metadata = images.to(device), metadata.to(device)
        logits = model(images, metadata=metadata)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
    """LBFGS fit of a single scalar T minimizing NLL of logits/T vs labels."""
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    nll = nn.CrossEntropyLoss()
    # strong_wolfe line search matters here: without it, LBFGS on this
    # simple 1-D problem can converge to a worse-than-baseline point in
    # fewer iterations instead of the (well-behaved, near-convex) NLL
    # minimum -- verified empirically while testing this against a
    # synthetic overconfidence case.
    optimizer = torch.optim.LBFGS(
        [temperature], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / temperature.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(min=1e-3).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--val-csv",
        default=None,
        help="Manifest CSV to calibrate on (default: val_split.csv next to the checkpoint).",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = XAISkinNet(ckpt.get("cfg", cfg)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_csv = args.val_csv or (Path(args.checkpoint).parent / "val_split.csv")
    if not Path(val_csv).exists():
        raise FileNotFoundError(
            f"No validation manifest at {val_csv}. Re-run training (it now saves "
            "val_split.csv), or pass --val-csv explicitly."
        )
    val_df = pd.read_csv(val_csv)
    val_df["filepath"] = val_df["filepath"].apply(Path)

    val_ds = HAM10000Dataset(val_df, get_transforms(cfg, train=False))
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    logits, labels = collect_logits(model, val_loader, device)
    probs_before = torch.softmax(logits, dim=1).numpy()
    ece_before = expected_calibration_error(probs_before, labels.numpy())

    temperature = fit_temperature(logits, labels)

    probs_after = torch.softmax(logits / temperature, dim=1).numpy()
    ece_after = expected_calibration_error(probs_after, labels.numpy())

    print(f"\nFitted temperature: {temperature:.4f}")
    print(f"ECE before scaling: {ece_before:.4f}")
    print(f"ECE after scaling:  {ece_after:.4f}")
    if ece_after > ece_before:
        print(
            "Note: ECE didn't improve -- with a small/synthetic validation "
            "split this can happen by chance; re-check once trained on the "
            "full dataset."
        )

    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent / "temperature.json"
    out_path.write_text(json.dumps({
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "n_val_samples": len(val_df),
    }, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
