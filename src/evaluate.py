import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.ham10000 import HAM10000Dataset, get_transforms
from src.models.xai_skinnet import XAISkinNet
from src.utils import expected_calibration_error, get_device, load_config

# Simple flip/rotate test-time augmentation views. Each is applied to the
# already-eval-transformed (resized+normalized) tensor, so it's cheap and
# doesn't need a second PIL pipeline.
_TTA_VIEWS = {
    "original": lambda t: t,
    "hflip": lambda t: TF.hflip(t),
    "vflip": lambda t: TF.vflip(t),
    "rot90": lambda t: torch.rot90(t, k=1, dims=[-2, -1]),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--test-csv",
        default=None,
        help="Path to a manifest CSV (e.g. runs/<run>/test_split.csv from training). "
        "If omitted, looks next to the checkpoint.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Average predictions over hflip/vflip/rot90 views for a small, standard "
        "accuracy bump at inference-time cost. Off by default to match single-pass "
        "numbers reported in the paper/PDF.",
    )
    parser.add_argument(
        "--temperature-file",
        default=None,
        help="Path to a temperature.json from src/calibrate.py. If omitted, looks "
        "next to the checkpoint and uses it automatically when present.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = XAISkinNet(ckpt.get("cfg", cfg)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    temperature = 1.0
    temp_path = Path(args.temperature_file) if args.temperature_file else (
        Path(args.checkpoint).parent / "temperature.json"
    )
    if temp_path.exists():
        temperature = json.loads(temp_path.read_text())["temperature"]
        print(f"Applying calibrated temperature T={temperature:.4f} from {temp_path}")

    test_csv = args.test_csv or (Path(args.checkpoint).parent / "test_split.csv")
    test_df = pd.read_csv(test_csv)
    # filepath column needs to be a Path again after CSV round-trip
    test_df["filepath"] = test_df["filepath"].apply(Path)

    class_names = cfg["data"]["class_names"]
    test_ds = HAM10000Dataset(test_df, get_transforms(cfg, train=False))
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    views = _TTA_VIEWS if args.tta else {"original": _TTA_VIEWS["original"]}
    if args.tta:
        print(f"TTA enabled: averaging over {list(views.keys())}")

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels, metadata in tqdm(test_loader, desc="Evaluating"):
            images, metadata = images.to(device), metadata.to(device)

            probs_sum = None
            for view_fn in views.values():
                logits = model(view_fn(images), metadata=metadata) / temperature
                probs = softmax(logits, dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            probs = (probs_sum / len(views)).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    ece = expected_calibration_error(all_probs, all_labels)

    accuracy = (all_labels == all_preds).mean()
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(len(class_names)), zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError as e:
        auc = float("nan")
        print(f"AUC could not be computed: {e}")

    print("\n=== Overall ===")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Macro AUC-ROC:   {auc:.4f}")
    print(f"ECE:             {ece:.4f}  (lower is better-calibrated; run src/calibrate.py "
          f"to fit a temperature if this looks high)")

    print("\n=== Per-class (Table III equivalent) ===")
    per_class = pd.DataFrame(
        {
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    print(per_class.to_string(index=False))

    out_dir = Path(args.checkpoint).parent
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("XAI-SkinNet Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    print(f"\nSaved per-class metrics + confusion matrix to {out_dir}")


if __name__ == "__main__":
    main()
