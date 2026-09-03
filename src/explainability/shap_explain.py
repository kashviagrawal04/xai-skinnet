"""
SHAP feature attribution over the 512-D fused feature vector.

Rather than running DeepExplainer over the full dual-branch model (slow,
and SHAP's DeepLIFT-style rules don't play well with attention gates or
ViT self-attention), we split the pipeline in two:
  1. Run every image through the trained CNN+ViT+fusion stack once to get
     its 512-D fused vector (cheap, done for background + query samples).
  2. Run DeepExplainer only on the classifier head (fused vector -> logits),
     which is a plain MLP DeepSHAP handles natively.

This matches the paper's framing ("SHAP applied to the fusion layer") and
is the standard practical way to make DeepSHAP tractable on a hybrid
CNN-ViT backbone.

Usage:
    python -m src.explainability.shap_explain --checkpoint runs/xai_skinnet_v1/best.pt \
        --data-root data/raw --n-background 50 --image path/to/lesion.jpg
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import torch
from torch.utils.data import DataLoader

from src.datasets.ham10000 import HAM10000Dataset, build_manifest, get_transforms
from src.models.xai_skinnet import XAISkinNet
from src.utils import get_device, load_config


def compute_fused_features(model, loader, device):
    feats = []
    with torch.no_grad():
        for images, _labels, metadata in loader:
            images, metadata = images.to(device), metadata.to(device)
            feats.append(model.fused_features(images, metadata=metadata).cpu())
    return torch.cat(feats, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image", default=None, help="Single image to explain; defaults to first test sample")
    parser.add_argument("--n-background", type=int, default=50)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--age", type=float, default=None,
        help="Patient age, only used with --image when model.use_metadata is on. "
        "Ignored (defaults to neutral) if omitted.",
    )
    parser.add_argument("--sex", default=None, choices=["male", "female"])
    parser.add_argument("--localization", default=None, help="e.g. 'back', 'scalp', 'face'")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = XAISkinNet(ckpt.get("cfg", cfg)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    class_names = cfg["data"]["class_names"]
    transform = get_transforms(cfg, train=False)

    # Background sample for SHAP's expectation baseline.
    df = build_manifest(args.data_root, class_names)
    background_df = df.sample(n=min(args.n_background, len(df)), random_state=cfg["seed"])
    background_ds = HAM10000Dataset(background_df, transform)
    background_loader = DataLoader(background_ds, batch_size=16, shuffle=False)
    background_fused = compute_fused_features(model, background_loader, device).to(device)

    explainer = shap.DeepExplainer(model.classifier, background_fused)

    # Query image: user-supplied, or the first row of the manifest.
    if args.image:
        from PIL import Image

        from src.datasets.ham10000 import encode_metadata_dict

        query_img = Image.open(args.image).convert("RGB")
        query_tensor = transform(query_img).unsqueeze(0).to(device)
        # For a manually supplied image there's no manifest row to pull
        # age/sex/localization from -- use CLI flags if given, else the
        # model's own neutral default (see XAISkinNet.default_metadata).
        query_metadata = torch.from_numpy(
            encode_metadata_dict(age=args.age, sex=args.sex, localization=args.localization)
        ).unsqueeze(0).to(device)
    else:
        query_row = df.iloc[[0]]
        query_ds = HAM10000Dataset(query_row, transform)
        query_tensor, _, query_metadata = query_ds[0]
        query_tensor = query_tensor.unsqueeze(0).to(device)
        query_metadata = query_metadata.unsqueeze(0).to(device)

    with torch.no_grad():
        query_fused = model.fused_features(query_tensor, metadata=query_metadata)
        logits = model.classifier(query_fused)
        pred_idx = logits.argmax(dim=1).item()

    shap_values = explainer.shap_values(query_fused)
    # shap_values: list of arrays (one per class) or a single array depending on shap version
    if isinstance(shap_values, list):
        class_shap = shap_values[pred_idx][0]
    else:
        class_shap = shap_values[0, :, pred_idx]

    top_k = 15
    order = pd.Series(class_shap).abs().sort_values(ascending=False).index[:top_k]

    plt.figure(figsize=(8, 6))
    plt.barh(
        [f"fused_dim_{i}" for i in order][::-1],
        class_shap[order][::-1],
        color=["#d62728" if v > 0 else "#1f77b4" for v in class_shap[order][::-1]],
    )
    plt.xlabel(f"SHAP value (impact on '{class_names[pred_idx]}' prediction)")
    plt.title("Top contributing fused-feature dimensions")
    plt.tight_layout()

    out_path = Path(args.out) if args.out else Path("shap_explanation.png")
    plt.savefig(out_path, dpi=150)
    print(f"Predicted class: {class_names[pred_idx]}")
    print(f"SHAP plot saved to {out_path}")
    print(
        "\nNote: these are contributions of the 512-D *fused* feature "
        "dimensions, not raw dermoscopic attributes (border, pigmentation, "
        "etc.) — mapping specific fused dims to clinical attributes for the "
        "paper's qualitative narrative requires a separate correlation "
        "analysis against known lesion descriptors, done once per trained "
        "model."
    )


if __name__ == "__main__":
    main()
