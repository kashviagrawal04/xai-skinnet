"""
Precompute a background set of fused-feature vectors for SHAP, once, right
after training — so the deployed backend doesn't need the raw HAM10000
images or metadata at inference time, just this one small tensor file.

Usage:
    python -m scripts.export_shap_background --checkpoint runs/xai_skinnet_v1/best.pt \
        --data-root data/raw --n-background 50
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.ham10000 import HAM10000Dataset, build_manifest, get_transforms
from src.models.xai_skinnet import XAISkinNet
from src.utils import get_device, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--n-background", type=int, default=50)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = XAISkinNet(ckpt.get("cfg", cfg)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    class_names = cfg["data"]["class_names"]
    df = build_manifest(args.data_root, class_names)
    sample_df = df.sample(n=min(args.n_background, len(df)), random_state=cfg["seed"])

    ds = HAM10000Dataset(sample_df, get_transforms(cfg, train=False))
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    feats = []
    with torch.no_grad():
        for images, _labels, metadata in loader:
            images, metadata = images.to(device), metadata.to(device)
            feats.append(model.fused_features(images, metadata=metadata).cpu())
    background = torch.cat(feats, dim=0)

    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent / "shap_background.pt"
    torch.save(background, out_path)
    print(f"Saved {background.shape[0]} background fused vectors to {out_path}")


if __name__ == "__main__":
    main()
