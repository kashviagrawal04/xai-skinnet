"""
Quantitative faithfulness evaluation for the explainability module.

The Project Review PDF's Objective 5 calls for validating "explanation
quality against dermatologist-annotated ground truth" — that full
human-annotation study is future work (Section 12) requiring clinician
time we don't have. What this script gives instead is the standard
*automatic* faithfulness check used in the XAI literature when human
annotation isn't yet available, so the "does the explanation actually
track what the model uses" question has a real, reproducible number
attached rather than being asserted from the heatmaps looking plausible:

1. Grad-CAM deletion/insertion (Petsiuk et al., RISE, 2018 convention):
   - Deletion: progressively blur out the image in Grad-CAM-ranked order
     (most-important pixels first) and track how fast the predicted
     class's probability collapses. A faithful heatmap should cause a
     *fast* drop -> low deletion AUC.
   - Insertion: the reverse — start fully blurred, reveal pixels in
     Grad-CAM-ranked order, track how fast probability recovers. A
     faithful heatmap should cause a *fast* rise -> high insertion AUC.

2. SHAP ablation consistency: zero out each of the top-K |SHAP|-ranked
   fused-feature dimensions one at a time and measure the resulting drop
   in predicted-class probability, then Spearman-correlate |SHAP value|
   against |probability drop| across dimensions. A well-behaved SHAP
   attribution should correlate positively — dimensions SHAP calls
   "important" should actually move the prediction when removed.

Usage:
    python -m src.explainability.xai_eval --checkpoint runs/xai_skinnet_v1/best.pt \
        --data-root data/raw --test-csv runs/xai_skinnet_v1/test_split.csv --n-samples 30
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from scipy.stats import spearmanr
from tqdm import tqdm

from src.datasets.ham10000 import HAM10000Dataset, get_transforms
from src.models.xai_skinnet import XAISkinNet
from src.utils import disable_inplace_ops, get_device, load_config

N_STEPS = 10  # deletion/insertion granularity: reveal/remove in 10% chunks


def _blurred(image_np: np.ndarray) -> np.ndarray:
    """Standard RISE-style baseline: a heavily blurred version of the image,
    rather than black pixels — a fully black patch is itself a spurious,
    high-contrast signal a CNN can key off, which would make deletion look
    more "faithful" than it really is."""
    k = max(3, (image_np.shape[0] // 8) | 1)  # odd kernel size
    return cv2.GaussianBlur(image_np, (k, k), 0)


def _order_pixels_by_cam(cam: np.ndarray) -> np.ndarray:
    """Flat pixel indices, most-important first."""
    return np.argsort(-cam.flatten())


@torch.no_grad()
def _predict_prob(model, tensor, target_idx, device):
    logits = model(tensor.to(device))
    probs = torch.softmax(logits, dim=1)[0, target_idx].item()
    return probs


def deletion_insertion(model, cam_engine, raw_resized, transform_fn, target_idx, device):
    """Returns (deletion_probs, insertion_probs), each length N_STEPS+1, and their AUCs."""
    h, w, _ = raw_resized.shape
    input_tensor = transform_fn(Image.fromarray((raw_resized * 255).astype(np.uint8))).unsqueeze(0)
    grayscale_cam = cam_engine(input_tensor=input_tensor.to(device), targets=None)[0]  # [H', W'] at model res
    cam_resized = cv2.resize(grayscale_cam, (w, h))
    order = _order_pixels_by_cam(cam_resized)

    baseline = _blurred(raw_resized)
    n_pixels = h * w
    chunk = n_pixels // N_STEPS

    deletion_probs, insertion_probs = [], []
    deleted = raw_resized.copy()
    inserted = baseline.copy()

    for step in range(N_STEPS + 1):
        n_reveal = min(step * chunk, n_pixels)
        idx = order[:n_reveal]
        rows, cols = np.unravel_index(idx, (h, w))

        # Deletion: start from the original, replace top-ranked pixels with baseline.
        deleted = raw_resized.copy()
        deleted[rows, cols] = baseline[rows, cols]
        del_tensor = transform_fn(Image.fromarray((np.clip(deleted, 0, 1) * 255).astype(np.uint8))).unsqueeze(0)
        deletion_probs.append(_predict_prob(model, del_tensor, target_idx, device))

        # Insertion: start from the baseline, reveal top-ranked pixels from the original.
        inserted = baseline.copy()
        inserted[rows, cols] = raw_resized[rows, cols]
        ins_tensor = transform_fn(Image.fromarray((np.clip(inserted, 0, 1) * 255).astype(np.uint8))).unsqueeze(0)
        insertion_probs.append(_predict_prob(model, ins_tensor, target_idx, device))

    fractions = np.linspace(0, 1, N_STEPS + 1)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # np.trapz removed in NumPy 2.0
    deletion_auc = float(trapz(deletion_probs, fractions))
    insertion_auc = float(trapz(insertion_probs, fractions))
    return deletion_probs, insertion_probs, deletion_auc, insertion_auc


def shap_consistency(model, explainer, fused, target_idx, device, top_k=20):
    """Zero-ablate each of the top-K |SHAP|-ranked fused dims individually,
    measure the resulting drop in predicted-class probability, and
    Spearman-correlate against |SHAP value|."""
    shap_values = explainer.shap_values(fused)
    class_shap = shap_values[target_idx][0] if isinstance(shap_values, list) else shap_values[0, :, target_idx]

    with torch.no_grad():
        base_prob = torch.softmax(model.classifier(fused), dim=1)[0, target_idx].item()

    order = np.argsort(-np.abs(class_shap))[:top_k]
    prob_drops = []
    for dim in order:
        ablated = fused.clone()
        ablated[0, dim] = 0.0
        with torch.no_grad():
            p = torch.softmax(model.classifier(ablated), dim=1)[0, target_idx].item()
        prob_drops.append(base_prob - p)

    shap_mags = np.abs(class_shap[order])
    corr, pvalue = spearmanr(shap_mags, np.abs(prob_drops))
    return float(corr) if not np.isnan(corr) else 0.0, float(pvalue), prob_drops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--n-background", type=int, default=30, help="Background set size for the SHAP explainer used in the consistency check.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = XAISkinNet(ckpt.get("cfg", cfg)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    disable_inplace_ops(model)

    class_names = cfg["data"]["class_names"]
    transform = get_transforms(cfg, train=False)
    size = cfg["data"]["image_size"]

    test_csv = args.test_csv or (Path(args.checkpoint).parent / "test_split.csv")
    test_df = pd.read_csv(test_csv)
    test_df["filepath"] = test_df["filepath"].apply(Path)

    rng = np.random.RandomState(cfg["seed"])
    sample_df = test_df.sample(n=min(args.n_samples, len(test_df)), random_state=rng)

    cam_engine = GradCAM(model=model, target_layers=[model.cnn.grad_cam_target_layer()])

    # --- Grad-CAM deletion/insertion, averaged over the sample ---
    all_deletion, all_insertion = [], []
    deletion_aucs, insertion_aucs = [], []
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Grad-CAM faithfulness"):
        raw_image = Image.open(row["filepath"]).convert("RGB")
        raw_resized = np.array(raw_image.resize((size, size))).astype(np.float32) / 255.0

        with torch.no_grad():
            pred_idx = model(transform(raw_image).unsqueeze(0).to(device)).argmax(dim=1).item()

        del_probs, ins_probs, del_auc, ins_auc = deletion_insertion(
            model, cam_engine, raw_resized, transform, pred_idx, device
        )
        all_deletion.append(del_probs)
        all_insertion.append(ins_probs)
        deletion_aucs.append(del_auc)
        insertion_aucs.append(ins_auc)

    mean_deletion = np.mean(all_deletion, axis=0)
    mean_insertion = np.mean(all_insertion, axis=0)

    # --- SHAP ablation consistency, on a (smaller) background + sample set ---
    background_df = test_df.sample(n=min(args.n_background, len(test_df)), random_state=rng)
    background_ds = HAM10000Dataset(background_df, transform)
    bg_feats = []
    with torch.no_grad():
        for i in range(len(background_ds)):
            img, _, meta = background_ds[i]
            bg_feats.append(model.fused_features(img.unsqueeze(0).to(device), metadata=meta.unsqueeze(0).to(device)))
    background_fused = torch.cat(bg_feats, dim=0)
    explainer = shap.DeepExplainer(model.classifier, background_fused)

    shap_sample_df = sample_df.head(min(10, len(sample_df)))  # SHAP is slower; smaller subset
    shap_ds = HAM10000Dataset(shap_sample_df, transform)
    correlations = []
    for i in range(len(shap_ds)):
        img, _, meta = shap_ds[i]
        with torch.no_grad():
            fused = model.fused_features(img.unsqueeze(0).to(device), metadata=meta.unsqueeze(0).to(device))
            pred_idx = model.classifier(fused).argmax(dim=1).item()
        corr, pvalue, _ = shap_consistency(model, explainer, fused, pred_idx, device)
        correlations.append(corr)

    results = {
        "n_gradcam_samples": len(sample_df),
        "deletion_auc_mean": float(np.mean(deletion_aucs)),
        "deletion_auc_std": float(np.std(deletion_aucs)),
        "insertion_auc_mean": float(np.mean(insertion_aucs)),
        "insertion_auc_std": float(np.std(insertion_aucs)),
        "n_shap_samples": len(shap_ds),
        "shap_ablation_spearman_mean": float(np.mean(correlations)) if correlations else None,
        "shap_ablation_spearman_per_sample": correlations,
    }

    print("\n=== Grad-CAM faithfulness ===")
    print(f"Deletion AUC:  {results['deletion_auc_mean']:.4f} +/- {results['deletion_auc_std']:.4f}  (lower = more faithful)")
    print(f"Insertion AUC: {results['insertion_auc_mean']:.4f} +/- {results['insertion_auc_std']:.4f}  (higher = more faithful)")
    print("\n=== SHAP ablation consistency ===")
    print(f"Mean Spearman(|SHAP|, |prob drop on ablation|): {results['shap_ablation_spearman_mean']:.4f}  "
          f"(closer to +1 = SHAP-important dims genuinely move the prediction)")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "xai_faithfulness.json").write_text(json.dumps(results, indent=2))

    fractions = np.linspace(0, 1, N_STEPS + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(fractions, mean_deletion, marker="o", label=f"Deletion (AUC={results['deletion_auc_mean']:.3f})")
    plt.plot(fractions, mean_insertion, marker="o", label=f"Insertion (AUC={results['insertion_auc_mean']:.3f})")
    plt.xlabel("Fraction of pixels revealed/removed (Grad-CAM order)")
    plt.ylabel("Predicted-class probability")
    plt.title("Grad-CAM Deletion/Insertion Faithfulness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "xai_faithfulness_curves.png", dpi=150)

    print(f"\nSaved metrics + curves to {out_dir}")


if __name__ == "__main__":
    main()
