"""
Grad-CAM spatial explanation for a single image, hooked on the last conv
layer of the EfficientNet-B4 branch (see CNNBranch.grad_cam_target_layer).

Usage:
    python -m src.explainability.gradcam --checkpoint runs/xai_skinnet_v1/best.pt --image path/to/lesion.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.datasets.ham10000 import get_transforms
from src.models.xai_skinnet import XAISkinNet
from src.utils import disable_inplace_ops, get_device, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--target-class", type=int, default=None, help="Defaults to predicted class")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = XAISkinNet(ckpt.get("cfg", cfg)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    class_names = cfg["data"]["class_names"]
    transform = get_transforms(cfg, train=False)

    raw_image = Image.open(args.image).convert("RGB")
    size = cfg["data"]["image_size"]
    raw_resized = np.array(raw_image.resize((size, size))).astype(np.float32) / 255.0

    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    disable_inplace_ops(model)
    target_layer = model.cnn.grad_cam_target_layer()

    cam = GradCAM(model=model, target_layers=[target_layer])

    targets = None
    if args.target_class is not None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(args.target_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    overlay = show_cam_on_image(raw_resized, grayscale_cam, use_rgb=True)

    with torch.no_grad():
        logits = model(input_tensor)
        pred_idx = logits.argmax(dim=1).item()
    pred_class = class_names[pred_idx]

    out_path = Path(args.out) if args.out else Path(args.image).with_suffix(".gradcam.png")
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Predicted class: {pred_class}")
    print(f"Grad-CAM overlay saved to {out_path}")


if __name__ == "__main__":
    main()
