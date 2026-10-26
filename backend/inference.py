"""
Wraps the trained XAI-SkinNet model for serving: a single `predict()` call
returns the class probabilities, a Grad-CAM overlay (base64 PNG), and the
top SHAP-contributing fused features.

Loads once at API startup — see backend/app.py.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from src.datasets.ham10000 import get_transforms
from src.models.xai_skinnet import XAISkinNet
from src.utils import disable_inplace_ops, get_device, load_config


class _MetaWrapper(nn.Module):
    """Lets pytorch-grad-cam (which calls model(x) with one argument) see the
    same clinical metadata the real prediction used."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.metadata = None

    def forward(self, x):
        return self.model(x, metadata=self.metadata)


def _normalize_map(m: np.ndarray) -> np.ndarray:
    """Scale a heatmap to [0,1], clipping at the 99th percentile so a single
    hot pixel can't flatten the rest of the map."""
    m = m - m.min()
    hi = np.percentile(m, 99)
    if hi <= 1e-8:
        return np.zeros_like(m)
    return np.clip(m / hi, 0.0, 1.0)


class InferenceEngine:
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        checkpoint_path: str = "runs/xai_skinnet_v2_meta/best.pt",
        shap_background_path: str | None = None,
        use_tta: bool = True,
    ):
        self.cfg = load_config(config_path)
        self.device = get_device()

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = XAISkinNet(ckpt.get("cfg", self.cfg)).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # Calibrated temperature fitted by src/calibrate.py (sits next to best.pt).
        # Applied to logits so the confidence shown to users matches what was
        # evaluated (ECE 0.024 after calibration vs 0.061 before).
        self.temperature = 1.0
        temp_path = Path(checkpoint_path).parent / "temperature.json"
        if temp_path.exists():
            self.temperature = float(json.loads(temp_path.read_text())["temperature"])
            print(f"Applying calibrated temperature T={self.temperature:.4f}")
        else:
            print(f"No temperature.json at {temp_path} - using T=1.0 (uncalibrated).")

        # Same flip/rotate views as `src.evaluate --tta` (90.1% vs 89.7% test acc).
        self.use_tta = use_tta
        self._views = [
            lambda t: t,
            lambda t: TF.hflip(t),
            lambda t: TF.vflip(t),
            lambda t: torch.rot90(t, k=1, dims=[-2, -1]),
        ] if use_tta else [lambda t: t]

        self.class_names = self.cfg["data"]["class_names"]
        self.transform = get_transforms(self.cfg, train=False)
        self.size = self.cfg["data"]["image_size"]

        from pytorch_grad_cam import GradCAMPlusPlus

        disable_inplace_ops(self.model)
        # Grad-CAM++ on an earlier EfficientNet stage (14x14 feature map instead
        # of the 7x7 conv_head) gives finer, less blob-like localisation.
        self._meta_wrapper = _MetaWrapper(self.model)
        stages = self.model.cnn.backbone.blocks
        self._gradcam = GradCAMPlusPlus(
            model=self._meta_wrapper,
            target_layers=[stages[-3] if len(stages) >= 3 else stages[-1]],
        )

        self.shap_explainer = None
        bg_path = shap_background_path or (Path(checkpoint_path).parent / "shap_background.pt")
        if Path(bg_path).exists():
            import shap

            background = torch.load(bg_path, map_location=self.device)
            self.shap_explainer = shap.DeepExplainer(self.model.classifier, background)
        else:
            print(
                f"No SHAP background found at {bg_path} — /predict will skip SHAP. "
                "Run scripts/export_shap_background.py once after training to enable it."
            )

    def predict(
        self,
        image_bytes: bytes,
        top_k_shap: int = 5,
        age: float | None = None,
        sex: str | None = None,
        localization: str | None = None,
    ) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        metadata = None
        if self.model.use_metadata:
            from src.datasets.ham10000 import encode_metadata_dict

            # age/sex/localization are optional form fields (see backend/app.py);
            # any omitted ones fall back to the model's neutral default encoding
            # rather than guessing, since a wrong guess would misleadingly sway
            # the prediction.
            metadata = torch.from_numpy(
                encode_metadata_dict(age=age, sex=sex, localization=localization)
            ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs_sum = 0
            for view in self._views:
                logits = self.model(view(tensor), metadata=metadata) / self.temperature
                probs_sum = probs_sum + torch.softmax(logits, dim=1)
            probs = (probs_sum / len(self._views))[0].cpu().numpy()
            # Branch weights the fusion gate assigned to this input.
            _, gate = self.model(tensor, metadata=metadata, return_attn=True)
            gate = gate[0].detach().cpu().numpy()
        if gate.shape[0] == 1:  # image-only model: sigmoid gate on the CNN branch
            branch_weights = {"cnn": float(gate[0]), "vit": float(1 - gate[0]), "metadata": 0.0}
        else:
            branch_weights = {"cnn": float(gate[0]), "vit": float(gate[1]), "metadata": float(gate[2])}
        pred_idx = int(probs.argmax())

        result = {
            "predicted_class": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {c: float(p) for c, p in zip(self.class_names, probs)},
            "gradcam_image_base64": self._explanation_overlay(
                image, tensor, pred_idx, metadata, branch_weights
            ),
            "branch_weights": branch_weights,
            "shap_top_features": self._shap_top_features(tensor, pred_idx, top_k_shap, metadata),
        }
        return result

    def _cnn_map(self, tensor, target_idx, metadata) -> np.ndarray:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        # Uses the real metadata (via the wrapper), so the map reflects the same
        # prediction the user sees.
        self._meta_wrapper.metadata = metadata
        cam = self._gradcam(input_tensor=tensor, targets=[ClassifierOutputTarget(target_idx)])[0]
        return _normalize_map(cam)

    def _vit_map(self, tensor) -> np.ndarray | None:
        """Attention rollout from the [CLS] token to the image patches."""
        try:
            vit = self.model.vit.backbone
            blocks = vit.blocks
            attns, hooks, prev = [], [], []
            for blk in blocks:
                prev.append(blk.attn.fused_attn)
                blk.attn.fused_attn = False  # so the attention matrix is materialised
                hooks.append(blk.attn.attn_drop.register_forward_hook(
                    lambda m, i, o: attns.append(o.detach())
                ))
            try:
                with torch.no_grad():
                    self.model.vit(tensor)
            finally:
                for h in hooks:
                    h.remove()
                for blk, f in zip(blocks, prev):
                    blk.attn.fused_attn = f
            if len(attns) != len(blocks):
                return None
            n = attns[0].shape[-1]
            eye = torch.eye(n, device=attns[0].device)
            rollout = eye
            for a in attns:
                a = a.mean(dim=1)[0] + eye
                a = a / a.sum(dim=-1, keepdim=True)
                rollout = a @ rollout
            n_prefix = getattr(vit, "num_prefix_tokens", 1)
            patches = rollout[0, n_prefix:]
            side = int(round(patches.numel() ** 0.5))
            if side * side != patches.numel():
                return None
            m = patches.reshape(1, 1, side, side)
            m = F.interpolate(m, size=(self.size, self.size), mode="bicubic", align_corners=False)
            return _normalize_map(m[0, 0].cpu().numpy())
        except Exception as e:  # explanation must never break prediction
            print(f"ViT attention map unavailable: {e}")
            return None

    def _explanation_overlay(self, pil_image, tensor, target_idx, metadata, branch_weights) -> str:
        """Heatmap = Grad-CAM++ (CNN) and attention rollout (ViT), blended by the
        weights the model's own fusion gate gave each branch for this image."""
        cnn_map = self._cnn_map(tensor, target_idx, metadata)
        vit_map = self._vit_map(tensor)
        w_cnn, w_vit = branch_weights["cnn"], branch_weights["vit"]
        if vit_map is None or (w_cnn + w_vit) < 1e-6:
            heat = cnn_map
        else:
            heat = (w_cnn * cnn_map + w_vit * vit_map) / (w_cnn + w_vit)
        heat = _normalize_map(cv2.GaussianBlur(heat.astype(np.float32), (0, 0), sigmaX=4))

        raw = np.array(pil_image.resize((self.size, self.size))).astype(np.float32) / 255.0
        color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)[..., ::-1] / 255.0
        # Opacity follows heat strength: cold areas stay untinted, hot areas glow.
        alpha = (np.clip((heat - 0.25) / 0.75, 0, 1) * 0.65)[..., None]
        overlay = np.uint8(255 * (raw * (1 - alpha) + color * alpha))

        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _shap_top_features(self, tensor: torch.Tensor, target_idx: int, top_k: int, metadata=None):
        if self.shap_explainer is None:
            return None

        with torch.no_grad():
            fused = self.model.fused_features(tensor, metadata=metadata)

        shap_values = self.shap_explainer.shap_values(fused)
        if isinstance(shap_values, list):
            class_shap = shap_values[target_idx][0]
        else:
            class_shap = shap_values[0, :, target_idx]

        order = np.argsort(-np.abs(class_shap))[:top_k]
        return [
            {"feature": f"fused_dim_{i}", "shap_value": float(class_shap[i])} for i in order
        ]
