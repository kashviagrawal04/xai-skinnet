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
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.datasets.ham10000 import get_transforms
from src.models.xai_skinnet import XAISkinNet
from src.utils import disable_inplace_ops, get_device, load_config


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

        from pytorch_grad_cam import GradCAM

        disable_inplace_ops(self.model)
        self._gradcam = GradCAM(
            model=self.model,
            target_layers=[self.model.cnn.grad_cam_target_layer()],
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
        pred_idx = int(probs.argmax())

        result = {
            "predicted_class": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {c: float(p) for c, p in zip(self.class_names, probs)},
            "gradcam_image_base64": self._gradcam_overlay(image, tensor, pred_idx),
            "shap_top_features": self._shap_top_features(tensor, pred_idx, top_k_shap, metadata),
        }
        return result

    def _gradcam_overlay(self, pil_image: Image.Image, tensor: torch.Tensor, target_idx: int) -> str:
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        # Note: pytorch-grad-cam's hooks call `self.model(input_tensor)` with
        # no way to thread the `metadata` kwarg through, so this overlay
        # always uses the model's neutral default metadata even when real
        # age/sex/localization were supplied for the prediction itself. The
        # spatial "where" the model looked is still valid; it just isn't
        # metadata-conditioned. SHAP (below) doesn't have this limitation
        # since we call the classifier head directly.
        raw_resized = np.array(pil_image.resize((self.size, self.size))).astype(np.float32) / 255.0
        grayscale_cam = self._gradcam(input_tensor=tensor, targets=[ClassifierOutputTarget(target_idx)])[0]
        overlay = show_cam_on_image(raw_resized, grayscale_cam, use_rgb=True)

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
