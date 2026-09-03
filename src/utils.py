import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def disable_inplace_ops(model):
    """
    pytorch-grad-cam's backward hooks conflict with in-place activations
    (SiLU/ReLU(inplace=True), common in EfficientNet blocks) — it raises
    'Output ... is a view and is being modified inplace'. Call this once,
    right after building the model and before creating a GradCAM instance,
    to flip every in-place-capable module to out-of-place. Safe to call on
    a model already in eval mode; has no effect on outputs, only autograd
    bookkeeping.
    """
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Standard ECE (Guo et al. 2017): bins predictions by confidence and
    measures the gap between confidence and actual accuracy in each bin,
    weighted by bin size. Lower is better-calibrated. Relevant here
    because a clinician-facing tool (Section 7's deployment goal) needs
    predicted confidence to actually mean something, not just an accurate
    argmax -- see src/calibrate.py.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        bin_size = in_bin.sum()
        if bin_size == 0:
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (bin_size / n) * abs(bin_acc - bin_conf)
    return float(ece)


class AverageMeter:
    """Tracks running average of a scalar (loss, accuracy, etc.)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)
