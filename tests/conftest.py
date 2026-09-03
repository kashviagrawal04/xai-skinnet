"""
Shared fixtures for the smoke-test suite.

Builds a small synthetic HAM10000-shaped dataset (same file layout the real
data/download_data.py produces) so every stage of the pipeline — manifest
building, splitting, the dataset/transforms, the model, training step,
Grad-CAM, SHAP, calibration, and the XAI faithfulness metrics — can be
exercised end-to-end without the real ~3GB download or a GPU. This mirrors
what the team already did for the Project Review (Section 8.1: "tested
against a synthetic sample dataset") but as a checked-in, repeatable suite
instead of a one-off manual run.
"""
import random

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.utils import load_config

N_PER_CLASS = 14  # enough for a 0.7/0.15/0.15 stratified 3-way split to succeed reliably
CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
SEXES = ["male", "female", None]
SITES = ["back", "scalp", "face", None]


@pytest.fixture(scope="session")
def synthetic_data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("ham_synth")
    img_dir1 = root / "HAM10000_images_part_1"
    img_dir2 = root / "HAM10000_images_part_2"
    img_dir1.mkdir()
    img_dir2.mkdir()

    rng = random.Random(42)
    rows = []
    for cls in CLASSES:
        for i in range(N_PER_CLASS):
            image_id = f"ISIC_{cls}_{i:04d}"
            arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
            img = Image.fromarray(arr)
            target_dir = img_dir1 if i % 2 == 0 else img_dir2
            img.save(target_dir / f"{image_id}.jpg")
            rows.append(
                {
                    "lesion_id": f"HAM_{cls}_{i}",
                    "image_id": image_id,
                    "dx": cls,
                    "dx_type": "histo",
                    "age": rng.choice([25.0, 40.0, 60.0, np.nan]),
                    "sex": rng.choice(SEXES),
                    "localization": rng.choice(SITES),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(root / "HAM10000_metadata.csv", index=False)
    return root


@pytest.fixture(scope="session")
def base_cfg():
    cfg = load_config("configs/config.yaml")
    # Keep image_size at 224: ViT-B/16 (non-dynamic) requires it to match
    # the backbone's fixed patch-embedding grid.
    cfg["model"]["pretrained"] = False  # no network access / HF hub in CI
    cfg["train"]["batch_size"] = 2
    cfg["train"]["epochs"] = 1
    cfg["train"]["freeze_epochs"] = 0
    cfg["train"]["num_workers"] = 0
    cfg["data"]["train_split"] = 0.7
    cfg["data"]["val_split"] = 0.15
    cfg["data"]["test_split"] = 0.15
    return cfg


@pytest.fixture(scope="session")
def metadata_cfg(base_cfg):
    cfg = {**base_cfg, "model": {**base_cfg["model"], "use_metadata": True}}
    return cfg


@pytest.fixture(scope="session")
def plain_model(base_cfg):
    from src.models.xai_skinnet import XAISkinNet

    torch.manual_seed(0)
    model = XAISkinNet(base_cfg)
    model.eval()
    return model


@pytest.fixture(scope="session")
def metadata_model(metadata_cfg):
    from src.models.xai_skinnet import XAISkinNet

    torch.manual_seed(0)
    model = XAISkinNet(metadata_cfg)
    model.eval()
    return model
