"""
HAM10000 dataset loading, splitting, transforms, and class-imbalance handling.

Expects `data_root` to contain (as produced by data/download_data.py, or by
Kaggle's "Add Data" for the kmader/skin-cancer-mnist-ham10000 dataset):
    HAM10000_metadata.csv
    HAM10000_images_part_1/*.jpg
    HAM10000_images_part_2/*.jpg
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

# HAM10000's `dx` column uses these short codes.
DX_TO_CLASS = {
    "mel": "MEL",
    "nv": "NV",
    "bcc": "BCC",
    "akiec": "AKIEC",
    "bkl": "BKL",
    "df": "DF",
    "vasc": "VASC",
}

# --- Clinical metadata encoding (age / sex / localization) ---------------
# HAM10000_metadata.csv ships these three columns alongside dx. They're not
# used by the paper's dual-branch architecture, but the project's own
# "Future Work" (Section 12) calls out adding patient metadata as an
# optional multimodal input, so we encode it here and wire it through an
# opt-in third branch (see src/models/metadata_branch.py, config key
# `model.use_metadata`). When that flag is off, this encoding is computed
# but simply never used by the model.
SEX_CATEGORIES = ["male", "female", "unknown"]
LOCALIZATION_CATEGORIES = [
    "back", "lower extremity", "trunk", "upper extremity", "abdomen",
    "face", "chest", "foot", "unknown", "neck", "scalp", "hand", "ear",
    "genital", "acral",
]
# [age_norm, age_missing] + one-hot(sex) + one-hot(localization)
METADATA_DIM = 2 + len(SEX_CATEGORIES) + len(LOCALIZATION_CATEGORIES)


def _one_hot(value, categories: list) -> np.ndarray:
    vec = np.zeros(len(categories), dtype=np.float32)
    idx = categories.index(value) if value in categories else categories.index("unknown")
    vec[idx] = 1.0
    return vec


def encode_metadata_dict(age=None, sex=None, localization=None) -> np.ndarray:
    """Encode raw (age, sex, localization) values into a fixed-size vector.

    Handles missing values gracefully (common in HAM10000 — ~50+ rows have
    no recorded age) via an explicit `age_missing` flag rather than
    silently imputing a value the model can't distinguish from real data.
    """
    age_missing = age is None or (isinstance(age, float) and np.isnan(age))
    age_norm = 0.5 if age_missing else float(np.clip(age, 0, 100) / 100.0)
    sex_vec = _one_hot(sex if isinstance(sex, str) else None, SEX_CATEGORIES)
    loc_vec = _one_hot(localization if isinstance(localization, str) else None, LOCALIZATION_CATEGORIES)
    return np.concatenate([[age_norm], [float(age_missing)], sex_vec, loc_vec]).astype(np.float32)


def encode_metadata_row(row) -> np.ndarray:
    age = row["age"] if "age" in row and pd.notna(row["age"]) else None
    sex = row["sex"] if "sex" in row and pd.notna(row["sex"]) else None
    localization = row["localization"] if "localization" in row and pd.notna(row["localization"]) else None
    return encode_metadata_dict(age=age, sex=sex, localization=localization)


def _find_image_dirs(data_root: Path):
    candidates = [
        data_root / "HAM10000_images_part_1",
        data_root / "HAM10000_images_part_2",
    ]
    dirs = [d for d in candidates if d.exists()]
    if not dirs:
        # Kaggle's "Add Data" sometimes nests one level deeper.
        dirs = [d for d in data_root.rglob("HAM10000_images_part_*") if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(
            f"Could not find HAM10000_images_part_1/2 under {data_root}. "
            "Check --data-root or re-run data/download_data.py."
        )
    return dirs


def _find_metadata_csv(data_root: Path) -> Path:
    direct = data_root / "HAM10000_metadata.csv"
    if direct.exists():
        return direct
    found = list(data_root.rglob("HAM10000_metadata.csv"))
    if not found:
        raise FileNotFoundError(f"Could not find HAM10000_metadata.csv under {data_root}")
    return found[0]


def build_manifest(data_root: str, class_names: list) -> pd.DataFrame:
    """Reads the metadata CSV and resolves each image_id to a file path + label index."""
    data_root = Path(data_root)
    csv_path = _find_metadata_csv(data_root)
    df = pd.read_csv(csv_path)

    image_dirs = _find_image_dirs(data_root)
    path_lookup = {}
    for d in image_dirs:
        for f in d.glob("*.jpg"):
            path_lookup[f.stem] = f

    df["class_name"] = df["dx"].map(DX_TO_CLASS)
    df["filepath"] = df["image_id"].map(path_lookup)
    missing = df["filepath"].isna().sum()
    if missing:
        print(f"Warning: {missing} rows in metadata had no matching image file; dropping them.")
    df = df.dropna(subset=["filepath", "class_name"]).reset_index(drop=True)

    name_to_idx = {name: i for i, name in enumerate(class_names)}
    df["label"] = df["class_name"].map(name_to_idx)
    return df


def split_manifest(df: pd.DataFrame, train_split: float, val_split: float, seed: int = 42):
    try:
        train_df, temp_df = train_test_split(
            df, train_size=train_split, stratify=df["label"], random_state=seed
        )
        relative_val = val_split / (1 - train_split)
        val_df, test_df = train_test_split(
            temp_df, train_size=relative_val, stratify=temp_df["label"], random_state=seed
        )
    except ValueError as e:
        # sklearn's stratify requires every class to survive both split
        # points with >=2 members. On the full HAM10000 (rarest class DF
        # has 115 images) this is never an issue; it only bites on small
        # custom/synthetic datasets, where the default error message
        # doesn't say what to actually do about it.
        counts = df["label"].value_counts()
        raise ValueError(
            f"{e}\n\nHint: a 3-way stratified split (train/val/test) needs every "
            f"class to have enough samples to survive two successive splits. "
            f"Smallest class here has {counts.min()} samples. For a train/val/test "
            f"split of {train_split}/{val_split}/{1 - train_split - val_split:.2f}, "
            f"aim for at least ~10 samples in your rarest class, or reduce "
            f"num_classes / merge rare classes for a quick smoke test."
        ) from e
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def get_transforms(cfg: dict, train: bool):
    size = cfg["data"]["image_size"]
    mean = cfg["data"]["imagenet_mean"]
    std = cfg["data"]["imagenet_std"]
    aug = cfg["augmentation"]

    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=tuple(aug["random_crop_scale"])),
                transforms.RandomHorizontalFlip(p=aug["hflip_p"]),
                transforms.RandomVerticalFlip(p=aug["vflip_p"]),
                transforms.RandomRotation(aug["rotation_deg"]),
                transforms.ColorJitter(
                    brightness=aug["color_jitter"],
                    contrast=aug["color_jitter"],
                    saturation=aug["color_jitter"],
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label"])
        # Always computed (cheap) so every consumer (train/eval/explain)
        # can use a single 3-tuple unpacking regardless of whether
        # `model.use_metadata` is enabled — the model itself ignores this
        # tensor when the metadata branch is off.
        metadata = torch.from_numpy(encode_metadata_row(row))
        return image, label, metadata


def make_sampler(df: pd.DataFrame, num_classes: int) -> WeightedRandomSampler:
    """
    Minority-class oversampling via inverse-frequency weighted sampling.

    This is the practical stand-in for "SMOTE" referenced in the paper —
    true SMOTE interpolation in raw pixel space isn't meaningful for
    dermoscopic images, so oversampling + weighted loss (see
    class_weights below) reproduces the intended effect on training
    dynamics without synthesizing garbage images.
    """
    counts = df["label"].value_counts().reindex(range(num_classes), fill_value=0).values
    counts = np.maximum(counts, 1)
    class_weight = 1.0 / counts
    sample_weights = df["label"].map(lambda c: class_weight[c]).to_numpy().copy()
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def class_weights_for_loss(df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights for nn.CrossEntropyLoss(weight=...)."""
    counts = df["label"].value_counts().reindex(range(num_classes), fill_value=0).values
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
