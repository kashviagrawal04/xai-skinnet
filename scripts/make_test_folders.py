"""Build a demo folder of REAL held-out test images, one sub-folder per lesion class.

    test_images/
        MEL/   NV/   BCC/   AKIEC/   BKL/   DF/   VASC/

It reproduces the exact test split used in training (same seed and function as
src/train.py), so none of these images were seen during training. Use them to
demo the web app: upload one image and compare the prediction with the folder name.

Usage (from the project folder):
    python scripts/make_test_folders.py --data-root "C:\\path\\to\\HAM10000" --per-class 5

--data-root is the folder that contains HAM10000_metadata.csv and the image folders.
If you have runs/.../test_split.csv from training, pass --test-csv to use it instead.
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.ham10000 import build_manifest, split_manifest  # noqa: E402
import yaml  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", help="Folder with HAM10000_metadata.csv and the images")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", default="test_images")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--test-csv", help="Optional test_split.csv saved by training")
    a = ap.parse_args()

    with open(a.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    names = cfg["data"]["class_names"]

    if a.test_csv:
        test_df = pd.read_csv(a.test_csv)
    else:
        if not a.data_root:
            ap.error("give --data-root (or --test-csv)")
        df = build_manifest(a.data_root, names)
        _, _, test_df = split_manifest(
            df, cfg["data"]["train_split"], cfg["data"]["val_split"], seed=cfg["seed"]
        )
    print(f"Test images available: {len(test_df)}")

    out = Path(a.out)
    for cls in names:
        d = out / cls
        d.mkdir(parents=True, exist_ok=True)
        rows = test_df[test_df["class_name"] == cls].sort_values("image_id").head(a.per_class)
        for _, r in rows.iterrows():
            shutil.copy2(r["filepath"], d / f"{r['image_id']}.jpg")
        print(f"{cls:6s} {len(rows)} images -> {d}")
    print(f"\nDone. Open the '{out}' folder and upload images from it into the app.")


if __name__ == "__main__":
    main()
