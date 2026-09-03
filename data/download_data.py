"""
Download HAM10000 (or ISIC 2019) into data/raw.

Two methods, tried in this order:

1. **kagglehub** (preferred) -- `pip install kagglehub`, then just:
       import kagglehub
       path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
   This is what Kaggle's own dataset page snippet gives you, and it's what
   this script uses under the hood. On Kaggle Notebooks it needs no
   credentials at all; on Colab/local it still needs a Kaggle API token
   (see below) but handles caching/versioning for you.

2. **kaggle CLI** (fallback) -- used automatically if kagglehub isn't
   installed, via `kaggle datasets download`.

Either way you need a free Kaggle account and API token at
~/.kaggle/kaggle.json (or KAGGLE_USERNAME / KAGGLE_KEY env vars) unless
you're running inside a Kaggle Notebook, where auth is automatic. See
https://www.kaggle.com/docs/api for how to get a token.

Usage:
    python data/download_data.py --dataset ham10000 --out data/raw
    python data/download_data.py --dataset isic2019 --out data/raw
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASETS = {
    "ham10000": "kmader/skin-cancer-mnist-ham10000",
    "isic2019": "andrewmvd/isic-2019",
}


def _copy_contents(src_dir: Path, dst_dir: Path):
    """kagglehub caches datasets outside the repo (e.g. ~/.cache/kagglehub);
    copy/link the actual files into --out so --data-root points somewhere
    predictable, matching what data/download_data.py has always produced."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        target = dst_dir / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _download_via_kagglehub(slug: str, out_dir: Path) -> bool:
    try:
        import kagglehub
    except ImportError:
        return False

    print(f"Using kagglehub to fetch {slug} ...")
    cache_path = Path(kagglehub.dataset_download(slug))
    print(f"kagglehub cached the dataset at: {cache_path}")
    _copy_contents(cache_path, out_dir)
    return True


def _download_via_kaggle_cli(slug: str, out_dir: Path):
    print(f"Using the kaggle CLI to fetch {slug} ...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir)],
            check=True,
        )
    except FileNotFoundError:
        print(
            "Neither kagglehub nor the 'kaggle' CLI is available. Run "
            "`pip install kagglehub` (preferred) or `pip install kaggle`, "
            "and make sure ~/.kaggle/kaggle.json exists.",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            "Kaggle download failed — check that kaggle.json is present and valid, "
            "and that you've accepted the dataset's terms on kaggle.com.",
            file=sys.stderr,
        )
        sys.exit(e.returncode)

    zip_path = out_dir / f"{slug.split('/')[-1]}.zip"
    if not zip_path.exists():
        candidates = list(out_dir.glob("*.zip"))
        if not candidates:
            print("Downloaded, but no .zip found to extract — check out_dir manually.")
            return
        zip_path = candidates[0]

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument(
        "--force-cli", action="store_true",
        help="Skip kagglehub even if installed, and use the kaggle CLI instead.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = DATASETS[args.dataset]
    print(f"Downloading {slug} -> {out_dir} ...")

    used_kagglehub = False if args.force_cli else _download_via_kagglehub(slug, out_dir)
    if not used_kagglehub:
        _download_via_kaggle_cli(slug, out_dir)

    print("Done. Contents of", out_dir)
    for p in sorted(out_dir.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
