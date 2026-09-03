#!/usr/bin/env bash
# Convenience wrapper: download data (if missing), train, then evaluate.
# Usage: bash scripts/run_train.sh [data_root]
set -e

DATA_ROOT="${1:-data/raw}"

if [ ! -f "$DATA_ROOT/HAM10000_metadata.csv" ]; then
    echo "HAM10000 not found at $DATA_ROOT — downloading..."
    python data/download_data.py --dataset ham10000 --out "$DATA_ROOT"
fi

python -m src.train --config configs/config.yaml --data-root "$DATA_ROOT"

RUN_NAME=$(python -c "import yaml; print(yaml.safe_load(open('configs/config.yaml'))['run_name'])")
python -m src.evaluate --config configs/config.yaml \
    --checkpoint "runs/${RUN_NAME}/best.pt" \
    --data-root "$DATA_ROOT"
