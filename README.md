# XAI-SkinNet

Explainable AI for Multi-Class Skin Lesion Classification using a hybrid
CNN (EfficientNet-B4) + Vision Transformer (ViT-B/16) architecture with
attention-gated feature fusion, Grad-CAM spatial explanations, and SHAP
feature attribution.

Suhani Singhania, Kashvi Agrawal, Riya Jain — VIT Vellore

This repo is the reference implementation backing the IEEE paper /
project review deck. It's written to run unmodified on **Google Colab**,
**Kaggle Notebooks**, or a **local/lab GPU machine** — nothing is
hardcoded to one environment.

## 1. Setup

```bash
git clone <your-repo-url>
cd xai-skinnet
pip install -r requirements.txt
```

On Colab, run that same block in a cell (prefix with `!`). On Kaggle, add
this repo as a dataset/utility script or just `!pip install -r
requirements.txt` after cloning.

## 2. Get the data (HAM10000)

You need a free Kaggle account and API token (`kaggle.json`) either way —
Kaggle hosts the cleanest copy of HAM10000.

**Easiest path — `kagglehub`** (this is what Kaggle's own "Copy code" button
on the dataset page gives you):
```python
import kagglehub
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
print("Path to dataset files:", path)
```
`data/download_data.py` now uses this automatically if `kagglehub` is
installed (`pip install kagglehub`, already in requirements.txt), and
copies the files into `--out` so `--data-root` stays predictable:

```bash
# Colab: upload kaggle.json first, e.g.
#   from google.colab import files; files.upload()
#   mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

python data/download_data.py --dataset ham10000 --out data/raw
```

This downloads and unzips into `data/raw/`, giving you:
```
data/raw/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
```

Kaggle Notebooks: skip the script entirely — instead add the
`kmader/skin-cancer-mnist-ham10000` dataset via "Add Data" in the UI (or
just run the `kagglehub` snippet above, no token needed inside a Kaggle
Notebook), and point `--data-root` at
`/kaggle/input/skin-cancer-mnist-ham10000` in the commands below.

ISIC 2019 (optional, for the cross-dataset generalization numbers) is
fetched the same way with `--dataset isic2019`.

**Where to actually run training:** this repo (including everything in
this fork — the metadata branch, calibration, TTA, XAI faithfulness eval)
was developed and unit-tested here against a small synthetic dataset (see
"Testing" below) since this environment has no GPU and no network access
to Kaggle. Real training needs to happen somewhere with both — **Kaggle
Notebooks is the path of least friction**: GPU is free, `kagglehub`/Kaggle
auth is automatic, and you can add this repo as a GitHub-linked notebook
or just upload the zip. Colab (upload `kaggle.json` once) works equally
well. A quickstart cell sequence for either:

```python
# Cell 1 — get the code (skip if you uploaded/cloned it another way)
!git clone <your-repo-url> && cd xai-skinnet

# Cell 2 — dependencies
!pip install -r requirements.txt -q

# Cell 3 — data (Kaggle Notebooks: this needs no token at all)
import kagglehub
data_path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
print(data_path)

# Cell 4 — train (point --data-root at whatever Cell 3 printed)
!python -m src.train --config configs/config.yaml --data-root {data_path}
```

## 3. Train

```bash
python -m src.train --config configs/config.yaml --data-root data/raw
```

Key knobs live in `configs/config.yaml` (epochs, batch size, LR, freeze
schedule, output dir). Checkpoints and logs land in `runs/<run_name>/`,
including `test_split.csv` **and `val_split.csv`** (the latter is what
`src/calibrate.py` fits temperature scaling on — see below).

Training follows the paper's recipe: AdamW (lr=2e-4, wd=1e-4), cosine
annealing, 60 epochs, batch size 32, weighted cross-entropy for class
imbalance, and the first 10 epochs train only the fusion head + last
block of each branch (lower layers frozen) before unfreezing.

**Note on "SMOTE":** true pixel-space SMOTE on dermoscopic images is not
meaningful (interpolating raw images produces garbage). What the
pipeline actually does — and what the paper's numbers reflect — is
minority-class oversampling via a `WeightedRandomSampler` combined with
weighted cross-entropy loss, which is the standard practical stand-in
and gives the equivalent effect on convergence. This is implemented in
`src/datasets/ham10000.py::make_sampler`.

## 4. Evaluate

```bash
python -m src.evaluate --config configs/config.yaml --checkpoint runs/<run_name>/best.pt --data-root data/raw
```

Prints accuracy, macro precision/recall/F1, macro-AUC (one-vs-rest), and
**Expected Calibration Error (ECE)**, and writes a confusion matrix image
+ per-class table — matching Tables I and III in the paper.

Two optional flags:
- `--tta` — averages predictions over horizontal-flip/vertical-flip/90°
  rotation views for a small, standard accuracy bump at extra inference
  cost. Off by default so baseline numbers stay comparable to a plain
  single-pass run.
- `--temperature-file` — applies a fitted temperature (see below) before
  computing metrics. Auto-detected from `runs/<run_name>/temperature.json`
  if you've already run calibration; pass explicitly to override.

### Calibration (temperature scaling)

A model can be accurate and still badly overconfident — a "92% confident"
prediction should be right about 92% of the time, which matters for a
clinician-facing tool. Fit a single scalar temperature on the held-out
validation split (never the test split, to avoid leakage into the
reported numbers):

```bash
python -m src.calibrate --checkpoint runs/<run_name>/best.pt --data-root data/raw
```

Reports ECE before/after and writes `runs/<run_name>/temperature.json`,
which `src/evaluate.py` then picks up automatically.

## 5. Explainability (Grad-CAM + SHAP)

```bash
python -m src.explainability.gradcam --checkpoint runs/<run_name>/best.pt --image path/to/lesion.jpg
python -m src.explainability.shap_explain --checkpoint runs/<run_name>/best.pt --data-root data/raw --n-background 50
```

Grad-CAM hooks the last conv block of the EfficientNet-B4 branch and
saves an overlay heatmap. SHAP uses `DeepExplainer` on the fusion-layer
output with a background sample drawn from the training set, and saves
a waterfall plot per queried image.

### Does the explanation actually track what the model uses?

Objective 5 (Section 3 of the PDF) calls for validating explanation
quality against dermatologist-annotated ground truth — that's a real
human-annotation study, out of scope for now. What's implemented instead
is the standard *automatic* faithfulness check used in the XAI literature
when annotations aren't available yet: Grad-CAM deletion/insertion AUC
(Petsiuk et al., RISE, 2018) and a SHAP ablation-consistency check
(Spearman correlation between |SHAP value| and the actual prediction
change when that feature is zeroed out):

```bash
python -m src.explainability.xai_eval --checkpoint runs/<run_name>/best.pt \
    --data-root data/raw --test-csv runs/<run_name>/test_split.csv --n-samples 30
```

Writes `xai_faithfulness.json` + a deletion/insertion curve plot next to
the checkpoint. This gives you a real number for the paper/review instead
of "the heatmaps look plausible."

## 6. Metadata branch (optional multimodal extension)

Section 12's Future Work calls out incorporating patient metadata (age,
lesion location, Fitzpatrick skin type) as an optional input — HAM10000's
own metadata CSV already has age/sex/localization, so this is implemented
as an opt-in third branch rather than left for later. Off by default (the
architecture in the PDF is the plain CNN+ViT pair); enable it with:

```yaml
model:
  use_metadata: true
  metadata_embed_dim: 64
```

When enabled, age/sex/localization are encoded into a small fixed vector
(`src/datasets/ham10000.py::encode_metadata_row`) and fused via a 3-way
softmax attention gate (`AttentionGatedFusion3` in `src/models/fusion.py`)
alongside the CNN and ViT branches — the natural generalization of the
paper's 2-way gate. `src/train.py`/`src/evaluate.py` use it automatically
once the flag is on; nothing else changes on your end. Grad-CAM's overlay
is computed with a neutral default metadata vector regardless (the
pytorch-grad-cam library has no hook for extra forward-pass arguments),
noted explicitly in the code and in the backend's `/predict` response
path — SHAP and the prediction itself do use real supplied metadata.

## 7. Deployment (optional web demo)

A minimal full-stack demo: upload a lesion image in the browser, get back
the predicted class, a Grad-CAM heatmap overlay, and the top SHAP feature
contributions.

### One-time: export a SHAP background

The backend needs a small precomputed set of "background" feature vectors
to run SHAP at request time (so it doesn't need the raw dataset on the
server). Run this once after training:

```bash
python -m scripts.export_shap_background --checkpoint runs/xai_skinnet_v1/best.pt --data-root data/raw
```

This writes `runs/xai_skinnet_v1/shap_background.pt`. If you skip this
step, `/predict` still works — it just omits `shap_top_features`.

### Backend (FastAPI)

```bash
pip install -r backend/requirements.txt   # on top of the main requirements.txt
uvicorn backend.app:app --reload --port 8000
```

Run this from the repo root (not inside `backend/`) so `src` and `backend`
are both importable. Config is via environment variables, all optional:

```bash
export XAI_CONFIG_PATH=configs/config.yaml
export XAI_CHECKPOINT_PATH=runs/xai_skinnet_v1/best.pt
export XAI_SHAP_BACKGROUND_PATH=runs/xai_skinnet_v1/shap_background.pt
```

Check it's up: `curl http://localhost:8000/health`

`POST /predict` takes a multipart `file` field (plus optional `age`,
`sex`, `localization` form fields, used only if the loaded checkpoint was
trained with `model.use_metadata: true` — see Section 6) and returns:
```json
{
  "predicted_class": "MEL",
  "confidence": 0.92,
  "probabilities": { "MEL": 0.92, "NV": 0.03, "...": "..." },
  "gradcam_image_base64": "<base64 PNG>",
  "shap_top_features": [{ "feature": "fused_dim_146", "shap_value": 0.31 }, "..."]
}
```

### Frontend (React + Vite)

```bash
cd frontend
npm install
cp .env.example .env      # points VITE_API_URL at your backend
npm run dev               # http://localhost:5173
```

`npm run build` produces a static `dist/` you can serve from anything
(nginx, Vercel, GitHub Pages, etc.) — just make sure `VITE_API_URL` points
at wherever the FastAPI backend actually runs.

**Before any real deployment:** the backend's CORS is wide open
(`allow_origins=["*"]`) for local dev convenience — narrow it to your
actual frontend's origin in `backend/app.py` first, and note this demo
has no auth, so don't expose it publicly as-is with patient data.

## Repo layout

```
configs/config.yaml         hyperparameters, paths, run name
data/download_data.py       Dataset fetcher (kagglehub, falls back to kaggle CLI)
src/datasets/ham10000.py    Dataset, transforms, class-imbalance sampler, metadata encoding
src/models/cnn_branch.py    EfficientNet-B4 feature extractor
src/models/vit_branch.py    ViT-B/16 feature extractor
src/models/fusion.py        Attention-gated fusion (2-way, and 3-way for metadata)
src/models/metadata_branch.py  Optional age/sex/localization MLP encoder
src/models/xai_skinnet.py   Full model (branches + fusion + head)
src/train.py                Training loop, freeze/unfreeze schedule
src/evaluate.py             Metrics: accuracy/F1/AUC/ECE + confusion matrix, TTA, calibration
src/calibrate.py            Temperature-scaling calibration (Guo et al. 2017)
src/explainability/gradcam.py     Grad-CAM
src/explainability/shap_explain.py SHAP DeepExplainer + waterfall plots
src/explainability/xai_eval.py    Grad-CAM deletion/insertion + SHAP ablation-consistency
src/utils.py                 Seeding, config loading, ECE, misc helpers
scripts/export_shap_background.py Precompute SHAP background (run once post-training)
backend/app.py               FastAPI app: /health, /predict
backend/inference.py         Loads checkpoint, serves prediction + Grad-CAM + SHAP
frontend/                    React + Vite upload/results UI
tests/                       Pytest suite against a synthetic HAM10000-shaped dataset
```

## Testing

The full pipeline — manifest building, stratified splitting, the metadata
encoding, the model (with and without the metadata branch), a real
forward+backward training step, Grad-CAM, SHAP, the faithfulness metrics,
and calibration — is exercised by a pytest suite against a small synthetic
dataset generated on the fly, so it runs without the real ~3GB download or
a GPU (a CPU forward pass through the full hybrid model takes well under a
second once built):

```bash
pip install pytest
python -m pytest tests/ -q
```

This is the same idea as the "tested against a synthetic sample dataset"
note in the Project Review PDF (Section 8.1), just checked in as a
repeatable suite instead of a one-off manual run — useful both for CI and
as a fast way to sanity-check any future change before spending GPU hours
on a real training run.

## Compute expectations

- Full 60-epoch run: a T4 (free Colab) will take considerably longer
  than the paper's A100 run — budget several hours, or reduce `epochs`
  in the config for a first working pass before a full run.
- `batch_size: 32` at 224×224 fits comfortably on a T4/P100; drop to 16
  if you hit CUDA OOM on a smaller GPU.
