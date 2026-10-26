"""
FastAPI backend for XAI-SkinNet.

Run from the repo root (so `src` and `backend` are both importable):
    uvicorn backend.app:app --reload --port 8000

Config via environment variables (all optional, sensible defaults shown):
    XAI_CONFIG_PATH=configs/config.yaml
    XAI_CHECKPOINT_PATH=runs/xai_skinnet_v2_meta/best.pt
    XAI_SHAP_BACKGROUND_PATH=runs/xai_skinnet_v2_meta/shap_background.pt
"""
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import InferenceEngine

CONFIG_PATH = os.environ.get("XAI_CONFIG_PATH", "configs/config.yaml")
CHECKPOINT_PATH = os.environ.get("XAI_CHECKPOINT_PATH", "runs/xai_skinnet_v2_meta/best.pt")
SHAP_BACKGROUND_PATH = os.environ.get("XAI_SHAP_BACKGROUND_PATH")  # None -> inferred from checkpoint dir

app = FastAPI(title="XAI-SkinNet API", version="1.0")

# Dev-friendly CORS. Narrow this to your actual frontend origin before any
# real deployment (e.g. allow_origins=["https://your-frontend.example.com"]).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: InferenceEngine | None = None


@app.on_event("startup")
def load_model():
    global engine
    try:
        engine = InferenceEngine(
            config_path=CONFIG_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            shap_background_path=SHAP_BACKGROUND_PATH,
        )
        print("Model loaded successfully.")
    except FileNotFoundError as e:
        # Server still starts so /health reports the real problem instead of
        # crashing the whole process — useful when standing this up before
        # training has produced a checkpoint yet.
        print(f"Model NOT loaded: {e}")
        engine = None


@app.get("/health")
def health():
    return {"status": "ok" if engine else "model_not_loaded"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    # Optional clinical metadata — only used if the loaded checkpoint was
    # trained with `model.use_metadata: true`; silently ignored otherwise.
    # All optional: an unanswered field is more honest than a guessed one.
    age: float | None = Form(default=None),
    sex: str | None = Form(default=None),
    localization: str | None = Form(default=None),
):
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — check that XAI_CHECKPOINT_PATH points to a valid checkpoint.",
        )
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file (jpg/png).")

    image_bytes = await file.read()
    try:
        return engine.predict(image_bytes, age=age, sex=sex, localization=localization)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
