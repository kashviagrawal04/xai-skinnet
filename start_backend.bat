@echo off
echo Starting XAI-SkinNet backend...
set XAI_CONFIG_PATH=configs/config.yaml
set XAI_CHECKPOINT_PATH=runs/xai_skinnet_v2_meta/best.pt
uvicorn backend.app:app --reload --port 8000
