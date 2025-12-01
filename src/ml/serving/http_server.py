# ml/serving/http_server.py
"""
FastAPI server for scoring windows.

Endpoint:
  POST /score
  body: { "window": [[...], ...], "model_type": "torchscript", "model_path": "models/model.ts" }
  (model_type/model_path can be omitted if server started with preloaded model)

Example run:
  uvicorn ml.serving.http_server:app --host 0.0.0.0 --port 8000
"""

import os
from typing import Optional, List
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ml.serving.predictor import Predictor

app = FastAPI(title="GraphQL Anomaly Detector - ML Scoring")

# Optionally preload a model via environment variables
PRELOAD_MODEL_TYPE = os.environ.get("PRELOAD_MODEL_TYPE")
PRELOAD_MODEL_PATH = os.environ.get("PRELOAD_MODEL_PATH")
PRELOAD_DEVICE = os.environ.get("PRELOAD_DEVICE", "cpu")

_predictor: Optional[Predictor] = None
if PRELOAD_MODEL_TYPE and PRELOAD_MODEL_PATH:
    try:
        _predictor = Predictor(PRELOAD_MODEL_TYPE, PRELOAD_MODEL_PATH, device=PRELOAD_DEVICE)
        print(f"Preloaded predictor: {PRELOAD_MODEL_TYPE} @ {PRELOAD_MODEL_PATH}")
    except Exception as e:
        print(f"Failed to preload predictor: {e}")

class ScoreRequest(BaseModel):
    window: List[List[float]]  # seq_len x feature_dim
    model_type: Optional[str] = None
    model_path: Optional[str] = None
    device: Optional[str] = "cpu"

class ScoreResponse(BaseModel):
    score: float

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    global _predictor
    if req.model_type and req.model_path:
        # ad-hoc load (stateless)
        if not os.path.exists(req.model_path):
            raise HTTPException(status_code=400, detail=f"Model path not found: {req.model_path}")
        try:
            predictor = Predictor(req.model_type, req.model_path, device=req.device or "cpu")
            s = predictor.predict(np.array(req.window, dtype=np.float32))
            return {"score": float(s)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    else:
        if _predictor is None:
            raise HTTPException(status_code=500, detail="No predictor loaded. Provide model_type and model_path in request or set PRELOAD_MODEL_TYPE/PRELOAD_MODEL_PATH env vars.")
        try:
            s = _predictor.predict(np.array(req.window, dtype=np.float32))
            return {"score": float(s)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

@app.get("/health")
def health():
    return {"status": "ok", "predictor_loaded": _predictor is not None}
