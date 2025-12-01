# ml/serving/predictor.py
"""
Predictor helper that can load TorchScript or ONNX and perform inference on a window.

Usage:
  from ml.serving.predictor import Predictor
  p = Predictor(model_type="torchscript", model_path="models/model.ts", device="cpu")
  score = p.predict(window)  # window: np.ndarray (seq_len, feature_dim)
"""

import os
import numpy as np
import torch

try:
    import onnxruntime as ort
    _ONNXRT_AVAILABLE = True
except Exception:
    _ONNXRT_AVAILABLE = False

class Predictor:
    def __init__(self, model_type: str, model_path: str, device: str = "cpu"):
        """
        model_type: "torchscript" or "onnx"
        model_path: path to file
        """
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.device = device
        self._load()

    def _load(self):
        if self.model_type == "torchscript":
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
        elif self.model_type == "onnx":
            if not _ONNXRT_AVAILABLE:
                raise RuntimeError("onnxruntime is not available. Install onnxruntime to serve ONNX models.")
            self.ort_session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        else:
            raise ValueError("model_type must be 'torchscript' or 'onnx'")

    def predict(self, window: np.ndarray):
        """
        window: (seq_len, feature_dim) or (1, seq_len, feature_dim)
        returns float score in [0,1]
        """
        if isinstance(window, list):
            window = np.asarray(window, dtype=np.float32)
        if isinstance(window, np.ndarray):
            arr = window.astype(np.float32)
        else:
            raise ValueError("window must be numpy ndarray or list")

        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]  # (1, seq_len, feature_dim)

        if self.model_type == "torchscript":
            with torch.no_grad():
                inp = torch.from_numpy(arr).to(self.device)
                out = self.model(inp)
                if isinstance(out, torch.Tensor):
                    out = out.cpu().numpy()
                score = float(np.asarray(out).reshape(-1)[0])
                return score
        else:
            # ONNX
            input_name = self.ort_session.get_inputs()[0].name
            out = self.ort_session.run(None, {input_name: arr})
            score = float(np.asarray(out[0]).reshape(-1)[0])
            return score
