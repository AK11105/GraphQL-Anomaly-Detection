# ml/export/export_onnx.py
"""
Export a trained PyTorch model checkpoint to ONNX.

Usage:
  python -m ml.export.export_onnx --checkpoint ml/checkpoints/best_model.pt --output models/model.onnx --model-type lstm
"""

import argparse
import torch
import os
from ml.models.lstm_detector import LSTMDetector
from ml.models.transformer_detector import TransformerDetector

def load_model_from_checkpoint(path: str, model_type: str, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("cfg") or {}
    if model_type.lower() == "lstm":
        model = LSTMDetector.from_config(cfg)
    elif model_type.lower() == "transformer":
        model = TransformerDetector.from_config(cfg)
    else:
        raise ValueError("Unknown model type")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-type", default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args(argv)

    model, cfg = load_model_from_checkpoint(args.checkpoint, args.model_type, device=args.device)

    seq_len = cfg.get("seq_len", 20)
    feature_dim = cfg.get("feature_dim", getattr(model, "feature_dim", None))
    if feature_dim is None:
        raise RuntimeError("Could not infer feature_dim from checkpoint cfg or model. Provide a valid checkpoint.")

    dummy = torch.randn(1, seq_len, feature_dim, device=args.device)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["score"],
        opset_version=args.opset,
        dynamic_axes={
            "input": {0: "batch_size", 1: "seq_len"},
            "score": {0: "batch_size"}
        }
    )
    print(f"ONNX model written to: {args.output}")
