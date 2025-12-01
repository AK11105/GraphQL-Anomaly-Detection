# ml/export/export_torchscript.py
"""
Export a trained model checkpoint to TorchScript.

Usage:
  python -m ml.export.export_torchscript --checkpoint ml/checkpoints/best_model.pt --output models/model.ts --model-type lstm
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
    args = parser.parse_args(argv)

    model, cfg = load_model_from_checkpoint(args.checkpoint, args.model_type, device=args.device)

    # Dummy input: (1, seq_len, feature_dim) - try to infer seq_len from cfg or use 20
    seq_len = cfg.get("seq_len", 20)
    feature_dim = cfg.get("feature_dim", getattr(model, "feature_dim", None))
    if feature_dim is None:
        raise RuntimeError("Could not infer feature_dim from checkpoint cfg or model. Provide a valid checkpoint.")

    dummy = torch.randn(1, seq_len, feature_dim, device=args.device)
    traced = torch.jit.trace(model, dummy)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    traced.save(args.output)
    print(f"TorchScript model written to: {args.output}")

if __name__ == "__main__":
    main()
