import os
import torch
import torch.onnx

from transformers import AutoTokenizer

from src.ml.models.feature_resmlp import FeatureResMLP
from src.ml.models.attentive_bilstm import AttentiveBiLSTM
from src.ml.models.sota_transformer import SOTATransformerClassifier
from src.ml.models.ensemble_head import StrongEnsembleHead

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

DEVICE = "cpu"  # export ONNX on CPU

ARTIFACT_DIR = "src/ml/artifacts/"

FEATURE_MODEL_CKPT = os.path.join(ARTIFACT_DIR, "feature_resmlp_best.pt")
BILSTM_MODEL_CKPT  = os.path.join(ARTIFACT_DIR, "bilstm_best.pt")
TRANS_MODEL_CKPT   = os.path.join(ARTIFACT_DIR, "transformer_best.pt")
ENSEMBLE_CKPT      = os.path.join(ARTIFACT_DIR, "ensemble_best.pt")

TRANSFORMER_MODEL_NAME = "roberta-base"
MAX_SEQ_LEN = 512

FEATURE_KEYS = [
    "num_fields", "num_fragments", "num_directives", "num_aliases",
    "num_operations", "num_mutations", "num_subscriptions",
    "num_variables", "num_arguments", "num_introspection_ops",
    "query_depth", "avg_depth", "branching_factor", "node_count",
    "num_nested_selections", "estimated_cost", "complexity_score",
    "entropy", "query_length", "num_tokens", "has_error",
]


def export_feature_resmlp_onnx():
    print("Exporting FeatureResMLP to ONNX...")
    input_dim = len(FEATURE_KEYS)

    model = FeatureResMLP(input_dim=input_dim)
    model.load_state_dict(torch.load(FEATURE_MODEL_CKPT, map_location=DEVICE))
    model.to(DEVICE).eval()

    example_input = torch.randn(1, input_dim, device=DEVICE)

    onnx_path = os.path.join(ARTIFACT_DIR, "feature_resmlp.onnx")

    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=13,
    )
    print("Saved:", onnx_path)


def export_bilstm_onnx():
    print("Exporting AttentiveBiLSTM to ONNX...")

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    vocab_size = tokenizer.vocab_size

    model = AttentiveBiLSTM(vocab_size=vocab_size)
    model.load_state_dict(torch.load(BILSTM_MODEL_CKPT, map_location=DEVICE))
    model.to(DEVICE).eval()

    example_input_ids = torch.randint(
        low=0, high=vocab_size, size=(1, MAX_SEQ_LEN), dtype=torch.long, device=DEVICE
    )
    example_attn_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long, device=DEVICE)

    onnx_path = os.path.join(ARTIFACT_DIR, "bilstm.onnx")

    torch.onnx.export(
        model,
        (example_input_ids, example_attn_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"},
        },
        opset_version=13,
    )
    print("Saved:", onnx_path)


def export_transformer_onnx():
    print("Exporting SOTATransformerClassifier to ONNX...")

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    vocab_size = tokenizer.vocab_size

    model = SOTATransformerClassifier(model_name=TRANSFORMER_MODEL_NAME)
    model.load_state_dict(torch.load(TRANS_MODEL_CKPT, map_location=DEVICE))
    model.to(DEVICE).eval()

    example_input_ids = torch.randint(
        low=0, high=vocab_size, size=(1, MAX_SEQ_LEN), dtype=torch.long, device=DEVICE
    )
    example_attn_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long, device=DEVICE)

    onnx_path = os.path.join(ARTIFACT_DIR, "transformer.onnx")

    torch.onnx.export(
        model,
        (example_input_ids, example_attn_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"},
        },
        opset_version=13,
    )
    print("Saved:", onnx_path)


def export_ensemble_onnx():
    print("Exporting StrongEnsembleHead to ONNX...")

    model = StrongEnsembleHead(in_dim=3)
    model.load_state_dict(torch.load(ENSEMBLE_CKPT, map_location=DEVICE))
    model.to(DEVICE).eval()

    example_p_feature = torch.rand(1, device=DEVICE)
    example_p_lstm = torch.rand(1, device=DEVICE)
    example_p_transformer = torch.rand(1, device=DEVICE)

    onnx_path = os.path.join(ARTIFACT_DIR, "ensemble.onnx")

    torch.onnx.export(
        model,
        (example_p_feature, example_p_lstm, example_p_transformer),
        onnx_path,
        input_names=["p_feature", "p_lstm", "p_transformer"],
        output_names=["logits"],
        dynamic_axes={
            "p_feature": {0: "batch_size"},
            "p_lstm": {0: "batch_size"},
            "p_transformer": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=13,
    )
    print("Saved:", onnx_path)


if __name__ == "__main__":
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print("Exporting ONNX models on device:", DEVICE)

    export_feature_resmlp_onnx()
    export_bilstm_onnx()
    export_transformer_onnx()
    export_ensemble_onnx()

    print("✅ ONNX export complete.")
