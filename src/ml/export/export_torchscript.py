import os
import torch

from transformers import AutoTokenizer

from src.ml.models.feature_resmlp import FeatureResMLP
from src.ml.models.attentive_bilstm import AttentiveBiLSTM
from src.ml.models.sota_transformer import SOTATransformerClassifier
from src.ml.models.ensemble_head import StrongEnsembleHead

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

DEVICE = "cpu"  # export on CPU for maximum compatibility

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


def export_feature_resmlp():
    print("Exporting FeatureResMLP to TorchScript...")
    input_dim = len(FEATURE_KEYS)

    model = FeatureResMLP(input_dim=input_dim)
    model.load_state_dict(torch.load(FEATURE_MODEL_CKPT, map_location=DEVICE))
    model.to(DEVICE).eval()

    example_input = torch.randn(1, input_dim, device=DEVICE)

    ts_path = os.path.join(ARTIFACT_DIR, "feature_resmlp_torchscript.pt")
    scripted = torch.jit.trace(model, example_inputs=example_input)
    scripted.save(ts_path)
    print("Saved:", ts_path)


def export_bilstm():
    print("Exporting AttentiveBiLSTM to TorchScript...")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    vocab_size = tokenizer.vocab_size

    model = AttentiveBiLSTM(vocab_size=vocab_size)
    model.load_state_dict(torch.load(BILSTM_MODEL_CKPT, map_location=DEVICE))
    model.to(DEVICE).eval()

    example_input_ids = torch.randint(
        low=0, high=vocab_size, size=(1, MAX_SEQ_LEN), dtype=torch.long, device=DEVICE
    )
    example_attn_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long, device=DEVICE)

    ts_path = os.path.join(ARTIFACT_DIR, "bilstm_torchscript.pt")
    scripted = torch.jit.trace(model, (example_input_ids, example_attn_mask))
    scripted.save(ts_path)
    print("Saved:", ts_path)


def export_transformer():
    print("Exporting SOTATransformerClassifier to TorchScript...")

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    model = SOTATransformerClassifier(model_name=TRANSFORMER_MODEL_NAME)
    model.load_state_dict(torch.load(TRANS_MODEL_CKPT, map_location=DEVICE))
    model.to(DEVICE).eval()

    vocab_size = tokenizer.vocab_size

    example_input_ids = torch.randint(
        low=0, high=vocab_size, size=(1, MAX_SEQ_LEN), dtype=torch.long, device=DEVICE
    )
    example_attn_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long, device=DEVICE)

    ts_path = os.path.join(ARTIFACT_DIR, "transformer_torchscript.pt")
    # transformers can be finicky with script; trace is usually safer for inference graph
    scripted = torch.jit.trace(model, (example_input_ids, example_attn_mask))
    scripted.save(ts_path)
    print("Saved:", ts_path)


def export_ensemble():
    print("Exporting StrongEnsembleHead to TorchScript...")

    model = StrongEnsembleHead(in_dim=3)
    model.load_state_dict(torch.load(ENSEMBLE_CKPT, map_location=DEVICE))
    model.to(DEVICE).eval()

    example_p_feature = torch.rand(1, device=DEVICE)
    example_p_lstm = torch.rand(1, device=DEVICE)
    example_p_transformer = torch.rand(1, device=DEVICE)

    ts_path = os.path.join(ARTIFACT_DIR, "ensemble_torchscript.pt")
    scripted = torch.jit.trace(
        model,
        (example_p_feature, example_p_lstm, example_p_transformer),
    )
    scripted.save(ts_path)
    print("Saved:", ts_path)


if __name__ == "__main__":
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print("Exporting TorchScript models on device:", DEVICE)

    export_feature_resmlp()
    export_bilstm()
    export_transformer()
    export_ensemble()

    print("✅ TorchScript export complete.")
