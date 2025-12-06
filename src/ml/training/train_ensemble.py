import copy
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from transformers import AutoTokenizer

from src.ml.models.feature_resmlp import FeatureResMLP
from src.ml.models.attentive_bilstm import AttentiveBiLSTM
from src.ml.models.sota_transformer import SOTATransformerClassifier
from src.ml.models.ensemble_head import StrongEnsembleHead

from src.ml.training.dataloaders import (
    build_feature_dataloaders,
    build_text_dataloaders,
)

# ============================================================
# ========================= CONFIG ===========================
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "roberta-base"
ARTIFACT_DIR = "src/ml/artifacts/"

FEATURE_MODEL_PATH = ARTIFACT_DIR + "feature_resmlp_best.pt"
BILSTM_MODEL_PATH  = ARTIFACT_DIR + "bilstm_best.pt"
TRANS_MODEL_PATH   = ARTIFACT_DIR + "transformer_best.pt"

ENSEMBLE_OUT_PATH = ARTIFACT_DIR + "ensemble_best.pt"

FEATURE_TRAIN = "dataset/features/train.jsonl"
SEQ_TRAIN     = "dataset/sequences/train.jsonl"

KFOLDS = 5
EPOCHS = 50
BATCH_SIZE = 512

FEATURE_KEYS = [
    "num_fields", "num_fragments", "num_directives", "num_aliases",
    "num_operations", "num_mutations", "num_subscriptions",
    "num_variables", "num_arguments", "num_introspection_ops",
    "query_depth", "avg_depth", "branching_factor", "node_count",
    "num_nested_selections", "estimated_cost", "complexity_score",
    "entropy", "query_length", "num_tokens", "has_error",
]

# ============================================================
# ======================= METRICS ============================
# ============================================================

def compute_metrics(logits, labels):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()

    auc = roc_auc_score(y, probs)
    preds = (probs >= 0.5).astype(int)

    p, r, f, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )

    return {
        "auc": float(auc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
    }

# ============================================================
# =================== LOAD BASE MODELS =======================
# ============================================================

def load_base_models(tokenizer):
    feature_model = FeatureResMLP(input_dim=len(FEATURE_KEYS))
    feature_model.load_state_dict(torch.load(FEATURE_MODEL_PATH, map_location="cpu"))
    feature_model.to(DEVICE).eval()

    bilstm_model = AttentiveBiLSTM(vocab_size=tokenizer.vocab_size)
    bilstm_model.load_state_dict(torch.load(BILSTM_MODEL_PATH, map_location="cpu"))
    bilstm_model.to(DEVICE).eval()

    transformer_model = SOTATransformerClassifier(model_name=MODEL_NAME)
    transformer_model.load_state_dict(torch.load(TRANS_MODEL_PATH, map_location="cpu"))
    transformer_model.to(DEVICE).eval()

    return feature_model, bilstm_model, transformer_model

# ============================================================
# =================== BUILD OOF PREDS ========================
# ============================================================

def generate_oof_predictions(feature_loader, text_loader, feature_model, bilstm_model, transformer_model):
    p_feature, p_lstm, p_transformer, labels = [], [], [], []

    with torch.no_grad():
        for (fx, fy), batch in zip(feature_loader, text_loader):
            fx = fx.to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            y = batch["labels"].float().to(DEVICE)

            logit_f = feature_model(fx)
            logit_l = bilstm_model(input_ids, attn_mask)
            logit_t = transformer_model(input_ids, attn_mask)

            p_feature.append(torch.sigmoid(logit_f))
            p_lstm.append(torch.sigmoid(logit_l))
            p_transformer.append(torch.sigmoid(logit_t))
            labels.append(y)

    return (
        torch.cat(p_feature),
        torch.cat(p_lstm),
        torch.cat(p_transformer),
        torch.cat(labels),
    )

# ============================================================
# ======================= MAIN TRAIN =========================
# ============================================================

def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    feature_loader, _ = build_feature_dataloaders(
        FEATURE_TRAIN, FEATURE_TRAIN, FEATURE_KEYS, BATCH_SIZE, BATCH_SIZE
    )

    text_loader, _ = build_text_dataloaders(
        SEQ_TRAIN, SEQ_TRAIN, tokenizer, 512, 64, 64
    )

    feature_model, bilstm_model, transformer_model = load_base_models(tokenizer)

    p_feature, p_lstm, p_transformer, labels = generate_oof_predictions(
        feature_loader,
        text_loader,
        feature_model,
        bilstm_model,
        transformer_model,
    )

    ensemble = StrongEnsembleHead(in_dim=3).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(ensemble.parameters(), lr=5e-3, weight_decay=1e-2)

    best_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        ensemble.train()

        logits = ensemble(
            p_feature=p_feature.to(DEVICE),
            p_lstm=p_lstm.to(DEVICE),
            p_transformer=p_transformer.to(DEVICE),
        )

        loss = criterion(logits, labels.to(DEVICE))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics = compute_metrics(logits, labels)

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(ensemble.state_dict(), ENSEMBLE_OUT_PATH)
            saved = True
        else:
            saved = False

        print(f"[ENSEMBLE] Epoch {epoch} | {metrics} | Saved={saved}")

    print("✅ Ensemble training complete.")

# ============================================================
# =========================== RUN ============================
# ============================================================

if __name__ == "__main__":
    print("DEVICE:", DEVICE)
    train()
