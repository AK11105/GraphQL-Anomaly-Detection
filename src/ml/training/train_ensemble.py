# src/ml/training/train_ensemble.py

import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer

from src.ml.models.feature_resmlp import FeatureResMLP
from src.ml.models.attentive_bilstm import AttentiveBiLSTM
from src.ml.models.sota_transformer import SOTATransformerClassifier
from src.ml.models.ensemble_head import StrongEnsembleHead

from src.ml.training.dataloader import (
    build_feature_dataloaders,
    build_text_dataloaders,
)
from src.ml.training.metrics import compute_classification_metrics


# ============================================================
# ========================= CONFIG ===========================
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "roberta-base"
ARTIFACT_DIR = Path("src/ml/artifacts")

FEATURE_MODEL_PATH = ARTIFACT_DIR / "feature_resmlp_best.pt"
BILSTM_MODEL_PATH  = ARTIFACT_DIR / "bilstm_best.pt"
TRANS_MODEL_PATH   = ARTIFACT_DIR / "transformer_best.pt"

ENSEMBLE_MODEL_PATH = ARTIFACT_DIR / "ensemble_best.pt"
ENSEMBLE_METRICS_PATH = ARTIFACT_DIR / "ensemble_metrics.json"

FEATURE_TRAIN = "dataset/features/train.jsonl"
SEQ_TRAIN     = "dataset/sequences/train.jsonl"

EPOCHS = 50
BATCH_SIZE = 512
LR = 5e-3
WEIGHT_DECAY = 1e-2

FEATURE_KEYS = [
    "num_fields", "num_fragments", "num_directives", "num_aliases",
    "num_operations", "num_mutations", "num_subscriptions",
    "num_variables", "num_arguments", "num_introspection_ops",
    "query_depth", "avg_depth", "branching_factor", "node_count",
    "num_nested_selections", "estimated_cost", "complexity_score",
    "entropy", "query_length", "num_tokens", "has_error",
]


# ============================================================
# =================== LOAD BASE MODELS =======================
# ============================================================

def load_base_models(tokenizer):
    feature_model = FeatureResMLP(input_dim=len(FEATURE_KEYS))
    feature_model.load_state_dict(torch.load(FEATURE_MODEL_PATH, map_location="cpu"))
    feature_model.to(DEVICE).eval()

    bilstm_model = AttentiveBiLSTM(
        vocab_size=tokenizer.vocab_size,
        pad_idx=tokenizer.pad_token_id,
    )
    bilstm_model.load_state_dict(torch.load(BILSTM_MODEL_PATH, map_location="cpu"))
    bilstm_model.to(DEVICE).eval()

    transformer_model = SOTATransformerClassifier(model_name=MODEL_NAME)
    transformer_model.load_state_dict(torch.load(TRANS_MODEL_PATH, map_location="cpu"))
    transformer_model.to(DEVICE).eval()

    return feature_model, bilstm_model, transformer_model


# ============================================================
# =================== OOF PREDICTIONS ========================
# ============================================================

@torch.no_grad()
def generate_predictions(feature_loader, text_loader,
                         feature_model, bilstm_model, transformer_model):

    p_feature, p_lstm, p_transformer, labels = [], [], [], []

    for (fx, _), batch in tqdm(
        zip(feature_loader, text_loader),
        total=min(len(feature_loader), len(text_loader)),
        desc="[ENSEMBLE] Generating base predictions",
    ):
        fx = fx.to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        y = batch["labels"].to(DEVICE)

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
# ======================= TRAIN ==============================
# ============================================================

def train():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    feature_loader, _ = build_feature_dataloaders(
        FEATURE_TRAIN,
        FEATURE_TRAIN,
        FEATURE_KEYS,
        batch_size_train=BATCH_SIZE,
        batch_size_val=BATCH_SIZE,
    )

    text_loader, _ = build_text_dataloaders(
        SEQ_TRAIN,
        SEQ_TRAIN,
        tokenizer,
        max_len=256,
        batch_size_train=64,
        batch_size_val=64,
    )

    feature_model, bilstm_model, transformer_model = load_base_models(tokenizer)

    p_feature, p_lstm, p_transformer, labels = generate_predictions(
        feature_loader,
        text_loader,
        feature_model,
        bilstm_model,
        transformer_model,
    )

    ensemble = StrongEnsembleHead(in_dim=3).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        ensemble.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_auc = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        ensemble.train()

        logits = ensemble(
            p_feature=p_feature.to(DEVICE),
            p_lstm=p_lstm.to(DEVICE),
            p_transformer=p_transformer.to(DEVICE),
        )

        loss = criterion(logits, labels.to(DEVICE))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_true = labels.cpu().numpy()

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred_proba=probs,
            threshold=0.5,
        )

        metrics.update({
            "epoch": epoch,
            "loss": float(loss.item()),
        })

        history.append(metrics)

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            torch.save(ensemble.state_dict(), ENSEMBLE_MODEL_PATH)
            saved = True
        else:
            saved = False

        print(
            f"[ENSEMBLE] Epoch {epoch:02d} | "
            f"Loss={loss.item():.4f} | "
            f"AUC={metrics['roc_auc']:.4f} | "
            f"P={metrics['precision']:.4f} | "
            f"R={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"FPR={metrics['fpr']:.4f} | "
            f"Saved={saved}"
        )

    with open(ENSEMBLE_METRICS_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"✅ Ensemble training complete. Best AUC: {best_auc:.4f}")
    print(f"📊 Metrics saved to: {ENSEMBLE_METRICS_PATH}")


# ============================================================
# =========================== RUN ============================
# ============================================================

if __name__ == "__main__":
    print("DEVICE:", DEVICE)
    train()
