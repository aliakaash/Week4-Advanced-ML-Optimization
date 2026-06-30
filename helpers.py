# utils/helpers.py
# Reusable helper functions for the ML project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import joblib
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ── Model Evaluation ───────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Return a dict of all required metrics."""
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model"    : model_name,
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score" : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC"  : round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else "N/A",
    }
    return metrics

# ── Save / Load ────────────────────────────────────────────────────────────
def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved → {path}")

def load_model(path=MODEL_PATH):
    return joblib.load(path)

def save_scaler(scaler, path=SCALER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"✅ Scaler saved → {path}")

def load_scaler(path=SCALER_PATH):
    return joblib.load(path)

# ── Plotting ───────────────────────────────────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test, model_name="Model"):
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix – {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, top_n=15):
    if not hasattr(model, "feature_importances_"):
        print("Model does not support feature_importances_")
        return None
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    return fig
