"""Module 4 — Phase / Microstructure Classification Model.

Multi-class classifier predicting alloy phase structure from composition,
materials descriptors, and manufacturing process parameters.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FIGURES_DIR,
    METRICS_FILE,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_CLASSIFICATION_COL,
)
from src.process_features import run_process_features
from src.utils import load_model, save_model, update_metrics_json

PHASE_MODEL_FILE = MODELS_DIR / "phase_classifier.pkl"
PHASE_ENCODER_FILE = MODELS_DIR / "phase_label_encoder.pkl"


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select appropriate input feature columns for model training."""
    exclude = {
        "Structure",
        "Phases",
        "UTS",
        "Yield",
        "Youngs_Modulus",
        "Multiphase",
        "IM_Structure",
        "Synthesis_Route",
        "Hot_Cold_Working",
    }
    # Keep numeric columns only
    cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return sorted(cols)


def train_and_evaluate_phase_model(
    df: Optional[pd.DataFrame] = None,
    target_col: str = TARGET_CLASSIFICATION_COL,
) -> Dict[str, Any]:
    """Train regularized RandomForest and XGBoost classifiers for phase prediction with CV."""
    if df is None:
        df = run_process_features()

    # Filter rows where target is present
    df_clean = df.dropna(subset=[target_col]).copy()
    feature_cols = get_feature_columns(df_clean)

    X = df_clean[feature_cols].copy()
    y_raw = df_clean[target_col].astype(str).str.strip().copy()

    # Encode targets
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)

    # 5-fold cross-validation with shuffle for generalized out-of-fold evaluation
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Candidate models with anti-overfitting regularization
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=3,
        min_samples_split=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    xgb = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )

    # Evaluate RF via CV
    y_pred_rf = cross_val_predict(rf, X, y, cv=cv)
    acc_rf = accuracy_score(y, y_pred_rf)
    f1_rf = f1_score(y, y_pred_rf, average="macro", zero_division=0)

    # Evaluate XGB via CV
    y_pred_xgb = cross_val_predict(xgb, X, y, cv=cv)
    acc_xgb = accuracy_score(y, y_pred_xgb)
    f1_xgb = f1_score(y, y_pred_xgb, average="macro", zero_division=0)

    # Select best model based on macro F1
    if f1_xgb >= f1_rf:
        best_model_name = "XGBClassifier"
        best_model = xgb
        y_best_pred = y_pred_xgb
        best_acc = acc_xgb
        best_f1 = f1_xgb
    else:
        best_model_name = "RandomForestClassifier"
        best_model = rf
        y_best_pred = y_pred_rf
        best_acc = acc_rf
        best_f1 = f1_rf

    # Fit best model on all training data
    best_model.fit(X, y)

    # Save artifacts
    save_model(
        {
            "model": best_model,
            "model_name": best_model_name,
            "feature_cols": feature_cols,
            "classes": class_names,
        },
        PHASE_MODEL_FILE,
    )
    save_model(label_encoder, PHASE_ENCODER_FILE)

    # Generate and save Confusion Matrix figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y, y_best_pred)

    plt.figure(figsize=(10, 8))
    # Shorten labels for readable axis display
    short_labels = [c[:22] + "..." if len(c) > 22 else c for c in class_names]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
    )
    plt.title(f"Phase Prediction Confusion Matrix ({best_model_name}, Acc: {best_acc:.2%})")
    plt.xlabel("Predicted Phase")
    plt.ylabel("True Phase")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = FIGURES_DIR / "phase_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    metrics = {
        "best_model": best_model_name,
        "accuracy": round(float(best_acc), 4),
        "macro_f1": round(float(best_f1), 4),
        "rf_accuracy": round(float(acc_rf), 4),
        "rf_macro_f1": round(float(f1_rf), 4),
        "xgb_accuracy": round(float(acc_xgb), 4),
        "xgb_macro_f1": round(float(f1_xgb), 4),
        "num_classes": len(class_names),
        "classes": class_names,
        "feature_count": len(feature_cols),
        "confusion_matrix_path": str(cm_path),
    }

    update_metrics_json(METRICS_FILE, "phase_classification", metrics)
    return metrics


def predict_phase(sample_df: pd.DataFrame) -> Tuple[str, float]:
    """Predict phase / structure for single or multiple sample rows."""
    model_payload = load_model(PHASE_MODEL_FILE)
    model = model_payload["model"]
    feature_cols = model_payload["feature_cols"]
    class_names = model_payload["classes"]

    # Align columns
    X_sample = sample_df.reindex(columns=feature_cols, fill_value=0.0)

    # Predict
    pred_idx = model.predict(X_sample)[0]
    pred_class = class_names[pred_idx]

    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_sample)[0]
        confidence = float(np.max(probs))

    return pred_class, confidence


if __name__ == "__main__":
    metrics = train_and_evaluate_phase_model()
    print("Phase Model Training Results:", metrics)
