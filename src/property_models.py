"""Module 5 — Mechanical Property Prediction Models.

Independent regression models for:
1. Ultimate Tensile Strength (UTS, MPa)
2. Yield Strength (MPa)
3. Young's Modulus of Elasticity (E, GPa)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    METRICS_FILE,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_REGRESSION_COLS,
)
from src.phase_model import get_feature_columns
from src.process_features import run_process_features
from src.utils import load_model, save_model, update_metrics_json

MODEL_FILES = {
    "UTS": MODELS_DIR / "uts_model.pkl",
    "Yield": MODELS_DIR / "yield_model.pkl",
    "Youngs_Modulus": MODELS_DIR / "e_model.pkl",
}


def evaluate_regressor_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Evaluate regressor using K-Fold cross validation and return R2, MAE, RMSE."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
    }
    cv_res = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=False)

    return {
        "r2_mean": float(np.mean(cv_res["test_r2"])),
        "r2_std": float(np.std(cv_res["test_r2"])),
        "mae_mean": float(-np.mean(cv_res["test_neg_mae"])),
        "mae_std": float(np.std(cv_res["test_neg_mae"])),
        "rmse_mean": float(-np.mean(cv_res["test_neg_rmse"])),
        "rmse_std": float(np.std(cv_res["test_neg_rmse"])),
    }


def train_single_property_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> Tuple[Any, str, Dict[str, float]]:
    """Train and tune candidate regressors with anti-overfitting regularization, selecting the best model by CV R2."""
    valid_mask = df[target_col].notnull()
    X = df.loc[valid_mask, feature_cols].copy()
    y = df.loc[valid_mask, target_col].astype(float).copy()

    # Candidate 1: Regularized XGBRegressor
    xgb = XGBRegressor(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
    )
    xgb_metrics = evaluate_regressor_cv(xgb, X, y)

    # Candidate 2: Regularized RandomForestRegressor
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=RANDOM_STATE,
    )
    rf_metrics = evaluate_regressor_cv(rf, X, y)

    # Candidate 3: Regularized GradientBoostingRegressor
    gb = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    )
    gb_metrics = evaluate_regressor_cv(gb, X, y)

    candidates = [
        ("XGBRegressor", xgb, xgb_metrics),
        ("RandomForestRegressor", rf, rf_metrics),
        ("GradientBoostingRegressor", gb, gb_metrics),
    ]

    # Select candidate with highest CV R2
    candidates.sort(key=lambda item: item[2]["r2_mean"], reverse=True)
    best_name, best_model, best_metrics = candidates[0]

    # Fit best model on entire subset
    best_model.fit(X, y)

    return best_model, best_name, best_metrics


def train_and_evaluate_property_models(
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Train independent regressors for UTS, Yield, and Young's Modulus."""
    if df is None:
        df = run_process_features()

    feature_cols = get_feature_columns(df)
    results = {}

    for target in TARGET_REGRESSION_COLS:
        if target not in df.columns:
            continue

        best_model, best_name, metrics = train_single_property_model(df, target, feature_cols)

        # Save model artifact
        save_path = MODEL_FILES[target]
        save_model(
            {
                "model": best_model,
                "model_name": best_name,
                "target": target,
                "feature_cols": feature_cols,
                "metrics": metrics,
            },
            save_path,
        )

        results[target] = {
            "algorithm": best_name,
            "r2": round(metrics["r2_mean"], 4),
            "r2_std": round(metrics["r2_std"], 4),
            "mae": round(metrics["mae_mean"], 4),
            "mae_std": round(metrics["mae_std"], 4),
            "rmse": round(metrics["rmse_mean"], 4),
            "rmse_std": round(metrics["rmse_std"], 4),
            "model_path": str(save_path),
        }

    update_metrics_json(METRICS_FILE, "mechanical_property_models", results)
    return results


def predict_properties(sample_df: pd.DataFrame) -> Dict[str, float]:
    """Predict UTS, Yield, and Young's Modulus for a single sample."""
    predictions = {}
    for target, model_path in MODEL_FILES.items():
        if not model_path.exists():
            continue
        model_payload = load_model(model_path)
        model = model_payload["model"]
        feature_cols = model_payload["feature_cols"]

        X_sample = sample_df.reindex(columns=feature_cols, fill_value=0.0)
        pred_val = float(model.predict(X_sample)[0])
        predictions[target] = round(pred_val, 2)

    return predictions


if __name__ == "__main__":
    metrics = train_and_evaluate_property_models()
    print("Mechanical Property Regressors Training Results:")
    for prop, m in metrics.items():
        print(f"  {prop}: {m['algorithm']} - R2: {m['r2']} +/- {m['r2_std']}, MAE: {m['mae']}, RMSE: {m['rmse']}")
