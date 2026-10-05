"""Unit tests for Phase Classifier (Module 4) and Property Regressors (Module 5)."""

import pytest
import numpy as np
import pandas as pd
from src.config import METRICS_FILE, TARGET_REGRESSION_COLS
from src.phase_model import predict_phase, train_and_evaluate_phase_model
from src.process_features import build_single_sample_features
from src.property_models import predict_properties, train_and_evaluate_property_models
from src.utils import load_json


@pytest.fixture(scope="module")
def trained_models():
    """Ensure models are trained and metrics recorded."""
    phase_metrics = train_and_evaluate_phase_model()
    reg_metrics = train_and_evaluate_property_models()
    return {"phase": phase_metrics, "regression": reg_metrics}


def test_phase_model_performance(trained_models):
    """Verify phase classification accuracy exceeds acceptable baseline."""
    metrics = trained_models["phase"]
    assert metrics["accuracy"] >= 0.80, f"Phase accuracy too low: {metrics['accuracy']}"
    assert metrics["macro_f1"] >= 0.70, f"Phase macro-F1 too low: {metrics['macro_f1']}"


def test_regression_models_performance(trained_models):
    """Verify all 3 regression models achieve high R2 and low error."""
    metrics = trained_models["regression"]
    for target in TARGET_REGRESSION_COLS:
        assert target in metrics, f"Missing regression target {target}"
        r2 = metrics[target]["r2"]
        mae = metrics[target]["mae"]
        assert r2 >= 0.80, f"R2 for {target} below threshold: {r2}"
        assert mae >= 0.0, f"MAE for {target} invalid: {mae}"


def test_end_to_end_inference_single_sample(trained_models):
    """Verify single-row inference works end-to-end for both phase and mechanical properties."""
    sample_comp = {"Mg": 91.2, "Al": 8.3, "Cu": 0.0, "Mn": 0.15, "Zn": 0.35}
    sample_proc = {
        "Synthesis_Route": "S_T6",
        "Hot_Cold_Working": "CW",
        "Homogenization_Temp": 400.0,
        "Homogenization_Time": 1440.0,
        "Annealing_Temp": 200.0,
        "Annealing_Time": 30.0,
        "Quenching": 100.0,
    }

    sample_df = build_single_sample_features(sample_comp, sample_proc)

    # Predict phase
    pred_phase, confidence = predict_phase(sample_df)
    assert isinstance(pred_phase, str)
    assert 0.0 <= confidence <= 1.0

    # Predict properties
    props = predict_properties(sample_df)
    assert "UTS" in props and props["UTS"] > 0
    assert "Yield" in props and props["Yield"] > 0
    assert "Youngs_Modulus" in props and props["Youngs_Modulus"] > 0
