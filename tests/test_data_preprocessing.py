"""Unit tests for Module 1 — Data Preprocessing."""

import numpy as np
import pandas as pd
import pytest
from src.config import ELEMENTS, NUMERIC_PROCESS_COLS, SCALER_FILE
from src.data_preprocessing import (
    clean_and_impute_data,
    fit_and_save_scaler,
    load_raw_data,
    run_data_preprocessing,
    standardize_columns,
)
from src.utils import load_model


def test_composition_sum_tolerance():
    """Verify composition rows sum to ~100 (within +-0.5% tolerance)."""
    raw_df = load_raw_data()
    std_df = standardize_columns(raw_df)
    comp_sums = std_df[ELEMENTS].sum(axis=1)

    assert len(comp_sums) > 0
    assert np.all(np.abs(comp_sums - 100.0) <= 0.5), (
        f"Alloy compositions outside tolerance: min={comp_sums.min()}, max={comp_sums.max()}"
    )


def test_no_nans_in_feature_columns_post_pipeline(tmp_path):
    """Verify that feature columns contain zero NaNs post-pipeline."""
    interim_df, processed_df = run_data_preprocessing()

    feature_cols = [
        col for col in processed_df.columns
        if col not in ["Structure", "Phases", "UTS", "Yield", "Youngs_Modulus", "Multiphase", "IM_Structure"]
    ]

    nan_counts = processed_df[feature_cols].isna().sum()
    assert nan_counts.sum() == 0, f"Found NaNs in feature columns: {nan_counts[nan_counts > 0]}"


def test_scaler_normalization():
    """Verify standard scaler produces mean approximately 0 and standard deviation 1."""
    interim_df, processed_df = run_data_preprocessing()
    scaler_payload = load_model(SCALER_FILE)
    scaler = scaler_payload["scaler"]
    numeric_cols = scaler_payload["numeric_cols"]

    scaled_cols = [f"{col}_scaled" for col in numeric_cols]
    scaled_matrix = processed_df[scaled_cols].values

    means = np.mean(scaled_matrix, axis=0)
    stds = np.std(scaled_matrix, axis=0)

    assert np.allclose(means, 0.0, atol=1e-3), f"Scaled means deviate from 0: {means}"
    assert np.allclose(stds, 1.0, atol=1e-3), f"Scaled stds deviate from 1: {stds}"
