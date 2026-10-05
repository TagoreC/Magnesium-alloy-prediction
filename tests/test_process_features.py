"""Unit tests for Module 3 — Process Feature Engineering."""

import numpy as np
import pandas as pd
import pytest
from src.data_preprocessing import run_data_preprocessing
from src.process_features import (
    build_single_sample_features,
    compute_process_interactions,
    run_process_features,
)


def test_process_interactions_computation():
    """Verify calculation of thermal budget, homogenization exposure, and homologous temperature."""
    test_df = pd.DataFrame([{
        "Annealing_Temp": 200.0,
        "Annealing_Time": 60.0,
        "Homogenization_Temp": 400.0,
        "Homogenization_Time": 1440.0,
        "calc_Tm_K": 1000.0,
    }])

    res_df = compute_process_interactions(test_df)

    # Annealing_Thermal_Budget = 200 * 60 = 12000
    assert res_df["Annealing_Thermal_Budget"].iloc[0] == 12000.0
    # Homogenization_Exposure = 400 * 1440 = 576000
    assert res_df["Homogenization_Exposure"].iloc[0] == 576000.0
    # Homologous_Temp = (400 + 273.15) / 1000 = 0.67315
    assert pytest.approx(res_df["Homologous_Temp"].iloc[0], rel=1e-3) == 0.67315


def test_build_single_sample_features():
    """Verify single-sample inference feature generator produces valid non-empty DataFrame."""
    # Ensure preprocessing artifacts exist
    run_data_preprocessing()

    sample_comp = {"Mg": 92.0, "Al": 5.0, "Cu": 0.0, "Mn": 1.0, "Zn": 2.0}
    sample_proc = {
        "Synthesis_Route": "IM",
        "Hot_Cold_Working": "CW",
        "Homogenization_Temp": 400.0,
        "Homogenization_Time": 1440.0,
        "Annealing_Temp": 200.0,
        "Annealing_Time": 30.0,
        "Quenching": 100.0,
    }

    feat_row = build_single_sample_features(sample_comp, sample_proc)

    assert len(feat_row) == 1
    # Check that one-hot columns are created
    assert "Synthesis_Route_IM" in feat_row.columns
    assert feat_row["Synthesis_Route_IM"].iloc[0] == 1
    assert "Hot_Cold_Working_CW" in feat_row.columns
    assert feat_row["Hot_Cold_Working_CW"].iloc[0] == 1
    # Check that scaled features are present
    assert "Annealing_Temp_scaled" in feat_row.columns
    # Check that materials features are present
    assert "calc_dSmix" in feat_row.columns
