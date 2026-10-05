"""Module 3 — Process Feature Engineering.

Computes process interaction features:
- Annealing thermal budget: Annealing_Temp * Annealing_Time
- Homogenization thermal exposure: Homogenization_Temp * Homogenization_Time
- Homologous temperature: Homogenization_Temp_K / Tm_K
Combines composition, materials descriptors, and encoded process features
into the final model-ready dataset (data/processed/model_ready.csv).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_PROCESS_COLS,
    ELEMENTS,
    ENCODER_CATEGORIES_FILE,
    MODEL_READY_FILE,
    NUMERIC_PROCESS_COLS,
    SCALER_FILE,
    TARGET_CLASSIFICATION_COL,
    TARGET_REGRESSION_COLS,
)
from src.materials_features import add_materials_features, compute_all_materials_features
from src.utils import load_json, load_model


def compute_process_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer physically meaningful process interaction features."""
    df_out = df.copy()

    # Annealing thermal budget proxy: Annealing_Temp (°C) * Annealing_Time (min)
    if "Annealing_Temp" in df_out.columns and "Annealing_Time" in df_out.columns:
        df_out["Annealing_Thermal_Budget"] = (
            df_out["Annealing_Temp"] * df_out["Annealing_Time"]
        )

    # Homogenization thermal exposure: Homogenization_Temp (°C) * Homogenization_Time (min)
    if "Homogenization_Temp" in df_out.columns and "Homogenization_Time" in df_out.columns:
        df_out["Homogenization_Exposure"] = (
            df_out["Homogenization_Temp"] * df_out["Homogenization_Time"]
        )

    # Homologous temperature: (Homogenization_Temp in K) / (Tm in K)
    # Prefer dataset Tm or calc_Tm_K / calc_Tm
    tm_k_col = None
    if "calc_Tm_K" in df_out.columns:
        tm_k_col = "calc_Tm_K"
    elif "Tm" in df_out.columns:
        # Convert Tm (°C) to K
        df_out["Tm_K_temp"] = df_out["Tm"] + 273.15
        tm_k_col = "Tm_K_temp"

    if tm_k_col and "Homogenization_Temp" in df_out.columns:
        homog_k = df_out["Homogenization_Temp"] + 273.15
        df_out["Homologous_Temp"] = homog_k / df_out[tm_k_col]
        if "Tm_K_temp" in df_out.columns:
            df_out.drop(columns=["Tm_K_temp"], inplace=True)

    return df_out


def build_single_sample_features(
    composition: Dict[str, float],
    process_params: Dict[str, Any],
    encoder_categories_path: Path = ENCODER_CATEGORIES_FILE,
    scaler_path: Path = SCALER_FILE,
) -> pd.DataFrame:
    """Build exact feature vector for an inference-time single sample (e.g. from Dashboard or Optimizer).

    Args:
        composition: dict with keys 'Mg', 'Al', 'Cu', 'Mn', 'Zn' (wt.%)
        process_params: dict with process values ('Synthesis_Route', 'Hot_Cold_Working', etc.)
        encoder_categories_path: path to serialized encoder categories JSON
        scaler_path: path to fitted StandardScaler payload

    Returns:
        pd.DataFrame with a single row containing all model-ready features.
    """
    row_dict = {el: float(composition.get(el, 0.0)) for el in ELEMENTS}
    row_dict.update(process_params)

    sample_df = pd.DataFrame([row_dict])

    # 1. Materials features
    mat_feats = compute_all_materials_features(composition, is_weight_percent=True)
    for k, v in mat_feats.items():
        sample_df[k] = v

    # Also map to raw dataset column names if present
    sample_df["Density_calc"] = mat_feats["calc_Density"]
    sample_df["dHmix"] = mat_feats["calc_dHmix"]
    sample_df["dSmix"] = mat_feats["calc_dSmix"]
    sample_df["dGmix"] = mat_feats["calc_dGmix"]
    sample_df["Tm"] = mat_feats["calc_Tm"]
    sample_df["Atom_Size_Diff"] = mat_feats["calc_Atom_Size_Diff"]
    sample_df["Elect_Diff"] = mat_feats["calc_Elect_Diff"]
    sample_df["VEC"] = mat_feats["calc_VEC"]
    sample_df["Num_of_Elem"] = sum(1 for el in ELEMENTS if composition.get(el, 0.0) > 0.05)

    # 2. Process interaction features
    sample_df = compute_process_interactions(sample_df)

    # 3. Categorical one-hot encoding using saved categories
    categories = load_json(encoder_categories_path)
    for cat_col, valid_cats in categories.items():
        val = str(sample_df[cat_col].iloc[0]) if cat_col in sample_df else "Unknown"
        for cat in valid_cats:
            sample_df[f"{cat_col}_{cat}"] = 1 if val == cat else 0

    # 4. Feature scaling using saved scaler
    if scaler_path.exists():
        scaler_payload = load_model(scaler_path)
        scaler = scaler_payload["scaler"]
        num_cols = scaler_payload["numeric_cols"]
        if all(c in sample_df.columns for c in num_cols):
            scaled_vals = scaler.transform(sample_df[num_cols])
            for idx, c in enumerate(num_cols):
                sample_df[f"{c}_scaled"] = scaled_vals[0, idx]

    return sample_df


def run_process_features(
    input_df: Optional[pd.DataFrame] = None,
    output_path: Path = MODEL_READY_FILE,
) -> pd.DataFrame:
    """Run Module 3 process feature engineering and generate model_ready.csv."""
    if input_df is None:
        from src.data_preprocessing import run_data_preprocessing
        _, input_df = run_data_preprocessing()

    # Add materials features if not present
    if "calc_dSmix" not in input_df.columns:
        input_df = add_materials_features(input_df)

    # Add process interactions
    model_ready_df = compute_process_interactions(input_df)

    # Save to data/processed/model_ready.csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_ready_df.to_csv(output_path, index=False)

    return model_ready_df


if __name__ == "__main__":
    df_ready = run_process_features()
    print("Process feature engineering complete. Shape:", df_ready.shape)
