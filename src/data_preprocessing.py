"""Module 1 — Data Preprocessing Pipeline.

Handles raw dataset loading, schema standardization, missing-value imputation,
duplicate detection, categorical encoding, and feature scaling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    CATEGORICAL_PROCESS_COLS,
    CLEANED_DATA_FILE,
    ELEMENTS,
    ENCODER_CATEGORIES_FILE,
    METRICS_FILE,
    MODEL_READY_FILE,
    NUMERIC_PROCESS_COLS,
    RAW_DATA_FILE,
    RAW_TO_CANONICAL_COLUMNS,
    SCALER_FILE,
    TARGET_CLASSIFICATION_COL,
    TARGET_REGRESSION_COLS,
)
from src.utils import save_json, update_metrics_json, save_model


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load raw dataset from CSV or Excel file."""
    if filepath is None:
        filepath = RAW_DATA_FILE

    if not filepath.exists():
        # Fallback to xlsx if csv not found
        xlsx_fallback = filepath.with_suffix(".xlsx")
        if xlsx_fallback.exists():
            filepath = xlsx_fallback
        else:
            raise FileNotFoundError(f"Raw data file not found at {filepath}")

    if filepath.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean whitespace from column names and apply canonical naming."""
    # Strip whitespace from columns
    df = df.rename(columns=lambda col: col.strip())

    # Map to canonical names
    rename_map = {
        col: RAW_TO_CANONICAL_COLUMNS[col]
        for col in df.columns
        if col in RAW_TO_CANONICAL_COLUMNS
    }
    df = df.rename(columns=rename_map)

    # Standardize string values in categorical columns
    string_cols = df.select_dtypes(include=["object"]).columns
    for sc in string_cols:
        df[sc] = (
            df[sc]
            .astype(str)
            .str.strip()
            .str.replace(r"\s*,\s*", ",", regex=True)
        )

    return df


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate and return percentage of missing values per column."""
    missing_pct = (df.isnull().sum() / len(df) * 100.0).to_dict()
    return {k: round(float(v), 2) for k, v in missing_pct.items()}


def clean_and_impute_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Perform validation, drop invalid alloy records, and impute missing process features."""
    initial_count = len(df)

    # Validate compositions
    valid_comp_mask = df[ELEMENTS].notnull().all(axis=1) & (df[ELEMENTS].sum(axis=1) > 0)
    df = df[valid_comp_mask].copy()

    # Continuous process parameter imputation: median grouped by Synthesis_Route if possible
    for num_col in NUMERIC_PROCESS_COLS:
        if num_col in df.columns:
            if "Synthesis_Route" in df.columns:
                group_median = df.groupby("Synthesis_Route")[num_col].transform("median")
                df[num_col] = df[num_col].fillna(group_median)
            df[num_col] = df[num_col].fillna(df[num_col].median())

    # Categorical process parameter imputation: mode or 'Unknown'
    for cat_col in CATEGORICAL_PROCESS_COLS:
        if cat_col in df.columns:
            mode_val = df[cat_col].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[cat_col] = df[cat_col].fillna(fill_val)

    report = {
        "initial_rows": initial_count,
        "retained_rows": len(df),
        "dropped_invalid_rows": initial_count - len(df),
    }
    return df, report


def detect_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Detect and drop exact duplicates; flag near-duplicates (same composition + process)."""
    exact_duplicates = int(df.duplicated().sum())
    df_dedup = df.drop_duplicates().copy()

    # Near duplicates: same composition + process, differing targets
    key_features = [col for col in (ELEMENTS + CATEGORICAL_PROCESS_COLS + NUMERIC_PROCESS_COLS) if col in df_dedup.columns]
    near_dupes_mask = df_dedup.duplicated(subset=key_features, keep=False)
    near_duplicates_count = int(near_dupes_mask.sum())

    dup_report = {
        "exact_duplicates_dropped": exact_duplicates,
        "near_duplicates_flagged": near_duplicates_count,
    }
    return df_dedup, dup_report


def fit_and_save_scaler(
    df: pd.DataFrame, numeric_cols: List[str], scaler_path: Path = SCALER_FILE
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on continuous numeric features and save to disk."""
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[numeric_cols])

    # Build scaled dataframe
    df_scaled = df.copy()
    for idx, col in enumerate(numeric_cols):
        df_scaled[f"{col}_scaled"] = scaled_values[:, idx]

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(
        {"scaler": scaler, "numeric_cols": numeric_cols},
        scaler_path,
    )
    return df_scaled, scaler


def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: List[str] = CATEGORICAL_PROCESS_COLS,
    categories_path: Path = ENCODER_CATEGORIES_FILE,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """One-hot encode categorical features and save category lists for deterministic inference."""
    encoder_categories = {}
    df_encoded = df.copy()

    for col in cat_cols:
        if col in df.columns:
            # Sorted unique categories
            cats = sorted([str(val) for val in df[col].dropna().unique()])
            encoder_categories[col] = cats

            # Create deterministic one-hot columns
            for cat in cats:
                df_encoded[f"{col}_{cat}"] = (df[col].astype(str) == cat).astype(int)

    categories_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(encoder_categories, categories_path)
    return df_encoded, encoder_categories


def run_data_preprocessing(
    raw_file: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete Module 1 pipeline."""
    # 1. Load raw data
    raw_df = load_raw_data(raw_file)

    # 2. Standardize column names and text values
    std_df = standardize_columns(raw_df)

    # 3. Missing-value analysis
    missing_report = analyze_missing_values(std_df)

    # 4. Clean & Impute
    clean_df, clean_report = clean_and_impute_data(std_df)

    # 5. Duplicate detection
    dedup_df, dup_report = detect_duplicates(clean_df)

    # Save interim cleaned dataset
    CLEANED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    dedup_df.to_csv(CLEANED_DATA_FILE, index=False)

    # 6. Categorical encoding
    encoded_df, encoder_categories = encode_categoricals(dedup_df)

    # 7. Feature scaling for numeric process parameters
    available_numeric = [col for col in NUMERIC_PROCESS_COLS if col in encoded_df.columns]
    scaled_df, scaler = fit_and_save_scaler(encoded_df, available_numeric)

    # Log metrics
    update_metrics_json(
        METRICS_FILE,
        "preprocessing",
        {
            "missing_value_report": missing_report,
            "cleaning_report": clean_report,
            "duplicate_report": dup_report,
            "encoder_categories": encoder_categories,
            "total_processed_samples": len(scaled_df),
        },
    )

    return dedup_df, scaled_df


if __name__ == "__main__":
    df_interim, df_processed = run_data_preprocessing()
    print(f"Data preprocessing complete. Cleaned: {df_interim.shape}, Processed: {df_processed.shape}")
