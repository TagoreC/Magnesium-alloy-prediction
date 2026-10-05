"""Single entry-point pipeline orchestration script.

Executes end-to-end:
1. Data Preprocessing (Module 1)
2. Materials Feature Engineering (Module 2)
3. Process Feature Engineering (Module 3)
4. Phase Classification Training & Evaluation (Module 4)
5. Mechanical Property Regressors Training & Evaluation (Module 5)
6. Explainable AI SHAP, PDP & Permutation Importance (Module 6)
7. Final Metrics Reporting & JSON validation.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_FILE, MODEL_READY_FILE
from src.data_preprocessing import run_data_preprocessing
from src.explainability import generate_all_explainability_reports
from src.materials_features import run_materials_features
from src.phase_model import train_and_evaluate_phase_model
from src.process_features import run_process_features
from src.property_models import train_and_evaluate_property_models
from src.utils import load_json


def run_full_pipeline() -> None:
    """Run all pipeline stages in sequence."""
    start_time = time.time()
    print("=" * 70)
    print("AI-Driven Digital Manufacturing Framework for Mg-Based HEAs")
    print("Pipeline Execution Started")
    print("=" * 70)

    # Stage 1: Data Preprocessing
    print("\n[1/6] Running Data Preprocessing...")
    df_cleaned, df_preprocessed = run_data_preprocessing()
    print(f"  -> Cleaned dataset: {df_cleaned.shape}, Preprocessed: {df_preprocessed.shape}")

    # Stage 2: Materials Physics Features
    print("\n[2/6] Engineering Materials Physics Descriptors...")
    df_materials = run_materials_features(df_preprocessed)
    calc_cols = [c for c in df_materials.columns if c.startswith("calc_")]
    print(f"  -> Added {len(calc_cols)} physical descriptors: {calc_cols[:5]}...")

    # Stage 3: Process Features & Final Model Matrix
    print("\n[3/6] Engineering Process Interactions & Exporting Matrix...")
    df_ready = run_process_features(df_materials)
    print(f"  -> Model-ready dataset saved to {MODEL_READY_FILE} ({df_ready.shape})")

    # Stage 4: Phase Classification Model
    print("\n[4/6] Training Phase / Microstructure Classification Model...")
    phase_metrics = train_and_evaluate_phase_model(df_ready)
    print(f"  -> Best Phase Classifier: {phase_metrics['best_model']}")
    print(f"  -> Accuracy: {phase_metrics['accuracy']:.2%}, Macro-F1: {phase_metrics['macro_f1']:.4f}")

    # Stage 5: Mechanical Property Regression Models
    print("\n[5/6] Training Mechanical Property Regressors (UTS, Yield, E)...")
    prop_metrics = train_and_evaluate_property_models(df_ready)
    for prop, m in prop_metrics.items():
        print(f"  -> {prop} ({m['algorithm']}): R2={m['r2']} +/- {m['r2_std']}, MAE={m['mae']}, RMSE={m['rmse']}")

    # Stage 6: Explainability Suite
    print("\n[6/6] Generating Explainability Suite (SHAP, Permutation, PDP)...")
    xai_res = generate_all_explainability_reports(df_ready)
    print(f"  -> Generated SHAP & PDP figures for: {list(xai_res['shap_importance'].keys())}")

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"Pipeline Successfully Completed in {elapsed}s")
    print(f"Summary Metrics Saved: {METRICS_FILE}")
    print("=" * 70)

    # Print final JSON metrics summary
    if METRICS_FILE.exists():
        final_metrics = load_json(METRICS_FILE)
        print("\nPipeline Summary JSON:")
        import json
        print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    run_full_pipeline()
