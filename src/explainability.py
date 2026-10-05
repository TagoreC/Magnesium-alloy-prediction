"""Module 6 — Explainable AI (XAI).

Computes SHAP explanations (TreeExplainer), Permutation Importance,
Partial Dependence Plots (PDP), and generates an automated insights summary:
reports/explainability_summary.md and reports/figures/*_shap_summary.png.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import shap

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, METRICS_FILE, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR
from src.phase_model import PHASE_MODEL_FILE, get_feature_columns
from src.process_features import run_process_features
from src.property_models import MODEL_FILES
from src.utils import load_model, update_metrics_json

EXPLAINABILITY_SUMMARY_FILE = REPORTS_DIR / "explainability_summary.md"


def compute_shap_explanations(
    model: Any,
    X: pd.DataFrame,
    model_name: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute SHAP values using TreeExplainer and save global summary beeswarm/bar plot."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Use TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Multi-class output handling
    if isinstance(shap_values, list):
        # Average absolute SHAP values across classes
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        shap_plot_vals = shap_values[0]
    elif len(np.shape(shap_values)) == 3:
        # Shape: (samples, features, classes)
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        shap_plot_vals = shap_values[:, :, 0]
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_plot_vals = shap_values

    # Feature importance mapping
    feature_importance = {
        col: round(float(val), 6)
        for col, val in zip(X.columns, mean_abs_shap)
    }
    sorted_importance = dict(
        sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
    )

    # Plot summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_plot_vals, X, show=False, max_display=12)
    plt.title(f"SHAP Global Feature Attribution — {model_name}", fontsize=12, pad=15)
    plt.tight_layout()
    summary_plot_path = FIGURES_DIR / f"{model_name.lower()}_shap_summary.png"
    plt.savefig(summary_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    return shap_values, sorted_importance


def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
) -> Dict[str, float]:
    """Compute permutation importance on dataset as independent verification."""
    result = permutation_importance(
        model, X, y, n_repeats=5, random_state=42, n_jobs=1
    )
    perm_dict = {
        col: round(float(val), 6)
        for col, val in zip(X.columns, result.importances_mean)
    }
    return dict(sorted(perm_dict.items(), key=lambda item: item[1], reverse=True))


def generate_partial_dependence_plots(
    model: Any,
    X: pd.DataFrame,
    top_features: List[str],
    model_name: str,
) -> List[str]:
    """Generate Partial Dependence Plots (PDP) for top influential features."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for feat in top_features[:4]:
        if feat not in X.columns:
            continue
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            PartialDependenceDisplay.from_estimator(
                model, X, [feat], ax=ax, line_kw={"color": "#1f77b4", "linewidth": 2}
            )
            ax.set_title(f"PDP: {feat} on {model_name}", fontsize=11)
            plt.tight_layout()
            pdp_path = FIGURES_DIR / f"{model_name.lower()}_pdp_{feat}.png"
            plt.savefig(pdp_path, dpi=200, bbox_inches="tight")
            plt.close()
            saved_paths.append(str(pdp_path))
        except Exception as e:
            # Skip PDP if categorical or incompatible
            continue

    return saved_paths


def explain_single_prediction(
    model_name: str,
    sample_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Generate local SHAP explanation for an individual prediction."""
    if model_name in MODEL_FILES:
        payload = load_model(MODEL_FILES[model_name])
    elif model_name.lower() in ["phase", "structure"]:
        payload = load_model(PHASE_MODEL_FILE)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = payload["model"]
    feature_cols = payload["feature_cols"]
    X_sample = sample_df.reindex(columns=feature_cols, fill_value=0.0)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)

    if isinstance(shap_vals, list):
        # Predict class index
        pred_idx = model.predict(X_sample)[0]
        local_vals = shap_vals[pred_idx][0]
    elif len(np.shape(shap_vals)) == 3:
        pred_idx = model.predict(X_sample)[0]
        local_vals = shap_vals[0, :, pred_idx]
    else:
        local_vals = shap_vals[0]

    contributions = {
        col: round(float(val), 4)
        for col, val in zip(feature_cols, local_vals)
    }
    sorted_contrib = dict(
        sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    )

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[0])
    else:
        base_val = float(base_val)

    return {
        "base_value": base_val,
        "feature_contributions": sorted_contrib,
    }


def generate_all_explainability_reports(
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Execute complete explainability suite across all models and generate markdown summary."""
    if df is None:
        df = run_process_features()

    summary_records = {}
    shap_metrics = {}
    pdp_records = {}
    perm_metrics = {}

    # 1. Regressors (UTS, Yield, Youngs_Modulus)
    for target, model_file in MODEL_FILES.items():
        if not model_file.exists():
            continue
        payload = load_model(model_file)
        model = payload["model"]
        feature_cols = payload["feature_cols"]

        valid_mask = df[target].notnull()
        X = df.loc[valid_mask, feature_cols].copy()
        y = df.loc[valid_mask, target].astype(float).copy()

        # SHAP
        _, shap_imp = compute_shap_explanations(model, X, target)
        shap_metrics[target] = shap_imp

        # Permutation Importance
        perm_imp = compute_permutation_importance(model, X, y, target)
        perm_metrics[target] = perm_imp

        # PDP
        top_4 = list(shap_imp.keys())[:4]
        pdp_paths = generate_partial_dependence_plots(model, X, top_4, target)
        pdp_records[target] = pdp_paths

        summary_records[target] = {
            "top_shap": list(shap_imp.items())[:5],
            "top_perm": list(perm_imp.items())[:5],
        }

    # 2. Phase Classifier
    if PHASE_MODEL_FILE.exists():
        phase_payload = load_model(PHASE_MODEL_FILE)
        phase_model = phase_payload["model"]
        phase_cols = phase_payload["feature_cols"]
        from src.config import TARGET_CLASSIFICATION_COL
        from sklearn.preprocessing import LabelEncoder

        valid_mask = df[TARGET_CLASSIFICATION_COL].notnull()
        X_phase = df.loc[valid_mask, phase_cols].copy()
        le = LabelEncoder()
        y_phase = le.fit_transform(df.loc[valid_mask, TARGET_CLASSIFICATION_COL].astype(str).str.strip())

        _, phase_shap_imp = compute_shap_explanations(phase_model, X_phase, "Phase")
        shap_metrics["Phase"] = phase_shap_imp

        phase_perm = compute_permutation_importance(phase_model, X_phase, y_phase, "Phase")
        perm_metrics["Phase"] = phase_perm

        top_4_phase = list(phase_shap_imp.keys())[:4]
        pdp_records["Phase"] = generate_partial_dependence_plots(phase_model, X_phase, top_4_phase, "Phase")

        summary_records["Phase"] = {
            "top_shap": list(phase_shap_imp.items())[:5],
            "top_perm": list(phase_perm.items())[:5],
        }

    # Write explainability_summary.md
    md_content = [
        "# Explainable AI (XAI) Synthesis Report: Drivers of Phase & Mechanical Properties\n",
        "This report identifies which manufacturing parameters and materials physics descriptors ",
        "most strongly dictate microstructure and mechanical performance in Mg-based High-Entropy Alloys.\n\n",
        "## Key Influential Factors by Target\n",
    ]

    for target, rec in summary_records.items():
        md_content.append(f"### {target} Key Drivers\n")
        md_content.append("| Rank | SHAP Feature | Mean |SHAP| | Permutation Feature | Score |\n")
        md_content.append("|---|---|---|---|---|\n")
        for rank in range(min(5, len(rec["top_shap"]))):
            s_feat, s_val = rec["top_shap"][rank]
            p_feat, p_val = rec["top_perm"][rank]
            md_content.append(f"| {rank+1} | `{s_feat}` | {s_val:.4f} | `{p_feat}` | {p_val:.4f} |\n")
        md_content.append("\n")

    md_content.append("## Physical Interpretation\n")
    md_content.append(
        "- **Elemental Composition**: The balance of Mg content versus transition metals (Cu, Mn, Zn) "
        "and Al fundamentally governs both phase stability and solid-solution / intermetallic strengthening.\n"
        "- **Thermodynamic Descriptors**: Mixing enthalpy (ΔH_mix), mixing entropy (ΔS_mix), and Valence Electron "
        "Concentration (VEC) serve as primary discriminators between single-phase solid solutions, intermetallics, and quasicrystals.\n"
        "- **Process Parameters**: Thermal budget (`Annealing_Thermal_Budget`), homogenization exposure, "
        "and quenching cooling rates strongly dictate precipitation kinetics, grain boundary pinning, and resultant UTS / Yield strength.\n"
    )

    EXPLAINABILITY_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPLAINABILITY_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.writelines(md_content)

    # Save to metrics.json
    update_metrics_json(
        METRICS_FILE,
        "explainability",
        {
            "shap_importance": shap_metrics,
            "permutation_importance": perm_metrics,
            "pdp_figures": pdp_records,
            "summary_report_path": str(EXPLAINABILITY_SUMMARY_FILE),
        },
    )

    return {
        "shap_importance": shap_metrics,
        "permutation_importance": perm_metrics,
        "pdp_figures": pdp_records,
    }


if __name__ == "__main__":
    res = generate_all_explainability_reports()
    print("Explainability generation complete.")
