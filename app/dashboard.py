"""Module 8 — Interactive Digital Manufacturing Dashboard (Streamlit).

AI-Driven Digital Manufacturing Framework for Phase and Mechanical Property
Prediction & Inverse Design Optimization of Mg-Based High-Entropy Alloys.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Setup project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ELEMENTS,
    FIGURES_DIR,
    METRICS_FILE,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.explainability import explain_single_prediction
from src.materials_features import calculate_density, compute_all_materials_features
from src.optimization import optimize_alloy_and_process
from src.phase_model import PHASE_MODEL_FILE, predict_phase
from src.process_features import build_single_sample_features
from src.property_models import MODEL_FILES, predict_properties
from src.utils import load_json

# Page Configuration
st.set_page_config(
    page_title="Mg-HEA Digital Manufacturing Framework",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .simplex-ok {
        padding: 8px 12px;
        border-radius: 6px;
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        font-weight: 600;
    }
    .simplex-warn {
        padding: 8px 12px;
        border-radius: 6px;
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Header & Quick Navigation
st.sidebar.image(
    "https://img.icons8.com/fluency/96/atom.png",
    width=64,
)
st.sidebar.title("Mg-HEA Studio")
st.sidebar.markdown("**Digital Manufacturing & Property Prediction**")
st.sidebar.markdown("---")

# Quick Preset Compositions
st.sidebar.subheader("Presets / Benchmarks")
preset_choice = st.sidebar.selectbox(
    "Load Benchmark Alloy:",
    [
        "Custom (Manual Input)",
        "Equiatomic HEA (20Mg-20Al-20Cu-20Mn-20Zn)",
        "Mg-Rich Baseline AZ91 (91.2Mg-8.3Al-0.15Mn-0.35Zn)",
        "High-Strength Medium HEA (50Mg-12.5Al-12.5Cu-12.5Mn-12.5Zn)",
        "Ultra-Lightweight HEA (80Mg-10Al-5Cu-5Mn)",
    ],
)

presets = {
    "Equiatomic HEA (20Mg-20Al-20Cu-20Mn-20Zn)": {"Mg": 20.0, "Al": 20.0, "Cu": 20.0, "Mn": 20.0, "Zn": 20.0},
    "Mg-Rich Baseline AZ91 (91.2Mg-8.3Al-0.15Mn-0.35Zn)": {"Mg": 91.2, "Al": 8.3, "Cu": 0.0, "Mn": 0.15, "Zn": 0.35},
    "High-Strength Medium HEA (50Mg-12.5Al-12.5Cu-12.5Mn-12.5Zn)": {"Mg": 50.0, "Al": 12.5, "Cu": 12.5, "Mn": 12.5, "Zn": 12.5},
    "Ultra-Lightweight HEA (80Mg-10Al-5Cu-5Mn)": {"Mg": 80.0, "Al": 10.0, "Cu": 5.0, "Mn": 5.0, "Zn": 0.0},
}

default_comp = presets.get(preset_choice, {"Mg": 91.2, "Al": 8.3, "Cu": 0.0, "Mn": 0.15, "Zn": 0.35})

# Main Header
st.markdown("<div class='main-title'>AI-Driven Digital Manufacturing Framework</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Microstructure Phase & Mechanical Property Prediction, Explainable AI, and Inverse Design Optimization for Mg-Based High-Entropy Alloys</div>",
    unsafe_allow_html=True,
)

# Tabs
tab_predict, tab_explain, tab_optimize, tab_metrics = st.tabs(
    ["🔬 Property & Phase Prediction", "💡 Explainability (XAI)", "🎯 Inverse Design Optimizer", "📊 Model Metrics & Reports"]
)

with tab_predict:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("1. Alloy Composition (wt.%)")
        st.caption("Adjust elemental proportions. Total must sum to 100%.")

        c_mg = st.slider("Magnesium (Mg)", min_value=0.0, max_value=100.0, value=float(default_comp["Mg"]), step=0.1)
        c_al = st.slider("Aluminum (Al)", min_value=0.0, max_value=40.0, value=float(default_comp["Al"]), step=0.1)
        c_cu = st.slider("Copper (Cu)", min_value=0.0, max_value=40.0, value=float(default_comp["Cu"]), step=0.1)
        c_mn = st.slider("Manganese (Mn)", min_value=0.0, max_value=40.0, value=float(default_comp["Mn"]), step=0.05)
        c_zn = st.slider("Zinc (Zn)", min_value=0.0, max_value=40.0, value=float(default_comp["Zn"]), step=0.05)

        total_comp = round(c_mg + c_al + c_cu + c_mn + c_zn, 2)
        comp_dict = {"Mg": c_mg, "Al": c_al, "Cu": c_cu, "Mn": c_mn, "Zn": c_zn}

        if abs(total_comp - 100.0) <= 0.2:
            st.markdown(
                f"<div class='simplex-ok'>✓ Total Composition: {total_comp:.2f}% (Valid Simplex)</div>",
                unsafe_allow_html=True,
            )
        else:
            diff = 100.0 - total_comp
            st.markdown(
                f"<div class='simplex-warn'>⚠ Total Composition: {total_comp:.2f}% (Deviation: {diff:+.2f}%). Please adjust to 100%.</div>",
                unsafe_allow_html=True,
            )
            if st.button("Auto-Balance to 100% (Adjust Mg)"):
                st.session_state["adjusted_mg"] = max(0.0, round(c_mg + diff, 2))
                st.rerun()

    with col_right:
        st.subheader("2. Manufacturing Process Parameters")
        st.caption("Set thermomechanical processing and heat-treatment conditions.")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            synth_route = st.selectbox("Synthesis Route:", ["IM", "DC", "S_T6", "S_T4", "DMD"], index=2)
            hot_cold = st.selectbox("Working Mode:", ["CW (Cold Work)", "CR (Cold Roll)"], index=0)
            work_type = "CW" if "CW" in hot_cold else "CR"
            quench_rate = st.select_slider(
                "Quenching Rate (°C/s):",
                options=[8.333, 100.0, 500.0, 1000.0],
                value=100.0,
            )

        with p_col2:
            homog_temp = st.number_input("Homogenization Temp (°C):", min_value=150.0, max_value=550.0, value=400.0, step=10.0)
            homog_time = st.number_input("Homogenization Time (min):", min_value=30.0, max_value=2880.0, value=1440.0, step=60.0)
            anneal_temp = st.number_input("Annealing Temp (°C):", min_value=150.0, max_value=450.0, value=200.0, step=10.0)
            anneal_time = st.number_input("Annealing Time (min):", min_value=10.0, max_value=240.0, value=30.0, step=10.0)

        process_dict = {
            "Synthesis_Route": synth_route,
            "Hot_Cold_Working": work_type,
            "Homogenization_Temp": homog_temp,
            "Homogenization_Time": homog_time,
            "Annealing_Temp": anneal_temp,
            "Annealing_Time": anneal_time,
            "Quenching": quench_rate,
        }

    st.markdown("---")

    # Predict Trigger
    predict_btn = st.button("🚀 Run Prediction & Characterization", type="primary", use_container_width=True)

    if predict_btn or "last_prediction" in st.session_state:
        # Build features
        sample_df = build_single_sample_features(comp_dict, process_dict)

        # Predict phase & properties
        pred_phase, phase_conf = predict_phase(sample_df)
        props = predict_properties(sample_df)
        density = calculate_density(comp_dict, is_weight_fraction=True)

        uts = props.get("UTS", 0.0)
        yield_str = props.get("Yield", 0.0)
        e_mod = props.get("Youngs_Modulus", 0.0)
        spec_strength = round(uts / density, 1) if density > 0 else 0.0
        spec_mod = round(e_mod / density, 1) if density > 0 else 0.0

        st.session_state["last_prediction"] = {
            "sample_df": sample_df,
            "phase": pred_phase,
            "phase_conf": phase_conf,
            "props": props,
            "density": density,
            "spec_strength": spec_strength,
            "spec_mod": spec_mod,
        }

        st.subheader("3. Predicted Material Characterization")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.markdown(
                f"<div class='metric-box'><div class='metric-label'>Microstructure Phase</div><div class='metric-val' style='font-size:1.1rem; color:#1E40AF;'>{pred_phase}</div><small>Confidence: {phase_conf:.0%}</small></div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"<div class='metric-box'><div class='metric-label'>UTS</div><div class='metric-val' style='color:#059669;'>{uts:.0f}</div><small>MPa</small></div>",
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"<div class='metric-box'><div class='metric-label'>Yield Strength</div><div class='metric-val' style='color:#D97706;'>{yield_str:.0f}</div><small>MPa</small></div>",
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"<div class='metric-box'><div class='metric-label'>Young's Modulus</div><div class='metric-val' style='color:#7C3AED;'>{e_mod:.1f}</div><small>GPa</small></div>",
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                f"<div class='metric-box'><div class='metric-label'>Density (ρ)</div><div class='metric-val'>{density:.3f}</div><small>g/cm³</small></div>",
                unsafe_allow_html=True,
            )
        with m6:
            st.markdown(
                f"<div class='metric-box'><div class='metric-label'>Specific Strength</div><div class='metric-val'>{spec_strength:.0f}</div><small>MPa·cm³/g</small></div>",
                unsafe_allow_html=True,
            )

        # Physics Descriptors Expander
        with st.expander("🔍 View Computed Physics-Informed Descriptors"):
            mat_desc = compute_all_materials_features(comp_dict, is_weight_percent=True)
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Mixing Entropy (ΔS_mix)", f"{mat_desc['calc_dSmix']:.2f} J/mol·K")
            d1.metric("Atomic Size Diff (δ)", f"{mat_desc['calc_Atom_Size_Diff']:.2f} %")
            d2.metric("Mixing Enthalpy (ΔH_mix)", f"{mat_desc['calc_dHmix']:.1f} J/mol")
            d2.metric("Electronegativity Diff (Δχ)", f"{mat_desc['calc_Elect_Diff']:.3f}")
            d3.metric("Melting Point (Tm)", f"{mat_desc['calc_Tm']:.1f} °C")
            d3.metric("VEC", f"{mat_desc['calc_VEC']:.2f}")
            d4.metric("Gibbs Energy (ΔG_mix @1573K)", f"{mat_desc['calc_dGmix']:.0f} J/mol")
            homog_exposure = homog_temp * homog_time
            d4.metric("Homogenization Exposure", f"{homog_exposure:,.0f} °C·min")

# Tab 2: Explainability
with tab_explain:
    st.subheader("Explainable AI (XAI) Attribution")
    st.caption("Inspect local SHAP contributions for the current alloy and global feature significance across the entire domain.")

    if "last_prediction" in st.session_state:
        target_to_explain = st.selectbox(
            "Select Property Target for Local SHAP Attribution:",
            ["UTS", "Yield", "Youngs_Modulus"],
        )
        try:
            local_shap = explain_single_prediction(
                target_to_explain, st.session_state["last_prediction"]["sample_df"]
            )
            top_contrib = list(local_shap["feature_contributions"].items())[:8]

            feat_names = [f[0] for f in reversed(top_contrib)]
            feat_vals = [f[1] for f in reversed(top_contrib)]
            colors = ["#10B981" if v > 0 else "#EF4444" for v in feat_vals]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(feat_names, feat_vals, color=colors)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(f"Local SHAP Feature Contributions for {target_to_explain} (Base: {local_shap['base_value']:.1f})")
            ax.set_xlabel("SHAP Value (Impact on Prediction)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"Could not render local SHAP plot: {e}")
    else:
        st.info("Run a prediction in Tab 1 to view local SHAP attributions.")

    st.markdown("---")
    st.subheader("Global Explanations & Partial Dependence")
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("**Global SHAP Summary Beeswarm**")
        shap_choice = st.selectbox("Model Summary:", ["uts", "yield", "youngs_modulus", "phase"])
        shap_img_path = FIGURES_DIR / f"{shap_choice}_shap_summary.png"
        if shap_img_path.exists():
            st.image(str(shap_img_path), use_container_width=True)
        else:
            st.info("SHAP summary plot will appear after running the pipeline.")

    with g_col2:
        st.markdown("**Phase Confusion Matrix & PDP**")
        cm_path = FIGURES_DIR / "phase_confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Confusion matrix will appear after training.")

# Tab 3: Inverse Design Optimization
with tab_optimize:
    st.subheader("Inverse Design & Manufacturing Optimization")
    st.caption("Discover optimal compositions and thermomechanical conditions maximizing structural and lightweight performance.")

    opt_col1, opt_col2 = st.columns([1, 2], gap="large")

    with opt_col1:
        st.markdown("**1. Objective & Weights**")
        obj_type = st.selectbox(
            "Objective Type:",
            [
                ("weighted_properties", "Standard Weighted Properties (UTS + Yield + E)"),
                ("specific_strength", "Maximize Specific Strength (UTS / Density)"),
                ("specific_modulus", "Maximize Specific Modulus (E / Density)"),
                ("combined_specific", "Combined Lightweight Performance"),
            ],
            format_func=lambda x: x[1],
        )[0]

        w_uts = st.slider("Weight: UTS", 0.0, 1.0, 0.33, 0.05)
        w_yield = st.slider("Weight: Yield Strength", 0.0, 1.0, 0.33, 0.05)
        w_e = st.slider("Weight: Young's Modulus", 0.0, 1.0, 0.34, 0.05)

        # Normalize weights
        w_sum = w_uts + w_yield + w_e
        if w_sum > 0:
            w_uts, w_yield, w_e = w_uts / w_sum, w_yield / w_sum, w_e / w_sum

        apply_density_cap = st.checkbox("Enforce Maximum Density Cap (Lightweighting)")
        max_rho = None
        if apply_density_cap:
            max_rho = st.slider("Max Density (g/cm³):", 1.8, 3.5, 2.5, 0.1)

        run_opt_btn = st.button("⚡ Run Multi-Objective Optimizer", type="primary", use_container_width=True)

    with opt_col2:
        st.markdown("**2. Top Recommended Alloy & Process Candidates**")
        if run_opt_btn:
            with st.spinner("Searching combinatorial composition & processing space via Differential Evolution..."):
                candidates = optimize_alloy_and_process(
                    w_uts=w_uts,
                    w_yield=w_yield,
                    w_e=w_e,
                    objective_type=obj_type,
                    max_density=max_rho,
                    n_candidates=5,
                    pop_size=30,
                    generations=15,
                )

            if candidates:
                cand_df = pd.DataFrame(candidates)
                # Display table
                display_cols = [
                    "Mg", "Al", "Cu", "Mn", "Zn",
                    "Synthesis_Route", "Hot_Cold_Working", "Homogenization_Temp", "Annealing_Temp",
                    "Predicted_Phase", "Predicted_UTS", "Predicted_Yield", "Predicted_E",
                    "Calculated_Density", "Specific_Strength", "Objective_Score",
                ]
                st.dataframe(
                    cand_df[display_cols].style.format({
                        "Mg": "{:.1f}%",
                        "Al": "{:.1f}%",
                        "Cu": "{:.1f}%",
                        "Mn": "{:.1f}%",
                        "Zn": "{:.1f}%",
                        "Homogenization_Temp": "{:.0f}°C",
                        "Annealing_Temp": "{:.0f}°C",
                        "Predicted_UTS": "{:.0f} MPa",
                        "Predicted_Yield": "{:.0f} MPa",
                        "Predicted_E": "{:.1f} GPa",
                        "Calculated_Density": "{:.3f} g/cm³",
                        "Specific_Strength": "{:.0f}",
                        "Objective_Score": "{:.1f}",
                    }),
                    use_container_width=True,
                )
                st.success("Found 5 distinct high-performing candidate alloys satisfying the composition simplex constraint.")
            else:
                st.error("No valid candidates found within constraints.")
        else:
            st.info("Configure weights on the left and click 'Run Multi-Objective Optimizer' to discover candidate alloys.")

# Tab 4: Metrics & Reports
with tab_metrics:
    st.subheader("Model Validation Metrics & Experiment Logs")
    if METRICS_FILE.exists():
        metrics_data = load_json(METRICS_FILE)
        st.json(metrics_data)
    else:
        st.info("Metrics report will be populated after executing run_pipeline.py.")
