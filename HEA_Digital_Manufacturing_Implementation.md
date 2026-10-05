# AI-Driven Digital Manufacturing Framework for Phase and Mechanical Property Prediction of Mg-Based High-Entropy Alloys

**Purpose of this document:** This is a complete implementation specification intended to be handed to an autonomous coding agent (e.g. Antigravity) to build the project end-to-end with minimal further clarification. It defines scope, data contracts, module-by-module logic, model architecture, file structure, and acceptance criteria.

---

## 1. Project Summary

Build a machine-learning pipeline + interactive dashboard that, given the **composition** and **manufacturing process parameters** of a Mg-based High-Entropy Alloy (HEA), predicts:

1. **Phase / microstructure classification**
2. **Mechanical properties**: Ultimate Tensile Strength (UTS), Yield Strength, Young's Modulus (E)
3. **Explainability** of which features drive predictions (SHAP, permutation importance, PDP)
4. **Optimized manufacturing/composition recommendations** that maximize a weighted objective of UTS, Yield, and E (optionally normalized by density for specific strength/modulus)

Final deliverable: an interactive dashboard where a user inputs composition + process parameters and receives predicted phase, mechanical properties, and optimization suggestions.

---

## 2. Tech Stack

- **Language:** Python 3.10+
- **Data handling:** pandas, numpy
- **ML:** scikit-learn, xgboost (or lightgbm) for classification/regression
- **Explainability:** shap
- **Optimization:** scipy.optimize, or a simple genetic algorithm (deap) / grid search over discretized process space
- **Dashboard:** Streamlit (fastest path to an interactive UI matching the mockup)
- **Experiment tracking (optional):** simple CSV/JSON logs; no need for MLflow unless requested
- **Testing:** pytest for data-pipeline and feature-engineering unit tests

---

## 3. Repository Structure

```
hea-digital-manufacturing/
│
├── data/
│   ├── raw/                      # original dataset(s), untouched
│   ├── interim/                  # after cleaning/imputation
│   └── processed/                # final feature matrix ready for modeling
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # column names, constants, physical property tables
│   ├── data_preprocessing.py     # Module 1
│   ├── materials_features.py     # Module 2
│   ├── process_features.py       # Module 3
│   ├── phase_model.py            # Module 4
│   ├── property_models.py        # Module 5
│   ├── explainability.py         # Module 6
│   ├── optimization.py           # Module 7
│   └── utils.py
│
├── models/                       # serialized trained models (.pkl / .joblib)
│
├── notebooks/                    # exploratory analysis (optional, not production path)
│
├── app/
│   └── dashboard.py               # Streamlit app (Module 8 / Final Application)
│
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_materials_features.py
│   ├── test_process_features.py
│   └── test_models.py
│
├── reports/
│   ├── figures/                  # SHAP plots, PDPs, confusion matrices
│   └── metrics.json              # final evaluation numbers
│
├── requirements.txt
├── README.md
└── run_pipeline.py                # single entry point: preprocessing → training → evaluation → save
```

---

## 4. Dataset Contract

The agent should first inspect the actual dataset (CSV) supplied by the user and map its columns to the schema below. If a column is missing, note it in `data/raw/COLUMN_MAPPING.md` and proceed with the columns that exist.

**Expected raw columns (rename to these canonical names in `config.py`):**

| Canonical name | Type | Description |
|---|---|---|
| `Mg, Al, Cu, Mn, Zn, ...` (one column per element) | float (at.% or wt.%) | Composition, must sum to 100 |
| `Synthesis_Route` | categorical | e.g. Casting, Powder Metallurgy, Additive Manufacturing |
| `Hot_Cold_Working` | categorical | e.g. Hot Rolled, Cold Rolled, As-Cast |
| `Quenching` | categorical/boolean | Yes/No or quench medium |
| `Homogenization_Temp` | float (°C) | continuous |
| `Homogenization_Time` | float (h) | continuous |
| `Annealing_Temp` | float (°C) | continuous |
| `Annealing_Time` | float (h) | continuous |
| `Density` | float (g/cm³) | measured or estimated |
| `Phase` / `Structure` | categorical target | e.g. FCC, BCC, HCP, dual-phase |
| `UTS` | float (MPa) | target |
| `Yield` | float (MPa) | target |
| `Youngs_Modulus` | float (GPa) | target |

If some targets are missing for many rows, build the regression/classification modules to train independently on the subset of rows where that target is available (do not drop rows globally for one missing target).

---

## 5. Module 1 — Data Preprocessing (`src/data_preprocessing.py`)

Pipeline steps, in order:

1. **Load** raw CSV(s) from `data/raw/`.
2. **Missing-value analysis**: report % missing per column (`reports/metrics.json["missing_value_report"]`). Strategy:
   - Composition columns: drop row if all-zero or NaN (invalid alloy record).
   - Continuous process params: median imputation, grouped by `Synthesis_Route` if possible.
   - Categorical process params: mode imputation or an explicit `"Unknown"` category.
   - Targets (UTS/Yield/E/Phase): do NOT impute — leave NaN, handle per-model as in Section 4.
3. **Duplicate analysis**: drop exact duplicate rows; flag near-duplicates (same composition + process, different targets) for manual review in a report, don't auto-drop.
4. **Categorical encoding**:
   - Low-cardinality (`Synthesis_Route`, `Hot_Cold_Working`, `Quenching`): One-Hot Encoding.
   - Keep an ordinal/label-encoded copy too if a tree-based model is used (trees don't need one-hot).
5. **Feature scaling**: StandardScaler for continuous features, fit on train split only, applied to val/test via the same fitted scaler. Save the scaler object to `models/scaler.pkl`.
6. Output: `data/interim/cleaned.csv` and `data/processed/model_ready.csv`.

Unit tests (`tests/test_data_preprocessing.py`): verify no NaNs remain in feature columns post-pipeline, verify composition rows sum to ~100 (±0.5 tolerance), verify scaler produces mean≈0/std≈1 on train.

---

## 6. Module 2 — Materials Feature Engineering (`src/materials_features.py`)

Compute the following physics-informed descriptors from composition (requires an elemental property lookup table in `config.py`: atomic radius, melting point, electronegativity (Pauling), VEC, molar volume):

- **Mixing entropy:**

  ΔS_mix = -R · Σ cᵢ·ln(cᵢ)   (cᵢ = atomic fraction of element i, R = 8.314 J/mol·K)

- **Mixing enthalpy** (ΔH_mix) via Miedema-model pairwise binary enthalpy table (Ωᵢⱼ) if available; otherwise use a published enthalpy lookup for the relevant element pairs:

  ΔH_mix = Σᵢ≠ⱼ 4·Ωᵢⱼ·cᵢ·cⱼ

- **Gibbs free energy of mixing** (at a reference temperature, e.g. melting point average):

  ΔG_mix = ΔH_mix − T·ΔS_mix

- **Atomic-size difference (δ):**

  δ = sqrt( Σ cᵢ·(1 − rᵢ / r̄)² ) · 100,   r̄ = Σ cᵢ·rᵢ

- **Electronegativity difference (Δχ):**

  Δχ = sqrt( Σ cᵢ·(χᵢ − χ̄)² ),   χ̄ = Σ cᵢ·χᵢ

- **Valence Electron Concentration (VEC):**

  VEC = Σ cᵢ·VECᵢ

- **Melting temperature (rule-of-mixtures estimate):**

  T_m = Σ cᵢ·T_m,i

- **Density** (rule-of-mixtures, if not already measured):

  ρ = 1 / Σ (wᵢ / ρᵢ)   (wᵢ = weight fraction)

Implement each as a pure function taking a composition dict/row and the elemental property table, returning a float. Vectorize over the full DataFrame with `.apply` or numpy broadcasting for performance. Output appended as new columns to `data/processed/model_ready.csv`.

Unit tests: verify ΔS_mix for a known 5-element equiatomic alloy matches the textbook value (R·ln5); verify δ = 0 for a single-element "alloy" edge case.

---

## 7. Module 3 — Process Feature Engineering (`src/process_features.py`)

- One-hot encode `Synthesis_Route`, `Hot_Cold_Working`, `Quenching` (reuse Module 1's encoder, keep consistent column order — save the encoder's category list to `models/encoder_categories.json` so inference-time single-row inputs (from the dashboard) can be encoded identically).
- Keep `Homogenization_Temp`, `Homogenization_Time`, `Annealing_Temp`, `Annealing_Time` as scaled numeric features.
- Optionally engineer interaction terms: `Annealing_Temp × Annealing_Time` (thermal budget proxy), `Homogenization_Temp / T_m` (homologous temperature — physically meaningful for HEAs).

---

## 8. Module 4 — Phase Prediction (`src/phase_model.py`)

- **Input:** composition + Module 2 features + Module 3 features.
- **Target:** `Phase` / `Structure` (multi-class categorical).
- **Model:** start with `RandomForestClassifier` as baseline, then `XGBClassifier` for comparison. Use stratified k-fold CV (k=5) given likely class imbalance.
- **Class imbalance handling:** class_weight='balanced' or SMOTE if a class has <10 samples (check counts first, log a warning if any class is too small to model reliably).
- **Metrics:** accuracy, macro-F1, confusion matrix (save plot to `reports/figures/`).
- **Save:** best model to `models/phase_classifier.pkl`.

---

## 9. Module 5 — Mechanical Property Prediction (`src/property_models.py`)

Three independent regression models, each trained only on rows where that target is non-null:

| Model | Target | Suggested algorithm |
|---|---|---|
| Model 1 | UTS (MPa) | XGBRegressor (or GradientBoostingRegressor) |
| Model 2 | Yield (MPa) | XGBRegressor |
| Model 3 | Young's Modulus (GPa) | XGBRegressor |

- Train/val/test split: 70/15/15, or k-fold CV if dataset is small (<200 rows — likely for HEA data, so prefer 5-fold CV with reported mean±std).
- **Metrics:** R², MAE, RMSE, reported per model.
- **Hyperparameter tuning:** `RandomizedSearchCV` or `Optuna` over n_estimators, max_depth, learning_rate, subsample.
- **Save:** each model to `models/uts_model.pkl`, `models/yield_model.pkl`, `models/e_model.pkl`.

---

## 10. Module 6 — Explainable AI (`src/explainability.py`)

For each trained model (phase classifier + 3 regressors):

1. **SHAP**: `TreeExplainer` (fast for tree models). Produce:
   - Global summary/beeswarm plot → `reports/figures/{model}_shap_summary.png`
   - Save mean |SHAP value| per feature to `reports/metrics.json["shap_importance"][model_name]`
2. **Permutation importance**: `sklearn.inspection.permutation_importance` on held-out test set, as a cross-check against SHAP.
3. **Partial Dependence Plots (PDP)**: for the top 4–5 most important features per model → `reports/figures/{model}_pdp_{feature}.png`.

Produce a short auto-generated markdown summary (`reports/explainability_summary.md`) answering: *"Which manufacturing parameters and material descriptors affect the predicted properties?"* — a ranked list per target.

---

## 11. Module 7 — Optimization (`src/optimization.py`)

**Objective:**

  maximize  f = w₁·UTS(x) + w₂·Yield(x) + w₃·E(x)

where `x` = (composition vector, process parameter vector), and UTS/Yield/E are predicted by the Module 5 surrogate models (not measured — this is a surrogate-model optimization).

**Constraints:**

  Σᵢ cᵢ = 100%
  cᵢ ≥ 0 for all i
  (optional) cᵢ within realistic bounds observed in the training data, to avoid extrapolation
  (optional) Density(x) ≤ some max, if optimizing for lightweighting

**Optional derived objectives** (implement as alternate objective functions the user can select in the dashboard):

  Specific Strength = UTS / Density
  Specific Modulus  = E / Density

**Approach:**
- Since composition is constrained to a simplex (sum to 100) and process params include categoricals, use a **genetic algorithm** (`deap` or `scipy.optimize.differential_evolution` with a penalty term for the simplex constraint) rather than gradient-based methods.
- Weights w₁, w₂, w₃ should be exposed as dashboard sliders (default equal weighting).
- Output: top-5 candidate (composition, process) combinations with their predicted UTS/Yield/E, ranked by objective value.

---

## 12. Module 8 — Dashboard (`app/dashboard.py`, Streamlit)

Layout mirrors the requested mockup:

**Section 1 — Alloy Composition** (sliders/number inputs per element, must sum to 100 — show a live running total with a warning if ≠100)

**Section 2 — Manufacturing Parameters** (Annealing Temp, Annealing Time, Working type dropdown, Quenching toggle, Homogenization Temp/Time)

**Section 3 — Predict button** → on click:
- Rebuild the Module 2/3 feature vector for the single input row (reuse the exact saved scaler/encoder from training — critical for correctness)
- Run phase classifier + 3 regressors
- Display: Predicted UTS, Yield, E, Phase, Structure

**Section 4 — Explainability panel**: show a per-prediction SHAP force/waterfall plot (local explanation) for this specific input, not just the global summary.

**Section 5 — Process Optimization panel**: button to run Module 7 optimizer with adjustable weights (w₁, w₂, w₃), display top-5 recommended parameter sets in a table.

Run with: `streamlit run app/dashboard.py`

---

## 13. Pipeline Orchestration (`run_pipeline.py`)

Single script that runs, in order, and is idempotent (safe to re-run):

```
1. data_preprocessing.run()        → data/processed/model_ready.csv
2. materials_features.run()        → appends descriptor columns
3. process_features.run()          → appends encoded process columns
4. phase_model.train_and_eval()    → models/phase_classifier.pkl
5. property_models.train_and_eval()→ models/{uts,yield,e}_model.pkl
6. explainability.generate_all()   → reports/figures/*, reports/explainability_summary.md
7. print final metrics summary to console + reports/metrics.json
```

---

## 14. Evaluation / Acceptance Criteria

- [ ] Pipeline runs end-to-end via `python run_pipeline.py` with no errors on the supplied dataset
- [ ] `reports/metrics.json` contains accuracy/F1 for phase model and R²/MAE/RMSE for all 3 regressors
- [ ] At least one SHAP summary plot generated per model
- [ ] Dashboard launches and produces a prediction for a manually entered composition within 2 seconds
- [ ] Optimizer returns 5 distinct candidate solutions that satisfy Σcᵢ=100 within numerical tolerance
- [ ] All unit tests in `tests/` pass

---

## 15. Suggested Build Order (for the coding agent)

1. Scaffold repo structure + `requirements.txt`
2. Load and inspect actual dataset → write `COLUMN_MAPPING.md`, finalize `config.py`
3. Module 1 (preprocessing) + tests
4. Module 2 (materials features) + tests
5. Module 3 (process features)
6. Module 4 (phase model) + evaluation
7. Module 5 (three regressors) + evaluation
8. Module 6 (explainability)
9. Module 7 (optimization)
10. `run_pipeline.py` end-to-end smoke test
11. Module 8 (Streamlit dashboard) wired to saved models/scaler/encoder
12. Final polish: README with setup + run instructions, `reports/` sanity check

---

## 16. Notes / Open Questions for the User (agent should surface these, not silently assume)

- Exact dataset file name/location and its real column names (map against Section 4's contract)
- Which elements appear in the alloy system (affects `config.py` elemental property table)
- Whether density is measured or must be estimated via rule-of-mixtures
- Preferred weighting scheme for the optimization objective (default: equal weights, user-adjustable in dashboard)
