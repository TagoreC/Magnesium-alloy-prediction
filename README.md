# AI-Driven Digital Manufacturing Framework for Mg-Based High-Entropy Alloys

An end-to-end machine learning pipeline and interactive Streamlit digital manufacturing dashboard for predicting microstructure phases and mechanical properties of Magnesium-based High-Entropy Alloys (HEAs), with explainable AI (SHAP / PDP) and inverse design optimization.

---

## 📋 Recommended Execution Order

You can run the entire framework in **one single command** or execute **step-by-step modularly**.

### Option 1: Single-Command Full Pipeline (Recommended)
Run all modules (preprocessing, physics features, process features, phase modeling, regression modeling, and explainability):
```powershell
python run_pipeline.py
```

Then launch the interactive digital manufacturing dashboard:
```powershell
streamlit run app/dashboard.py
```

---

### Option 2: Step-by-Step Modular Execution

If you wish to run and inspect individual modules one by one, execute them in this exact order:

| Step | Command | Description | Output Artifacts |
|---|---|---|---|
| **1** | `python -m src.data_preprocessing` | Cleans dataset, analyzes missing values, standardizes column names, one-hot encodes categoricals, fits standard scaler | `data/interim/cleaned.csv`, `models/scaler.pkl`, `models/encoder_categories.json` |
| **2** | `python -m src.materials_features` | Computes physics-informed thermodynamic descriptors ($\Delta S_{\text{mix}}$, $\Delta H_{\text{mix}}$, $\Delta G_{\text{mix}}$, $\delta$, $\Delta \chi$, $\text{VEC}$, $T_m$, $\rho$) | Vectorized physical descriptors appended to features |
| **3** | `python -m src.process_features` | Computes process interaction features (thermal budget, homologous temperature, homogenization exposure) | `data/processed/model_ready.csv` |
| **4** | `python -m src.phase_model` | Trains multi-class phase/microstructure classifier with Stratified K-Fold CV & class balancing | `models/phase_classifier.pkl`, `models/phase_label_encoder.pkl`, `reports/figures/phase_confusion_matrix.png` |
| **5** | `python -m src.property_models` | Trains 3 independent regressors for UTS, Yield Strength, and Young's Modulus (E) | `models/uts_model.pkl`, `models/yield_model.pkl`, `models/e_model.pkl` |
| **6** | `python -m src.explainability` | Runs TreeExplainer SHAP summaries, permutation importance, and Partial Dependence Plots (PDP) | `reports/figures/*_shap_summary.png`, `reports/explainability_summary.md` |
| **7** | `python -m src.optimization` | Runs inverse design multi-objective Genetic Algorithm / Differential Evolution to find top-5 candidates | Console candidate recommendations ranked by score |
| **8** | `streamlit run app/dashboard.py` | Launches the interactive web dashboard | Web application at `http://localhost:8501` |

---

### Running Automated Unit Tests
To verify all pipeline logic and tests:
```powershell
python -m pytest tests/ -v
```
All unit tests in `tests/` verify:
1. `test_data_preprocessing.py`: No NaNs post-pipeline, composition sums to 100%, scaler normalization.
2. `test_materials_features.py`: Equiatomic 5-element $\Delta S_{\text{mix}} = R \ln(5)$, pure metal $\delta = 0$, VEC, and rule-of-mixtures density.
3. `test_process_features.py`: Thermal budgets, homologous temperature, deterministic single-row inference vector builder.
4. `test_models.py`: Phase classification and mechanical property regressors performance thresholds and end-to-end inference.

---

## 🏗 Repository Structure

```
DM/
├── data/
│   ├── raw/                      # Original dataset & column mapping
│   │   ├── Phase_field_updated_model_data.csv
│   │   ├── Phase_field_updated_model_data.xlsx
│   │   └── COLUMN_MAPPING.md
│   ├── interim/
│   │   └── cleaned.csv           # Cleaned & standardized intermediate data
│   └── processed/
│       └── model_ready.csv       # Final matrix with materials & process features
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Elemental properties, Miedema table, canonical names
│   ├── utils.py                  # Serialization, model persistence, JSON helpers
│   ├── data_preprocessing.py     # Module 1: Preprocessing & scaling
│   ├── materials_features.py     # Module 2: Physics-informed HEA descriptors
│   ├── process_features.py       # Module 3: Process interaction engineering
│   ├── phase_model.py            # Module 4: Multi-class phase classifier
│   ├── property_models.py        # Module 5: UTS, Yield, and E regressors
│   ├── explainability.py         # Module 6: SHAP, Permutation, and PDP
│   └── optimization.py           # Module 7: Genetic Algorithm / Differential Evolution
│
├── models/                       # Serialized trained models & scalers
│   ├── scaler.pkl
│   ├── encoder_categories.json
│   ├── phase_classifier.pkl
│   ├── phase_label_encoder.pkl
│   ├── uts_model.pkl
│   ├── yield_model.pkl
│   └── e_model.pkl
│
├── app/
│   └── dashboard.py              # Module 8: Interactive Streamlit web application
│
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_materials_features.py
│   ├── test_process_features.py
│   └── test_models.py
│
├── reports/
│   ├── figures/                  # Confusion matrices, SHAP summary beeswarms, PDPs
│   ├── metrics.json              # Evaluation metrics (Accuracy, F1, R2, MAE, RMSE)
│   └── explainability_summary.md # Top feature rankings and physical insights
│
├── requirements.txt
├── README.md
└── run_pipeline.py               # Single entry-point orchestration script
```

---

## 🌟 Dashboard Features (`app/dashboard.py`)

1. **Alloy Composition Sliders**: Interactive controls for `Mg`, `Al`, `Cu`, `Mn`, `Zn` with live simplex ($\sum c_i = 100\%$) validation and auto-balance assist.
2. **Manufacturing Process Controls**: Synthesis route, working mode (CW/CR), quenching cooling rate, homogenization temperature/time, and annealing temperature/time.
3. **Instant Property Prediction**: Real-time evaluation of:
   - Predicted Microstructure Phase with classification confidence
   - Ultimate Tensile Strength (UTS, MPa)
   - Yield Strength (MPa)
   - Young's Modulus ($E$, GPa)
   - Rule-of-mixtures Density ($\rho$, g/cm³)
   - Specific Strength ($\text{UTS}/\rho$) & Specific Modulus ($E/\rho$)
4. **Interactive Explainability**: Local SHAP feature attribution bar chart explaining which parameters drove the active prediction.
5. **Inverse Design Optimizer**: Multi-objective candidate search with customizable weights and density caps, producing top-5 distinct candidate alloys.
