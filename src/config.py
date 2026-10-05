"""Configuration, elemental lookup tables, column names, and path definitions."""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Default Raw Data File
RAW_DATA_FILE = RAW_DATA_DIR / "Phase_field_updated_model_data.csv"
CLEANED_DATA_FILE = INTERIM_DATA_DIR / "cleaned.csv"
MODEL_READY_FILE = PROCESSED_DATA_DIR / "model_ready.csv"
METRICS_FILE = REPORTS_DIR / "metrics.json"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
ENCODER_CATEGORIES_FILE = MODELS_DIR / "encoder_categories.json"

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Column Schema Definitions
ELEMENTS = ["Mg", "Al", "Cu", "Mn", "Zn"]

# Raw to Canonical Column Rename Mapping
RAW_TO_CANONICAL_COLUMNS = {
    "Mg ": "Mg",
    "Al ": "Al",
    "Cu ": "Cu",
    "Mn ": "Mn",
    "Zn ": "Zn",
    "Mg": "Mg",
    "Al": "Al",
    "Cu": "Cu",
    "Mn": "Mn",
    "Zn": "Zn",
    "Synthesis_Route": "Synthesis_Route",
    "Hot_Cold_Working": "Hot_Cold_Working",
    "Homogenization_Temp": "Homogenization_Temp",
    "Homogenization_Time": "Homogenization_Time",
    "Annealing_Temp": "Annealing_Temp",
    "Annealing_Time": "Annealing_Time",
    "Quenching": "Quenching",
    "Structure": "Structure",
    "Phases": "Phases",
    "UTS": "UTS",
    "Yield": "Yield",
    "E": "Youngs_Modulus",
    "Density_calc": "Density_calc",
    "dHmix": "dHmix",
    "dSmix": "dSmix",
    "dGmix": "dGmix",
    "Tm": "Tm",
    "Atom_Size_Diff": "Atom_Size_Diff",
    "Elect_Diff": "Elect_Diff",
    "VEC": "VEC",
    "n_Para": "n_Para",
    "Num_of_Elem": "Num_of_Elem",
}

# Categorical and Continuous Feature Lists
CATEGORICAL_PROCESS_COLS = ["Synthesis_Route", "Hot_Cold_Working"]
NUMERIC_PROCESS_COLS = [
    "Homogenization_Temp",
    "Homogenization_Time",
    "Annealing_Temp",
    "Annealing_Time",
    "Quenching",
]

TARGET_REGRESSION_COLS = ["UTS", "Yield", "Youngs_Modulus"]
TARGET_CLASSIFICATION_COL = "Structure"

# Elemental Properties Table
# Units:
# - atomic_radius: Angstroms (10^-10 m)
# - melting_point_c: Celsius (°C)
# - melting_point_k: Kelvin (K)
# - electronegativity: Pauling scale
# - vec: Valence Electron Concentration
# - molar_mass: g/mol
# - density: g/cm^3
ELEMENTAL_PROPERTIES = {
    "Mg": {
        "atomic_radius": 1.60,
        "melting_point_c": 650.0,
        "melting_point_k": 923.15,
        "electronegativity": 1.31,
        "vec": 2.0,
        "molar_mass": 24.305,
        "density": 1.738,
    },
    "Al": {
        "atomic_radius": 1.43,
        "melting_point_c": 660.0,
        "melting_point_k": 933.15,
        "electronegativity": 1.61,
        "vec": 3.0,
        "molar_mass": 26.982,
        "density": 2.700,
    },
    "Cu": {
        "atomic_radius": 1.28,
        "melting_point_c": 1085.0,
        "melting_point_k": 1358.15,
        "electronegativity": 1.90,
        "vec": 11.0,
        "molar_mass": 63.546,
        "density": 8.960,
    },
    "Mn": {
        "atomic_radius": 1.27,
        "melting_point_c": 1246.0,
        "melting_point_k": 1519.15,
        "electronegativity": 1.55,
        "vec": 7.0,
        "molar_mass": 54.938,
        "density": 7.210,
    },
    "Zn": {
        "atomic_radius": 1.34,
        "melting_point_c": 419.5,
        "melting_point_k": 692.65,
        "electronegativity": 1.65,
        "vec": 12.0,
        "molar_mass": 65.380,
        "density": 7.140,
    },
}

# Binary Mixing Enthalpies ΔH_mix (kJ/mol) based on Miedema's model (Takeuchi & Inoue 2005)
# Ω_ij = 4 * ΔH_mix(i, j)
BINARY_ENTHALPY_TABLE_KJ_MOL = {
    ("Mg", "Al"): -5.0,
    ("Mg", "Cu"): -3.0,
    ("Mg", "Mn"): 4.0,
    ("Mg", "Zn"): -4.0,
    ("Al", "Cu"): -28.0,
    ("Al", "Mn"): -19.0,
    ("Al", "Zn"): 1.0,
    ("Cu", "Mn"): 4.0,
    ("Cu", "Zn"): 1.0,
    ("Mn", "Zn"): 0.0,
}

# Physical Constants
GAS_CONSTANT_R = 8.314462  # J / (mol * K)
REFERENCE_TEMPERATURE_K = 1573.0  # Reference temperature for Gibbs free energy (K)
