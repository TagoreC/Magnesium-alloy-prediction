"""Module 7 — Inverse Alloy & Process Optimization.

Multi-objective surrogate optimization using Genetic Algorithm / Differential Evolution
to find optimal alloy compositions and manufacturing parameter combinations that maximize
weighted strength, stiffness, or specific (density-normalized) properties.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from src.config import (
    CATEGORICAL_PROCESS_COLS,
    ELEMENTS,
    NUMERIC_PROCESS_COLS,
)
from src.materials_features import calculate_density
from src.phase_model import predict_phase
from src.process_features import build_single_sample_features
from src.property_models import predict_properties


SYNTHESIS_ROUTES = ["IM", "DC", "S_T6", "S_T4", "DMD"]
WORKING_TYPES = ["CW", "CR"]

# Realistic parameter search bounds from experimental space
PARAM_BOUNDS = {
    # Elements: initial bounds for Dirichlet / Simplex search
    "Mg": (20.0, 95.5),
    "Al": (0.0, 20.0),
    "Cu": (0.0, 20.0),
    "Mn": (0.0, 20.0),
    "Zn": (0.0, 20.0),
    # Process continuous variables
    "Homogenization_Temp": (150.0, 500.0),
    "Homogenization_Time": (60.0, 1440.0),
    "Annealing_Temp": (200.0, 300.0),
    "Annealing_Time": (30.0, 120.0),
    "Quenching": (8.33, 1000.0),
}


def normalize_composition_simplex(raw_comp: np.ndarray) -> np.ndarray:
    """Project positive values onto the 100% simplex sum."""
    clipped = np.clip(raw_comp, 0.0, None)
    total = np.sum(clipped)
    if total <= 1e-6:
        # Fallback to pure Mg
        res = np.zeros_like(clipped)
        res[0] = 100.0
        return res
    norm = (clipped / total) * 100.0
    return np.round(norm, 2)


def evaluate_candidate(
    comp_dict: Dict[str, float],
    proc_dict: Dict[str, Any],
    w_uts: float = 0.33,
    w_yield: float = 0.33,
    w_e: float = 0.34,
    objective_type: str = "weighted_properties",
    max_density: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate surrogate objective and constraints for a candidate alloy and process."""
    # Build full feature vector
    sample_df = build_single_sample_features(comp_dict, proc_dict)

    # Predictions
    props = predict_properties(sample_df)
    uts = props.get("UTS", 0.0)
    yield_str = props.get("Yield", 0.0)
    e_mod = props.get("Youngs_Modulus", 0.0)

    # Predicted Phase
    pred_phase, phase_conf = predict_phase(sample_df)

    # Density & specific metrics
    density = calculate_density(comp_dict, is_weight_fraction=True)
    specific_strength = uts / density if density > 0 else 0.0
    specific_modulus = e_mod / density if density > 0 else 0.0

    # Objective calculation
    penalty = 0.0
    if max_density is not None and density > max_density:
        penalty += (density - max_density) * 1000.0

    # Normalization references (approximate median ranges for balanced weighting)
    # UTS ~ 300, Yield ~ 180, E ~ 95
    norm_uts = uts / 300.0
    norm_yield = yield_str / 180.0
    norm_e = e_mod / 95.0

    if objective_type == "specific_strength":
        score = specific_strength - penalty
    elif objective_type == "specific_modulus":
        score = specific_modulus - penalty
    elif objective_type == "combined_specific":
        score = (w_uts * (specific_strength / 150.0) +
                 w_yield * (yield_str / density / 90.0) +
                 w_e * (specific_modulus / 50.0)) - penalty
    else:  # default weighted_properties
        score = (w_uts * norm_uts + w_yield * norm_yield + w_e * norm_e) * 100.0 - penalty

    candidate_data = {
        "Mg": comp_dict["Mg"],
        "Al": comp_dict["Al"],
        "Cu": comp_dict["Cu"],
        "Mn": comp_dict["Mn"],
        "Zn": comp_dict["Zn"],
        "Synthesis_Route": proc_dict["Synthesis_Route"],
        "Hot_Cold_Working": proc_dict["Hot_Cold_Working"],
        "Homogenization_Temp": proc_dict["Homogenization_Temp"],
        "Homogenization_Time": proc_dict["Homogenization_Time"],
        "Annealing_Temp": proc_dict["Annealing_Temp"],
        "Annealing_Time": proc_dict["Annealing_Time"],
        "Quenching": proc_dict["Quenching"],
        "Predicted_Phase": pred_phase,
        "Phase_Confidence": round(phase_conf, 3),
        "Predicted_UTS": round(uts, 1),
        "Predicted_Yield": round(yield_str, 1),
        "Predicted_E": round(e_mod, 1),
        "Calculated_Density": round(density, 3),
        "Specific_Strength": round(specific_strength, 1),
        "Specific_Modulus": round(specific_modulus, 1),
        "Objective_Score": round(float(score), 2),
    }

    return float(score), candidate_data


def optimize_alloy_and_process(
    w_uts: float = 0.33,
    w_yield: float = 0.33,
    w_e: float = 0.34,
    objective_type: str = "weighted_properties",
    max_density: Optional[float] = None,
    n_candidates: int = 5,
    pop_size: int = 40,
    generations: int = 25,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run genetic optimization to discover top-ranked candidate alloy & manufacturing parameters.

    Returns:
        List of distinct top-N candidate dictionaries satisfying sum(c_i) = 100.
    """
    rng = np.random.default_rng(random_seed)
    candidates_pool: List[Tuple[float, Dict[str, Any]]] = []

    # Parameter vector format (12 dimensions):
    # [c_Mg, c_Al, c_Cu, c_Mn, c_Zn, route_idx, work_idx, homog_temp, homog_time, anneal_temp, anneal_time, quench]
    bounds = [
        PARAM_BOUNDS["Mg"],
        PARAM_BOUNDS["Al"],
        PARAM_BOUNDS["Cu"],
        PARAM_BOUNDS["Mn"],
        PARAM_BOUNDS["Zn"],
        (0, len(SYNTHESIS_ROUTES) - 1e-4),
        (0, len(WORKING_TYPES) - 1e-4),
        PARAM_BOUNDS["Homogenization_Temp"],
        PARAM_BOUNDS["Homogenization_Time"],
        PARAM_BOUNDS["Annealing_Temp"],
        PARAM_BOUNDS["Annealing_Time"],
        PARAM_BOUNDS["Quenching"],
    ]

    def decode_vector(vec: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Any]]:
        raw_comp = vec[:5]
        norm_comp = normalize_composition_simplex(raw_comp)
        # Fix exact sum to 100.0
        diff = 100.0 - np.sum(norm_comp)
        norm_comp[0] = round(norm_comp[0] + diff, 2)

        comp_dict = {el: float(val) for el, val in zip(ELEMENTS, norm_comp)}

        route_idx = int(vec[5]) % len(SYNTHESIS_ROUTES)
        work_idx = int(vec[6]) % len(WORKING_TYPES)

        proc_dict = {
            "Synthesis_Route": SYNTHESIS_ROUTES[route_idx],
            "Hot_Cold_Working": WORKING_TYPES[work_idx],
            "Homogenization_Temp": round(float(vec[7]), 1),
            "Homogenization_Time": round(float(vec[8]), 1),
            "Annealing_Temp": round(float(vec[9]), 1),
            "Annealing_Time": round(float(vec[10]), 1),
            "Quenching": round(float(vec[11]), 1),
        }
        return comp_dict, proc_dict

    def objective_fn(vec: np.ndarray) -> float:
        comp_dict, proc_dict = decode_vector(vec)
        score, cand_info = evaluate_candidate(
            comp_dict,
            proc_dict,
            w_uts=w_uts,
            w_yield=w_yield,
            w_e=w_e,
            objective_type=objective_type,
            max_density=max_density,
        )
        candidates_pool.append((score, cand_info))
        # Minimize negative objective
        return -score

    # Run Differential Evolution with simplex mapping
    differential_evolution(
        objective_fn,
        bounds=bounds,
        strategy="best1bin",
        maxiter=generations,
        popsize=pop_size,
        seed=random_seed,
        polish=False,
    )

    # Also evaluate known high-performing seeds from dataset
    seed_compositions = [
        {"Mg": 20.0, "Al": 20.0, "Cu": 20.0, "Mn": 20.0, "Zn": 20.0},
        {"Mg": 45.0, "Al": 13.75, "Cu": 13.75, "Mn": 13.75, "Zn": 13.75},
        {"Mg": 50.0, "Al": 12.5, "Cu": 12.5, "Mn": 12.5, "Zn": 12.5},
        {"Mg": 80.0, "Al": 10.0, "Cu": 5.0, "Mn": 5.0, "Zn": 0.0},
        {"Mg": 91.2, "Al": 8.3, "Cu": 0.0, "Mn": 0.15, "Zn": 0.35},
        {"Mg": 92.05, "Al": 5.3, "Cu": 0.0, "Mn": 0.15, "Zn": 2.5},
    ]
    for s_comp in seed_compositions:
        for route in ["IM", "S_T6", "DMD"]:
            s_proc = {
                "Synthesis_Route": route,
                "Hot_Cold_Working": "CW",
                "Homogenization_Temp": 400.0,
                "Homogenization_Time": 1440.0,
                "Annealing_Temp": 200.0,
                "Annealing_Time": 30.0,
                "Quenching": 100.0,
            }
            s_score, s_info = evaluate_candidate(
                s_comp, s_proc, w_uts, w_yield, w_e, objective_type, max_density
            )
            candidates_pool.append((s_score, s_info))

    # Sort descending by objective score
    candidates_pool.sort(key=lambda item: item[0], reverse=True)

    # Filter for diverse, distinct solutions
    unique_candidates: List[Dict[str, Any]] = []
    seen_compositions = []

    for score, cand in candidates_pool:
        comp_vec = np.array([cand["Mg"], cand["Al"], cand["Cu"], cand["Mn"], cand["Zn"]])
        # Verify diversity: difference > 1.5 wt% from previously selected candidates
        is_distinct = True
        for prev in seen_compositions:
            if np.max(np.abs(comp_vec - prev)) < 1.5:
                is_distinct = False
                break

        if is_distinct:
            seen_compositions.append(comp_vec)
            unique_candidates.append(cand)

        if len(unique_candidates) >= n_candidates:
            break

    return unique_candidates


if __name__ == "__main__":
    top_candidates = optimize_alloy_and_process(
        w_uts=0.33, w_yield=0.33, w_e=0.34, n_candidates=5
    )
    print(f"Top {len(top_candidates)} Optimized Candidate Recommendations:")
    for idx, cand in enumerate(top_candidates, 1):
        print(
            f"[{idx}] Score: {cand['Objective_Score']} | Mg: {cand['Mg']}%, Al: {cand['Al']}%, "
            f"Cu: {cand['Cu']}%, Mn: {cand['Mn']}%, Zn: {cand['Zn']}% | "
            f"UTS: {cand['Predicted_UTS']} MPa, Yield: {cand['Predicted_Yield']} MPa, E: {cand['Predicted_E']} GPa | "
            f"Phase: {cand['Predicted_Phase']}"
        )
