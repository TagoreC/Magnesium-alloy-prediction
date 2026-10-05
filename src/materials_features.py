"""Module 2 — Materials Feature Engineering.

Computes physics-informed metallurgical descriptors from alloy elemental composition:
- Mixing entropy (ΔS_mix)
- Mixing enthalpy (ΔH_mix via Miedema pairwise lookup)
- Gibbs free energy of mixing (ΔG_mix)
- Atomic-size disparity (δ)
- Electronegativity disparity (Δχ)
- Valence Electron Concentration (VEC)
- Melting temperature (T_m)
- Rule-of-mixtures density (ρ)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    BINARY_ENTHALPY_TABLE_KJ_MOL,
    ELEMENTAL_PROPERTIES,
    ELEMENTS,
    GAS_CONSTANT_R,
    REFERENCE_TEMPERATURE_K,
)


def weight_to_atomic_fractions(
    composition: Dict[str, float],
    props: Dict[str, Dict[str, float]] = ELEMENTAL_PROPERTIES,
) -> Dict[str, float]:
    """Convert weight percentage/fractions to normalized atomic fractions (sum = 1.0)."""
    moles = {}
    for el, wt in composition.items():
        if el in props and wt > 0:
            moles[el] = wt / props[el]["molar_mass"]

    total_moles = sum(moles.values())
    if total_moles <= 0:
        return {el: 0.0 for el in composition}

    return {el: moles.get(el, 0.0) / total_moles for el in composition}


def calculate_mixing_entropy(
    atomic_fractions: Dict[str, float],
    r_gas: float = GAS_CONSTANT_R,
) -> float:
    """Calculate mixing entropy ΔS_mix = -R * Σ (c_i * ln(c_i)) in J/(mol·K)."""
    s_mix = 0.0
    for el, c_i in atomic_fractions.items():
        if c_i > 1e-12:
            s_mix += c_i * np.log(c_i)
    return float(-r_gas * s_mix)


def calculate_mixing_enthalpy(
    atomic_fractions: Dict[str, float],
    enthalpy_table: Dict[Tuple[str, str], float] = BINARY_ENTHALPY_TABLE_KJ_MOL,
    in_joules: bool = True,
) -> float:
    """Calculate mixing enthalpy ΔH_mix = Σ_{i!=j} 4 * ΔH_ij * c_i * c_j.

    Output in J/mol by default (or kJ/mol if in_joules=False).
    """
    elements = list(atomic_fractions.keys())
    n = len(elements)
    h_mix_kj = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            el1, el2 = elements[i], elements[j]
            c1, c2 = atomic_fractions.get(el1, 0.0), atomic_fractions.get(el2, 0.0)
            if c1 > 0 and c2 > 0:
                pair = (el1, el2) if (el1, el2) in enthalpy_table else (el2, el1)
                delta_h_ij = enthalpy_table.get(pair, 0.0)
                # Ω_ij = 4 * delta_h_ij
                h_mix_kj += 4.0 * delta_h_ij * c1 * c2

    return float(h_mix_kj * 1000.0 if in_joules else h_mix_kj)


def calculate_gibbs_free_energy(
    h_mix: float,
    s_mix: float,
    temperature_k: float = REFERENCE_TEMPERATURE_K,
) -> float:
    """Calculate Gibbs free energy of mixing ΔG_mix = ΔH_mix - T * ΔS_mix (J/mol)."""
    return float(h_mix - (temperature_k * s_mix))


def calculate_atomic_size_difference(
    atomic_fractions: Dict[str, float],
    props: Dict[str, Dict[str, float]] = ELEMENTAL_PROPERTIES,
) -> float:
    """Calculate atomic size difference parameter δ = sqrt(Σ c_i * (1 - r_i / r_bar)^2) * 100 (%)."""
    active = {el: c for el, c in atomic_fractions.items() if c > 1e-12 and el in props}
    if not active:
        return 0.0

    # Average atomic radius r_bar
    r_bar = sum(c * props[el]["atomic_radius"] for el, c in active.items())
    if r_bar <= 0:
        return 0.0

    variance = sum(c * ((1.0 - (props[el]["atomic_radius"] / r_bar)) ** 2) for el, c in active.items())
    return float(np.sqrt(max(0.0, variance)) * 100.0)


def calculate_electronegativity_difference(
    atomic_fractions: Dict[str, float],
    props: Dict[str, Dict[str, float]] = ELEMENTAL_PROPERTIES,
) -> float:
    """Calculate electronegativity disparity Δχ = sqrt(Σ c_i * (χ_i - χ_bar)^2)."""
    active = {el: c for el, c in atomic_fractions.items() if c > 1e-12 and el in props}
    if not active:
        return 0.0

    chi_bar = sum(c * props[el]["electronegativity"] for el, c in active.items())
    variance = sum(c * ((props[el]["electronegativity"] - chi_bar) ** 2) for el, c in active.items())
    return float(np.sqrt(max(0.0, variance)))


def calculate_vec(
    atomic_fractions: Dict[str, float],
    props: Dict[str, Dict[str, float]] = ELEMENTAL_PROPERTIES,
) -> float:
    """Calculate Valence Electron Concentration VEC = Σ c_i * VEC_i."""
    return float(sum(c * props[el]["vec"] for el, c in atomic_fractions.items() if el in props))


def calculate_melting_temperature(
    atomic_fractions: Dict[str, float],
    props: Dict[str, Dict[str, float]] = ELEMENTAL_PROPERTIES,
    in_kelvin: bool = False,
) -> float:
    """Calculate melting temperature via rule-of-mixtures T_m = Σ c_i * T_{m,i}."""
    key = "melting_point_k" if in_kelvin else "melting_point_c"
    return float(sum(c * props[el][key] for el, c in atomic_fractions.items() if el in props))


def calculate_density(
    composition: Dict[str, float],
    props: Dict[str, Dict[str, float]] = ELEMENTAL_PROPERTIES,
    is_weight_fraction: bool = True,
) -> float:
    """Calculate density via rule-of-mixtures ρ = 1 / Σ (w_i / ρ_i) in g/cm^3."""
    if is_weight_fraction:
        total_w = sum(composition.get(el, 0.0) for el in props)
        if total_w <= 0:
            return 0.0
        norm_w = {el: composition.get(el, 0.0) / total_w for el in props}
    else:
        # Convert atomic fractions to weight fractions
        m_total = sum(c * props[el]["molar_mass"] for el, c in composition.items() if el in props)
        if m_total <= 0:
            return 0.0
        norm_w = {
            el: (composition.get(el, 0.0) * props[el]["molar_mass"]) / m_total
            for el in props
        }

    sum_inv_rho = sum(w / props[el]["density"] for el, w in norm_w.items() if w > 0)
    return float(1.0 / sum_inv_rho if sum_inv_rho > 0 else 0.0)


def compute_all_materials_features(
    comp_dict: Dict[str, float],
    is_weight_percent: bool = True,
) -> Dict[str, float]:
    """Compute complete suite of physics-informed materials features for a single composition."""
    # Normalize inputs
    if is_weight_percent:
        total = sum(comp_dict.get(el, 0.0) for el in ELEMENTS)
        w_dict = {el: (comp_dict.get(el, 0.0) / total if total > 0 else 0.0) for el in ELEMENTS}
        at_dict = weight_to_atomic_fractions(w_dict)
    else:
        total = sum(comp_dict.get(el, 0.0) for el in ELEMENTS)
        at_dict = {el: (comp_dict.get(el, 0.0) / total if total > 0 else 0.0) for el in ELEMENTS}
        w_dict = at_dict  # will be converted in calculate_density

    s_mix = calculate_mixing_entropy(at_dict)
    h_mix = calculate_mixing_enthalpy(at_dict)
    t_m_c = calculate_melting_temperature(at_dict, in_kelvin=False)
    t_m_k = calculate_melting_temperature(at_dict, in_kelvin=True)
    g_mix = calculate_gibbs_free_energy(h_mix, s_mix, temperature_k=REFERENCE_TEMPERATURE_K)
    delta = calculate_atomic_size_difference(at_dict)
    delta_chi = calculate_electronegativity_difference(at_dict)
    vec = calculate_vec(at_dict)
    density = calculate_density(comp_dict, is_weight_fraction=is_weight_percent)

    return {
        "calc_dSmix": s_mix,
        "calc_dHmix": h_mix,
        "calc_dGmix": g_mix,
        "calc_Tm": t_m_c,
        "calc_Tm_K": t_m_k,
        "calc_Atom_Size_Diff": delta,
        "calc_Elect_Diff": delta_chi,
        "calc_VEC": vec,
        "calc_Density": density,
    }


def add_materials_features(
    df: pd.DataFrame,
    element_cols: Optional[list] = None,
) -> pd.DataFrame:
    """Vectorize materials feature computation across all rows in a DataFrame."""
    if element_cols is None:
        element_cols = ELEMENTS

    features_list = []
    for _, row in df.iterrows():
        comp = {el: float(row[el]) for el in element_cols if el in row}
        feats = compute_all_materials_features(comp, is_weight_percent=True)
        features_list.append(feats)

    feats_df = pd.DataFrame(features_list, index=df.index)
    return pd.concat([df, feats_df], axis=1)


def run_materials_features(
    input_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run Module 2 feature engineering step."""
    if input_df is None:
        from src.data_preprocessing import run_data_preprocessing
        _, input_df = run_data_preprocessing()

    df_with_features = add_materials_features(input_df)
    return df_with_features


if __name__ == "__main__":
    df_result = run_materials_features()
    print("Materials features engineered:", [c for c in df_result.columns if c.startswith("calc_")])
