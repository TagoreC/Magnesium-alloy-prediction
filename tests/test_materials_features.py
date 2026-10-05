"""Unit tests for Module 2 — Materials Feature Engineering."""

import numpy as np
import pytest
from src.config import GAS_CONSTANT_R
from src.materials_features import (
    calculate_atomic_size_difference,
    calculate_density,
    calculate_mixing_entropy,
    calculate_vec,
    compute_all_materials_features,
)


def test_mixing_entropy_5_element_equiatomic():
    """Verify ΔS_mix for a known 5-element equiatomic alloy matches the textbook value R·ln(5)."""
    equiatomic = {
        "Mg": 0.2,
        "Al": 0.2,
        "Cu": 0.2,
        "Mn": 0.2,
        "Zn": 0.2,
    }
    s_mix = calculate_mixing_entropy(equiatomic, r_gas=GAS_CONSTANT_R)
    expected = GAS_CONSTANT_R * np.log(5.0)

    assert pytest.approx(s_mix, rel=1e-5) == expected, (
        f"Equiatomic ΔS_mix {s_mix} does not match expected {expected}"
    )


def test_atomic_size_diff_single_element_edge_case():
    """Verify atomic-size difference δ = 0 for a pure/single-element 'alloy' edge case."""
    pure_mg = {"Mg": 1.0, "Al": 0.0, "Cu": 0.0, "Mn": 0.0, "Zn": 0.0}
    delta_pure = calculate_atomic_size_difference(pure_mg)
    assert delta_pure == 0.0, f"Pure metal atomic size disparity must be 0.0, got {delta_pure}"


def test_vec_calculation():
    """Verify VEC calculation for pure Mg (VEC=2) and pure Al (VEC=3)."""
    assert pytest.approx(calculate_vec({"Mg": 1.0})) == 2.0
    assert pytest.approx(calculate_vec({"Al": 1.0})) == 3.0


def test_density_rule_of_mixtures_pure_metal():
    """Verify rule of mixtures density yields exact single-element densities for pure metals."""
    rho_mg = calculate_density({"Mg": 100.0, "Al": 0.0, "Cu": 0.0, "Mn": 0.0, "Zn": 0.0})
    assert pytest.approx(rho_mg, abs=1e-3) == 1.738

    rho_al = calculate_density({"Mg": 0.0, "Al": 100.0, "Cu": 0.0, "Mn": 0.0, "Zn": 0.0})
    assert pytest.approx(rho_al, abs=1e-3) == 2.700


def test_compute_all_features_completeness():
    """Verify all expected feature keys are present and finite."""
    comp = {"Mg": 90.0, "Al": 8.0, "Cu": 0.0, "Mn": 1.0, "Zn": 1.0}
    feats = compute_all_materials_features(comp)

    expected_keys = [
        "calc_dSmix",
        "calc_dHmix",
        "calc_dGmix",
        "calc_Tm",
        "calc_Tm_K",
        "calc_Atom_Size_Diff",
        "calc_Elect_Diff",
        "calc_VEC",
        "calc_Density",
    ]
    for key in expected_keys:
        assert key in feats, f"Missing key: {key}"
        assert np.isfinite(feats[key]), f"Feature {key} is non-finite: {feats[key]}"
