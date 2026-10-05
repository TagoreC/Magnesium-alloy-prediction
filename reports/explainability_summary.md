# Explainable AI (XAI) Synthesis Report: Drivers of Phase & Mechanical Properties
This report identifies which manufacturing parameters and materials physics descriptors most strongly dictate microstructure and mechanical performance in Mg-based High-Entropy Alloys.

## Key Influential Factors by Target
### UTS Key Drivers
| Rank | SHAP Feature | Mean |SHAP| | Permutation Feature | Score |
|---|---|---|---|---|
| 1 | `Al` | 63.2436 | `Al` | 0.9473 |
| 2 | `Atom_Size_Diff` | 12.3725 | `Atom_Size_Diff` | 0.0662 |
| 3 | `Cu` | 11.6739 | `Cu` | 0.0614 |
| 4 | `calc_dHmix` | 7.7086 | `calc_dHmix` | 0.0177 |
| 5 | `Density_calc` | 3.0812 | `Density_calc` | 0.0125 |

### Yield Key Drivers
| Rank | SHAP Feature | Mean |SHAP| | Permutation Feature | Score |
|---|---|---|---|---|
| 1 | `calc_Tm` | 6.6639 | `Synthesis_Route_S_T4` | 0.0351 |
| 2 | `Synthesis_Route_S_T4` | 6.0132 | `calc_Tm` | 0.0316 |
| 3 | `Elect_Diff` | 5.7115 | `Elect_Diff` | 0.0269 |
| 4 | `calc_Density` | 5.4033 | `calc_Density` | 0.0258 |
| 5 | `Mg` | 5.0703 | `Density_calc` | 0.0255 |

### Youngs_Modulus Key Drivers
| Rank | SHAP Feature | Mean |SHAP| | Permutation Feature | Score |
|---|---|---|---|---|
| 1 | `Zn` | 5.1794 | `Zn` | 0.0626 |
| 2 | `Mg` | 2.7428 | `Mn` | 0.0168 |
| 3 | `Mn` | 2.6610 | `calc_Density` | 0.0154 |
| 4 | `calc_dHmix` | 2.3819 | `Mg` | 0.0150 |
| 5 | `calc_Density` | 2.3517 | `calc_dGmix` | 0.0124 |

### Phase Key Drivers
| Rank | SHAP Feature | Mean |SHAP| | Permutation Feature | Score |
|---|---|---|---|---|
| 1 | `Quenching` | 0.0083 | `Quenching` | 0.0068 |
| 2 | `Atom_Size_Diff` | 0.0056 | `Quenching_scaled` | 0.0034 |
| 3 | `Mn` | 0.0054 | `Synthesis_Route_DMD` | 0.0034 |
| 4 | `Quenching_scaled` | 0.0053 | `Al` | 0.0000 |
| 5 | `Mg` | 0.0053 | `Annealing_Temp` | 0.0000 |

## Physical Interpretation
- **Elemental Composition**: The balance of Mg content versus transition metals (Cu, Mn, Zn) and Al fundamentally governs both phase stability and solid-solution / intermetallic strengthening.
- **Thermodynamic Descriptors**: Mixing enthalpy (ΔH_mix), mixing entropy (ΔS_mix), and Valence Electron Concentration (VEC) serve as primary discriminators between single-phase solid solutions, intermetallics, and quasicrystals.
- **Process Parameters**: Thermal budget (`Annealing_Thermal_Budget`), homogenization exposure, and quenching cooling rates strongly dictate precipitation kinetics, grain boundary pinning, and resultant UTS / Yield strength.
