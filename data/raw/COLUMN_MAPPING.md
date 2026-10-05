# Raw Column Mapping Specification

This document maps the columns from the raw dataset (`data/raw/Phase_field_updated_model_data.csv.xlsx`) to the canonical schema defined in Section 4 of `HEA_Digital_Manufacturing_Implementation.md`.

## Composition Columns
The raw dataset contains 5 elements with trailing whitespace in column names:

| Raw Column Name | Canonical Name | Type | Units / Range | Description |
|---|---|---|---|---|
| `'Mg '` | `Mg` | float | 20.0 - 95.34 wt.% | Magnesium base content |
| `'Al '` | `Al` | float | 0.0 - 20.0 wt.% | Aluminum alloy content |
| `'Cu '` | `Cu` | float | 0.0 - 20.0 wt.% | Copper alloy content |
| `'Mn '` | `Mn` | float | 0.0 - 20.0 wt.% | Manganese alloy content |
| `'Zn '` | `Zn` | float | 0.0 - 20.0 wt.% | Zinc alloy content |

## Process Parameter Columns

| Raw Column Name | Canonical Name | Type | Values / Units | Description |
|---|---|---|---|---|
| `Synthesis_Route` | `Synthesis_Route` | categorical | IM, DC, S_T6, S_T4, DMD | Primary synthesis method |
| `Hot_Cold_Working` | `Hot_Cold_Working` | categorical | CW, CR | Cold Working or Cold Rolling |
| `Quenching` | `Quenching` | float | 8.33 - 1000.0 °C/s | Quenching cooling rate |
| `Homogenization_Temp` | `Homogenization_Temp` | float | 150 - 500 °C | Homogenization temperature |
| `Homogenization_Time` | `Homogenization_Time` | float | 60 - 1440 min | Homogenization duration |
| `Annealing_Temp` | `Annealing_Temp` | float | 200 - 300 °C | Annealing temperature |
| `Annealing_Time` | `Annealing_Time` | float | 30 - 120 min | Annealing duration |

## Target Columns

| Raw Column Name | Canonical Name | Type | Range | Description |
|---|---|---|---|---|
| `Structure` | `Structure` | categorical | 10 classes | Multi-class phase/microstructure target |
| `Phases` | `Phases` | categorical | Intermetallic phases | Secondary constituent phases |
| `UTS` | `UTS` | float | 210 - 717 MPa | Ultimate Tensile Strength |
| `Yield` | `Yield` | float | 90 - 450 MPa | Yield Strength |
| `E` | `Youngs_Modulus` | float | 35.5 - 151.0 GPa | Young's Modulus of elasticity |

## Precomputed Descriptors in Raw Dataset

The raw file also includes reference calculated physical descriptors:
- `Density_calc`: Rule-of-mixtures density (g/cm³)
- `dHmix`: Enthalpy of mixing (J/mol)
- `dSmix`: Entropy of mixing (J/mol·K)
- `dGmix`: Gibbs free energy of mixing (J/mol) at 1573 K
- `Tm`: Rule-of-mixtures melting temperature (°C)
- `Atom_Size_Diff`: Atomic size disparity descriptor
- `Elect_Diff`: Electronegativity disparity descriptor
- `VEC`: Valence electron concentration
- `n_Para`: Parameter descriptor
- `Num_of_Elem`: Number of constituent alloying elements (3, 4, or 5)
