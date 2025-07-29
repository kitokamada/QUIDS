# QUIDS
Q/U Integrated Dust Shells
(**QUIDS**) is a Python package for generating synthetic Stokes **Q** and **U** polarization maps from 3D dust density and Galactic magnetic field shell data. It computes polarized emission by integrating over spherical shells using input polarization angles derived from Galactic magnetic field models (e.g., UF23, JF12).

---

## Features


- Computes:
  - Stokes Q and U maps per shell
  - Total integrated Q/U and polarized intensity maps P
- Supports HEALPix format and Galactic coordinates
- Normalized visualization with `healpy.mollview`
- Easily selects and integrates over custom radial ranges


---

## 📦 Requirements

- `numpy`
- `jax`
- `healpy`
- `astropy`
- `matplotlib`

---
## 🌌 `UF23-Calculation/writeUF23Grid_earth.cpp`: Field Grid Generator for GMF Dust Modeling

### 📌 Purpose

`writeUF23Grid_earth.cpp` is a standalone example script demonstrating how to use the **UF23 Galactic Magnetic Field (GMF) model** in heliocentric coordinates (with Earth at the origin) to generate a 3D vector field grid. The output from this code provides magnetic field components \( B_x, B_y, B_z \) on a uniform Cartesian grid, which can later be interpolated onto spherical shells for **Stokes Q/U map modeling** of dust emission aligned with the GMF.

This serves as a key **preprocessing step** for building synthetic sky polarization maps used in Galactic dust foreground modeling.

---

### ⚙️ How to Compile

Generate your GMF model (eg.UF23):

-- Make sure you have a C++17-compatible compiler (e.g., `g++`) 
Navigate to the folder where writeUF23Grid_earth.cpp and UF23Field.h (and UF23Field.cpp) are located.
Run:(1)  g++ -std=c++17 -O3 writeUF23Grid_earth.cpp UF23Field.cc -o writeUF23Grid_earth
    (2) ./writeUF23Grid_earth

---
## `scripts/gif_shell_mapper.py` — Interpolate GMF Cube onto Spherical Shells

This script interpolates Galactic Magnetic Field (GMF) data from a 3D Cartesian grid onto spherical shells using HEALPix angular sampling. The result is a set of FITS files, each storing magnetic field vectors over the sky at a given radius.

### 🔧 Function
```python
generate_shell_fits(
    gmf_filename: str,
    output_dir: str,
    healpix_order: int,
    r_max_pc: float,
    r_spacing_pc: float
)

## 📓 Example Notebook "notebooks/GMF_shell_mapper_operator.ipynb":

this example demonstrates how to:

1. Generate shell-wise magnetic field FITS maps from a GMF cube using `generate_shell_fits`.
2. Load a shell file and visualize the field using `healpy`.


---
##  `generate_log_spherical_shell_coordinates.py` — step 2, Generate Log-Spaced Shell Coordinates


This script generates **logarithmically spaced spherical shells** sampled in angular direction using a **HEALPix grid**, and saves their **Galactic Cartesian coordinates (X, Y, Z)** and radii \( R \) in FITS binary tables. These shells serve as the spatial foundation for magnetic field sampling and polarization modeling.

---

### 🔧 Function
```python
generate_log_spherical_shells(
    output_folder: str,
    healpix_order: int,
    r_min_pc: float,
    r_max_pc: float,
    num_shells: int
)

## 📓 Example Notebook: "notebooks/shell_coord_builder_operator.ipynb"

This notebook demonstrates how to generate **logarithmically spaced spherical shell coordinates** sampled with a HEALPix grid and save them as binary FITS tables.

---
### 🔧 Function
```python
generate_log_spherical_shells(
    output_folder=OUTPUT_FOLDER,
    healpix_order=HEALPIX_ORDER,
    r_min_pc=R_MIN_PC,
    r_max_pc=R_MAX_PC,
    num_shells=NUM_SHELLS
)

## `scripts/compute_GMF_angles_jax.py` — Step 2.5: Compute Polarization & Inclination Angles (JAX)

This script calculates **polarization angles** and **inclination angles** for each point on a spherical shell using the magnetic field vector and the line-of-sight (LOS) direction. It is **JAX-accelerated** for efficient computation across full-sky HEALPix shells.

This is **Step 2.5** in the dust emission modeling pipeline.
Special Contribution from Dr. Gina Panopoulou and Gina Panopoulou's research group : https://gpanopoulou.github.io
---

### 🔧 Function

```python
process_all_shells(coord_folder, field_folder, output_folder)

## 📓 Example Notebook: "notebooks/angle_calculation_operator.ipynb" Compute Polarization & Inclination Angles (JAX Accelerated)

This notebook demonstrates how to compute **polarization angles** and **inclination angles** from GMF vector fields on spherical shells using JAX. This is **Step 2.5** in the dust emission modeling pipeline.

---

## `integrate_GMF_QU_shells_jax.py` — Step 3: Integrate Stokes Q and U Over Shells

This script performs Step 3 of the dust emission modeling pipeline: it integrates the **GMF-induced Stokes Q and U parameters** across all spherical shells using the polarization and inclination angles computed in Step 2.5.

The result is a pair of full-sky HEALPix maps representing the projected magnetic field structure from the integrated shell geometry. **No dust model is applied at this stage.**

---

### 🧠 Physical Basis

For each shell pixel, the Stokes parameters are computed as:
\[
Q_i = \sin^2(\alpha_i) \cdot \cos(2\beta_i), \quad
U_i = \sin^2(\alpha_i) \cdot \sin(2\beta_i)
\]
Then summed over all shells to produce:
\[
Q_{\text{total}} = \sum_i Q_i, \quad
U_{\text{total}} = \sum_i U_i
\]

---

### 🔧 Functions

#### `sum_QU_over_shells_jax(folder_path)`
- Loads shell FITS files with `Inclination_Angle_deg` and `Polarization_Angle_deg`.
- Computes per-shell Q and U maps.
- Sums all contributions using **JAX** for efficiency.
- Returns: `Q_total`, `U_total` (HEALPix maps)

#### `save_QU_to_fits(Q, U, output_filename, nside)`
- Saves the integrated Q and U maps as a FITS binary table.

#### `plot_QU_maps(Q, U, output_prefix)`
- Visualizes the Q and U maps with Mollweide projection.
- Saves PNG figures: `{output_prefix}_Q.png` and `{output_prefix}_U.png`

---

### 📥 Input
- Folder of angle FITS files from Step 2.5 (e.g. `fits_shell_angles_jax_log/`)
  - Each file contains: `Polarization_Angle_deg`, `Inclination_Angle_deg`

### 📤 Output
- **FITS file**: Combined Q and U maps
- **PNG plots**: Visualizations of Q and U in Mollweide projection

---

## 📓 Example Notebook: Integrate Stokes Q and U Over GMF Shells

This notebook demonstrates **Step 3** of the dust emission modeling pipeline: integrating the **Stokes Q and U parameters** over spherical shells based on GMF-induced polarization geometry. It uses the output angle maps from Step 2.5 to construct full-sky Q/U HEALPix maps.

---
from integrate_GMF_QU_shells_jax import (
    sum_QU_over_shells_jax,
    save_QU_to_fits,
    plot_QU_maps,
)






