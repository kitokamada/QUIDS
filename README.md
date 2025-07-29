# 🌌 QUIDS: Q/U Integrated Dust Shells

**QUIDS** is a Python package for generating synthetic Stokes **Q** and **U** polarization maps from 3D dust density and Galactic magnetic field (GMF) shell data. It computes polarized emission by integrating over spherical shells using polarization angles derived from GMF models (e.g., UF23, JF12).

---

## 🧭 Pipeline Overview

| Step | Description |
|------|-------------|
| **Step 0** | *(Optional)* Generate 3D GMF vector cube  |
| **Step 1** | Interpolate GMF vectors onto spherical HEALPix shells |
| **Step 2** | Generate log-spaced spherical shell coordinates |
| **Step 2.5** | Compute polarization & inclination angles using JAX > 🧑‍🔬 Developed in collaboration with [Dr. Gina Panopoulou](https://gpanopoulou.github.io), whose methods and research in magnetic field modeling contributed to the implementation of polarization angle calculations in Step 2.5.|
| **Step 3** | Integrate Q and U across shells (GMF-only) |
| **Step 4** | Integrate Q and U across shells, weighted by dust density |

Each step has a matching example notebook in the `notebooks/` directory for easy experimentation.

---

## Features

- Computes:
  - Per-shell Stokes Q and U maps
  - Total integrated Q/U maps
  - (Optional) Polarized intensity \( P = \sqrt{Q^2 + U^2} \)
- Supports:
  - HEALPix format
  - Galactic coordinate system
- Accelerated by [JAX](https://github.com/google/jax)
- Easily integrates over custom radial ranges
- Built-in Mollweide visualization with `healpy.mollview`

---

## 📦 Requirements

- `numpy`
- `jax`
- `healpy`
- `astropy`
- `matplotlib`

---

## 🛠️ Step 0 (Optional): UF23 Field Grid Generator

### 📄 `UF23-Calculation/writeUF23Grid_earth.cpp`

Standalone C++ script that generates a heliocentric GMF grid using the **UF23 model**.

#### 🔧 Purpose

Produces a 3D Cartesian grid of GMF vectors (\( B_x, B_y, B_z \)) centered on Earth. This grid is later interpolated onto spherical shells for use in polarization modeling.

#### 🧪 Compile and Run

```bash
g++ -std=c++17 -O3 writeUF23Grid_earth.cpp UF23Field.cc -o writeUF23Grid_earth
./writeUF23Grid_earth
```


## 🔁 Step 1: Interpolate GMF Onto Spherical Shells

### 📄 `scripts/GMF_shell_mapper.py`
This script interpolates Galactic Magnetic Field (GMF) data from a 3D Cartesian grid (Earth-centered) onto spherical shells using HEALPix angular sampling. The result is a set of FITS files, each storing magnetic field vectors over the sky at a given radius.
### 🧠 Purpose
To prepare shell-wise magnetic field data as a function of radius and sky position, enabling downstream modeling of dust polarization and emission using line-of-sight integration.
### 🔧 Function

```python
generate_shell_fits(
    gmf_filename: str,
    output_dir: str,
    healpix_order: int,
    r_max_pc: float,
    r_spacing_pc: float
)
```
## 📓 Step 1 Example Notebook:
**File**: `notebooks/GMF_shell_mapper_operator.ipynb`  
This notebook demonstrates how to use `generate_shell_fits` from `gif_shell_mapper.py` to interpolate Galactic Magnetic Field (GMF) data onto a set of radial spherical shells sampled in HEALPix.

---

### 🧪 What It Does

- Loads a 3D GMF cube centered at Earth
- Defines radial shell spacing and angular resolution (HEALPix)
- Interpolates the magnetic field components \((B_x, B_y, B_z)\) at each direction and radius
- Saves shell-wise FITS files containing GMF vectors
- Visualizes one shell's field component using `healpy.mollview`

---

## 🔁 Step 2: Generate Log-Spaced Shell Coordinates

### 📄 `scripts/generate_log_spherical_shell_coordinates.py`
This script generates **logarithmically spaced spherical shell coordinates** sampled in angular direction using a **HEALPix grid**, and saves them in FITS binary tables.

Each output file corresponds to one shell and contains the Galactic Cartesian coordinates (X, Y, Z) and the constant radius \( R \) for every pixel.
### 🧠 Purpose
To define a spherical coordinate system over which Galactic Magnetic Field vectors can be projected and analyzed. These shells are used in Step 2.5 to compute polarization and inclination angles per pixel.
### 🔧 Function

```python
generate_log_spherical_shells(
    output_folder: str,
    healpix_order: int,
    r_min_pc: float,
    r_max_pc: float,
    num_shells: int
)
```


## 📓 Step 2 Example Notebook
**File**: `notebooks/shell_coord_builder_operator.ipynb`  
This notebook demonstrates how to generate **logarithmically spaced spherical shell coordinates** using the `generate_log_spherical_shells` function from `generate_log_spherical_shell_coordinates.py`.

These coordinates define the positions (X, Y, Z) in Galactic Cartesian space for each radial shell and are essential for projecting GMF vectors onto the shell surface.

---

### 🧪 What It Does

- Defines logarithmically spaced radial shells between `r_min` and `r_max`
- Uses HEALPix to sample angular directions on each shell
- Converts from spherical (r, θ, φ) to Cartesian (x, y, z) coordinates
- Saves each shell’s coordinates to a FITS binary table

---

---

## 🔁 Step 2.5: Compute Polarization & Inclination Angles

### 📄 `scripts/compute_GMF_angles_jax.py`

This script computes the **polarization angle** (β) and **inclination angle** (α) for each pixel on every shell using the Galactic Magnetic Field (GMF) vector and the line-of-sight (LOS) direction. It is **JAX-accelerated** for fast, vectorized computation across the full sky.

---

### 🧠 Purpose

To convert raw GMF vectors and LOS directions into angular quantities used in computing Stokes Q and U in Steps 3 and 4:
- **Polarization angle (β):** orientation of the B-field projection in the plane of the sky
- **Inclination angle (α):** angle between the B-field and the LOS

---

### 🔧 Function

```python
process_all_shells(coord_folder, field_folder, output_folder)
```

---

## 📓 Step 2.5 Example Notebook: Compute Polarization & Inclination Angles (JAX Accelerated)

**File**: `notebooks/angle_calculation_operator.ipynb`

This notebook demonstrates how to compute **polarization angles** (β) and **inclination angles** (α) for each shell using precomputed GMF vectors and spherical coordinates. The calculation uses `jax.jit` for high-performance full-sky evaluation.

---

### 🧪 What It Does

- Loads spherical shell coordinates (`X_PC`, `Y_PC`, `Z_PC`)
- Loads GMF field components from Step 1 (`B_x_uG`, `B_y_uG`, `B_z_uG`)
- Computes:
  - **Polarization angle** (β): orientation of the projected magnetic field
  - **Inclination angle** (α): angle between magnetic field and line of sight
- Saves these angle maps as shell-wise FITS files

---


## 🔁 Step 3: Integrate Stokes Q and U Over Shells (GMF Only)

### 📄 `scripts/integrate_GMF_QU_shells_jax.py`

This script performs **Step 3** of the dust emission modeling pipeline: it integrates the **GMF-derived Stokes Q and U parameters** over multiple spherical shells using precomputed inclination and polarization angles from Step 2.5.

---

### 🧠 Purpose

To construct full-sky synthetic Stokes Q and U maps by summing contributions from each shell. This step reveals the **pure magnetic geometry** effects on polarization, **without any dust density weighting**.

---

### 🔧 Functions

```python
sum_QU_over_shells_jax(folder_path)
```
---

## 📓 Step 3 Example Notebook: Integrate Stokes Q and U Over GMF Shells

**File**: `notebooks/integrate_GMF_QU_operator.ipynb`

This notebook demonstrates how to compute the **total integrated Stokes Q and U** maps over all shells using only the Galactic Magnetic Field geometry — without any dust weighting.

---

### 🧪 What It Does

- Loads inclination angle (α) and polarization angle (β) from per-shell FITS files
- Computes per-shell Stokes Q and U maps
- Sums over all shells to get integrated maps
- Saves the final Q/U maps as a FITS file
- Visualizes them using `healpy.mollview`

---

---

## 🔁 Step 4: Dust-Weighted Q/U Integration

### 📄 `scripts/integrate_QU_with_dust_jax.py`

This script performs **Step 4** of the QUIDS pipeline: integrating Stokes **Q** and **U** polarization maps over spherical shells, now weighted by a 3D **dust density distribution** provided as a FITS cube.

It extends Step 3 by modulating each shell's contribution to polarization using the spatially varying dust content, resulting in more realistic synthetic sky maps.

---

### 🧠 Purpose

To produce physically motivated polarization maps by combining magnetic alignment geometry with the spatial distribution of dust. This simulates what a satellite like Planck would observe for thermal dust emission in polarization.

---

### 🧮 Equations

For each shell:
\[
Q_i = \sin^2(\alpha_i) \cdot \cos(2\beta_i) \cdot n_d(r_i, \hat{n})
\]
\[
U_i = \sin^2(\alpha_i) \cdot \sin(2\beta_i) \cdot n_d(r_i, \hat{n})
\]

Then sum across all shells:
\[
Q_{\text{total}} = \sum_i Q_i, \quad U_{\text{total}} = \sum_i U_i
\]

---

### 🔧 Functions

```python
sum_QU_over_shells_jax(polar_folder, nd_fits_path)
```
---

## 📓 Step 4 Example Notebook:
**File**: `notebooks/dust_QU_shell_integrator_operator.ipynb`

This notebook demonstrates how to compute and visualize **dust-weighted Stokes Q and U** polarization maps over spherical shells using the final step of the QUIDS pipeline.

It combines the magnetic field geometry (via polarization and inclination angles) with the 3D dust density distribution to produce realistic synthetic sky maps.

---

### 🧪 What It Does

- Loads precomputed `Inclination_Angle_deg` and `Polarization_Angle_deg` per shell
- Loads 3D dust density cube from a FITS file
- Computes:
  - Per-shell dust-weighted Q and U maps
  - Final integrated Q and U maps
- Optionally saves full stacks of Q/U per shell
- Visualizes results using `healpy.mollview`

---




