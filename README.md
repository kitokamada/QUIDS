# 🌌 QUIDS: Q/U Integrated Dust Shells

**QUIDS** is a Python package for generating synthetic Stokes **Q** and **U** polarization maps from 3D dust density and Galactic magnetic field (GMF) shell data. It computes polarized emission by integrating over spherical shells using polarization angles derived from GMF models (e.g., UF23, JF12).

---

## 🧭 Pipeline Overview

| Step | Description |
|------|-------------|
| **Step 0** | *(Optional)* Generate 3D GMF vector cube  |
| **Step 1** | Interpolate GMF vectors onto spherical HEALPix shells |
| **Step 2** | Generate log-spaced spherical shell coordinates |
| **Step 2.5** | Compute polarization & inclination angles using JAX |
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


---

## 🔁 Step 1: Interpolate GMF Onto Spherical Shells

### 📄 `scripts/gif_shell_mapper.py`

This script interpolates Galactic Magnetic Field (GMF) data from a 3D Cartesian grid (Earth-centered) onto spherical shells using HEALPix angular sampling. The result is a set of FITS files, each storing magnetic field vectors over the sky at a given radius.

###  Purpose

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

