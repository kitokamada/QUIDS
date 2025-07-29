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

## 📓 Example Notebook: Generating and Visualizing Shell Magnetic Field Maps

Located in the `notebooks/` directory, this example demonstrates how to:

1. Generate shell-wise magnetic field FITS maps from a GMF cube using `generate_shell_fits`.
2. Load a shell file and visualize the field using `healpy`.

---


