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
## 🌌 `writeUF23Grid_earth.cpp`: Field Grid Generator for GMF Dust Modeling

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


