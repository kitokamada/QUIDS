# QUIDS
Q/U Integrated Dust Shells
(**QUIDS**) is a Python package for generating synthetic Stokes **Q** and **U** polarization maps from 3D dust density and Galactic magnetic field shell data. It computes polarized emission by integrating over spherical shells using input polarization angles derived from Galactic magnetic field models (e.g., UF23, JF12).

---

## Features

- Reads spherical shell data of:
  - Dust density (`n_d`) from a single FITS cube
  - Polarization angles (`Inclination`, `Polarization`) from per-shell FITS files
- Computes:
  - Stokes Q and U maps per shell
  - Total integrated Q/U and polarized intensity maps \( P = \sqrt{Q^2 + U^2} \)
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
