# gif_shell_mapper.py

import os
import numpy as np
import pandas as pd
import healpy as hp
from scipy.interpolate import RegularGridInterpolator
from astropy.io import fits


def generate_shell_fits(gmf_filename, output_dir, healpix_order, r_max_pc, r_spacing_pc):
    """
    Generate shell-wise HEALPix FITS files of magnetic field components.

    Parameters:
        gmf_filename   : str   – Path to GMF cube (.dat or .csv file)
        output_dir     : str   – Directory to store the output FITS files
        healpix_order  : int   – HEALPix order (e.g., 8 means NSIDE=256)
        r_max_pc       : float – Maximum radius in parsecs
        r_spacing_pc   : float – Radial spacing between shells in parsecs
    """

    print(f"📥 Loading GMF data from: {gmf_filename}")
    df = pd.read_csv(gmf_filename, sep="\t")
    df.columns = [col.strip() for col in df.columns]

    # Extract unique coordinate grid
    x_unique = np.sort(df['# X[kpc]'].unique())
    y_unique = np.sort(df['Y[kpc]'].unique())
    z_unique = np.sort(df['Z[kpc]'].unique())

    shape = (len(x_unique), len(y_unique), len(z_unique))
    Bx = df["B_X[μG]"].values.reshape(shape)
    By = df["B_Y[μG]"].values.reshape(shape)
    Bz = df["B_Z[μG]"].values.reshape(shape)

    # Interpolators
    interp_Bx = RegularGridInterpolator((x_unique, y_unique, z_unique), Bx, bounds_error=False, fill_value=np.nan)
    interp_By = RegularGridInterpolator((x_unique, y_unique, z_unique), By, bounds_error=False, fill_value=np.nan)
    interp_Bz = RegularGridInterpolator((x_unique, y_unique, z_unique), Bz, bounds_error=False, fill_value=np.nan)

    # Radial shells and HEALPix sampling
    r_vals = np.arange(0, r_max_pc + r_spacing_pc, r_spacing_pc)
    nside = hp.order2nside(healpix_order)
    npix = hp.nside2npix(nside)
    i_pix = np.arange(npix)
    theta, phi = hp.pix2ang(nside, i_pix, nest=False, lonlat=False)

    # Allocate storage
    results_Bx = []
    results_By = []
    results_Bz = []
    results_Btot = []

    print(f"📡 Interpolating magnetic field onto {len(r_vals)} radial shells...")
    for i, r in enumerate(r_vals):
        x = (r * np.sin(theta) * np.cos(phi)) / 1000  # convert pc to kpc
        y = (r * np.sin(theta) * np.sin(phi)) / 1000
        z = (r * np.cos(theta)) / 1000
        points = np.vstack((x, y, z)).T

        bx = interp_Bx(points)
        by = interp_By(points)
        bz = interp_Bz(points)
        btot = np.sqrt(bx**2 + by**2 + bz**2)

        results_Bx.append(bx)
        results_By.append(by)
        results_Bz.append(bz)
        results_Btot.append(btot)

    os.makedirs(output_dir, exist_ok=True)

    print(f"💾 Writing FITS files to: {output_dir}/")
    for i, r in enumerate(r_vals):
        r_arr = np.full_like(results_Bx[i], fill_value=r)

        cols = fits.ColDefs([
            fits.Column(name='radius_pc', array=r_arr, format='E'),
            fits.Column(name='B_x_uG', array=results_Bx[i], format='E'),
            fits.Column(name='B_y_uG', array=results_By[i], format='E'),
            fits.Column(name='B_z_uG', array=results_Bz[i], format='E'),
            fits.Column(name='B_total_uG', array=results_Btot[i], format='E')
        ])

        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header['NSIDE'] = nside
        hdu.header['RADIUS'] = r
        hdu.header['UNIT'] = 'microGauss'

        fname = os.path.join(output_dir, f"shell_r{i:04d}.fits")
        hdu.writeto(fname, overwrite=True)

    print(f"✅ Saved all {len(r_vals)} shell-wise FITS files to '{output_dir}/'.")


# Optional: Standalone run for debugging
if __name__ == "__main__":
    generate_shell_fits(
        gmf_filename="GMF_earth_centered_cube.dat",
        output_dir="fits_shell_fields",
        healpix_order=8,
        r_max_pc=2000,
        r_spacing_pc=2.5
    )
