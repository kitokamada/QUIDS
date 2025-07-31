"""
generate_log_spherical_shell_coordinates.py

This script generates log-spaced spherical shell coordinates sampled on a HEALPix grid
and saves them in FITS binary tables. Each file corresponds to a radial shell and contains
the Galactic Cartesian coordinates (X, Y, Z) and the shell radius R.
"""

import os
import numpy as np
import healpy as hp
from astropy.io import fits


def spherical_to_galactic_xyz(r, theta, phi):
    """
    Convert spherical Galactic coordinates (r, theta, phi)
    to Cartesian coordinates (x, y, z) where:
        +x points toward Galactic center (l=0)
        +y points toward l=90°
        +z points toward b=90°
    Parameters
    ----------
    r : float or array
        Radial distance
    theta : float or array
        Colatitude in radians (0 at north pole, π at south pole)
    phi : float or array
        Longitude in radians (from 0 to 2π)
    Returns
    -------
    x, y, z : float or arrays
        Cartesian coordinates
    """
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z


def generate_log_spherical_shells(output_folder, healpix_order,
                                   r_min_pc, r_max_pc, num_shells):
    """
    Generate log-spaced spherical shell coordinates sampled on a HEALPix grid
    and save them as FITS binary tables in the specified folder.

    Parameters
    ----------
    output_folder : str
        Directory to save the shell FITS files
    healpix_order : int
        HEALPix order (e.g., 8 ⇒ NSIDE=256)
    r_min_pc : float
        Minimum shell radius in parsecs
    r_max_pc : float
        Maximum shell radius in parsecs
    num_shells : int
        Number of radial shells to generate
    """
    nside = hp.order2nside(healpix_order)
    npix = hp.nside2npix(nside)
    print(f"📡 HEALPix NSIDE = {nside}, Npix = {npix}")

    i_pix = np.arange(npix)
    theta, phi = hp.pix2ang(nside, i_pix, nest=False, lonlat=False)

    r_vals = np.logspace(np.log10(r_min_pc), np.log10(r_max_pc), num=num_shells)
    os.makedirs(output_folder, exist_ok=True)

    print(f"🌀 Generating {num_shells} log-spaced spherical shells...")

    for i, r in enumerate(r_vals):
        x, y, z = spherical_to_galactic_xyz(r, theta, phi)
        r_col = np.full_like(x, r)

        # Prepare FITS binary table
        cols = [
            fits.Column(name='X_PC', format='E', array=x.astype(np.float32)),
            fits.Column(name='Y_PC', format='E', array=y.astype(np.float32)),
            fits.Column(name='Z_PC', format='E', array=z.astype(np.float32)),
            fits.Column(name='R_PC', format='E', array=r_col.astype(np.float32))
        ]
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header['NSIDE'] = nside
        hdu.header['RADIUS'] = r
        hdu.header['UNIT'] = 'pc'

        filename = f"shell_{i:04d}.fits"
        filepath = os.path.join(output_folder, filename)
        hdu.writeto(filepath, overwrite=True)

        print(f"✅ Saved: {filename}")

    print(f"🎉 All {num_shells} shell coordinates saved to '{output_folder}/'")


# Optional direct run
if __name__ == "__main__":
    generate_log_spherical_shells(
        output_folder="UF23_fits_shells_coordinate_log"
    )
