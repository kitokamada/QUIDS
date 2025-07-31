"""
integrate_GMF_QU_shells_jax.py

Step 3 of the dust emission modeling pipeline:
This script integrates the GMF-derived Stokes Q and U parameters over multiple
spherical shells. Each shell contains inclination (α) and polarization (β) angles
stored in FITS format. The integration uses:

    Q_i = sin²(α_i) * cos(2β_i)
    U_i = sin²(α_i) * sin(2β_i)

and sums over all shells to produce full-sky Q and U maps in HEALPix format.
This version does NOT yet include any dust model.

Dependencies:
- jax
- astropy
- healpy
- matplotlib
- numpy
"""

import os
import numpy as np
import jax.numpy as jnp
import healpy as hp
import matplotlib.pyplot as plt
from astropy.io import fits


def sum_QU_over_shells_jax(folder_path):
    """
    Compute total Stokes Q and U maps by integrating GMF contributions
    across all spherical shells.

    Parameters
    ----------
    folder_path : str
        Directory containing shell FITS files with 'Inclination_Angle_deg'
        and 'Polarization_Angle_deg' columns.

    Returns
    -------
    Q_total, U_total : jnp.ndarray
        Full-sky HEALPix maps of integrated Q and U.
    """
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.fits')])
    Q_total, U_total = None, None

    for fname in file_list:
        file_path = os.path.join(folder_path, fname)
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            alpha_deg = jnp.array(data['Inclination_Angle_deg'])
            beta_deg = jnp.array(data['Polarization_Angle_deg'])

            # Convert to radians
            alpha_rad = jnp.radians(alpha_deg)
            beta_rad = jnp.radians(beta_deg)

            # Compute shell Q and U
            sin2_alpha = jnp.sin(alpha_rad) ** 2
            Q_partial = sin2_alpha * jnp.cos(2 * beta_rad)
            U_partial = sin2_alpha * jnp.sin(2 * beta_rad)

            # Replace NaNs
            Q_partial = jnp.nan_to_num(Q_partial, nan=0.0)
            U_partial = jnp.nan_to_num(U_partial, nan=0.0)

            # Accumulate
            if Q_total is None:
                Q_total = Q_partial
                U_total = U_partial
            else:
                Q_total += Q_partial
                U_total += U_partial

    return Q_total, U_total


def save_QU_to_fits(Q_array, U_array, output_filename, nside):
    """
    Save Q and U HEALPix maps to a FITS file.

    Parameters
    ----------
    Q_array : ndarray
        Stokes Q map (float32)
    U_array : ndarray
        Stokes U map (float32)
    output_filename : str
        Output FITS filename
    nside : int
        HEALPix NSIDE value to include in header
    """
    col_Q = fits.Column(name='Q_integrated', array=Q_array.astype('float32'), format='E')
    col_U = fits.Column(name='U_integrated', array=U_array.astype('float32'), format='E')
    hdu = fits.BinTableHDU.from_columns([col_Q, col_U])

    hdu.header['NSIDE'] = nside
    hdu.header['PIXTYPE'] = 'HEALPIX'
    hdu.header['ORDERING'] = 'RING'
    hdu.header['COORDSYS'] = 'GALACTIC'
    hdu.header['OBJECT'] = 'Q_U_integrated_over_shells'

    hdu.writeto(output_filename, overwrite=True)
    print(f"✅ Saved Q and U integrated maps to {output_filename}")


def plot_QU_maps(Q_map, U_map, output_prefix="no_dust"):
    """
    Generate and save Mollweide plots of the Q and U maps.

    Parameters
    ----------
    Q_map : ndarray
        Stokes Q map
    U_map : ndarray
        Stokes U map
    output_prefix : str
        Prefix for output image files
    """
    Q_map = np.nan_to_num(Q_map)
    U_map = np.nan_to_num(U_map)

    hp.mollview(Q_map, title="Q Map (Integrated over shells)", unit="Q [arb. units]",
                norm='hist', cmap="RdBu", nest=False)
    plt.savefig(f"{output_prefix}_Q.png")

    hp.mollview(U_map, title="U Map (Integrated over shells)", unit="U [arb. units]",
                norm='hist', cmap="RdBu", nest=False)
    plt.savefig(f"{output_prefix}_U.png")


# Optional direct execution for testing
if __name__ == "__main__":
    folder = "UF23_fits_shell_angles_jax_log"
    output_filename = "UF23_QU_integrated_over_shells_log.fits"

    Q_map, U_map = sum_QU_over_shells_jax(folder)
    save_QU_to_fits(Q_map, U_map, output_filename)
    plot_QU_maps(Q_map, U_map, output_prefix="no_dust")
