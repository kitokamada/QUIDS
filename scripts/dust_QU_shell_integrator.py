import os
import numpy as np
import jax
import jax.numpy as jnp
import healpy as hp
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt

def sum_QU_over_shells_jax(polar_folder, nd_fits_path):
    """
    Compute total integrated Stokes Q and U maps over all shells using
    dust density and polarization angle maps.

    Parameters
    ----------
    polar_folder : str
        Folder containing per-shell FITS files with inclination and polarization angles.
    nd_fits_path : str
        Path to FITS file containing dust density shells (shape: [Nr, Npix], NESTED).

    Returns
    -------
    Q_total : jax.numpy.ndarray
        Integrated Stokes Q map (RING-ordered).
    U_total : jax.numpy.ndarray
        Integrated Stokes U map (RING-ordered).
    """
    # === Load all n_d shells from the single FITS file ===
    with fits.open(nd_fits_path) as hdul_nd:
        nd_data_all = hdul_nd[0].data  # shape: (Nr, Npix) — assumed NESTED

    polar_files = sorted([f for f in os.listdir(polar_folder) if f.endswith('.fits')])

    Q_total, U_total = None, None

    for polar_fname in polar_files:
        shell_index_str = polar_fname[-9:-5]  # e.g., '0123' from 'polar_0123.fits'
        try:
            shell_index = int(shell_index_str)
        except ValueError:
            print(f"Skipping {polar_fname}: invalid index format.")
            continue

        if shell_index >= nd_data_all.shape[0]:
            print(f"Skipping {polar_fname}: index {shell_index} out of bounds.")
            continue

        # === Reorder dust data from NESTED to RING ===
        nd_data_nest = nd_data_all[shell_index]
        nd_data_ring = hp.reorder(nd_data_nest, n2r=True)
        nd_data = jnp.array(nd_data_ring)
        nd_data = jnp.nan_to_num(nd_data, nan=0.0)

        # === Load polarization angles for this shell ===
        polar_path = os.path.join(polar_folder, polar_fname)
        with fits.open(polar_path) as hdul_polar:
            polar_data = hdul_polar[1].data
            alpha_deg = jnp.array(polar_data['Inclination_Angle_deg'])
            beta_deg = jnp.array(polar_data['Polarization_Angle_deg'])

            alpha_rad = jnp.radians(alpha_deg)
            beta_rad = jnp.radians(beta_deg)

            sin2_alpha = jnp.sin(alpha_rad) ** 2
            cos2_beta = jnp.cos(2 * beta_rad)
            sin2_beta = jnp.sin(2 * beta_rad)

            # === Compute weighted Q and U partial maps ===
            Q_partial = sin2_alpha * cos2_beta * nd_data
            U_partial = sin2_alpha * sin2_beta * nd_data

            Q_partial = jnp.nan_to_num(Q_partial, nan=0.0)
            U_partial = jnp.nan_to_num(U_partial, nan=0.0)

            if Q_total is None:
                Q_total = Q_partial
                U_total = U_partial
            else:
                Q_total += Q_partial
                U_total += U_partial

    return Q_total, U_total

def save_QU_to_fits(Q_array, U_array, output_filename):
    """
    Save Q and U maps into a single binary table FITS file.

    Parameters
    ----------
    Q_array : array-like
        Stokes Q map (1D, length = Npix).
    U_array : array-like
        Stokes U map (1D, length = Npix).
    output_filename : str
        Path to output FITS file.
    """
    col_Q = fits.Column(name='Q_integrated', array=Q_array.astype('float32'), format='E')
    col_U = fits.Column(name='U_integrated', array=U_array.astype('float32'), format='E')
    hdu = fits.BinTableHDU.from_columns([col_Q, col_U])

    hdu.header['NSIDE'] = 256
    hdu.header['PIXTYPE'] = 'HEALPIX'
    hdu.header['ORDERING'] = 'RING'
    hdu.header['COORDSYS'] = 'GALACTIC'
    hdu.header['OBJECT'] = 'Q_U_integrated_over_shells'

    hdu.writeto(output_filename, overwrite=True)
    print(f"Saved Q and U integrated maps to {output_filename}")


def compute_QU_per_shell_jax(polar_folder, nd_fits_path, assume_nested=True):
    # Load all n_d shells from single file
    with fits.open(nd_fits_path) as hdul_nd:
        nd_data_all = hdul_nd[0].data  # shape: (Nr, Npix)
    print(f"Loaded nd_data_all with shape: {nd_data_all.shape}")

    polar_files = sorted([f for f in os.listdir(polar_folder) if f.endswith('.fits')])
    Q_shells = []
    U_shells = []

    for polar_fname in polar_files:
        shell_index_str = polar_fname[-9:-5]  # e.g., '0123' from 'polar_0123.fits'
        try:
            shell_index = int(shell_index_str)
        except ValueError:
            print(f"Skipping {polar_fname}: invalid index format.")
            continue

        if shell_index >= nd_data_all.shape[0]:
            print(f"Skipping {polar_fname}: index {shell_index} out of bounds.")
            continue

        # Load and optionally reorder nd_data
        nd_data_np = np.nan_to_num(nd_data_all[shell_index], nan=0.0)
        if assume_nested:
            nd_data_np = hp.reorder(nd_data_np, n2r=True)  # Convert NESTED to RING
        nd_data = jnp.array(nd_data_np)

        # Load polarization angles
        polar_path = os.path.join(polar_folder, polar_fname)
        with fits.open(polar_path) as hdul_polar:
            polar_data = hdul_polar[1].data
            alpha_deg = jnp.array(polar_data['Inclination_Angle_deg'])
            beta_deg  = jnp.array(polar_data['Polarization_Angle_deg'])

            alpha_rad = jnp.radians(alpha_deg)
            beta_rad  = jnp.radians(beta_deg)

            sin2_alpha = jnp.sin(alpha_rad) ** 2
            cos2_beta = jnp.cos(2 * beta_rad)
            sin2_beta = jnp.sin(2 * beta_rad)

            Q_partial = sin2_alpha * cos2_beta * nd_data
            U_partial = sin2_alpha * sin2_beta * nd_data

            Q_partial = jnp.nan_to_num(Q_partial, nan=0.0)
            U_partial = jnp.nan_to_num(U_partial, nan=0.0)

            Q_shells.append(Q_partial)
            U_shells.append(U_partial)

    # Stack into arrays of shape (Nshells, Npix)
    Q_shells = jnp.stack(Q_shells, axis=0)
    U_shells = jnp.stack(U_shells, axis=0)

    print(f"Computed Q/U maps: shape = {Q_shells.shape}")
    return Q_shells, U_shells  # shape: (Nshells, Npix)





def save_QU_shells_to_fits(Q_shells, U_shells, q_filename="Q_shells.fits", u_filename="U_shells.fits"):
    # Convert from JAX to NumPy
    Q_shells_np = np.array(Q_shells)
    U_shells_np = np.array(U_shells)

    # Create FITS HDUs
    hdu_q = fits.PrimaryHDU(data=Q_shells_np)
    hdu_u = fits.PrimaryHDU(data=U_shells_np)

    # Write to disk
    hdu_q.writeto(q_filename, overwrite=True)
    hdu_u.writeto(u_filename, overwrite=True)

    print(f"✅ Saved: {q_filename} and {u_filename}")



