"""
compute_GMF_angles_jax.py

Step 2.5 in the dust emission modeling pipeline.
Computes polarization and inclination angles from GMF vector and coordinate data
stored in spherical shells. Each shell file is processed using JAX for speed.
Outputs are saved as FITS files with polarization and inclination angle maps.

Dependencies:
- jax
- astropy
- healpy
- numpy
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
from astropy.io import fits


@jax.jit
def polarization_anglev2_jax_array(ox, oy, oz, bx, by, bz):
    LOS = jnp.stack([ox, oy, oz], axis=1)
    LOS_unit = LOS / jnp.linalg.norm(LOS, axis=1, keepdims=True)

    z_axis = jnp.array([0.0, 0.0, 1.0])
    z_proj = z_axis - jnp.einsum('ij,j->i', LOS_unit, z_axis)[:, None] * LOS_unit
    z_proj_unit = z_proj / jnp.linalg.norm(z_proj, axis=1, keepdims=True)

    east_vec = jnp.cross(z_proj_unit, LOS_unit)
    east_unit = east_vec / jnp.linalg.norm(east_vec, axis=1, keepdims=True)

    B = jnp.stack([bx, by, bz], axis=1)
    B_unit = B / jnp.linalg.norm(B, axis=1, keepdims=True)

    B_north = jnp.einsum('ij,ij->i', B_unit, z_proj_unit)
    B_east = jnp.einsum('ij,ij->i', B_unit, east_unit)
    B_radial = jnp.abs(jnp.einsum('ij,ij->i', -LOS_unit, B_unit))

    Q = B_north**2 - B_east**2
    U = 2 * B_north * B_east

    pol_angle = 0.5 * jnp.degrees(jnp.arctan2(U, Q))
    pol_angle = jnp.where(pol_angle < 0, pol_angle + 180, pol_angle)
    incl_angle = 90 - jnp.degrees(jnp.arctan2(jnp.sqrt(1 - B_radial**2), B_radial))

    return pol_angle.astype(jnp.float32), incl_angle.astype(jnp.float32)


def process_shell_file(index, coord_folder, field_folder, output_folder):
    coord_filename = os.path.join(coord_folder, f"shell_{index:04d}.fits")
    field_filename = os.path.join(field_folder, f"shell_r{index:04d}.fits")

    with fits.open(coord_filename) as hdul:
        coord_data = hdul[1].data
        r = hdul[1].header.get('RADIUS', None)
        x = coord_data['X_PC']
        y = coord_data['Y_PC']
        z = coord_data['Z_PC']

    with fits.open(field_filename) as hdul:
        field_data = hdul[1].data
        Bx = field_data['B_x_uG']
        By = field_data['B_y_uG']
        Bz = field_data['B_z_uG']

    pol_angles, incl_angles = polarization_anglev2_jax_array(
        jnp.array(x), jnp.array(y), jnp.array(z),
        jnp.array(Bx), jnp.array(By), jnp.array(Bz)
    )

    cols = [
        fits.Column(name='Polarization_Angle_deg', array=np.array(pol_angles), format='E'),
        fits.Column(name='Inclination_Angle_deg', array=np.array(incl_angles), format='E'),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header['RADIUS'] = r
    hdu.header['UNIT'] = 'degrees'
    hdu.header['COMMENT'] = 'Polarization and inclination angle from B-field (JAX accelerated)'

    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.join(output_folder, f"angles_{index:04d}.fits")
    hdu.writeto(output_filename, overwrite=True)
    print(f"✅ Saved: {output_filename}")


def process_all_shells(coord_folder, field_folder, output_folder):
    total_files = len([f for f in os.listdir(coord_folder) if f.endswith('.fits')])
    for i in range(total_files):
        process_shell_file(i, coord_folder, field_folder, output_folder)
