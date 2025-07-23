#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "UF23Field.h"

int main() {
    // === Select UF23 GMF model ===
    UF23Field field(UF23Field::base);  // Options: base, cre10, etc.

    // === Define heliocentric cube (Earth at origin) ===
    double x_min = -2.0, x_max = 2.0, dx = 0.01;  // kpc
    double y_min = -2.0, y_max = 2.0, dy = 0.01;
    double z_min = -2.0, z_max = 2.0, dz = 0.01;

    int Nx = static_cast<int>((x_max - x_min) / dx + 1);
    int Ny = static_cast<int>((y_max - y_min) / dy + 1);
    int Nz = static_cast<int>((z_max - z_min) / dz + 1);

    std::cout << "Heliocentric cube shape: " << Nx << " x " << Ny << " x " << Nz
              << " = " << static_cast<long long>(Nx) * Ny * Nz << " points\n";

    std::ofstream fout("GMF_earth_centered_cube.dat");
    fout << std::fixed << std::setprecision(6);

    // === Write header ===
    fout << "# X[kpc]\tY[kpc]\tZ[kpc]\tB_X[μG]\tB_Y[μG]\tB_Z[μG]\tB_tot[μG]\n";

    for (int i = 0; i < Nx; ++i) {
        double x_helio = x_min + i * dx;
        for (int j = 0; j < Ny; ++j) {
            double y_helio = y_min + j * dy;
            for (int k = 0; k < Nz; ++k) {
                double z_helio = z_min + k * dz;

                // === Shift from heliocentric to Galactocentric ===
                double x_gal = x_helio - 8.2;
                double y_gal = y_helio;
                double z_gal = z_helio;

                Vector3 pos(x_gal, y_gal, z_gal);
                Vector3 B = field(pos);
                double Btot = std::sqrt(B.x * B.x + B.y * B.y + B.z * B.z);

                fout << x_helio << "\t" << y_helio << "\t" << z_helio << "\t"
                     << B.x << "\t" << B.y << "\t" << B.z << "\t" << Btot << "\n";
            }
        }
    }

    fout.close();
    std::cout << "✅ Done. Output written to GMF_earth_centered_cube.dat\n";
    return 0;
}
