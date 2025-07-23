#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "UF23Field.h"

int main() {
    // === Select GMF model ===
    UF23Field field(UF23Field::base);  // Options: base, cre10, expX, etc.

    // === Grid boundaries and resolution ===
    double x_min = -20.0, x_max = 20.0, dx = 0.1;
    double y_min = -20.0, y_max = 20.0, dy = 0.1;
    double z_min = -20.0, z_max = 20.0, dz = 0.1;

    // === Calculate grid shape ===
    int Nx = static_cast<int>((x_max - x_min) / dx + 1);
    int Ny = static_cast<int>((y_max - y_min) / dy + 1);
    int Nz = static_cast<int>((z_max - z_min) / dz + 1);
    std::cout << "Grid shape: " << Nx << " x " << Ny << " x " << Nz
              << " = " << static_cast<long long>(Nx) * Ny * Nz << " points\n";

    // === Open output file ===
    std::ofstream fout("GMF_grid.dat");
    fout << std::fixed << std::setprecision(6);

    // === Write header ===
    fout << "# X[kpc]\tY[kpc]\tZ[kpc]\tB_X[μG]\tB_Y[μG]\tB_Z[μG]\tB_tot[μG]\n";

    // === Main loop over all grid points ===
    for (int i = 0; i < Nx; ++i) {
        double x = x_min + i * dx;
        for (int j = 0; j < Ny; ++j) {
            double y = y_min + j * dy;
            for (int k = 0; k < Nz; ++k) {
                double z = z_min + k * dz;

                Vector3 pos(x, y, z);
                Vector3 B = field(pos);
                double Btot = std::sqrt(B.x * B.x + B.y * B.y + B.z * B.z);

                fout << x << "\t" << y << "\t" << z << "\t"
                     << B.x << "\t" << B.y << "\t" << B.z << "\t"
                     << Btot << "\n";
            }
        }
    }

    fout.close();
    std::cout << "✅ Done. Output written to GMF_grid.dat\n";
    return 0;
}
