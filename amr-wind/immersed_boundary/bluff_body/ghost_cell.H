#ifndef GHOST_CELL_H_
#define GHOST_CELL_H_

#include <AMReX_FArrayBox.H>
#include <cmath>

namespace amr_wind {
namespace ib {

/** GHOST CELL METHOD
 */

/*AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void ghost_cell(
    int i,
    int j,
    int k,
    amrex::Real xgc,
    amrex::Real ygc,
    amrex::Real zgc,
    amrex::Real nx,
    amrex::Real ny,
    amrex::Real nz,
    amrex::Real& velx,
    amrex::Real& vely,
    amrex::Real& velz) noexcept
{}
*/

/** VANDERMONDE MATRIX
 *
 *  phi(x1,x2,x3) = C1*x1*x2*x3 + C2*x1*x2 + C3*x2*x3 + C4*x1*x3 + C5*x1 + C6*x2
 * + C7*x3 + C8
 *
 *  THE Vandemonde Matrix takes the form (underscores denote the cell-centre
 * points)
 *
 *        [ x*y*z|_1     x*y|_1    y*z|_1 .....  z|_1      1  ]
 *        | x*y*z|_2     x*y|_2    y*z|_2 .....  z|_2      1  |
 *  [V] = | x*y*z|_3     x*y|_3    y*z|_3 .....  z|_3      1  |
 *        |   ...             ...         ...            ...  |
 *        [ x*y*z|_8     x*y|_8    y*z|_8 .....  z|_8      1  ]
 *
 *        8 ___________ 7
 *         /|         /|
 *      5 /_|______6_/ |
 *        | |        | |
 *        | |_4______| |
 *        | /        | / 3
 *        |/_________|/
 *        1          2
 *
 *            / y
 *      z ^  /
 *        | /
 *        |/----> x
 */

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void vandermonde_matrix(
    const amrex::Vector<amrex::Real>& x_loc_NIP,
    const amrex::Vector<amrex::Real>& y_loc_NIP,
    const amrex::Vector<amrex::Real>& z_loc_NIP,
    amrex::Array2D<amrex::Real, 0, 7, 0, 7>& Vmatrix)
{
    const int Np = 8;

    for (int row = 0; row < Np; row++) {
        amrex::Real xloc = x_loc_NIP[row];
        amrex::Real yloc = y_loc_NIP[row];
        amrex::Real zloc = z_loc_NIP[row];

        Vmatrix(row, 0) = xloc * yloc * zloc;
        Vmatrix(row, 1) = xloc * yloc;
        Vmatrix(row, 2) = xloc * zloc;
        Vmatrix(row, 3) = yloc * zloc;
        Vmatrix(row, 4) = xloc;
        Vmatrix(row, 5) = yloc;
        Vmatrix(row, 6) = zloc;
        Vmatrix(row, 7) = 1.0;
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void gauss_siedel_coeffVector(
    amrex::Array2D<amrex::Real, 0, 7, 0, 7>& Vmatrix,
    const amrex::Vector<amrex::Real>& phi,
    amrex::Vector<amrex::Real>& coeff)
{
    const int Np = 8;
    const int Niter = 100;
    int n = 0;

    amrex::Real sum_c1;
    amrex::Real sum_c2;

    while (n < Niter) {
        for (int row = 0; row < Np; row++) {
            sum_c1 = 0.0;
            sum_c2 = 0.0;

            amrex::Real inv_Vrr = 1.0 / Vmatrix(row, row);

            for (int col = 0; col <= (row - 1); col++) {
                sum_c1 = sum_c1 + Vmatrix(row, col) * coeff[col];
            }

            for (int col = row + 1; col <= (Np - 1); col++) {
                sum_c2 = sum_c2 + Vmatrix(row, col) * coeff[col];
            }

            coeff[row] = inv_Vrr * (phi[row] - sum_c1 - sum_c2);
        }
        n++;
    }
}

} // namespace ib
} // namespace amr_wind

#endif // GHOST_CELL.H
