#ifndef GHOST_CELL_H_
#define GHOST_CELL_H_

#include <AMReX_FArrayBox.H>
#include <cmath>

namespace amr_wind {
namespace ib {

/** GHOST CELL METHOD
 */

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void ghost_cell(
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
    amrex::Real xIP,
    amrex::Real yIP,
    amrex::Real zIP,
    amrex::Real dx,
    amrex::Real dy,
    amrex::Real dz,
    amrex::Vector<amrex::Real> Vmatrix)
{
    // Find neighboring cells by defining a 8x3 vector
}

} // namespace ib
} // namespace amr_wind

#endif // GHOST_CELL.H
