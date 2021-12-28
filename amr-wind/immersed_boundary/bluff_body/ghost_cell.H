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
    amrex::Real& velx,
    amrex::Real& vely,
    amrex::Real& velz) noexcept
{}

} // namespace ib
} // namespace amr_wind

#endif // GHOST_CELL.H
