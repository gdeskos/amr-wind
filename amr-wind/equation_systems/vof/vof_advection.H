#ifndef VOF_ADVECTION_H_
#define VOF_ADVECTION_H_

#include <AMReX_FArrayBox.H>
#include "amr-wind/equation_systems/vof/volume_fractions.H"
#include <cmath>

namespace amr_wind {
namespace multiphase {

using namespace amrex;

void lagrangian_split_advection(
    int lev,
    Box const& bx,
    amrex::Array4<Real> const& cc,
    amrex::Array4<amrex::Real const> const& umac,
    amrex::Array4<amrex::Real const> const& vmac,
    amrex::Array4<amrex::Real const> const& wmac,
    amrex::Array4<amrex::Real const> const& fq,
    BCRec const* pbc,
    int const* iconserv,
    Real* p,
    Vector<amrex::Geometry> geom,
    Real dt)
{
    BL_PROFILE("amr-wind::multiphase::lagrangian_split_advection");
}

// onto cell explicit linear mapping
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void volume_fraction_balance(
    const int i,
    const int j,
    const int k,
    const int dim,
    amrex::Array4<amrex::Real> const& volfrac,
    amrex::Array4<amrex::Real> const& vol_in,
    amrex::Array4<amrex::Real> const& vol_cell,
    amrex::Array4<amrex::Real> const& vol_out,
    const int bclo,
    const int bchi,
    const int domlo,
    const int domhi)
{
    using namespace amrex;

    if (dim == 1) {
        // Flux BCs for y-direction
        if (bclo == BCType::ext_dir || bclo == BCType::hoextrap) {
            if (i == domlo) {
                vol_in(i - 1, j, k) = 0.0;
            }
        }

        if (bchi == BCType::ext_dir || bchi == BCType::hoextrap) {
            if (i == domhi) {
                vol_out(i + 1, j, k) = 0.0;
            }
        }
        volfrac(i, j, k) =
            vol_in(i - 1, j, k) + vol_cell(i, j, k) + vol_out(i + 1, j, k);
    } else if (dim == 2) {
        // Flux BCs for y-direction
        if (bclo == BCType::ext_dir || bclo == BCType::hoextrap) {
            if (j == domlo) {
                vol_in(i, j - 1, k) = 0.0;
            }
        }

        if (bchi == BCType::ext_dir || bchi == BCType::hoextrap) {
            if (j == domhi) {
                vol_out(i, j + 1, k) = 0.0;
            }
        }
        volfrac(i, j, k) =
            vol_in(i, j - 1, k) + vol_cell(i, j, k) + vol_out(i, j + 1, k);
    } else if (dim == 3) {
        // Flux BCs for z-direction
        if (bclo == BCType::ext_dir || bclo == BCType::hoextrap) {
            if (k == domlo) {
                vol_in(i, j, k - 1) = 0.0;
            }
        }
        if (bchi == BCType::ext_dir || bchi == BCType::hoextrap) {
            if (k == domhi) {
                vol_out(i, j, k + 1) = 0.0;
            }
        }
        volfrac(i, j, k) =
            vol_in(i, j, k - 1) + vol_cell(i, j, k) + vol_out(i, j, k + 1);
    }
    // Do clipping
    volfrac(i, j, k) = std::max(0.0, std::min(1.0, volfrac(i, j, k)));
}

} // namespace multiphase
} // namespace amr_wind

#endif