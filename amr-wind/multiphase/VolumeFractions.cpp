#include "amr-wind/multiphase/Multiphase.H"
#include "amr-wind/multiphase/VolumeFractions_K.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/equation_systems/BCOps.H"

namespace amr_wind {

void Multiphase::compute_interface_normal()
{
/** Computes the normal vector based on the mixed-Youngs centered (MYC) method
 *  The implementation follows that of Aulisa et al. 2007
 */

    const int nlevels = m_sim.repo().num_active_levels();
    const auto& geom = m_sim.mesh().Geom();

    for (int lev=0; lev < nlevels; ++lev) {
    
        auto& vof = (*m_vof)(lev);
        auto& normal = m_normal(lev);
        const amrex::Real dx = geom[lev].CellSize()[0];
        const amrex::Real dy = geom[lev].CellSize()[1];
        const amrex::Real dz = geom[lev].CellSize()[2];

        const amrex::Real idx = 1.0 / dx;
        const amrex::Real idy = 1.0 / dy;
        const amrex::Real idz = 1.0 / dz;

        for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& normal_arr = normal.array(mfi);
            const auto& fraction_arr = vof.const_array(mfi);

            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    mixed_Youngs_centered(i, j, k, idx, idy, idz, fraction_arr, normal_arr);
                    });
        }

    }
}


void Multiphase::compute_fraction_intercept()
{
/** Computes the fraction intercept alpha
 */

    const int nlevels = m_sim.repo().num_active_levels();
    const auto& geom = m_sim.mesh().Geom();

    for (int lev=0; lev < nlevels; ++lev) {
    
        auto& vof = (*m_vof)(lev);
        auto& normal = m_normal(lev);
        const amrex::Real dx = geom[lev].CellSize()[0];
        const amrex::Real dy = geom[lev].CellSize()[1];
        const amrex::Real dz = geom[lev].CellSize()[2];

        const amrex::Real idx = 1.0 / dx;
        const amrex::Real idy = 1.0 / dy;
        const amrex::Real idz = 1.0 / dz;

        for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& normal_arr = normal.array(mfi);
            const auto& fraction_arr = vof.const_array(mfi);

            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    mixed_Youngs_centered(i, j, k, idx, idy, idz, fraction_arr, normal_arr);
                    });
        }

    }

}

}