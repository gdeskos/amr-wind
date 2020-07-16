#include "amr-wind/multiphase/Multiphase.H"
#include "amr-wind/multiphase/VolumeFractions_K.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/equation_systems/BCOps.H"
#include "amr-wind/derive/derive_K.H"

namespace amr_wind {

// This should be templatized
/** Computes the normal vector based on the mixed-Youngs centered (MYC) method
 *  The implementation follows that of Aulisa et al. 2007
 */
void Multiphase::compute_interface_normal()
{

    const int nlevels = m_sim.repo().num_active_levels();
    const auto& geom = m_sim.mesh().Geom();

    for (int lev = 0; lev < nlevels; ++lev) {

        auto& vof = (*m_vof)(lev);
        auto& normal = m_normal(lev);
        const amrex::Real dx = geom[lev].CellSize()[0];
        const amrex::Real dy = geom[lev].CellSize()[1];
        const amrex::Real dz = geom[lev].CellSize()[2];

        for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& normal_arr = normal.array(mfi);
            const auto& fraction_arr = vof.const_array(mfi);

            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    mixed_Youngs_centered<StencilInterior>(
                        i, j, k, dx, dy, dz, fraction_arr, normal_arr);
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

    for (int lev = 0; lev < nlevels; ++lev) {

        auto& vof = (*m_vof)(lev);
        auto& normal = m_normal(lev);
        auto& intercept = m_intercept(lev);
        const amrex::Real dx = geom[lev].CellSize()[0];
        const amrex::Real dy = geom[lev].CellSize()[1];
        const amrex::Real dz = geom[lev].CellSize()[2];

        for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& normal_arr = normal.const_array(mfi);
            const auto& fraction_arr = vof.const_array(mfi);
            const auto& alpha_arr = intercept.array(mfi);
            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    alpha_arr(i, j, k) = volume_fraction_intercept(
                        i, j, k, dx, dy, dz, fraction_arr, normal_arr);
                });
        }
    }
}

void Multiphase::reconstruct_volume()
{

    const int nlevels = m_sim.repo().num_active_levels();
    const auto& geom = m_sim.mesh().Geom();

    for (int lev = 0; lev < nlevels; ++lev) {

        auto& vof = (*m_vof)(lev);
        auto& normal = m_normal(lev);
        auto& intercept = m_intercept(lev);
        const amrex::Real dx = geom[lev].CellSize()[0];
        const amrex::Real dy = geom[lev].CellSize()[1];
        const amrex::Real dz = geom[lev].CellSize()[2];
        const auto& problo = geom[lev].ProbLoArray();
        const auto& probhi = geom[lev].ProbHiArray();

        for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& normal_arr = normal.const_array(mfi);
            const auto& fraction_arr = vof.array(mfi);
            const auto& alpha_arr = intercept.const_array(mfi);

            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    amrex::Real x0 = problo[0] + i * dx;
                    amrex::Real y0 = problo[1] + j * dy;
                    amrex::Real z0 = problo[2] + k * dz;
                    compute_volume_fraction(
                        i, j, k, x0, y0, z0, dx, dy, dz, normal_arr, alpha_arr,
                        fraction_arr);
                });
        }
    }
}

void Multiphase::do_clipping()
{
    const int nlevels = m_sim.repo().num_active_levels();

    for (int lev = 0; lev < nlevels; ++lev) {

        auto& vof = (*m_vof)(lev);

        for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& fraction_arr = vof.array(mfi);

            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    amrex::Real eps = 1e-8;
                    if (fraction_arr(i, j, k) < eps) {
                        fraction_arr(i, j, k) = 0.0;
                    } else if (fraction_arr(i, j, k) > 1.0 - eps) {
                        fraction_arr(i, j, k) = 1.0;
                    }
                });
        }
    }
}

void Multiphase::levelset2vof()
{
    // Precompute the volume fraction in cell using the normal vector 
    precompute_volume_fractions_from_levelset();
    do_clipping();
    compute_interface_normal();
    // Compute volume of fluid intercept
    //compute_fraction_intercept();
    // Reconstruct the volume fractions
    //reconstruct_volume();
}

void Multiphase::precompute_volume_fractions_from_levelset()
{
    const int nlevels = m_sim.repo().num_active_levels();
    const auto& geom = m_sim.mesh().Geom();


    for (int lev = 0; lev < nlevels; ++lev) {
      auto& vof = (*m_vof)(lev);
      auto& levelset = m_levelset(lev);
      const amrex::Real dx = geom[lev].CellSize()[0];
      const amrex::Real dy = geom[lev].CellSize()[1];
      const amrex::Real dz = geom[lev].CellSize()[2];

      for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
          const auto& bx = mfi.tilebox();
          const auto& fraction_arr = vof.array(mfi);
          const auto& phi_arr = levelset.array(mfi);

          amrex::ParallelFor(
              bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                  amrex::Real Delta=std::sqrt(dx*dx+dy*dy+dz*dz); 
                  amrex::Real ratio = std::abs(phi_arr(i,j,k)/Delta);
                  if (phi_arr(i,j,k)>0.0 && ratio>1.0) {
                      fraction_arr(i, j, k) = 1.0;
                  } else if (phi_arr(i, j, k)>0.0 && ratio<1.0) {
                      fraction_arr(i, j, k) = ratio+0.5;
                  } else if (phi_arr(i,j,k)<0.0 && ratio<1.0){
                      fraction_arr(i,j,k) = 0.5-ratio;
                  } else if (phi_arr(i,j,k)<0.0 && ratio>1.0){
                      fraction_arr(i,j,k) = 0.0;
                  }
              });
      } 
    } 
}

} // namespace amr_wind
