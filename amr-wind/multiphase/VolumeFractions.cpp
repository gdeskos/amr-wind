#include "amr-wind/multiphase/Multiphase.H"
#include "amr-wind/equation_systems/vof/VolumeFractions_K.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/equation_systems/BCOps.H"
#include "amr-wind/derive/derive_K.H"

namespace amr_wind {

// This should be templatized
/** Computes the normal vector based on the mixed-Youngs centered (MYC) method
 *  The implementation follows that of Aulisa et al. 2007
 */
//void Multiphase::compute_interface_normal()
//{
//
//    const int nlevels = m_sim.repo().num_active_levels();
//    const auto& geom = m_sim.mesh().Geom();
//
//    for (int lev = 0; lev < nlevels; ++lev) {
//
//        auto& vof = (*m_vof)(lev);
//        auto& normal = m_normal(lev);
//        const amrex::Real dx = geom[lev].CellSize()[0];
//        const amrex::Real dy = geom[lev].CellSize()[1];
//        const amrex::Real dz = geom[lev].CellSize()[2];
//
//        for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
//            const auto& bx = mfi.tilebox();
//            const auto& normal_arr = normal.array(mfi);
//            const auto& fraction_arr = vof.const_array(mfi);
//
//            amrex::ParallelFor(
//                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
//                    mixed_Youngs_centered<StencilInterior>(
//                        i, j, k, dx, dy, dz, fraction_arr, normal_arr);
//                });
//        }
//    }
//}

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
    // Convert level set phi to VOF function
    const int nlevels = m_sim.repo().num_active_levels();
    const auto& geom = m_sim.mesh().Geom();
    /** We make our calculations from 1 to n-1 using growntilebox(-1)
     *  1) Compute the normal vector mx,my,mz using the levelset function
     *  2) Normalise the normal vector so that mx,my,mz>0 and mx+my+mz=1;
     *  3) Shift alpha to origin
     *  4) Get the fraction volume from alpha
     *  5) Do clipping - make sure that Volume fraction are between 0 and 1
    */

    for (int lev = 0; lev < nlevels; ++lev) {
      auto& vof = (*m_vof)(lev);
      auto& levelset = m_levelset(lev);
      auto& normal = m_normal(lev);
      auto& intercept = m_intercept(lev);

      for (amrex::MFIter mfi(levelset); mfi.isValid(); ++mfi) {
          const auto& bx = mfi.tilebox();
          const auto& cc = vof.array(mfi);
          const auto& ls = levelset.array(mfi);
          const auto& mxyz = normal.array(mfi);
          const auto& alpha_arr = intercept.array(mfi);
          amrex::ParallelFor(
              bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                 // Step (1) -- compute normals
                 // Compute x-direction normal 
                 amrex::Real mm1,mm2,mx,my,mz;
                 mm1 =  ls(i-1,j-1,k-1)+ls(i-1,j-1,k+1)+ls(i-1,j+1,k-1)
                        +ls(i-1,j+1,k+1)+2.0*(ls(i-1,j-1,k)+ls(i-1,j+1,k)
                        +ls(i-1,j,k-1)+ls(i-1,j,k+1))+4.0*ls(i-1,j,k);
                  mm2 = ls(i+1,j-1,k-1)+ls(i+1,j-1,k+1)+ls(i+1,j+1,k-1)
                        +ls(i+1,j+1,k+1)+2.0*(ls(i+1,j-1,k)+ls(i+1,j+1,k)
                        +ls(i+1,j,k-1)+ls(i+1,j,k+1))+4.0*ls(i+1,j,k);
                  mx = (mm1 - mm2)/32.0;
                 // Compute y-direction normal
                  mm1 = ls(i-1,j-1,k-1)+ls(i-1,j-1,k+1)+ls(i+1,j-1,k-1)
                        +ls(i+1,j-1,k+1)+2.0*(ls(i-1,j-1,k)+ls(i+1,j-1,k)
                        +ls(i,j-1,k-1)+ls(i,j-1,k+1))+4.0*ls(i,j-1,k);
                  mm2 = ls(i-1,j+1,k-1)+ls(i-1,j+1,k+1)+ls(i+1,j+1,k-1)
                        +ls(i+1,j+1,k+1)+2.0*(ls(i-1,j+1,k)+ls(i+1,j+1,k)
                        +ls(i,j+1,k-1)+ls(i,j+1,k+1))+4.0*ls(i,j+1,k);
                  my = (mm1 - mm2)/32.0;
                 // Compute z-direction normal
                  mm1 = ls(i-1,j-1,k-1)+ls(i-1,j+1,k-1)+ls(i+1,j-1,k-1)
                        +ls(i+1,j+1,k-1)+2.0*(ls(i-1,j,k-1)+ls(i+1,j,k-1)
                        +ls(i,j-1,k-1)+ls(i,j+1,k-1))+4.0*ls(i,j,k-1);
                  mm2 = ls(i-1,j-1,k+1)+ls(i-1,j+1,k+1)+ls(i+1,j-1,k+1)
                        +ls(i+1,j+1,k+1)+2.0*(ls(i-1,j,k+1)+ls(i+1,j,k+1)
                        +ls(i,j-1,k+1)+ls(i,j+1,k+1))+4.0*ls(i,j,k+1);
                  mz = (mm1 - mm2)/32.0;
                  // Step (2) 
                  mx=std::abs(mx);
                  my=std::abs(my);
                  mz=std::abs(mz);
                  amrex::Real normL1=mx+my+mz;
                  mx=mx/normL1;
                  my=my/normL1;
                  mz=mz/normL1;

                  mxyz(i,j,k,0)=mx;
                  mxyz(i,j,k,1)=my;
                  mxyz(i,j,k,2)=mz;

                  amrex::Real alpha = ls(i,j,k)/normL1;
                  alpha=alpha+0.50;

                  alpha_arr(i,j,k)=alpha;

                  if(alpha>=1.0){
                      cc(i,j,k)=1.0;
                  }else if (alpha<=0.0){
                      cc(i,j,k)=0.0;
                  }else{
                      cc(i,j,k)=FL3D(i, j, k, mx, my, mz,alpha, 0.0, 1.0);
                  }
              });
        }
    }
    do_clipping();
}

} // namespace amr_wind
