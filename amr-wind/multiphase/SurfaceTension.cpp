#include "amr-wind/multiphase/SurfaceTension.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/multiphase/Multiphase.H"

#include "AMReX_ParmParse.H"
#include "AMReX_Gpu.H"

namespace amr_wind {
namespace pde {
namespace icns {

SurfaceTension::SurfaceTension(const CFDSim& sim)
    : m_mesh(sim.mesh())
    , m_density(sim.repo().get_field("density"))
    , m_levelset(sim.repo().get_field("levelset"))
    , m_lsnormal(sim.repo().get_field("ls_normal"))
    , m_lscurv(sim.repo().get_field("ls_curvature"))
{
    amrex::ParmParse pp("incflo");
    pp.query("surface_tension_coeff", m_sigma);
}

SurfaceTension::~SurfaceTension() = default;

void SurfaceTension::operator()(
    const int lev,
    const amrex::MFIter& mfi,
    const amrex::Box& bx,
    const FieldState,
    const amrex::Array4<amrex::Real>& src_term) const
{
  
   auto& geom = m_mesh.Geom(lev);
   auto& density = m_density(lev);
   auto& levelset = m_levelset(lev);
   auto& normal = m_lsnormal(lev);
   auto& curvature = m_lscurv(lev);

   const auto& dx = geom.CellSizeArray();
   auto rho = density.array(mfi);
   auto phi = levelset.array(mfi);
   auto n = normal.array(mfi);
   auto kappa = curvature.array(mfi);
   const amrex::Real epsilon=2.*std::cbrt(dx[0]*dx[1]*dx[2]);

   amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        
        if (phi(i, j, k) > epsilon || phi(i,j,k)<-epsilon) {
            src_term(i, j, k, 0) += 0.;
            src_term(i, j, k, 1) += 0.;
            src_term(i, j, k, 2) += 0.;
        }else{
            const amrex::Real delta_st=1./(2*epsilon)*(1+std::cos(M_PI*phi(i,j,k)/epsilon));
            src_term(i, j, k, 0) += m_sigma*kappa(i,j,k)*delta_st*n(i,j,k,0)/rho(i,j,k);
            src_term(i, j, k, 1) += m_sigma*kappa(i,j,k)*delta_st*n(i,j,k,1)/rho(i,j,k);
            src_term(i, j, k, 2) += m_sigma*kappa(i,j,k)*delta_st*n(i,j,k,2)/rho(i,j,k);
        }
    }); 
}

} // namespace icns
} // namespace pde
} // namespace amr_wind
