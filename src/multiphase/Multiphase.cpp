#include "Multiphase.H"
#include "VolumeFractions_K.H"
#include "CFDSim.H"
#include "AMReX_ParmParse.H"
#include "trig_ops.H"

namespace amr_wind {


Multiphase::Multiphase(const CFDSim& sim)
    : m_levelset(sim.repo().get_field("levelset"))
    , m_vof(sim.repo().get_field("vof"))
    , m_density(sim.repo().get_field("density"))
    , m_velocity(sim.repo().get_field("velocity"))
{
    amrex::ParmParse pp("incflo");
    pp.query("ro_air", m_rho_air);
    pp.query("ro_water", m_rho_water);
    pp.query("multiphase_problem", m_multiphase_problem);
}

/** Initialize the vof and density fields at the beginning of the
 *  simulation.
 */
void Multiphase::initialize_fields(
    int level,
    const amrex::Geometry& geom)
{
    using namespace utils;

    auto& levelset=m_levelset(level);
    auto& vof = m_vof(level);
    auto& velocity = m_velocity(level);
    
    for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();

        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();
        const auto& probhi = geom.ProbHiArray();
        auto phi = levelset.array(mfi);
        auto vel = velocity.array(mfi);

        //Hardcoded constants at the moment
        const amrex::Real a=4.; 
        const amrex::Real b=2.; 
        const amrex::Real c=2.; 
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
            const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
            const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
            const amrex::Real x0= 0.5*(problo[0]+probhi[0]);
            const amrex::Real y0= 0.5*(problo[1]+probhi[1]);
            const amrex::Real z0= 0.5*(problo[2]+probhi[2]); 
            phi(i,j,k) = -((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0))
                         *(std::sqrt(x*x/(a*a)+y*y/(b*b)+z*z/(c*c))-1); 
            vel(i,j,k) = 0.;
        });
    }
    // compute density based on the volume fractions
    set_density(level,geom);
}

// Reconstruct the interface based on the levelset/tracking function approach

void Multiphase::set_density(
        int level,
        const amrex::Geometry& geom)
{
    using namespace utils;
    
    auto& level_set = m_levelset(level);
    auto& density = m_density(level);

    const amrex::Real dx = geom.CellSize()[0];
    const amrex::Real dy = geom.CellSize()[1];
    const amrex::Real dz = geom.CellSize()[2];
    const amrex::Real ds = std::cbrt(dx * dy * dz);
    const amrex::Real epsilon = 2.*ds;

    for (amrex::MFIter mfi(level_set); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();

        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();
        auto phi = level_set.array(mfi);
        auto Density = density.array(mfi);

        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (phi(i,j,k)>epsilon){
            const amrex::Real H=1.0;
            Density(i,j,k) = m_rho_water*H+m_rho_air*(1.-H);
            }else if (phi(i,j,k)<-epsilon){
            const amrex::Real H=0.;
            Density(i,j,k) = m_rho_water*H+m_rho_air*(1.-H);
            }else{
            const amrex::Real H=0.5*(1+phi(i,j,k)/(2*epsilon)+1./M_PI*std::sin(phi(i,j,k)*M_PI/epsilon));
            Density(i,j,k) = m_rho_water*H+m_rho_air*(1.-H);
            }
        });
    }

}

} // namespace amr_wind
