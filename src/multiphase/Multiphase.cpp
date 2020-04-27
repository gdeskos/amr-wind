#include "Multiphase.H"
#include "VolumeFractions_K.H"
#include "CFDSim.H"
#include "AMReX_ParmParse.H"
#include "trig_ops.H"

namespace amr_wind {


Multiphase::Multiphase(const CFDSim& sim)
    : m_vof(sim.repo().get_field("vof"))
    , m_density(sim.repo().get_field("density"))
    , m_velocity(sim.repo().get_field("velocity"))
{
    amrex::ParmParse pp("incflo");
    pp.query("ro_air", m_rho_air);
    pp.query("ro_water", m_rho_water);
    pp.query("vof_problem", m_vof_problem);
}

/** Initialize the vof and density fields at the beginning of the
 *  simulation.
 */
void Multiphase::initialize_fields(
    int level,
    const amrex::Geometry& geom)
{
    using namespace utils;

    auto& vof = m_vof(level);
    auto& velocity = m_velocity(level);
    
    for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();

        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();
        auto F = vof.array(mfi);
        auto vel = velocity.array(mfi);

        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
            const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
            const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
            F(i,j,k) =  1.;
            vel(i,j,k) = 0.;
        });
    }
}

void Multiphase::set_density(
        int level,
        const amrex::Geometry& geom)
{
    using namespace utils;
    
    auto& vof = m_vof(level);
    auto& density = m_density(level);

    for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();

        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();
        auto VolumeFraction = vof.array(mfi);
        auto Density = density.array(mfi);

        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            Density(i,j,k) = m_rho_water*VolumeFraction(i,j,k)+m_rho_air*(1.-VolumeFraction(i,j,k));
        });
    }

}

} // namespace amr_wind
