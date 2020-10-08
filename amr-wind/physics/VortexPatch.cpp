#include "amr-wind/physics/VortexPatch.H"
#include "amr-wind/physics/VortexPatchFieldInit.H"
#include "amr-wind/CFDSim.H"

namespace amr_wind {

VortexPatch::VortexPatch(CFDSim& sim)
    : m_sim(sim)
    , m_velocity(sim.repo().get_field("velocity"))
    , m_density(sim.repo().get_field("density"))
{
    // Register levelset equation
    auto& levelset_eqn = sim.pde_manager().register_transport_pde("Levelset");

    // Defer getting levelset field until PDE has been registered
    m_levelset = &(levelset_eqn.fields().field);

    // Instantiate the VortexPatch field initializer
    m_field_init.reset(new VortexPatchFieldInit());
}

/** Initialize the velocity and levelset fields at the beginning of the
 *  simulation.
 *
 *  \sa amr_wind::VortexPatchFieldInit
 */
void VortexPatch::initialize_fields(
    int level,
    const amrex::Geometry& geom)
{
    auto& velocity = m_velocity(level);
    auto& density = m_density(level);
    auto& levelset = (*m_levelset)(level);

    for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();

        (*m_field_init)(
            vbx, geom, velocity.array(mfi), density.array(mfi),
            levelset.array(mfi));
    }
}


void VortexPatch::post_advance_work()
{
    const auto& time = m_sim.time().current_time();
    amrex::Print() << time <<std::endl;
    const int nlevels = m_sim.repo().num_active_levels();
    const auto& geom = m_sim.mesh().Geom();

    //Overriding the velocity field
    for (int lev = 0; lev < nlevels; ++lev) {
        for (amrex::MFIter mfi(m_velocity(lev)); mfi.isValid(); ++mfi) {
            const auto& vbx = mfi.validbox();
            const auto& dx = geom[lev].CellSizeArray(); 
            const auto& problo = geom[lev].ProbLoArray();
            
            auto vel = m_velocity(lev).array(mfi);
            amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    const amrex::Real x = problo[0] + (i+0.5)*dx[0];
                    const amrex::Real y = problo[1] + (j+0.5)*dx[1];
                    const amrex::Real z = problo[2] + (k+0.5)*dx[2];
                    vel(i,j,k,0) = 2.0*std::sin(M_PI*x)*std::sin(M_PI*x)
                                   *std::sin(2.0*M_PI*y)*std::sin(2.0*M_PI*z)*std::cos(M_PI*time/m_TT);
                    vel(i,j,k,1) = -std::sin(M_PI*y)*std::sin(M_PI*y)
                                   *std::sin(2.0*M_PI*x)*std::sin(2.0*M_PI*z)*std::cos(M_PI*time/m_TT);
                    vel(i,j,k,2) = -std::sin(M_PI*z)*std::sin(M_PI*z)
                                   *std::sin(2.0*M_PI*x)*std::sin(2.0*M_PI*y)*std::cos(M_PI*time/m_TT);
                });
            }


    }
}

} // namespace amr_wind