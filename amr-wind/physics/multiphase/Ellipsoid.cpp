#include "amr-wind/physics/multiphase/Ellipsoid.H"
#include "amr-wind/CFDSim.H"
#include "AMReX_ParmParse.H"
#include "amr-wind/fvm/gradient.H"
#include "amr-wind/core/field_ops.H"

namespace amr_wind {

Ellipsoid::Ellipsoid(CFDSim& sim)
    : m_sim(sim)
    , m_velocity(sim.repo().get_field("velocity"))
    , m_levelset(sim.repo().get_field("levelset"))
{}

/** Initialize the velocity and levelset fields at the beginning of the
 *  simulation.
 */
void Ellipsoid::initialize_fields(int level, const amrex::Geometry& geom)
{
    auto& velocity = m_velocity(level);
    auto& levelset = m_levelset(level);
    velocity.setVal(0.0, 0, AMREX_SPACEDIM);

    for (amrex::MFIter mfi(levelset); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();

        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();

        const amrex::Real xc = m_loc[0];
        const amrex::Real yc = m_loc[1];
        const amrex::Real zc = m_loc[2];
        const amrex::Real eps = m_eps;
        const amrex::Real a = m_a;
        const amrex::Real b = m_b;
        const amrex::Real c = m_c;

        auto phi = levelset.array(mfi);

        amrex::ParallelFor(
            vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
                const amrex::Real z = problo[2] + (k + 0.5) * dx[2];

                phi(i, j, k) =
                    (eps + (x - xc) * (x - xc) + (y - yc) * (y - yc) +
                     (z - zc) * (z - zc)) *
                    (std::sqrt(
                         x * x / (a * a) + y * y / (b * b) + z * z / (c * c)) -
                     1);
            });
    }
}

void Ellipsoid::pre_advance_work() {}

void Ellipsoid::post_advance_work()
{

    const int nlevels = m_sim.repo().num_active_levels();

    // Overriding the velocity field
    for (int lev = 0; lev < nlevels; ++lev) {
        auto& velocity = m_velocity(lev);
        velocity.setVal(0.0, 0, AMREX_SPACEDIM);
    }
}

} // namespace amr_wind
