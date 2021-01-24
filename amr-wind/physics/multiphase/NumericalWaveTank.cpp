#include "amr-wind/physics/multiphase/NumericalWaveTank.H"
#include "amr-wind/CFDSim.H"
#include "AMReX_ParmParse.H"
#include "amr-wind/fvm/gradient.H"
#include "amr-wind/core/field_ops.H"

namespace amr_wind {

NumericalWaveTank::NumericalWaveTank(CFDSim& sim)
    : m_velocity(sim.repo().get_field("velocity"))
    , m_levelset(sim.repo().get_field("levelset"))
{
    amrex::ParmParse pp(identifier());
    pp.query("amplitude", m_amplitude);
    pp.query("wavelength", m_wavelength);
    pp.query("water_level", m_waterlevel);
}

/** Initialize the velocity and levelset fields at the beginning of the
 *  simulation.
 *
 */
void NumericalWaveTank::initialize_fields(
    int level, const amrex::Geometry& geom)
{
    auto& velocity = m_velocity(level);
    velocity.setVal(0.0, 0, AMREX_SPACEDIM);

    auto& levelset = m_levelset(level);
    const auto& dx = geom.CellSizeArray();
    const auto& problo = geom.ProbLoArray();
    const amrex::Real alpha = m_amplitude;
    const amrex::Real lambda = m_wavelength;
    const amrex::Real water_level = m_waterlevel;

    for (amrex::MFIter mfi(levelset); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();
        auto phi = levelset.array(mfi);
        auto vel = velocity.array(mfi);

        amrex::ParallelFor(
            vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
                const amrex::Real kappa = 2.0 * M_PI / lambda;
                const amrex::Real eps = alpha * kappa;
                // Compute free surface amplitude
                const amrex::Real eta =
                    water_level +
                    alpha *
                        ((1.0 - 1.0 / 16.0 * eps * eps) * std::cos(kappa * x) +
                         0.5 * eps * std::cos(2.0 * kappa * x) +
                         3.0 / 8.0 * eps * eps * std::cos(3.0 * kappa * x));
                phi(i, j, k) = eta - z;
                // compute velocities
                const amrex::Real g = 9.81;
                const amrex::Real Omega =
                    std::sqrt(g * kappa * (1.0 + eps * eps));
                if (z < eta) {
                    vel(i, j, k, 0) = Omega * alpha * std::exp(kappa * z) *
                                      std::cos(kappa * x);
                    vel(i, j, k, 2) = Omega * alpha * std::exp(kappa * z) *
                                      std::sin(kappa * x);
                }
            });
    }
}

} // namespace amr_wind
