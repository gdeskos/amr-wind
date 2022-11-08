#include "amr-wind/physics/multiphase/MultiPhase.H"
#include "amr-wind/physics/multiphase/WaterWaves.H"
#include "amr-wind/utilities/trig_ops.H"
#include "amr-wind/CFDSim.H"
#include "AMReX_ParmParse.H"

namespace amr_wind {

WaterWaves::WaterWaves(CFDSim& sim)
    : m_sim(sim)
    , m_velocity(sim.repo().get_field("velocity"))
    , m_levelset(sim.repo().get_field("levelset"))
    , m_density(sim.repo().get_field("density"))
{

    {
        std::string turbulence_model;
        amrex::ParmParse pp("turbulence");
        pp.query("model", turbulence_model);
        if (turbulence_model == "Laminar") {
            m_laminar = true;
        }
    }

    {
        amrex::ParmParse pp("WaterWaves");
        pp.query("wave_steepness", m_wave_steepness);
        pp.query("wave_length", m_wave_length);
        pp.query("water_surface_level", m_water_level);

        pp.query("is_air_precursor", m_air_precursor);

        if (m_laminar) {
            pp.query("air_max_velocity", m_air_U0);
            pp.query("air_velocity_perturbation_magnitude", m_air_pert);
        } else {
            pp.query("air_max_velocity", m_air_U0);
            pp.query("air_velocity_perturbation_magnitude", m_air_pert);
            pp.query("tke0", m_tke0);
            pp.query("sdr0", m_sdr0);
        }
    }
}
WaterWaves::~WaterWaves() = default;

/** Initialize the velocity and vof fields at the beginning of the
 *  simulation.
 */
void WaterWaves::initialize_fields(int level, const amrex::Geometry& geom)
{

    auto& velocity = m_velocity(level);
    auto& levelset = m_levelset(level);
    auto& density = m_density(level);

    const auto& mphase = m_sim.physics_manager().get<MultiPhase>();
    const amrex::Real rho1 = mphase.rho1();
    const amrex::Real rho2 = mphase.rho2();
    velocity.setVal(0.0, 0, AMREX_SPACEDIM);

    const auto& dx = geom.CellSizeArray();
    const auto& problo = geom.ProbLoArray();
    const auto& probhi = geom.ProbHiArray();
    const amrex::Real epsilon = m_wave_steepness;
    const amrex::Real lambda = m_wave_length;
    const amrex::Real water_level = m_water_level;
    const amrex::Real U0 = m_air_U0;
    const amrex::Real vel_pert = m_air_pert;

    if (!m_laminar) {
        auto& tke = m_sim.repo().get_field("tke")(level);
        auto& sdr = m_sim.repo().get_field("sdr")(level);
        tke.setVal(m_tke0);
        sdr.setVal(m_sdr0);
    }

    for (amrex::MFIter mfi(levelset); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.growntilebox();
        auto vel = velocity.array(mfi);
        auto phi = levelset.array(mfi);
        auto rho = density.array(mfi);

        amrex::ParallelFor(
            vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                const amrex::Real y = problo[1] + (k + 0.5) * dx[1];
                const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
                const amrex::Real kappa = 2.0 * utils::pi() / lambda;
                const amrex::Real alpha = epsilon / kappa;
                // Compute free surface amplitude
                const amrex::Real eta =
                    water_level +
                    alpha * ((1.0 - 1.0 / 16.0 * epsilon * epsilon) *
                                 std::cos(kappa * x) +
                             0.5 * epsilon * std::cos(2.0 * kappa * x) +
                             3.0 / 8.0 * epsilon * epsilon *
                                 std::cos(3.0 * kappa * x));
                phi(i, j, k) = eta - z;
                // compute velocities
                const amrex::Real g = 9.81;
                const amrex::Real Omega =
                    std::sqrt(g * kappa * (1.0 + epsilon * epsilon));
                if (z < eta) {
                    vel(i, j, k, 0) = Omega * alpha * std::exp(kappa * z) *
                                      std::cos(kappa * x);
                    vel(i, j, k, 2) = Omega * alpha * std::exp(kappa * z) *
                                      std::sin(kappa * x);
                } else {
                    const amrex::Real Lx = probhi[0] - problo[0];
                    const amrex::Real Ly = probhi[1] - problo[1];
                    const amrex::Real Lz = probhi[2] - problo[2];
                    const amrex::Real pert_u =
                        vel_pert * U0 * Lx *
                        std::sin(4.0 * utils::pi() / Lx * x) *
                        std::cos(4.0 * utils::pi() / Ly * y) *
                        std::sin(4.0 * utils::pi() / Lz * z);
                    const amrex::Real pert_v =
                        -vel_pert * U0 * Ly *
                        std::cos(4.0 * utils::pi() / Lx * x) *
                        std::sin(4.0 * utils::pi() / Ly * y) *
                        std::sin(4.0 * utils::pi() / Lz * z);

                    const amrex::Real pert_w = 0;
                    vel(i, j, k, 0) =
                        (z - eta) / (probhi[2] - eta) * U0 + pert_u;
                    vel(i, j, k, 1) = pert_v;
                    vel(i, j, k, 2) = pert_w;
                }
                // compute density
                amrex::Real smooth_heaviside;
                amrex::Real eps = std::cbrt(2. * dx[0] * dx[1] * dx[2]);
                if (phi(i, j, k) > eps) {
                    smooth_heaviside = 1.0;
                } else if (phi(i, j, k) < -eps) {
                    smooth_heaviside = 0.;
                } else {
                    smooth_heaviside =
                        0.5 * (1.0 + phi(i, j, k) / eps +
                               1.0 / utils::pi() *
                                   std::sin(phi(i, j, k) * utils::pi() / eps));
                }
                rho(i, j, k) =
                    rho1 * smooth_heaviside + rho2 * (1.0 - smooth_heaviside);
            });
    }
}

void WaterWaves::pre_advance_work()
{
    if (m_air_precursor) {
        freeze_waves_motion();
    }
}

void WaterWaves::post_advance_work()
{
    if (m_air_precursor) {
        freeze_waves_motion();
    }
}

void WaterWaves::freeze_waves_motion()
{
    const int nlevels = m_sim.repo().num_active_levels();

    for (int lev = 0; lev < nlevels; ++lev) {
        auto& velocity = m_velocity(lev);
        auto& density = m_density(lev);
        auto& vof = m_sim.repo().get_field("vof")(lev);

        for (amrex::MFIter mfi(velocity); mfi.isValid(); ++mfi) {
            const auto& vbx = mfi.validbox();
            const amrex::Array4<amrex::Real>& F = vof.array(mfi);
            const amrex::Array4<amrex::Real>& u = velocity.array(mfi);
            amrex::ParallelFor(
                vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    u(i, j, k, 0) = u(i, j, k, 0) * F(i, j, k);
                    u(i, j, k, 1) = u(i, j, k, 1) * F(i, j, k);
                    u(i, j, k, 2) = u(i, j, k, 2) * F(i, j, k);
                });
        }
    }
}

} // namespace amr_wind
