#include "amr-wind/air_sea_interaction/LinearWaveForcing.H"
#include "amr-wind/incflo.H"

#include "AMReX_ParmParse.H"
#include "AMReX_MultiFab.H"

namespace amr_wind {

LinearWaveForcing::LinearWaveForcing(CFDSim& sim)
    : m_sim(sim), m_velocity(sim.pde_manager().icns().fields().field)
{}

void LinearWaveForcing::post_init_actions()
{
    // Register Dirichlet boundary conditions for velocity
    m_velocity.register_custom_bc<WaveBCVel>();
}

void LinearWaveForcing::pre_advance_work() {}

WaveBCVel::WaveBCVel(Field& /*unused*/)
{
    amrex::ParmParse pp("LinearWaveForcing");
    pp.query("wave_length", m_wave_length);
    pp.query("wave_height", m_wave_height);
    pp.query("phase_speed", m_phase_speed);
}

void WaveBCVel::operator()(Field& velocity, const FieldState /*rho_state*/)
{
    BL_PROFILE("amr-wind::WaveBCVel");
    constexpr int idim = 2;
    const auto& repo = velocity.repo();
    const int nlevels = repo.num_active_levels();

    amrex::Orientation zlo(amrex::Direction::z, amrex::Orientation::low);
    amrex::Orientation zhi(amrex::Direction::z, amrex::Orientation::high);
    if ((velocity.bc_type()[zlo] != BC::moving_wall) &&
        (velocity.bc_type()[zhi] != BC::moving_wall)) {
        return;
    }
    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& geom = repo.mesh().Geom(lev);
        const auto& domain = geom.Domain();
        amrex::MFItInfo mfi_info{};
        auto problo = geom.ProbLo();
        auto dx = geom.CellSizeArray();
        auto& vel_lev = velocity(lev);

        if (amrex::Gpu::notInLaunchRegion()) {
            mfi_info.SetDynamic(true);
        }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(vel_lev, mfi_info); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.validbox();
            auto varr = vel_lev.array(mfi);
            const amrex::Real H = m_wave_height;
            const amrex::Real l = m_wave_length;
            const amrex::Real c = m_phase_speed;

            if (bx.smallEnd(idim) == domain.smallEnd(idim) &&
                velocity.bc_type()[zlo] == BC::moving_wall) {
                amrex::ParallelFor(
                    amrex::bdryLo(bx, idim),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                        // Dirichlet BC
                        const amrex::Real k_w = 2 * M_PI / l;
                        varr(i, j, k - 1, 0) =
                            c * H / 2. * k_w * std::cos(k_w * x);
                        varr(i, j, k - 1, 1) = 0.0;
                        varr(i, j, k - 1, 2) =
                            c * H / 2. * k_w * std::sin(k_w * x);
                    });
            }

            if (bx.bigEnd(idim) == domain.bigEnd(idim) &&
                velocity.bc_type()[zhi] == BC::moving_wall) {
                amrex::ParallelFor(
                    amrex::bdryHi(bx, idim),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        // Dirichlet BC
                        varr(i, j, k - 1, 0) = 1.0;
                        varr(i, j, k - 1, 1) = 0.0;
                        varr(i, j, k - 1, 2) = 0.0;
                    });
            }
        }
    }
}

} // namespace amr_wind