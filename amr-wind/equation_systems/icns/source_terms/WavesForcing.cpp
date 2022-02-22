#include "amr-wind/equation_systems/icns/source_terms/WavesForcing.H"
#include "amr-wind/CFDSim.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Gpu.H"

namespace amr_wind {
namespace pde {
namespace icns {

WavesForcing::WavesForcing(const CFDSim& sim) : m_sim(sim)
{
    amrex::ParmParse pp("WavesForcing");
    pp.get("magnitude", m_magnitude);
    pp.get("wavelength", m_lambda);
    pp.get("decay_length", m_decay_length);
}

WavesForcing::~WavesForcing() = default;

void WavesForcing::operator()(
    const int lev,
    const amrex::MFIter&,
    const amrex::Box& bx,
    const FieldState,
    const amrex::Array4<amrex::Real>& src_term) const
{
    const amrex::Real time = m_sim.time().current_time();

    const auto& geom = m_sim.mesh().Geom(lev);
    const auto& dx = geom.CellSizeArray();
    const auto& problo = geom.ProbLoArray();

    const amrex::Real kappa = 2. * utils::pi() / m_lambda;
    const amrex::Real omega =
        std::sqrt(kappa * 9.81); // Linear deep-wave theory
    const amrex::Real Ldecay = m_decay_length;
    const amrex::Real mag = m_magnitude;

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
        const amrex::Real z = problo[2] + (k + 0.5) * dx[2];

        const amrex::Real phase = kappa * x - omega * time;
        const amrex::Real forcex =
            mag * std::exp(-z / Ldecay) * std::sin(phase);

        const amrex::Real forcez =
            mag * std::exp(-z / Ldecay) * std::cos(phase);
        src_term(i, j, k, 0) += forcex;
        src_term(i, j, k, 2) += forcez;
    });
}

} // namespace icns
} // namespace pde
} // namespace amr_wind
