#include "SurfaceTension.H"
#include "CFDSim.H"
#include "Multiphase.H"

#include "AMReX_ParmParse.H"
#include "AMReX_Gpu.H"

namespace amr_wind {
namespace pde {
namespace icns {

SurfaceTension::SurfaceTension(const CFDSim& sim) : m_time(sim.time())
{
    const auto& multi_phase = dynamic_cast<const amr_wind::Multiphase&>(
        sim.physics_manager()(amr_wind::Multiphase::identifier()));
    multi_phase.register_forcing_term(this);

}

SurfaceTension::~SurfaceTension() = default;

void SurfaceTension::operator()(
    const int,
    const amrex::MFIter&,
    const amrex::Box& bx,
    const FieldState,
    const amrex::Array4<amrex::Real>& src_term) const
{

}

} // namespace icns
} // namespace pde
} // namespace amr_wind
