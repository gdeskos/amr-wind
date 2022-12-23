#include "amr-wind/air_sea_interaction/VOFInterfaceDrag.H"
#include "amr-wind/incflo.H"

#include "AMReX_ParmParse.H"
#include "AMReX_MultiFab.H"

namespace amr_wind {

VOFInterfaceDrag::VOFInterfaceDrag(CFDSim& sim) : m_sim(sim)
{
    // Check that Multiphase is enabled
    if (!sim.physics_manager().contains("MultiPhase")) {
        amrex::Abort(
            "VOFInterfaceDrag requires Multiphase/VOF physics to be active");
    }
}

VOFInterfaceDrag::~VOFInterfaceDrag() = default;

} // namespace amr_wind