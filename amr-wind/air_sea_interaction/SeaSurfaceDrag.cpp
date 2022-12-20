#include "amr-wind/air_sea_interaction/SeaSurfaceDrag.H"
#include "amr-wind/incflo.H"

#include "AMReX_ParmParse.H"
#include "AMReX_MultiFab.H"

namespace amr_wind {

SeaSurfaceDrag::SeaSurfaceDrag(CFDSim& sim) : m_sim(sim)
{
    // Check that Multiphase is enabled
    if (!sim.physics_manager().contains("MultiPhase")) {
        amrex::Abort("SeaSurfaceDrag requires Multiphase physics to be active");
    }
}

SeaSurfaceDrag::~SeaSurfaceDrag() = default;

} // namespace amr_wind