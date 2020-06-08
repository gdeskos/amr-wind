#include "amr-wind/equation_systems/clsvof/vof.H"
#include "amr-wind/equation_systems/AdvOp_Godunov.H"
#include "amr-wind/equation_systems/AdvOp_MOL.H"
#include "amr-wind/equation_systems/BCOps.H"
#include "amr-wind/equation_systems/clsvof/vof_ops.H"

namespace amr_wind {
namespace pde {

template class PDESystem<VOF, fvm::Godunov>;
template class PDESystem<VOF, fvm::MOL>;



} // namespace pde
} // namespace amr_wind
