#include "amr-wind/equation_systems/clsvof/levelset.H"
#include "amr-wind/equation_systems/AdvOp_Godunov.H"
#include "amr-wind/equation_systems/AdvOp_MOL.H"
#include "amr-wind/equation_systems/BCOps.H"
#include "amr-wind/equation_systems/clsvof/levelset_ops.H"

namespace amr_wind {
namespace pde {

template class PDESystem<LevelSet, fvm::Godunov>;
template class PDESystem<LevelSet, fvm::MOL>;

} // namespace pde
} // namespace amr_wind
