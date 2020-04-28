#include "clsvof/levelset.H"
#include "AdvOp_Godunov.H"
#include "AdvOp_MOL.H"
#include "BCOps.H"
#include "clsvof/levelset_ops.H"

namespace amr_wind {
namespace pde {

template class PDESystem<LevelSet, fvm::Godunov>;
template class PDESystem<LevelSet, fvm::MOL>;

} // namespace pde
} // namespace amr_wind
