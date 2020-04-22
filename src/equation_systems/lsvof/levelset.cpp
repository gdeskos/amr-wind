#include "lsvof/levelset.H"
#include "AdvOp_Godunov.H"
#include "AdvOp_MOL.H"

namespace amr_wind {
namespace pde {

template class PDESystem<LS, fvm::Godunov>;
template class PDESystem<LS, fvm::MOL>;

} // namespace pde
} // namespace amr_wind
