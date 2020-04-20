#include "vof/vof.H"
#include "AdvOp_Godunov.H"
#include "AdvOp_MOL.H"

namespace amr_wind {
namespace pde {

template class PDESystem<VOF, fvm::Godunov>;
template class PDESystem<VOF, fvm::MOL>;

} // namespace pde
} // namespace amr_wind
