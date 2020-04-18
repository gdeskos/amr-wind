#include "vof/vof.H"

namespace amr_wind {
namespace pde {

template class PDESystem<VOF, fvm::Godunov>;
template class PDESystem<VOF, fvm::MOL>;

} // namespace pde
} // namespace amr_wind
