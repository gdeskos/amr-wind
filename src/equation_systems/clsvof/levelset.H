#ifndef LEVELSET_H
#define LEVELSET_H

#include "PDETraits.H"
#include "SchemeTraits.H"
#include "PDEHelpers.H"
#include "PDE.H"

namespace amr_wind {
namespace pde {

struct LevelSet : ScalarTransport
{
    static std::string pde_name() { return "LevelSet"; }
    static std::string var_name() { return "levelset"; }
    
    static constexpr bool multiply_rho = false;
    static constexpr bool has_diffusion = false;    
};


} // namespace pde
} // namespace amr_wind


#endif /* LEVELSET_H */
