#ifndef LEVELSET_H
#define LEVELSET_H

#include "PDETraits.H"
#include "SchemeTraits.H"
#include "PDEHelpers.H"
#include "PDE.H"

namespace amr_wind {
namespace pde {

struct LS : ScalarTransport
{
    static std::string pde_name() { return "LS"; }
    static std::string var_name() { return "ls"; }
};


} // namespace pde
} // namespace amr_wind


#endif /* VOF_H */
