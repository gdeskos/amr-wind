#ifndef VOF_H
#define VOF_H

#include "PDETraits.H"
#include "SchemeTraits.H"
#include "PDEHelpers.H"
#include "PDE.H"

namespace amr_wind {
namespace pde {

struct VOF : ScalarTransport
{
    static std::string pde_name() { return "VOF"; }
    static std::string var_name() { return "vof"; }
};


} // namespace pde
} // namespace amr_wind


#endif /* VOF_H */
