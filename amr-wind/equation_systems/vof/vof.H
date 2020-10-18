#ifndef VOF_H
#define VOF_H

#include "amr-wind/equation_systems/PDETraits.H"
#include "amr-wind/equation_systems/SchemeTraits.H"
#include "amr-wind/equation_systems/PDEHelpers.H"
#include "amr-wind/equation_systems/PDE.H"

namespace amr_wind {
namespace pde {

/**
 *  \defgroup vof VOF transport equation
 *  \ingroup eqsys
 */

/**
 *  \defgroup vof_src VOF source terms
 *
 *  Source terms used in VOF transport equation
 *  \ingroup vof
 */

/** Characteristics of VOF transport equation system
 *  \ingroup vof
 */
struct VOF : ScalarTransport
{

    static std::string pde_name() { return "VOF"; }
    static std::string var_name() { return "vof"; }

    static constexpr bool multiply_rho = false;
    static constexpr bool has_diffusion = false;

    static constexpr amrex::Real default_bc_value = 0.0;
};

} // namespace pde
} // namespace amr_wind

#endif /* VOF_H */
