#ifndef WKDATA_H
#define WKDATA_H

#include "amr-wind/utilities/trig_ops.H"

namespace amr_wind {
/** Structure that holds information for ocean-waves
 * \ingroup we_abl
 *
 *
 */

struct WKData
{
    amrex::Real amp{1.6};     // Wave amplitude
    amrex::Real lambda{100.}; // Wavelength
    amrex::Real c{12.5};      // Wave speed
    amrex::Real time{0.0};

    amrex::Real u_wave(amrex::Real x) const
    {
        amrex::Real k = 2. * utils::pi() / lambda;

        return amp * k * c * std::cos(k * (x - c * time));
    }

    amrex::Real w_wave(amrex::Real x) const
    {
        amrex::Real k = 2. * utils::pi() / lambda;
        return amp * k * c * std::sin(k * (x - c * time));
    }
};

} // namespace amr_wind

#endif /* WKDATA_H */