#ifndef WAVE_DRAG_MODEL_H
#define WAVE_DRAG_MODEL_H

#include "amr-wind/utilities/trig_ops.H"

namespace amr_wind {

/** Wave Drag Model
 *
 * Aiyer, A. K., Deike, L., & Mueller, M. E. (2023). A Sea Surface-Based Drag
 * Model for Large-Eddy Simulation of Wind-Wave Interaction. Journal of the
 * Atmospheric Sciences, 49–62. https://doi.org/10.1175/JAS-D-21-0329.1
 */
struct WaveDragModel
{

    amrex::Real CD{0.01}; ///< Wave drag coefficient
    amrex::Real A{0.01};  ///< Wave amplitude
    amrex::Real k{0.01};  ///< Wave number
    amrex::Real c{1.0};   ///< Wave phase speed
    amrex::Real P{1.2};   ///< Drag coefficient constant 1
    amrex::Real Q{6.0};   ///< Drag coefficient constant 2
};

} // namespace amr_wind

#endif /* MODATA_H */