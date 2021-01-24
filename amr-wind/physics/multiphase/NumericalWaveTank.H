#ifndef NumericalWaveTank_H
#define NumericalWaveTank_H

#include "amr-wind/core/Physics.H"
#include "amr-wind/core/Field.H"

/** Multiphase Sloshing Tank physics
 *  \ingroup mulitphase_physics
 *
 */

namespace amr_wind {

class NumericalWaveTank : public Physics::Register<NumericalWaveTank>
{
public:
    static const std::string identifier() { return "NumericalWaveTank"; }

    NumericalWaveTank(CFDSim& sim);

    virtual ~NumericalWaveTank() = default;

    //! Initialize the levelset and velocity fields for Numerical Wave Tank
    //! simulations
    void initialize_fields(int level, const amrex::Geometry& geom) override;

    void post_init_actions() override {}

    void post_regrid_actions() override {}

    void pre_advance_work() override {}

    void post_advance_work() override {}

private:
    Field& m_velocity;
    Field& m_levelset;

    //! Initial free surface amplitude magnitude
    amrex::Real m_amplitude{0.1};

    //! Initial free surface wavelength
    amrex::Real m_wavelength{2.0};

    //! Initial zero-level free-surface water depth
    amrex::Real m_waterlevel{0.0};
};

} // namespace amr_wind
#endif /* NumericalWaveTank_H */