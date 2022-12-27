#ifndef LINEAR_WAVE_FORCING_H
#define LINEAR_WAVE_FORCING_H

#include "amr-wind/CFDSim.H"
#include "amr-wind/core/Physics.H"
#include "amr-wind/core/SimTime.H"
#include "amr-wind/core/FieldBCOps.H"
#include "amr-wind/wind_energy/MOData.H"

namespace amr_wind {

class LinearWaveForcing : public Physics::Register<LinearWaveForcing>
{
public:
    static std::string identifier() { return "LinearWaveForcing"; }

    explicit LinearWaveForcing(CFDSim& sim);

    ~LinearWaveForcing() = default;

    void post_init_actions() override;

    void post_regrid_actions() override {}

    void
    initialize_fields(int /* level */, const amrex::Geometry& /*geom*/) override
    {}

    void pre_advance_work() override;

    void post_advance_work() override {}

private:
    const CFDSim& m_sim;

    Field& m_velocity;
    Field& m_eddy_viscosity;
    Field& m_lwf_src;

    const MOData& m_mo;
};

/** Applies Dirichlet boundary conditions
 */
class WaveBCVel : public FieldBCIface
{
public:
    WaveBCVel(Field& velocity, const SimTime& time);

    void operator()(Field& velocity, const FieldState rho_state) override;

private:
    //! Instance of Simulation Time class
    const SimTime& m_time;

    //! Wave length
    amrex::Real m_wave_length{100.};

    //! Wave height
    amrex::Real m_wave_height{3.2};

    //! Wave phase speed
    amrex::Real m_phase_speed{12.5};
};

} // namespace amr_wind
#endif /* LINEAR_WAVE_FORCING_H */
