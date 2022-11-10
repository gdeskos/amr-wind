#ifndef NWB_H
#define NWB_H

#include "amr-wind/core/Physics.H"
#include "amr-wind/core/SimTime.H"

/**
 *  \defgroup Water Waves modeling
 *  WaterWaves modeling
 *
 *  This group documents all the wave mechanics related physics models available
 * in AMR-Wind
 *
 *  \ingroup physics
 */

namespace amr_wind {

class WaterWaves : public Physics::Register<WaterWaves>
{
public:
    static std::string identifier() { return "WaterWaves"; }

    explicit WaterWaves(CFDSim& sim);

    ~WaterWaves() override;

    void initialize_fields(int level, const amrex::Geometry& geom) override;

    void post_init_actions() override {}

    void post_regrid_actions() override {}

    void pre_advance_work() override;

    void post_advance_work() override {}

    void pre_pressure_correction_work() override;

    void freeze_waves_motion();

    void reset_waves_velocity();

private:
    const CFDSim& m_sim;

    Field& m_velocity;
    Field& m_levelset;
    Field& m_density;

    //! initial tke value
    amrex::Real m_tke0{0.000001};

    //! initial sdr value
    amrex::Real m_sdr0{1.0};

    //! Initial free surface amplitude magnitude
    amrex::Real m_wave_steepness{0.1};

    //! Initial free surface wavelength
    amrex::Real m_wave_length{1.0};

    //! Initial zero-level free-surface water depth
    amrex::Real m_water_level{0.0};

    //! Initial air velocity shear profile
    amrex::Real m_air_U0{1.};

    //! Initial air near wall perturbation level
    amrex::Real m_air_pert{0.2};

    bool m_laminar{false};

    bool m_air_precursor{false};

    int m_restart_vof_init{0};
};

} // namespace amr_wind

#endif /* NWB_H */
