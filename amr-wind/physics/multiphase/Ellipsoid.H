#ifndef Ellipsoid_H
#define Ellipsoid_H

#include "amr-wind/core/Physics.H"
#include "amr-wind/core/Field.H"

/** Multiphase Droplet physics
 *  \ingroup mulitphase_physics
 *
 */

namespace amr_wind {

class Ellipsoid : public Physics::Register<Ellipsoid>
{
public:
    static const std::string identifier() { return "Ellipsoid"; }

    Ellipsoid(CFDSim& sim);

    virtual ~Ellipsoid() = default;

    //! Initialize the levelset and velocity fields for Vortex Patch simulations
    void initialize_fields(int level, const amrex::Geometry& geom) override;

    void post_init_actions() override {}

    void post_regrid_actions() override {}

    void pre_advance_work() override;

    void post_advance_work() override;

private:
    const CFDSim& m_sim;

    Field& m_velocity;
    Field& m_levelset;

    //! Initial Ellipsoid location
    amrex::Vector<amrex::Real> m_loc{{3.5, 2.0, 2.0}};

    //! Ellipsoid axis length
    amrex::Real m_a{4.0};
    amrex::Real m_b{2.0};
    amrex::Real m_c{2.0};
    //! Ellipsoid bump radius
    amrex::Real m_eps{0.1};
};

} // namespace amr_wind

#endif /* ZalesakDisk_H */
