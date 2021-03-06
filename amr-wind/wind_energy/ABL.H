#ifndef ABL_H
#define ABL_H

#include <memory>

#include "amr-wind/core/Physics.H"
#include "amr-wind/wind_energy/ABLFieldInit.H"
#include "amr-wind/wind_energy/ABLWallFunction.H"
#include "amr-wind/core/SimTime.H"
#include "amr-wind/utilities/FieldPlaneAveraging.H"

class incflo;

namespace amr_wind {

namespace pde {
namespace icns {
class ABLForcing;
}
}

/** Atmospheric Boundary Layer physics
 */
class ABL : public Physics::Register<ABL>
{
public:
    static const std::string identifier() { return "ABL"; }

    ABL(CFDSim& sim);

    virtual ~ABL();

    const ABLWallFunction& abl_wall_function() const
    { return m_abl_wall_func; }

    void post_init_actions() override;

    void post_regrid_actions() override {}

    void initialize_fields(
        int level,
        const amrex::Geometry& geom) override;

    void pre_advance_work() override;

    void post_advance_work() override {}

    void register_forcing_term(pde::icns::ABLForcing* forcing) const
    {
        m_abl_forcing = forcing;
    }
    
private:
    const CFDSim& m_sim;

    Field& m_velocity;
    Field& m_mueff;
    Field& m_density;
    Field* m_temperature{nullptr};

    FieldPlaneAveraging m_pa;
    ABLWallFunction m_abl_wall_func;

    //! ABL field initializer instance
    std::unique_ptr<ABLFieldInit> m_field_init;

    //! Reference to ABL forcing term if present
    mutable pde::icns::ABLForcing* m_abl_forcing{nullptr};
    
};

}

#endif /* ABL_H */
