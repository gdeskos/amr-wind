#ifndef VolumeFractions_H
#define VolumeFractions_H

#include "Physics.H"
#include "Field.H"

namespace amr_wind {

/** VOF init file 
 */
class VolumeFractions : public Physics::Register<VolumeFractions>
{
public:
    static const std::string identifier() { return "VolumeFractions"; }

    VolumeFractions(const CFDSim& sim);

    virtual ~VolumeFractions() = default;

    //! Initialize the temperature and velocity fields for BoussinesqBubble simulations
    void initialize_fields(
        int level,
        const amrex::Geometry& geom) override;

    void pre_advance_work() override {}


private:
    
    void set_density(int level,
        const amrex::Geometry& geom);

    Field& m_vof;
    Field& m_density;

    //! initial density value
    amrex::Real m_rho_air{1.2041};
    amrex::Real m_rho_water{997.};
    int m_vof_problem{1};

};

}

#endif /* VolumeFractions_H */
