#ifndef Multiphase_H
#define Multiphase_H

#include "Physics.H"
#include "Field.H"
#include "clsvof/levelset.H"

namespace amr_wind {

/** VOF init file 
 */
class Multiphase : public Physics::Register<Multiphase>
{
public:
    static const std::string identifier() { return "Multiphase"; }

    Multiphase(const CFDSim& sim);

    virtual ~Multiphase() = default;

    //! Initialize velocity, volume fractions and density for multiphase simulations 
    void initialize_fields(
        int level,
        const amrex::Geometry& geom) override;

    void post_init_actions() override {}

    void pre_advance_work() override;


private:
    const CFDSim& m_sim;

    void set_density(int level,const amrex::Geometry& geom);
    //void compute_normal_and_curvature(int level,const amrex::Geometry& geom);
     
    Field& m_levelset;
    Field& m_vof;
    Field& m_density;
    Field& m_velocity;
    //Field& m_lsnormal;
    //Field& m_lscurv;
   
    //! initial density value
    amrex::Real m_rho_air{1.2041};
    amrex::Real m_rho_water{997.};
    int m_multiphase_problem{1};

};

}

#endif /* Multiphase_H */
