#ifndef Multiphase_H
#define Multiphase_H

#include "amr-wind/core/Physics.H"
#include "amr-wind/core/Field.H"
#include "amr-wind/equation_systems/clsvof/levelset.H"

namespace amr_wind {


/** Multiphase physics 
 */
class Multiphase : public Physics::Register<Multiphase>
{
public:
    static const std::string identifier() { return "Multiphase"; }

    Multiphase(CFDSim& sim);

    virtual ~Multiphase() = default;

    void post_init_actions() override;

    void post_regrid_actions() override {}

    //! Initialize velocity, volume fractions and density for multiphase simulations 
    void initialize_fields(
        int level,
        const amrex::Geometry& geom) override;

    void pre_advance_work() override;

    void post_advance_work() override;

private:
    const CFDSim& m_sim;

    void set_density(int level,const amrex::Geometry& geom);
    
    void compute_normals_and_curvature();
    
    void compute_surface_tension();

    void reconstruct_vof();  

    Field& m_density;
    Field& m_velocity;

    // Define multiphase fields
    Field* m_levelset{nullptr};
    Field* m_vof{nullptr};

    Field& m_lsnormal;
    Field& m_lscurv;
    Field& m_surface_tension;

    //! initial density value
    amrex::Real m_rho_air{1.};
    amrex::Real m_rho_water{1000.};
    amrex::Real m_sigma{0.07};

    int m_multiphase_problem{1};

    // Cases parameters -- TODO this shouldnt be here
    amrex::Real m_dambreak_d{0.1461};
    amrex::Real m_dambreak_h{0.292};

};

}

#endif /* Multiphase_H */
