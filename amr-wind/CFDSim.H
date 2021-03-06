#ifndef CFDSIM_H
#define CFDSIM_H

#include "AMReX_AmrCore.H"
#include "amr-wind/core/SimTime.H"
#include "amr-wind/core/FieldRepo.H"
#include "amr-wind/equation_systems/PDEBase.H"
#include "amr-wind/core/Physics.H"

namespace amr_wind {
class IOManager;
class PostProcessManager;

namespace turbulence {
class TurbulenceModel;
}

/** Data structures for a CFD simulation
 *
 *  CFDSim is a thin wrapper that holds all the necessary objects for a CFD
 *  simulation. The key data members within this object are:
 *
 *  - mesh (amrex::AmrCore) The AMR mesh hierarchy data structure
 *  - time (SimTime)        The time object
 *  - repo (FieldRepo)      The field repository
 *  - pde_manager (PDEMgr)  PDE manager interface
 */
class CFDSim
{
public:
    CFDSim(amrex::AmrCore& mesh);

    ~CFDSim();

    //! Return the AMR mesh hierarchy
    amrex::AmrCore& mesh() { return m_mesh; }
    const amrex::AmrCore& mesh() const { return m_mesh; }

    //! Return simulation time control
    SimTime& time() { return m_time; }
    const SimTime& time() const { return m_time; }

    //! Return the field repository
    FieldRepo& repo() { return m_repo; }
    FieldRepo& repo() const { return m_repo; }

    pde::PDEMgr& pde_manager() { return m_pde_mgr; }
    const pde::PDEMgr& pde_manager() const { return m_pde_mgr; }

    PhysicsMgr& physics_manager() { return m_physics_mgr; }
    const PhysicsMgr& physics_manager() const { return m_physics_mgr; }

    PhysicsMgr::TypeVector& physics() { return m_physics_mgr.objects(); }
    const PhysicsMgr::TypeVector& physics() const { return m_physics_mgr.objects(); }

    turbulence::TurbulenceModel& turbulence_model() { return *m_turbulence; }
    const turbulence::TurbulenceModel& turbulence_model() const
    { return *m_turbulence; }

    IOManager& io_manager() { return *m_io_mgr; }
    const IOManager& io_manager() const { return * m_io_mgr; }

    PostProcessManager& post_manager() { return *m_post_mgr; }
    const PostProcessManager& post_manager() const  { return *m_post_mgr; }

    void create_turbulence_model();

    void init_physics();

private:
    amrex::AmrCore& m_mesh;

    SimTime m_time;

    mutable FieldRepo m_repo;

    pde::PDEMgr m_pde_mgr;

    PhysicsMgr m_physics_mgr;

    std::unique_ptr<turbulence::TurbulenceModel> m_turbulence;

    std::unique_ptr<IOManager> m_io_mgr;

    std::unique_ptr<PostProcessManager> m_post_mgr;
};

}

#endif /* CFDSIM_H */
