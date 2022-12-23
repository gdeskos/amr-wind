#ifndef VOF_INTERFACE_DRAG_H
#define VOF_INTERFACE_DRAG_H

#include "amr-wind/core/Physics.H"
#include "amr-wind/core/SimTime.H"
namespace amr_wind {

class VOFInterfaceDrag : public Physics::Register<VOFInterfaceDrag>
{
public:
    static std::string identifier() { return "VOFInterfaceDrag"; }

    explicit VOFInterfaceDrag(CFDSim& sim);

    ~VOFInterfaceDrag() override;

    void post_init_actions() override {}

    void post_regrid_actions() override {}

    void initialize_fields(int, const amrex::Geometry&) override {}

    void pre_advance_work() override {}

    void post_advance_work() override {}

private:
    const CFDSim& m_sim;
};

} // namespace amr_wind
#endif /* VOF_INTERFACE_DRAG_H */
