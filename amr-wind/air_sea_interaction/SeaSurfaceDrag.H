#ifndef SEA_SURFACE_DRAG_H
#define SEA_SURFACE_DRAG_H

#include "amr-wind/core/Physics.H"
#include "amr-wind/core/SimTime.H"
namespace amr_wind {

class SeaSurfaceDrag : public Physics::Register<SeaSurfaceDrag>
{
public:
    static std::string identifier() { return "SeaSurfaceDrag"; }

    explicit SeaSurfaceDrag(CFDSim& sim);

    ~SeaSurfaceDrag() override;

    void post_init_actions() override {}

    void post_regrid_actions() override {}

    void initialize_fields(int, const amrex::Geometry&) override {}

    void pre_advance_work() override {}

    void post_advance_work() override {}

private:
    const CFDSim& m_sim;
};

} // namespace amr_wind
#endif /* SEA_SURFACE_DRAG_H */
