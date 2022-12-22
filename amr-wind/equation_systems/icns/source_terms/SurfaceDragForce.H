#ifndef SURFACE_DRAG_FORCE_H
#define SURFACE_DRAG_FORCE_H

#include "amr-wind/core/FieldRepo.H"
#include "amr-wind/equation_systems/icns/MomentumSource.H"
#include "amr-wind/wind_energy/MOData.H"

namespace amr_wind::pde::icns {

/** Apply a user-defined drag force as a source term
 *  \ingroup icns_src
 *
 */
class SurfaceDragForce : public MomentumSource::Register<SurfaceDragForce>
{
public:
    static std::string identifier() { return "SurfaceDragForce"; }

    explicit SurfaceDragForce(const CFDSim& sim);

    ~SurfaceDragForce() override;

    virtual void operator()(
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Box& bx,
        const FieldState fstate,
        const amrex::Array4<amrex::Real>& src_term) const override;

private:
    const Field& m_surf_drag_src;
};

} // namespace amr_wind::pde::icns

#endif /* SURFACE_DRAG_FORCE_H */
