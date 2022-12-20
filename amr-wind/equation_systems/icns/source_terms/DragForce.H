#ifndef DRAG_FORCE_H
#define DRAG_FORCE_H

#include "amr-wind/equation_systems/icns/MomentumSource.H"
#include "amr-wind/core/SimTime.H"

namespace amr_wind::pde::icns {

/** Apply a user-defined drag force as a source term
 *  \ingroup icns_src
 *
 */
class DragForce : public MomentumSource::Register<DragForce>
{
public:
    static std::string identifier() { return "DragForce"; }

    explicit DragForce(const CFDSim& /*sim*/);

    ~DragForce() override;

    void operator()(
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Box& bx,
        const FieldState fstate,
        const amrex::Array4<amrex::Real>& src_term) const override;

private:
    const Field& m_drag_src;
};

} // namespace amr_wind::pde::icns

#endif /* DRAG_FORCE_H */
