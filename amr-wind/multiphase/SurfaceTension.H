#ifndef SURFACETENSION_H
#define SURFACETENSION_H

#include "amr-wind/equation_systems/icns/MomentumSource.H"
#include "amr-wind/core/SimTime.H"
#include "amr-wind/multiphase/Multiphase.H"

namespace amr_wind {
namespace pde {
namespace icns {

class SurfaceTension : public MomentumSource::Register<SurfaceTension>
{
public:
    static const std::string identifier() { return "SurfaceTension"; }

    SurfaceTension(const CFDSim& sim);

    virtual ~SurfaceTension();

    virtual void operator()(
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Box& bx,
        const FieldState fstate,
        const amrex::Array4<amrex::Real>& src_term) const override;


private:
    const amrex::AmrCore& m_mesh;

    Field& m_density;
    Field& m_levelset;
    Field& m_lsnormal;
    Field& m_lscurv;
    
    amrex::Real m_sigma{0.07}; // surface tension coefficient
    
};

} // namespace icns
} // namespace pde
} // namespace amr_wind

#endif /* SURFACETENSION_H */

