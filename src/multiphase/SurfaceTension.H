#ifndef SURFACETENSION_H
#define SURFACETENSION_H

#include "icns/MomentumSource.H"
#include "SimTime.H"
#include "Multiphase.H"

class Multiphase;

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
    const SimTime& m_time;

    amrex::Real sigma; // surface tension coefficient

};

} // namespace icns
} // namespace pde
} // namespace amr_wind

#endif /* SURFACETENSION_H */

