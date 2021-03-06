#ifndef BOUSSINESQBUOYANCY_H
#define BOUSSINESQBUOYANCY_H

#include "amr-wind/core/FieldRepo.H"
#include "amr-wind/equation_systems/icns/MomentumSource.H"

namespace amr_wind {
namespace pde {
namespace icns {

class BoussinesqBuoyancy : public MomentumSource::Register<BoussinesqBuoyancy>
{
public:
    static const std::string identifier() { return "BoussinesqBuoyancy"; }

    BoussinesqBuoyancy(const CFDSim& sim);

    virtual ~BoussinesqBuoyancy();

    virtual void operator()(
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Box& bx,
        const FieldState fstate,
        const amrex::Array4<amrex::Real>& src_term) const override;

private:
    const Field& m_temperature;

    amrex::Vector<amrex::Real> m_gravity{{0.0, 0.0, -9.81}};

    //! Reference temperature (Kelvin)
    amrex::Real m_ref_theta{300.0};

    //! Thermal expansion coefficient
    amrex::Real m_beta{0.0};
};
} // namespace icns

} // namespace pde
} // namespace amr_wind

#endif /* BOUSSINESQBUOYANCY_H */
