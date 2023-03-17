#ifndef HURRICANE_FORCING_H
#define HURRICANE_FORCING_H

#include "amr-wind/core/FieldRepo.H"
#include "amr-wind/equation_systems/icns/MomentumSource.H"
#include "amr-wind/utilities/FieldPlaneAveraging.H"

namespace amr_wind::pde::icns {

/** Idealized Hurricane forcing source term
 *  Apply mesoscale tendency source term to drive the flow
 * and match the hurricane cyclic/rotational forcing at large
 * distances from the eye-wall. This implements the "Simple"
 * method of George Bryan.
 * Details can be found in:
 * "A simple Method for Simulating Wind Profiles in the Boundary Layer
 * of Tropical Cyclones. Bryant et al. Boundary-Layer Meteorol (2017)
 * DOI 10.1007/s10546-016-0207-0
 */

class HurricaneForcing : public MomentumSource::Register<HurricaneForcing>
{
public:
    static std::string identifier() { return "HurricaneForcing"; }

    explicit HurricaneForcing(const CFDSim& sim);

    ~HurricaneForcing() override;

    void operator()(
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Box& bx,
        const FieldState fstate,
        const amrex::Array4<amrex::Real>& src_term) const override;

    void mean_velocity_init(const VelPlaneAveraging& /*vavg*/);

    void mean_velocity_update(const VelPlaneAveraging& /*vavg*/);

private:
    //! Mesh
    const amrex::AmrCore& m_mesh;

    amrex::Gpu::DeviceVector<amrex::Real> m_vel_ht;
    amrex::Gpu::DeviceVector<amrex::Real> m_vel_vals;

    //! Axis over which averages are computed
    int m_axis{2};

    //! Gradient velocity
    amrex::Real m_V{40.0}; // V=40 m/s is the default value

    //! Gradient wind radial decay
    amrex::Real m_dVdR{-0.0008};

    //! Reference gradient wind zero height in metres
    amrex::Real m_Vzh{18000.};

    //! Eye-wall radial distance
    amrex::Vector<amrex::Real> m_X{
        {-40000.0, 0.0,
         0.0}}; // 40km to the west is taken as the default value. The wind farm
                // is assumed to start at (0,0,0)

    //! Relative angle theta
    amrex::Real  m_theta{0.0};

    //! Coriolis factor
    amrex::Real m_coriolis_factor{0.0005};
};

} // namespace amr_wind::pde::icns

#endif /* HURRICANE_FORCING_H */