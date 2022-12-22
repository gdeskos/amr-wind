#include "amr-wind/equation_systems/icns/source_terms/SurfaceDragForce.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/core/FieldUtils.H"
#include "amr-wind/wind_energy/ABL.H"
#include "amr-wind/wind_energy/ShearStress.H"

#include "AMReX_ParmParse.H"
#include "AMReX_Gpu.H"

namespace amr_wind::pde::icns {

/** Surface Drag Force
 */
SurfaceDragForce::SurfaceDragForce(const CFDSim& sim)
    : m_surf_drag_src(sim.repo().get_field("surface_drag"))
{
    if (!sim.physics_manager().contains("LinearWaveForcing") ||
        !sim.physics_manager().contains("SeaSurfaceDrag")) {
        amrex::Abort(
            "SurfaceDragForce requires LinearWaveForcing or SeaSurfaceDrag "
            "physics to be active");
    }
}

SurfaceDragForce::~SurfaceDragForce() = default;

void SurfaceDragForce::operator()(
    const int lev,
    const amrex::MFIter& mfi,
    const amrex::Box& bx,
    const FieldState /*fstate*/,
    const amrex::Array4<amrex::Real>& src_term) const
{

    const auto varr = m_surf_drag_src(lev).const_array(mfi);

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        src_term(i, j, k, 0) += varr(i, j, k, 0);
        src_term(i, j, k, 1) += varr(i, j, k, 1);
        src_term(i, j, k, 2) += varr(i, j, k, 2);
    });

    /*
    const auto& geom = m_mesh.Geom(lev);
    const auto& dx = m_mesh.Geom(lev).CellSizeArray();
    const auto& domain = geom.Domain();

    const int idir = m_axis;

    // FIXME: only doing zlo for now
    if (!(bx.smallEnd(idir) == domain.smallEnd(idir))) return;
    if (idir != 2) return;

    // FIXME: should not be doing this for every box
    auto tau = ShearStressMoeng(m_mo);

    const auto& vel_arr =
        m_vel.state(field_impl::dof_state(fstate))(lev).const_array(mfi);

    amrex::ParallelFor(
        amrex::bdryLo(bx, idir),

        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real uu = vel_arr(i, j, k, 0);
            const amrex::Real vv = vel_arr(i, j, k, 1);

            const amrex::Real u_wall = vel_arr(i, j, k - 1, 0);
            const amrex::Real v_wall = vel_arr(i, j, k - 1, 1);

            const amrex::Real ur = uu - u_wall;
            const amrex::Real vr = vv - v_wall;
            const amrex::Real wspdr = std::sqrt(ur * ur + vr * vr);

            const amrex::Real tau_D_xz = 0.0;
            const amrex::Real tau_D_yz = 0.0;

            src_term(i, j, k, 0) +=
                (tau_D_xz - tau.calc_vel_x(ur, wspdr)) / dx[2];
            src_term(i, j, k, 1) +=
                (tau_D_yz - tau.calc_vel_y(vr, wspdr)) / dx[2];
            //        src_term(i, j, k, 2) += 0.0;
        });
        */
}

} // namespace amr_wind::pde::icns
