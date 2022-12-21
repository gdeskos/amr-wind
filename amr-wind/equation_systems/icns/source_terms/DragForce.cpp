#include "amr-wind/equation_systems/icns/source_terms/DragForce.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/core/FieldUtils.H"
#include "amr-wind/wind_energy/ABL.H"
#include "amr-wind/wind_energy/ShearStress.H"

#include "AMReX_ParmParse.H"
#include "AMReX_Gpu.H"

namespace amr_wind::pde::icns {

/** Drag Force
 */
DragForce::DragForce(const CFDSim& sim)
    : m_mesh(sim.mesh())
    , m_vel(sim.repo().get_field("velocity"))
    , m_mo(sim.physics_manager().get<amr_wind::ABL>().abl_wall_function().mo())
{}

DragForce::~DragForce() = default;

void DragForce::operator()(
    const int lev,
    const amrex::MFIter& mfi,
    const amrex::Box& bx,
    const FieldState fstate,
    const amrex::Array4<amrex::Real>& src_term) const
{
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
            const amrex::Real wspd = std::sqrt(ur * ur + vr * vr);

            src_term(i, j, k, 0) -= tau.calc_vel_x(uu - u_wall, wspd) / dx[2];
            src_term(i, j, k, 1) -= tau.calc_vel_y(vv - v_wall, wspd) / dx[2];
            //        src_term(i, j, k, 2) += 0.0;
        });
}

} // namespace amr_wind::pde::icns
