#include "amr-wind/immersed_boundary/bluff_body/bluff_body_ops.H"
#include "amr-wind/core/MultiParser.H"
#include "amr-wind/utilities/ncutils/nc_interface.H"
#include "amr-wind/utilities/io_utils.H"

#include "amr-wind/fvm/gradient.H"
#include "amr-wind/core/field_ops.H"

// Used for mms
#include "amr-wind/physics/ConvectingTaylorVortex.H"

#include "AMReX_ParmParse.H"

namespace amr_wind {
namespace ib {
namespace bluff_body {

void read_inputs(
    BluffBodyBaseData& wdata, IBInfo&, const ::amr_wind::utils::MultiParser& pp)
{
    pp.query("has_wall_model", wdata.has_wall_model);
    pp.query("is_moving", wdata.is_moving);
    pp.query("is_mms", wdata.is_mms);
    pp.queryarr("vel_bc", wdata.vel_bc);
}

void init_data_structures(BluffBodyBaseData&) {}

void apply_mms_vel(CFDSim& sim)
{
    const int nlevels = sim.repo().num_active_levels();

    auto& levelset = sim.repo().get_field("ib_levelset");
    auto& velocity = sim.repo().get_field("velocity");
    auto& m_conv_taylor_green =
        sim.physics_manager().get<ctv::ConvectingTaylorVortex>();

    const amrex::Real u0 = m_conv_taylor_green.get_u0();
    const amrex::Real v0 = m_conv_taylor_green.get_v0();
    const amrex::Real omega = m_conv_taylor_green.get_omega();

    amrex::Real t = sim.time().new_time();

    auto& geom = sim.mesh().Geom();

    for (int lev = 0; lev < nlevels; ++lev) {

        const auto& dx = geom[lev].CellSizeArray();
        const auto& problo = geom[lev].ProbLoArray();

        for (amrex::MFIter mfi(levelset(lev)); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.growntilebox();
            auto phi = levelset(lev).array(mfi);
            auto varr = velocity(lev).array(mfi);
            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                    const amrex::Real y = problo[1] + (j + 0.5) * dx[1];

                    if (phi(i, j, k) <= 0) {
                        varr(i, j, k, 0) =
                            u0 - std::cos(utils::pi() * (x - u0 * t)) *
                                     std::sin(utils::pi() * (y - v0 * t)) *
                                     std::exp(-2.0 * omega * t);
                        varr(i, j, k, 1) =
                            v0 + std::sin(utils::pi() * (x - u0 * t)) *
                                     std::cos(utils::pi() * (y - v0 * t)) *
                                     std::exp(-2.0 * omega * t);
                        varr(i, j, k, 2) = 0.0;
                    }
                });
        }
    }
}

void apply_mms_mac_vel(CFDSim& sim)
{
    const int nlevels = sim.repo().num_active_levels();
    auto& geom = sim.mesh().Geom();
    auto& levelset = sim.repo().get_field("ib_levelset");

    std::unique_ptr<ScratchField> ls_xf, ls_yf, ls_zf;
    ls_xf = sim.repo().create_scratch_field(1, 0, amr_wind::FieldLoc::XFACE);
    ls_yf = sim.repo().create_scratch_field(1, 0, amr_wind::FieldLoc::YFACE);
    ls_zf = sim.repo().create_scratch_field(1, 0, amr_wind::FieldLoc::ZFACE);

    amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> ls_face(
        nlevels);

    for (int lev = 0; lev < nlevels; ++lev) {
        ls_face[lev][0] = &(*ls_xf)(lev);
        ls_face[lev][1] = &(*ls_yf)(lev);
        ls_face[lev][2] = &(*ls_zf)(lev);

        amrex::average_cellcenter_to_face(
            ls_face[lev], levelset(lev), geom[lev]);
    }

    // These are phase fields
    auto& umac = sim.repo().get_field("u_mac");
    auto& vmac = sim.repo().get_field("v_mac");
    auto& wmac = sim.repo().get_field("w_mac");

    // ctv related variables
    auto& m_conv_taylor_green =
        sim.physics_manager().get<ctv::ConvectingTaylorVortex>();
    const amrex::Real u0 = m_conv_taylor_green.get_u0();
    const amrex::Real v0 = m_conv_taylor_green.get_v0();
    const amrex::Real omega = m_conv_taylor_green.get_omega();

    // mac velocities should be at t{n+1/2};
    amrex::Real t = 0.5 * (sim.time().current_time() + sim.time().new_time());

    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& dx = geom[lev].CellSizeArray();
        const auto& problo = geom[lev].ProbLoArray();

        for (amrex::MFIter mfi(levelset(lev)); mfi.isValid(); ++mfi) {
            const auto& xbx = mfi.nodaltilebox(0);
            const auto& ybx = mfi.nodaltilebox(1);
            const auto& zbx = mfi.nodaltilebox(2);

            auto phi_xf = ls_face[lev][0]->array(mfi);
            auto umac_arr = umac(lev).array(mfi);
            amrex::ParallelFor(
                xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    const amrex::Real x = problo[0] + i * dx[0];
                    const amrex::Real y = problo[1] + (j + 0.5) * dx[1];

                    if (phi_xf(i, j, k) <= 0.) {
                        umac_arr(i, j, k) =
                            u0 - std::cos(utils::pi() * (x - u0 * t)) *
                                     std::sin(utils::pi() * (y - v0 * t)) *
                                     std::exp(-2.0 * omega * t);
                    }
                });

            auto phi_yf = ls_face[lev][1]->array(mfi);
            auto vmac_arr = vmac(lev).array(mfi);
            amrex::ParallelFor(
                ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                    const amrex::Real y = problo[1] + j * dx[1];

                    if (phi_yf(i, j, k) <= 0.) {
                        vmac_arr(i, j, k) =
                            v0 + std::sin(utils::pi() * (x - u0 * t)) *
                                     std::cos(utils::pi() * (y - v0 * t)) *
                                     std::exp(-2.0 * omega * t);
                    }
                });

            auto phi_zf = ls_face[lev][2]->array(mfi);
            auto wmac_arr = wmac(lev).array(mfi);
            amrex::ParallelFor(
                zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    if (phi_zf(i, j, k) <= 0.) {
                        wmac_arr(i, j, k) = 0.;
                    }
                });
        }
    }
}

void apply_dirichlet_vel(CFDSim& sim, amrex::Vector<amrex::Real>& vel_bc)
{
    const int nlevels = sim.repo().num_active_levels();
    auto& geom = sim.mesh().Geom();

    auto& velocity = sim.repo().get_field("velocity");
    auto& levelset = sim.repo().get_field("ib_levelset");
    levelset.fillpatch(sim.time().current_time());
    auto& normal = sim.repo().get_field("ib_normal");
    fvm::gradient(normal, levelset);
    field_ops::normalize(normal);
    normal.fillpatch(sim.time().current_time());

    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& dx = geom[lev].CellSizeArray();
        // const auto& problo = geom[lev].ProbLoArray();
        // Defining the "ghost-cell" band distance
        amrex::Real phi_b = std::cbrt(dx[0] * dx[1] * dx[2]);

        for (amrex::MFIter mfi(levelset(lev)); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            auto varr = velocity(lev).array(mfi);
            auto phi_arr = levelset(lev).array(mfi);
            auto norm_arr = normal(lev).array(mfi);

            amrex::Real velx = vel_bc[0];
            amrex::Real vely = vel_bc[1];
            amrex::Real velz = vel_bc[2];

            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    // Pure solid-body points
                    if (phi_arr(i, j, k) < -phi_b) {
                        varr(i, j, k, 0) = velx;
                        varr(i, j, k, 1) = vely;
                        varr(i, j, k, 2) = velz;
                        norm_arr(i, j, k, 0) = 0.;
                        norm_arr(i, j, k, 1) = 0.;
                        norm_arr(i, j, k, 2) = 0.;
                        // This determines the ghost-cells
                    } else if (
                        phi_arr(i, j, k) < 0 && phi_arr(i, j, k) >= -phi_b) {
                        varr(i, j, k, 0) = velx;
                        varr(i, j, k, 1) = vely;
                        varr(i, j, k, 2) = velz;
                    } else {
                        norm_arr(i, j, k, 0) = 0.;
                        norm_arr(i, j, k, 1) = 0.;
                        norm_arr(i, j, k, 2) = 0.;
                    }
                });
        }
    }
}

void prepare_netcdf_file(
    const std::string& ncfile,
    const BluffBodyBaseData& meta,
    const IBInfo& info)
{
    amrex::ignore_unused(ncfile, meta, info);
}

void write_netcdf(
    const std::string& ncfile,
    const BluffBodyBaseData& meta,
    const IBInfo& info,
    const amrex::Real time)
{
    amrex::ignore_unused(ncfile, meta, info, time);
}

} // namespace bluff_body
} // namespace ib
} // namespace amr_wind
