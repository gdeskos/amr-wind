#include "amr-wind/physics/multiphase/MultiPhase.H"
#include "amr-wind/CFDSim.H"
#include "AMReX_ParmParse.H"

#include "amr-wind/fvm/gradient.H"
#include "amr-wind/core/field_ops.H"
#include "amr-wind/physics/multiphase/VolumeFraction_K.H"

namespace amr_wind {

MultiPhase::MultiPhase(CFDSim& sim)
    : m_sim(sim)
    , m_velocity(sim.pde_manager().icns().fields().field)
    , m_mueff(sim.pde_manager().icns().fields().mueff)
    , m_density(sim.repo().get_field("density"))
{
    // Register levelset equation
    auto& levelset_eqn = sim.pde_manager().register_transport_pde("Levelset");
    // auto& vof_eqn = sim.pde_manager().register_transport_pde("VOF");

    // Defer getting levelset field until PDE has been registered
    m_levelset = &(levelset_eqn.fields().field);
    // m_vof = &(vof_eqn.fields().field);

    amrex::ParmParse pp_multiphase("MultiPhase");
    pp_multiphase.query("density_fluid1", m_rho1);
    pp_multiphase.query("density_fluid2", m_rho2);
    pp_multiphase.query("viscosity_fluid1", m_mu1);
    pp_multiphase.query("viscosity_fluid2", m_mu2);
}

void MultiPhase::initialize_fields(int level, const amrex::Geometry& geom)
{
    set_multiphase_properties(level, geom);
}

void MultiPhase::post_init_actions()
{
    const int nlevels = m_sim.repo().num_active_levels();
    const auto& geom = m_sim.mesh().Geom();

    for (int lev = 0; lev < nlevels; ++lev) {
        set_multiphase_properties(lev, geom[lev]);
    }

    // levelset2vof();
}

void MultiPhase::post_advance_work()
{
    const int nlevels = m_sim.repo().num_active_levels();
    const auto& geom = m_sim.mesh().Geom();

    for (int lev = 0; lev < nlevels; ++lev) {
        set_multiphase_properties(lev, geom[lev]);
    }
    // levelset2vof();
}

void MultiPhase::set_multiphase_properties(
    int level, const amrex::Geometry& geom)
{
    auto& density = m_density(level);
    auto& mueff = m_mueff(level);
    auto& levelset = (*m_levelset)(level);

    for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();
        const auto& dx = geom.CellSizeArray();

        const amrex::Array4<amrex::Real>& phi = levelset.array(mfi);
        const amrex::Array4<amrex::Real>& rho = density.array(mfi);
        const amrex::Array4<amrex::Real>& mu = mueff.array(mfi);
        const amrex::Real eps = 2. * std::cbrt(dx[0] * dx[1] * dx[2]);
        const amrex::Real rho1 = m_rho1;
        const amrex::Real rho2 = m_rho2;
        const amrex::Real mu1 = m_mu1;
        const amrex::Real mu2 = m_mu2;
        amrex::ParallelFor(
            vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::Real H;
                // const amrex::Real H = (phi(i,j,k) > eps) ? 1.0 :
                // 0.5*(1+phi(i,j,k)/(2*epsilon)+1./M_PI*std::sin(phi(i,j,k)*M_PI/epsilon);
                if (phi(i, j, k) > eps) {
                    H = 1.0;
                } else if (phi(i, j, k) < -eps) {
                    H = 0.;
                } else {
                    H = 0.5 * (1 + phi(i, j, k) / (2 * eps) +
                               1. / M_PI * std::sin(phi(i, j, k) * M_PI / eps));
                }
                rho(i, j, k) = rho1 * H + rho2 * (1 - H);
                mu(i, j, k) = mu1 * H + mu2 * (1 - H);
            });
    }
}

void MultiPhase::levelset2vof()
{

    // Compute normal vectors
    auto& normal = m_sim.repo().get_field("interface_normal");
    fvm::gradient(normal, (*m_levelset));
    // field_ops::normalize(normal);

    // Convert level set phi to VOF function
    const int nlevels = m_sim.repo().num_active_levels();
    /** We make our calculations from 1 to n-1 using growntilebox(-1)
     *  1) Compute the normal vector mx,my,mz using the levelset function
     *  2) Normalise the normal vector so that mx,my,mz>0 and mx+my+mz=1;
     *  3) Shift alpha to origin
     *  4) Get the fraction volume from alpha
     *  5) Do clipping - make sure that Volume fraction are between 0 and 1
     */
    for (int lev = 0; lev < nlevels; ++lev) {
        auto& vof = (*m_vof)(lev);
        auto& levelset = (*m_levelset)(lev);
        auto& norm = normal(lev);

        for (amrex::MFIter mfi(levelset); mfi.isValid(); ++mfi) {
            const auto& vbx = mfi.validbox();
            const auto& cc = vof.array(mfi);
            const auto& ls = levelset.array(mfi);
            const auto& mxyz = norm.array(mfi);
            amrex::ParallelFor(
                vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    amrex::Real mx = std::abs(mxyz(i, j, k, 0));
                    amrex::Real my = std::abs(mxyz(i, j, k, 1));
                    amrex::Real mz = std::abs(mxyz(i, j, k, 2));

                    amrex::Real normL1 = mx + my + mz;
                    mx = mx / normL1;
                    my = my / normL1;
                    mz = mz / normL1;

                    amrex::Real alpha = ls(i, j, k) / normL1;

                    alpha = alpha + 0.50;

                    if (alpha >= 1.0) {
                        cc(i, j, k) = 1.0;
                    } else if (alpha <= 0.0) {
                        cc(i, j, k) = 0.0;
                    } else {
                        cc(i, j, k) = FL3D(mx, my, mz, alpha, 0.0, 1.0);
                    }
                });
        }
    }
}

} // namespace amr_wind
