#include "amr-wind/multiphase/Multiphase.H"
#include "amr-wind/multiphase/VolumeFractions_K.H"
#include "amr-wind/CFDSim.H"
#include "AMReX_ParmParse.H"
#include "amr-wind/utilities/trig_ops.H"
#include "amr-wind/derive/derive_K.H"
#include "amr-wind/utilities/tensor_ops.H"
#include "amr-wind/equation_systems/BCOps.H"

namespace amr_wind {

Multiphase::Multiphase(CFDSim& sim)
    : m_sim(sim)
    //, m_velocity(sim.pde_manager().icns().fields().field)
    , m_velocity(sim.repo().get_field("velocity"))
    , m_density(sim.repo().get_field("density"))
    , m_normal(sim.repo().declare_cc_field("normal", AMREX_SPACEDIM, 1, 1))
    , m_curvature(sim.repo().declare_cc_field("curvature", 1, 1, 1))
    , m_intercept(sim.repo().declare_cc_field("intercept", 1, 1, 1))
    , m_levelset(sim.repo().declare_cc_field("levelset", 1, 1, 1))
    , m_surface_tension(
          sim.repo().declare_cc_field("surface_tension", AMREX_SPACEDIM, 1, 1))
{

    // Define the VOF equation system -- Apply Geometric VOF
    auto& vof_eqn = sim.pde_manager().register_transport_pde("VOF");
    m_vof = &(vof_eqn.fields().field);

    amrex::ParmParse pp("incflo");
    pp.query("rho_air", m_rho_air);
    pp.query("rho_water", m_rho_water);
    pp.query("multiphase_problem", m_multiphase_problem);
    pp.query("dambreak_box_h", m_dambreak_h);
    pp.query("dambreak_box_d", m_dambreak_d);
    pp.query("surface_tension_coeff", m_sigma);
}

/** Initialize the levelset function and density fields at the beginning of the
 *  simulation.
 */
void Multiphase::initialize_fields(int level, const amrex::Geometry& geom)
{
    using namespace utils;

    auto& vof = (*m_vof)(level);
    auto& levelset = m_levelset(level);
    auto& velocity = m_velocity(level);

    /** Initialise fields through a levelset function $\Phi$
        By convention the “inside” of the interface corresponds to Φ>0.
    */
    for (amrex::MFIter mfi(levelset); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();

        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();
        const auto& probhi = geom.ProbHiArray();
        auto phi = levelset.array(mfi);
        //auto vel = velocity.array(mfi);
        amrex::ParallelFor(
            vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
                const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
                const amrex::Real x0 = 0.5 * (problo[0] + probhi[0]);
                const amrex::Real y0 = 0.5 * (problo[1] + probhi[1]);
                const amrex::Real z0 = 0.5 * (problo[2] + probhi[2]);
                if (m_multiphase_problem == 1) { // Sphere
                    amrex::Real R=0.15;
                    phi(i, j, k) = R - std::sqrt((x -x0+R) * (x - x0+R) + 
                                   (y - y0+R) * (y - y0+R) + (z - z0+R) * (z - z0+R));
                } else if (m_multiphase_problem == 2) {
                    if (x < m_dambreak_d && z < m_dambreak_h) {
                        phi(i, j, k) =
                            std::min(m_dambreak_d - x, m_dambreak_h - z);
                    } else if (x < m_dambreak_d && z > m_dambreak_h) {
                        phi(i, j, k) = m_dambreak_h - z;
                    } else if (x > m_dambreak_d && z < m_dambreak_h) {
                        phi(i, j, k) = m_dambreak_d - x;
                    } else {
                        phi(i, j, k) =
                            std::min(m_dambreak_d - x, m_dambreak_h - z);
                    }
                }
            });
    }

}

void Multiphase::post_init_actions()
{
    const auto& time = m_sim.time().current_time();

    
    // Apply BC so that all fields are updated
    BCSrcTerm bc_levelset(m_levelset);
    bc_levelset();
    
    // From levelset to vof
    levelset2vof();

    BCSrcTerm bc_normal(m_normal);
    bc_normal();
    BCSrcTerm bc_curv(m_curvature);
    bc_curv();
    BCSrcTerm bc_surface_tension(m_surface_tension);
    bc_surface_tension();
     
    const int nlevels = m_sim.repo().num_active_levels();
    const auto& geom = m_sim.mesh().Geom();
    for (int lev = 0; lev < nlevels; ++lev) {
        set_density(lev, geom[lev]);
    }
}

void Multiphase::pre_advance_work()
{
    const auto& time = m_sim.time().current_time();

    (*m_vof).fillpatch(time);
    
    const int nlevels = m_sim.repo().num_active_levels();
    const auto& geom = m_sim.mesh().Geom();

    for (int lev = 0; lev < nlevels; ++lev) {
        set_density(lev, geom[lev]);
    }
    // Compute surface tension
    compute_surface_tension();
}

void Multiphase::post_advance_work()
{
    do_clipping();
}

void Multiphase::compute_surface_tension()
{
    const int nlevels = m_sim.repo().num_active_levels();
    const auto& geom = m_sim.mesh().Geom();

    for (int lev = 0; lev < nlevels; ++lev) {

        auto& density = m_density(lev);
        auto& vof = (*m_vof)(lev);
        auto& normal = m_normal(lev);
        auto& curvature = m_curvature(lev);
        auto& surface_tension = m_surface_tension(lev);
        const amrex::Real dx = geom[lev].CellSize()[0];
        const amrex::Real dy = geom[lev].CellSize()[1];
        const amrex::Real dz = geom[lev].CellSize()[2];
        const amrex::Real ds = std::cbrt(dx * dy * dz);
        const amrex::Real epsilon = 2. * ds;

        surface_tension.setVal(0.);

        for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            auto rho = density.array(mfi);
            auto F = vof.array(mfi);
            auto n = normal.array(mfi);
            auto kappa = curvature.array(mfi);
            auto ST = surface_tension.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                if (F(i, j, k) <= 1.) {
                    // const amrex::Real
                    // delta_st=1./(2*epsilon)*(1+std::cos(M_PI*phi(i,j,k)/epsilon));
                    ST(i, j, k, 0) =
                        0.; // m_sigma*kappa(i,j,k)*delta_st*n(i,j,k,0)/rho(i,j,k);
                    ST(i, j, k, 1) =
                        0.; // m_sigma*kappa(i,j,k)*delta_st*n(i,j,k,1)/rho(i,j,k);
                    ST(i, j, k, 2) =
                        0.; // m_sigma*kappa(i,j,k)*delta_st*n(i,j,k,2)/rho(i,j,k);
                }
            });
        }
    }
}

void Multiphase::set_density(int level, const amrex::Geometry& geom)
{
    using namespace utils;

    // auto& levelset = (*m_levelset)(level);
    auto& vof = (*m_vof)(level);
    auto& density = m_density(level);

    const amrex::Real dx = geom.CellSize()[0];
    const amrex::Real dy = geom.CellSize()[1];
    const amrex::Real dz = geom.CellSize()[2];
    const amrex::Real ds = std::cbrt(dx * dy * dz);
    const amrex::Real epsilon = 2. * ds;

    for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.tilebox();

        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();
        // auto phi = levelset.array(mfi);
        auto F = vof.array(mfi);
        auto Density = density.array(mfi);

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                Density(i, j, k) =
                    m_rho_water * F(i, j, k) + m_rho_air * (1 - F(i, j, k));
            });
    }
}

} // namespace amr_wind
