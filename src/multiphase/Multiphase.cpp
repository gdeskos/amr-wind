#include "Multiphase.H"
#include "VolumeFractions_K.H"
#include "CFDSim.H"
#include "AMReX_ParmParse.H"
#include "trig_ops.H"
#include "derive_K.H"
#include "tensor_ops.H"
#include "BCOps.H"

namespace amr_wind {

Multiphase::Multiphase(CFDSim& sim)
    : m_sim(sim)
    , m_velocity(sim.pde_manager().icns().fields().field)
    , m_density(sim.repo().get_field("density"))
    , m_lsnormal(sim.repo().declare_cc_field("ls_normal",AMREX_SPACEDIM,1,1))
    , m_lscurv(sim.repo().declare_cc_field("ls_curvature",1,1,1))
    , m_surface_tension(sim.repo().declare_cc_field("surface_tension",AMREX_SPACEDIM,1,1))
{
    // Define the levelset equation system
    auto& ls_eqn = sim.pde_manager().register_transport_pde("LevelSet");
    m_levelset = &(ls_eqn.fields().field);
    // Define the VOF equation system
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

/** Initialize the vof and density fields at the beginning of the
 *  simulation.
 */
void Multiphase::initialize_fields(int level, const amrex::Geometry& geom)
{
    using namespace utils;

    auto& levelset = (*m_levelset)(level);
    auto& vof = (*m_vof)(level);
    auto& velocity = m_velocity(level);

    for (amrex::MFIter mfi(levelset); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();

        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();
        const auto& probhi = geom.ProbHiArray();
        auto phi = levelset.array(mfi);
        auto F = vof.array(mfi);
        auto vel = velocity.array(mfi);

        // Hardcoded constants at the moment
        const amrex::Real a = 4.;
        const amrex::Real b = 2.;
        const amrex::Real c = 2.;
        amrex::ParallelFor(
            vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
                const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
                const amrex::Real x0 = 0.5 * (problo[0] + probhi[0]);
                const amrex::Real y0 = 0.5 * (problo[1] + probhi[1]);
                const amrex::Real z0 = 0.5 * (problo[2] + probhi[2]);
           if(m_multiphase_problem==1){ 
                phi(i, j, k) =
                    -((x - x0) * (x - x0) + (y - y0) * (y - y0) +
                      (z - z0) * (z - z0)) *
                    (std::sqrt(
                         x * x / (a * a) + y * y / (b * b) + z * z / (c * c)) -
                     1.);
                vel(i, j, k, 0) = 1.;
                vel(i, j, k, 1) = 0.;
                vel(i, j, k, 2) = 0.;
            }else if(m_multiphase_problem==2){
                if (x<m_dambreak_d && z<m_dambreak_h){
                    phi(i,j,k)=std::min(m_dambreak_d-x,m_dambreak_h-z); 
                    F(i,j,k)=1.;
                }else if(x<m_dambreak_d && z>m_dambreak_h){
                    phi(i,j,k)=m_dambreak_h-z;
                    F(i,j,k)=0.;
                }else if(x>m_dambreak_d && z<m_dambreak_h){
                    phi(i,j,k)=m_dambreak_d-x;
                    F(i,j,k)=0.;
                }else{
                    phi(i,j,k)=std::min(m_dambreak_d-x,m_dambreak_h-z); 
                    F(i,j,k)=0.;
                }

                vel(i, j, k, 0) = 0.;
                vel(i, j, k, 1) = 0.;
                vel(i, j, k, 2) = 0.;
            }
            });
    }
    // compute density based on the volume fractions
    set_density(level, geom);
}

void Multiphase::post_init_actions()
{
    BCScalar bc_ls((*m_levelset));
    bc_ls();
    BCSrcTerm bc_norm(m_lsnormal);
    bc_norm();
    BCSrcTerm bc_curv(m_lscurv);
    bc_curv();
}

void Multiphase::pre_advance_work()
{
    
    const int nlevels = m_sim.repo().num_active_levels();
    const auto& geom = m_sim.mesh().Geom();

    for (int lev = 0; lev < nlevels; ++lev) {
        set_density(lev, geom[lev]); 
    }
    compute_normals_and_curvature();
   
    /* Check the limits of levelset, lsnormal and curvature
    for (int lev = 0; lev < nlevels; ++lev) { 
        amrex::Print()<<(*m_levelset)(lev).min(0)<<" "<<(*m_levelset)(lev).max(0)<<std::endl; 
        amrex::Print()<<(m_lsnormal)(lev).min(0)<<" "<<(m_lsnormal)(lev).max(0)<<std::endl; 
        amrex::Print()<<(m_lscurv)(lev).min(0)<<" "<<(m_lscurv)(lev).max(0)<<std::endl; 
    }
    */
    
}

void Multiphase::compute_surface_tension()
{
    const int nlevels = m_sim.repo().num_active_levels();
    const auto& geom = m_sim.mesh().Geom();

   for(int lev=0; lev<nlevels; ++lev){ 
       
        auto& density = m_density(lev);
        auto& levelset = (*m_levelset)(lev);
        auto& normal = m_lsnormal(lev);
        auto& curvature = m_lscurv(lev);
        auto& surface_tension = m_surface_tension(lev);
        const amrex::Real dx = geom[lev].CellSize()[0];
        const amrex::Real dy = geom[lev].CellSize()[1];
        const amrex::Real dz = geom[lev].CellSize()[2];
        const amrex::Real ds = std::cbrt(dx * dy * dz);
        const amrex::Real epsilon = 2. * ds;

        surface_tension.setVal(0.);

       for (amrex::MFIter mfi(levelset); mfi.isValid(); ++mfi) {
           const auto& bx = mfi.tilebox();
           auto rho = density.array(mfi);
           auto phi = levelset.array(mfi);
           auto n = normal.array(mfi);
           auto kappa = curvature.array(mfi);
           auto ST=surface_tension.array(mfi);

           amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {  
             if (abs(phi(i, j, k)) < epsilon ) {
                 const amrex::Real delta_st=1./(2*epsilon)*(1+std::cos(M_PI*phi(i,j,k)/epsilon));
                 ST(i, j, k, 0) = m_sigma*kappa(i,j,k)*delta_st*n(i,j,k,0)/rho(i,j,k);
                 ST(i, j, k, 1) = m_sigma*kappa(i,j,k)*delta_st*n(i,j,k,1)/rho(i,j,k);
                 ST(i, j, k, 2) = m_sigma*kappa(i,j,k)*delta_st*n(i,j,k,2)/rho(i,j,k);
             }
         }); 
    }
    }
}
void Multiphase::compute_normals_and_curvature()
{
   
    const auto& time = m_sim.time().current_time();
    (*m_levelset).fillpatch(time);
    //populate gradient into lsnormal to avoid creating a temporary buffer     
    compute_gradient(m_lsnormal,(*m_levelset));
    
    m_lsnormal.fillpatch(time);
    m_lscurv.fillpatch(time);

    compute_curvature(m_lscurv,m_lsnormal);
    // now normalise the gradient of the levelset to get m_lsnormal
    normalize_field(m_lsnormal);
}


void Multiphase::set_density(int level, const amrex::Geometry& geom)
{
    using namespace utils;

    auto& levelset = (*m_levelset)(level);
    auto& vof = (*m_vof)(level);
    auto& density = m_density(level);

    const amrex::Real dx = geom.CellSize()[0];
    const amrex::Real dy = geom.CellSize()[1];
    const amrex::Real dz = geom.CellSize()[2];
    const amrex::Real ds = std::cbrt(dx * dy * dz);
    const amrex::Real epsilon = 2. * ds;

    for (amrex::MFIter mfi(levelset); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.tilebox();

        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();
        auto phi = levelset.array(mfi);
        auto F = vof.array(mfi);
        auto Density = density.array(mfi);

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            //    if (phi(i, j, k) > epsilon) {
            //        const amrex::Real H = 1.0;
            //        Density(i, j, k) = m_rho_water * H + m_rho_air * (1. - H);
            //    } else if (phi(i, j, k) < -epsilon) {
            //        const amrex::Real H = 0.;
            //        Density(i, j, k) = m_rho_water * H + m_rho_air * (1. - H);
            //    } else {
            //        const amrex::Real H =
            //            0.5 *
            //            (1 + phi(i, j, k) / (2 * epsilon) +
            //             1. / M_PI * std::sin(phi(i, j, k) * M_PI / epsilon));
            //        Density(i, j, k) = m_rho_water * H + m_rho_air * (1. - H);
            //    }
            Density(i,j,k)=m_rho_water*F(i,j,k)+m_rho_air*(1-F(i,j,k));
            });
    }
}

} // namespace amr_wind