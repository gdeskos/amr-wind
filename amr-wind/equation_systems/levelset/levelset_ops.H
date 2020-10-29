#ifndef LEVELSET_OPS_H
#define LEVELSET_OPS_H

#include "amr-wind/equation_systems/levelset/levelset.H"
#include "amr-wind/fvm/gradient.H"

namespace amr_wind {
namespace pde {

/** Field registration operator for Levelset equation system
 *  \ingroup levelset
 */
template <typename Scheme>
struct FieldRegOp<Levelset, Scheme>
{
    FieldRegOp(CFDSim& sim_in) : sim(sim_in) {}

    PDEFields operator()(const SimTime& time, const int probtype)
    {
        auto& repo = sim.repo();
        auto fields =
            create_fields_instance<Levelset, Scheme>(time, repo, probtype);

        auto& normal =
            repo.declare_cc_field("interface_normal", AMREX_SPACEDIM, 1, 1);
        auto& curvature = repo.declare_cc_field("interface_curvature", 1, 1, 1);

        normal.template register_fill_patch_op<FieldFillPatchOps<FieldBCNoOp>>(
            repo.mesh(), time, probtype);
        curvature
            .template register_fill_patch_op<FieldFillPatchOps<FieldBCNoOp>>(
                repo.mesh(), time, probtype);

        normal.fillpatch(time.current_time());
        curvature.fillpatch(time.current_time());

        // Read input parameters
        amrex::ParmParse pp_levelset("LevelSet");
        pp_levelset.query("reinit",do_reinit);
        pp_levelset.query("reinit_max_iter",max_reinit_iter);

        // Register fields for output/restart
        auto& iomgr = sim.io_manager();
        iomgr.register_io_var(fields.field.name());
        iomgr.register_io_var(normal.name());
        iomgr.register_io_var(curvature.name());

        return fields;
    }

    CFDSim& sim;
};

/** Right-hand side (RHS) evaluator for Levelset transport equation
 *  \ingroup levelset
 */
template <typename Scheme>
struct ComputeRHSOp<Levelset, Scheme>
{
    ComputeRHSOp(PDEFields& fields_in) : fields(fields_in) {}

    void predictor_rhs(const DiffusionType, const amrex::Real dt)
    {
        // Field states for diffusion and advection terms. In Godunov scheme
        // these terms only have one state.
        auto fstate = std::is_same<Scheme, fvm::Godunov>::value
                          ? FieldState::New
                          : FieldState::Old;

        const int nlevels = fields.repo.num_active_levels();
        auto& field = fields.field;
        auto& field_old = field.state(FieldState::Old);
        auto& conv_term = fields.conv_term.state(fstate);

        for (int lev = 0; lev < nlevels; ++lev) {
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(field(lev)); mfi.isValid(); ++mfi) {
                const auto& bx = mfi.tilebox();
                auto phi = field(lev).array(mfi);
                const auto phi_o = field_old(lev).const_array(mfi);
                const auto ddt_o = conv_term(lev).const_array(mfi);

                amrex::ParallelFor(
                    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        phi(i, j, k) = phi_o(i, j, k) + dt * ddt_o(i, j, k);
                        // Defer n+1/2 update to the time-stepping algorithm
                        // phi_nph(i, j, k) = 0.5 * (phi(i, j, k) + phi_o(i, j,
                        // k));
                    });
            }
        }
    
    // Re-initialize the levelset so that $|\nabla \cdot \phi| = 1$
    field.advance_states();
    field.state(amr_wind::FieldState::New).fillpatch(0.0);
    field_old.state(amr_wind::FieldState::Old).fillpatch(0.0);
    const auto& geom = fields.repo.mesh().Geom();
    auto& normal=fields.repo.get_field("interface_normal");
    fvm::gradient(normal,field_old);
    int iter=0;
    bool converged=false;
    while(!converged){
    // test convergence
    if (iter>max_reinit_iter-1 || !do_reinit)
        break;

    iter = iter + 1;

    for (int lev = 0; lev < nlevels; ++lev) {
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(field(lev)); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            auto phi = field(lev).array(mfi);
            auto dphi = normal(lev).array(mfi);
            const auto phi_o = field_old(lev).const_array(mfi);
            const auto& dx = geom[lev].CellSizeArray();
            amrex::Real eps = 1.0e-6;
            amrex::Real dl = std::sqrt(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]);

            amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real Dphi2 = dphi(i,j,k,0)*dphi(i,j,k,0)+dphi(i,j,k,1)*dphi(i,j,k,1)+dphi(i,j,k,2)*dphi(i,j,k,2);
            amrex::Real sign = phi_o(i,j,k)/std::sqrt(phi_o(i,j,k)*phi_o(i,j,k)+Dphi2*dl*dl);

                if(phi_o(i,j,k)*phi_o(i-1,j,k)<0 || 
                   phi_o(i,j,k)*phi_o(i+1,j,k)<0 ||
                   phi_o(i,j,k)*phi_o(i,j-1,k)<0 ||
                   phi_o(i,j,k)*phi_o(i,j+1,k)<0 ||
                   phi_o(i,j,k)*phi_o(i,j,k-1)<0 ||
                   phi_o(i,j,k)*phi_o(i,j,k+1)<0)
                {
                    amrex::Real Dijk = phi_o(i,j,k) / std::max(std::sqrt(Dphi2), eps);
                    phi(i,j,k) = phi(i,j,k) - dt/dl*(sign*amrex::Math::abs(phi(i,j,k))-Dijk);
                }else{
                    amrex::Real Gijk=interface_gradients(i, j, k, phi_o(i,j,k), dx[0], dx[1],dx[2],phi);
                    phi(i,j,k) = phi(i,j,k) - dt*sign*Gijk;
                }
            
            });
        } // mfi loop

    } // level loop



    }// while loop end
    amrex::Print()<<"  Level set re-initialization "<<std::setw(4)<<iter<<std::endl;       

    }

    void corrector_rhs(const DiffusionType, const amrex::Real dt)
    {
        const int nlevels = fields.repo.num_active_levels();
        auto& field = fields.field;
        auto& field_old = field.state(FieldState::Old);
        auto& conv_term = fields.conv_term;
        auto& conv_term_old = fields.conv_term.state(FieldState::Old);

        for (int lev = 0; lev < nlevels; ++lev) {
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(field(lev)); mfi.isValid(); ++mfi) {
                const auto& bx = mfi.tilebox();
                auto phi = field(lev).array(mfi);
                const auto phi_o = field_old(lev).const_array(mfi);
                const auto ddt = conv_term(lev).const_array(mfi);
                const auto ddt_o = conv_term_old(lev).const_array(mfi);

                amrex::ParallelFor(
                    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        phi(i, j, k) =
                            phi_o(i, j, k) +
                            dt * 0.5 * (ddt_o(i, j, k) + ddt(i, j, k));
                        // Defer n+1/2 update to time-stepping algorithm
                        // phi_nph(i, j, k) =
                        //     0.5 * (phi(i, j, k) + phi_o(i, j, k));
                    });
            }
        }
    }

    amrex::Real interface_gradients(int i, 
                                    int j, 
                                    int k, 
                                    amrex::Real phi0,
                                    amrex::Real dx,
                                    amrex::Real dy,
                                    amrex::Real dz,                        
                                    const amrex::Array4<amrex::Real>& phi)
    {
        amrex::Real a = (phi(i,j,k)-phi(i-1,j,k))/dx;
        amrex::Real b = (phi(i+1,j,k)-phi(i,j,k))/dx;
        amrex::Real c = (phi(i,j,k)-phi(i,j-1,k))/dy;
        amrex::Real d = (phi(i,j+1,k)-phi(i,j,k))/dy;
        amrex::Real e = (phi(i,j,k)-phi(i,j,k-1))/dz;
        amrex::Real f = (phi(i,j,k+1)-phi(i,j,k))/dz;
        amrex::Real ap = std::max(a,0.0); amrex::Real am = std::min(a,0.0);
        amrex::Real bp = std::max(b,0.0); amrex::Real bm = std::min(b,0.0);
        amrex::Real cp = std::max(c,0.0); amrex::Real cm = std::min(c,0.0);
        amrex::Real dp = std::max(d,0.0); amrex::Real dm = std::min(d,0.0);
        amrex::Real ep = std::max(e,0.0); amrex::Real em = std::min(e,0.0);
        amrex::Real fp = std::max(f,0.0); amrex::Real fm = std::min(f,0.0);

        return (phi0 > 0.0) ? std::sqrt(amrex::max(ap*ap,bm*bm)+amrex::max(cp*cp,dm*dm)+amrex::max(ep*ep,fm*fm))-1 
                          : std::sqrt(amrex::max(am*am,bp*bp)+amrex::max(cm*cm,dp*dp)+amrex::max(em*em,fp*fp))-1 ;
    }

    // data members
    PDEFields& fields;
};
} // namespace pde
} // namespace amr_wind

#endif /* LEVELSET_OPS_H */
