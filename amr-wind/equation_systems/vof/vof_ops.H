#ifndef VOF_OPS_H
#define VOF_OPS_H

#include "amr-wind/equation_systems/vof/vof.H"
#include "amr-wind/equation_systems/vof/VolumeFractions_K.H"
#include "amr-wind/equation_systems/vof/Sweep_K.H"

namespace amr_wind {
namespace pde {

template<typename Scheme>
struct FieldRegOp<VOF, Scheme>
{
    FieldRegOp(CFDSim& sim): repo(sim.repo()) {}

    PDEFields  operator()(const SimTime& time, const int probtype)
    {
        auto fields = create_fields_instance<VOF, Scheme>(time, repo, probtype);
        
        auto& normal = repo.declare_cc_field("normal",AMREX_SPACEDIM,1,1);
        auto& levelset = repo.declare_cc_field("levelset",1,1,1);
        auto& intercept = repo.declare_cc_field("intercept",1,1,1);
        auto& curvature = repo.declare_cc_field("curvature",1,1,1);

        levelset.template register_fill_patch_op<FieldFillPatchOps<FieldBCNoOp>>(repo.mesh(),time,probtype);
        normal.template register_fill_patch_op<FieldFillPatchOps<FieldBCNoOp>>(repo.mesh(),time,probtype);
        curvature.template register_fill_patch_op<FieldFillPatchOps<FieldBCNoOp>>(repo.mesh(),time,probtype);
        intercept.template register_fill_patch_op<FieldFillPatchOps<FieldBCNoOp>>(repo.mesh(),time,probtype);
     
        return fields;
    }
    FieldRepo& repo;

};


template<typename Scheme>
struct ComputeRHSOp<VOF, Scheme>
{
    ComputeRHSOp(PDEFields& fields_in) 
    : fields(fields_in) 
    //,u_mac(fields_in.repo.get_field("u_mac")) 
    //,v_mac(fields_in.repo.get_field("v_mac")) 
    //,w_mac(fields_in.repo.get_field("w_mac"))
    {}
        
    int tswap=0;

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
        
        // Do the temporal swap
        tswap = tswap +1;

        for (int lev = 0; lev < nlevels; ++lev) {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(field(lev)); mfi.isValid(); ++mfi) {
                const auto& bx = mfi.tilebox();
                auto vof = field(lev).array(mfi);
                const auto vof_o = field_old(lev).const_array(mfi);
                const auto ddt_o = conv_term(lev).const_array(mfi);
                 
                amrex::ParallelFor(
                    bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        if(tswap>3){
                            tswap=1;
                        }
                        if(tswap%3==0){
                            //swp(i,j,k,3,dt,dx,dy,dz,uz,vof);
                            //swp(i,j,k,1,dt,dx,dy,dz,ux,vof);
                            //swp(i,j,k,2,dt,dx,dy,dz,uy,vof);
                        }else if (tswap%2==0){
                            //swp(i,j,k,2,dt,dx,dy,dz,uy,vof);
                            //swp(i,j,k,3,dt,dx,dy,dz,uz,vof);
                            //swp(i,j,k,1,dt,dx,dy,dz,ux,vof);
                        }else if (tswap%3==0){
                            //swp(i,j,k,1,dt,dx,dy,dz,ux,vof);
                            //swp(i,j,k,2,dt,dx,dy,dz,uy,vof);
                            //swp(i,j,k,3,dt,dx,dy,dz,uz,vof);
                        }
                        vof(i, j, k) = vof_o(i, j, k) + dt * ddt_o(i, j, k);
                        // Defer n+1/2 update to the time-stepping algorithm
                        // vof_nph(i, j, k) = 0.5 * (vof(i, j, k) + vof_o(i, j, k));
                    });
            }
        }
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
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(field(lev)); mfi.isValid(); ++mfi) {
                const auto& bx = mfi.tilebox();
                auto vof = field(lev).array(mfi);
                const auto vof_o = field_old(lev).const_array(mfi);
                const auto ddt = conv_term(lev).const_array(mfi);
                const auto ddt_o = conv_term_old(lev).const_array(mfi);

                amrex::ParallelFor(
                    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        vof(i, j, k) =
                            vof_o(i, j, k) +
                            dt * 0.5 * (ddt_o(i, j, k) + ddt(i, j, k));
                        // Defer n+1/2 update to time-stepping algorithm
                        // vof_nph(i, j, k) =
                        //     0.5 * (vof(i, j, k) + vof_o(i, j, k));
                    });
            }
        }
    }

    // data members
    PDEFields& fields;
};
}
}

#endif /* VOF_OPS_H */
