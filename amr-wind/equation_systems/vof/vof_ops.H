#ifndef VOF_OPS_H
#define VOF_OPS_H

#include "amr-wind/equation_systems/PDETraits.H"
#include "amr-wind/equation_systems/PDEOps.H"
#include "amr-wind/equation_systems/PDEHelpers.H"
#include "amr-wind/equation_systems/AdvOp_Godunov.H"
#include "amr-wind/equation_systems/AdvOp_MOL.H"
#include "amr-wind/equation_systems/DiffusionOps.H"
#include "amr-wind/equation_systems/vof/vof.H"
#include "amr-wind/core/field_ops.H"
#include "amr-wind/fvm/gradient.H"
#include "amr-wind/fvm/divergence.H"
#include "AMReX_MultiFabUtil.H"

namespace amr_wind {
namespace pde {

void compression_mac_project(FieldRepo& repo, FieldState fstate, const bool);

template <typename Scheme>
struct FieldRegOp<VOF, Scheme>
{
    FieldRegOp(CFDSim& sim) :  repo(sim.repo()) {}

    PDEFields operator()(const SimTime& time, const int probtype)
    {
        
        auto fields = create_fields_instance<VOF, Scheme>(time, repo, probtype);

        auto& normal = repo.declare_cc_field("normal", AMREX_SPACEDIM, 1, 1);
        auto& levelset = repo.declare_cc_field("levelset", 1, 1, 1);
        auto& curvature = repo.declare_cc_field("curvature", 1, 1, 1);
        auto& velc = repo.declare_cc_field("compress_velocity", AMREX_SPACEDIM, 1, 1);

        levelset.template register_fill_patch_op<FieldFillPatchOps<FieldBCNoOp>>(
            repo.mesh(), time, probtype);
        normal.template register_fill_patch_op<FieldFillPatchOps<FieldBCNoOp>>(
            repo.mesh(), time, probtype);
        curvature.template register_fill_patch_op<FieldFillPatchOps<FieldBCNoOp>>(
            repo.mesh(), time, probtype);

        return fields;
    }
    FieldRepo& repo;
};

template<>
struct SrcTermOp<VOF> : SrcTermOpBase<VOF>
{
    SrcTermOp(PDEFields& fields_in)
    : SrcTermOpBase<VOF>(fields_in)
    , fields(fields_in)
    , normal(fields_in.repo.get_field("normal"))
    , velc(fields_in.repo.get_field("compress_velocity"))
    , vel(fields_in.repo.get_field("velocity"))
    {}

    void operator() (const FieldState fstate)
    {
        auto& repo = fields.repo;
        auto& geom = repo.mesh().Geom();

        auto& src_term = fields.src_term;
        auto& dof_field = fields.field.state(fstate);

        // Start by computing the normal vector
        fvm::gradient(normal,dof_field);
        field_ops::normalize(normal);
        //amrex::Real maxVel=field_ops::max_field_mag(vel);
        amrex::Real globalVelMag;
        field_ops::global_max_magnitude(vel,globalVelMag);
        for (int lev = 0; lev < repo.num_active_levels(); ++lev) {
            amrex::MFItInfo mfi_info;
        if (amrex::Gpu::notInLaunchRegion())
            mfi_info.EnableTiling(amrex::IntVect(1024, 1024, 1024))
                .SetDynamic(true);
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif

            for (amrex::MFIter mfi(dof_field(lev),mfi_info); mfi.isValid(); ++mfi)
            {
                const auto& bx = mfi.tilebox();
                // Define the compression velocity first at the faces
                // uc_mac=min(Ca*|u_mac|,u_mac|) etc

                auto normal_arr = normal(lev).array(mfi);
                auto vof_arr = dof_field(lev).array(mfi);
                auto vel_arr = vel(lev).array(mfi);
                auto velc_arr = velc(lev).array(mfi);
                amrex::ParallelFor(
                    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        amrex::Real magV=std::sqrt(vel_arr(i,j,k,0)*vel_arr(i,j,k,0)
                                                +  vel_arr(i,j,k,1)*vel_arr(i,j,k,1)
                                                +  vel_arr(i,j,k,2)*vel_arr(i,j,k,2));
                        amrex::Real C=1.5;
                        amrex::Real VC = std::min(C*magV,globalVelMag);
                        velc_arr(i,j,k)=normal_arr(i,j,k)*VC*vof_arr(i,j,k)*(1.0-vof_arr(i,j,k));
                        });
            }
        }
        fvm::divergence(src_term,velc);
    }

    PDEFields& fields;
    Field& normal;
    Field& vel;
    Field& velc;

};

template <typename Scheme>
struct ComputeRHSOp<VOF, Scheme>
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
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(field(lev)); mfi.isValid(); ++mfi) {
                const auto& bx = mfi.tilebox();
                auto vof = field(lev).array(mfi);
                const auto vof_o = field_old(lev).const_array(mfi);
                const auto ddt_o = conv_term(lev).const_array(mfi);
                    
                amrex::ParallelFor(
                    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                vof(i, j, k) = vof_o(i, j, k) + dt * ddt_o(i, j, k);
                // Defer n+1/2 update to the time-stepping algorithm
                // vof_nph(i, j, k) = 0.5 * (vof(i, j, k) + vof_o(i, j,k));
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



} // namespace pde
} // namespace amr_wind

#endif /* VOF_OPS_H */
