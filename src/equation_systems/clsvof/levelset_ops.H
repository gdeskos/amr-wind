#ifndef LevelSet_OPS_H
#define LevelSet_OPS_H

#include "PDETraits.H"
#include "PDEOps.H"
#include "PDEHelpers.H"
#include "DiffusionOps.H"
#include "clsvof/levelset.H"

namespace amr_wind {
namespace pde {

template<typename Scheme>
struct FieldRegOp<LevelSet, Scheme>
{
    FieldRegOp(FieldRepo& repo_in): repo(repo_in) {}

    PDEFields  operator()(const SimTime& time, const int probtype)
    {
        auto fields = create_fields_instance<LevelSet, Scheme>(time, repo, probtype);
        
        auto& ls_normal = repo.declare_cc_field("ls_normal",3,1,1);
        auto& ls_curv   = repo.declare_nd_field("ls_curvature",1,0,1);

        ls_normal.template register_fill_patch_op<FieldFillPatchOps<FieldBCNoOp>>(repo.mesh(),time,probtype);

        return fields;
    }
    FieldRepo& repo;
};

template<typename Scheme>
struct ComputeRHSOp<LevelSet, Scheme>
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
                auto levelset = field(lev).array(mfi);
                const auto levelset_o = field_old(lev).const_array(mfi);
                const auto ddt_o = conv_term(lev).const_array(mfi);

                amrex::ParallelFor(
                    bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        levelset(i, j, k) = levelset_o(i, j, k) + dt * ddt_o(i, j, k);
                        // Defer n+1/2 update to the time-stepping algorithm
                        // levelset_nph(i, j, k) = 0.5 * (levelset(i, j, k) + levelset_o(i, j, k));
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
                auto levelset = field(lev).array(mfi);
                const auto levelset_o = field_old(lev).const_array(mfi);
                const auto ddt = conv_term(lev).const_array(mfi);
                const auto ddt_o = conv_term_old(lev).const_array(mfi);

                amrex::ParallelFor(
                    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        levelset(i, j, k) =
                            levelset_o(i, j, k) +
                            dt * 0.5 * (ddt_o(i, j, k) + ddt(i, j, k));
                        // Defer n+1/2 update to time-stepping algorithm
                        // levelset_nph(i, j, k) =
                        //     0.5 * (levelset(i, j, k) + levelset_o(i, j, k));
                    });
            }
        }
    }

    // data members
    PDEFields& fields;
};

template <typename Scheme>
struct DiffusionOp<LevelSet, Scheme> 
    : public DiffSolverIface<typename LevelSet::MLDiffOp>
{
    
    static_assert(LevelSet::ndim == 1, "Re-initialisation operation masked as DiffusionOp invoked for LevelSet");
    static_assert(std::is_same<typename LevelSet::MLDiffOp, amrex::MLABecLaplacian>::value,
            "Invalid linear operator for scalar diffusion operator");

    DiffusionOp(PDEFields& fields):DiffSolverIface<typename LevelSet::MLDiffOp>(fields){}
    
    void compute_diff_term(const FieldState fstate){}

    void linsys_solve(const amrex::Real dt)
    {
    const auto& repo = this->m_pdefields.repo;
    const int nlevels = repo.num_active_levels();
    const auto& ls = repo.get_field("levelset");
    const auto& normal = repo.get_field("ls_normal");

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (int lev = 0; lev < nlevels; ++lev) {

            // Re-initiliasation of the level-set function according to 
            // Russo & Smereka 2000 -- A remark on Computing Distance Functions
            for (amrex::MFIter mfi(ls(lev), amrex::TilingIfNotGPU());
                 mfi.isValid(); ++mfi) {
                const auto& bx = mfi.tilebox();
                auto phi_arr = ls(lev).array(mfi);
                auto normal_arr = normal(lev).array(mfi);

                amrex::Real Dijk;
                amrex::ParallelFor(
                    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
                    {
                    //Copy initial value
                    });
            }
        }
    }
};

}
}

#endif /* LevelSet_OPS_H */
