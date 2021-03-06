#include "amr-wind/derive/derive_K.H"
#include "amr-wind/core/FieldRepo.H"
namespace amr_wind {

template<typename FType>
void compute_curvature(FType& curvf, const Field& field)
{
    const auto& repo = field.repo();
    const auto& geom_vec = repo.mesh().Geom();
    const auto ncomp = field.num_comp();
    
    if (ncomp>1) {amrex::Abort("Curvature can be computed only on a scalar field");}

    const int nlevels = repo.num_active_levels();
    for (int lev=0; lev < nlevels; ++lev) {
        const auto& geom = geom_vec[lev];
        const auto& domain = geom.Domain();

        const amrex::Real dx = geom.CellSize()[0];
        const amrex::Real dy = geom.CellSize()[1];
        const amrex::Real dz = geom.CellSize()[2];

        const amrex::Real idx = 1.0 / dx;
        const amrex::Real idy = 1.0 / dy;
        const amrex::Real idz = 1.0 / dz;

        for (amrex::MFIter mfi(field(lev)); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& curv_arr = curvf(lev).array(mfi);
            const auto& field_arr = field(lev).const_array(mfi);

            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    curv_arr(i,j,k)=curvature<StencilInterior>(
                      i, j, k, idx, idy, idz, field_arr);
                });

            // TODO: Check if the following is correct for `foextrap` BC types
            const auto& bxi = mfi.tilebox();
            int idim = 0;
            if (!geom.isPeriodic(idim)) {
                if (bxi.smallEnd(idim) == domain.smallEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = low[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);

                    auto bxlo = amrex::Box(low, hi);

                    amrex::ParallelFor(
                        bxlo, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            curvature<StencilILO>(
                              i, j, k, idx, idy, idz, field_arr);
                        });
                }

                if (bxi.bigEnd(idim) == domain.bigEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = hi[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);

                    auto bxhi = amrex::Box(low, hi);
                    
                    amrex::ParallelFor(
                        bxhi, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            curvature<StencilIHI>(
                              i, j, k, idx, idy, idz, field_arr);
                        });
                }
            } // if (!geom.isPeriodic)
             
            idim = 1;
            if (!geom.isPeriodic(idim)) {
                if (bxi.smallEnd(idim) == domain.smallEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = low[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);

                    auto bxlo = amrex::Box(low, hi);

                    amrex::ParallelFor(
                        bxlo, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            curvature<StencilJLO>(
                                i, j, k, idx, idy, idz, field_arr);
                        });
                }

                if (bxi.bigEnd(idim) == domain.bigEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = hi[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);

                    auto bxhi = amrex::Box(low, hi);

                    amrex::ParallelFor(
                        bxhi, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            curvature<StencilJHI>(
                                i, j, k, idx, idy, idz, field_arr);
                        });
                }
            } // if (!geom.isPeriodic)

            idim = 2;
            if (!geom.isPeriodic(idim)) {
                if (bxi.smallEnd(idim) == domain.smallEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = low[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);

                    auto bxlo = amrex::Box(low, hi);

                    amrex::ParallelFor(
                        bxlo, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            curvature<StencilKLO>(
                                i, j, k, idx, idy, idz, field_arr);
                        });
                }

                if (bxi.bigEnd(idim) == domain.bigEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = hi[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);

                    auto bxhi = amrex::Box(low, hi);

                    amrex::ParallelFor(
                        bxhi, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            curvature<StencilKHI>(
                                i, j, k, idx, idy, idz, field_arr);
                        });
                }
            } // if (!geom.isPeriodic)
            
        }
    }
}


template void compute_curvature<Field>(Field&, const Field&);
template void compute_curvature<ScratchField>(ScratchField&, const Field&);

}
