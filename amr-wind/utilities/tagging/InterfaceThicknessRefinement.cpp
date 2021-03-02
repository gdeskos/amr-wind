#include "amr-wind/utilities/tagging/InterfaceThicknessRefinement.H"
#include "amr-wind/CFDSim.H"

#include "AMReX.H"
#include "AMReX_ParmParse.H"

namespace amr_wind {

InterfaceThicknessRefinement::InterfaceThicknessRefinement(const CFDSim& sim)
    : m_sim(sim)
    , m_thickness_value(
          m_sim.mesh().maxLevel() + 1, std::numeric_limits<amrex::Real>::max())
{}

void InterfaceThicknessRefinement::initialize(const std::string&)
{
    std::string fname = "vof";

    const auto& repo = m_sim.repo();
    if (!repo.field_exists(fname)) {
        amrex::Abort("FieldRefinement: Cannot find field = " + fname);
    }
    m_vof = &(m_sim.repo().get_field(fname));
}

void InterfaceThicknessRefinement::operator()(
    int level, amrex::TagBoxArray& tags, amrex::Real time, int)
{
    const bool tag_field = level <= m_max_lev_field;

    if (!tag_field) return;

    m_vof->fillpatch(level, time, (*m_vof)(level), 1);

    const auto& mfab = (*m_vof)(level);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

    for (amrex::MFIter mfi(mfab, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        const auto& bx = mfi.tilebox();
        const auto& tag = tags.array(mfi);
        const auto& volfrac = mfab.const_array(mfi);

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // Implementing the Topology-oriented (TO) refinement criterion
                // of Chen & Yang 2014 (JCP)

                // It involves 26 neighbouring cells
                int N0 = 0;
                int N1 = 0;
                amrex::Real small_vof = 1e-6;
                if (volfrac(i, j, k) < 1. && volfrac(i, j, k) > 0.) {
                    /*
                    if (volfrac(i - 1, j, k) < small_vof &&
                        volfrac(i + 1, j, k) < small_vof &&
                        volfrac(i, j - 1, k) < small_vof &&
                        volfrac(i, j + 1, k) < small_vof &&
                        volfrac(i, j, k - 1) < small_vof &&
                        volfrac(i, j, k + 1) < small_vof) {
                        N0 = 1;
                    }
                    */
                    if (volfrac(i - 1, j - 1, k - 1) < small_vof &&
                        volfrac(i - 1, j - 1, k) < small_vof &&
                        volfrac(i - 1, j - 1, k + 1) < small_vof &&
                        volfrac(i - 1, j, k - 1) < small_vof &&
                        volfrac(i - 1, j, k) < small_vof &&
                        volfrac(i - 1, j, k + 1) < small_vof &&
                        volfrac(i - 1, j + 1, k - 1) < small_vof &&
                        volfrac(i - 1, j + 1, k) < small_vof &&
                        volfrac(i - 1, j + 1, k + 1) < small_vof &&
                        volfrac(i, j - 1, k - 1) < small_vof &&
                        volfrac(i, j - 1, k) < small_vof &&
                        volfrac(i, j - 1, k + 1) < small_vof &&
                        volfrac(i, j, k - 1) < small_vof &&
                        volfrac(i, j, k + 1) < small_vof &&
                        volfrac(i, j + 1, k - 1) < small_vof &&
                        volfrac(i, j + 1, k) < small_vof &&
                        volfrac(i, j + 1, k + 1) < small_vof &&
                        volfrac(i + 1, j - 1, k - 1) < small_vof &&
                        volfrac(i + 1, j - 1, k) < small_vof &&
                        volfrac(i + 1, j - 1, k + 1) < small_vof &&
                        volfrac(i + 1, j, k - 1) < small_vof &&
                        volfrac(i + 1, j, k) < small_vof &&
                        volfrac(i + 1, j, k + 1) < small_vof &&
                        volfrac(i + 1, j + 1, k - 1) < small_vof &&
                        volfrac(i + 1, j + 1, k) < small_vof &&
                        volfrac(i + 1, j + 1, k + 1) < small_vof) {

                        N0 = 1;
                    }

                    if (N0 > 0 || N1 > 0) tag(i, j, k) = amrex::TagBox::SET;
                }
            });
    }
}

} // namespace amr_wind
