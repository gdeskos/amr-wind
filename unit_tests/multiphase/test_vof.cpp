#include "aw_test_utils/MeshTest.H"
#include "amr-wind/equation_systems/vof/volume_fractions.H"
#include "aw_test_utils/iter_tools.H"
#include "aw_test_utils/test_utils.H"

namespace amr_wind_tests {

class VOFOpTest : public MeshTest
{};

namespace {

void initialize_volume_fractions(
    amrex::Geometry geom,
    const amrex::Box& bx,
    const int pdegree,
    amrex::Gpu::DeviceVector<amrex::Real>& c,
    amrex::Array4<amrex::Real>& volume_fractions)
{
    auto problo = geom.ProbLoArray();
    auto probhi = geom.ProbHiArray();
    auto dx = geom.CellSizeArray();

    const amrex::Real* c_ptr = c.data();

    // grow the box by 1 so that x,y,z go out of bounds and min(max()) corrects
    // it and it fills the ghosts with wall values
    amrex::ParallelFor(grow(bx, 1), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        const amrex::Real x = amrex::min(
            amrex::max(problo[0] + (i + 0.5) * dx[0], problo[0]), probhi[0]);
        const amrex::Real y = amrex::min(
            amrex::max(problo[1] + (j + 0.5) * dx[1], problo[1]), probhi[1]);
        const amrex::Real z = amrex::min(
            amrex::max(problo[2] + (k + 0.5) * dx[2], problo[2]), probhi[2]);
    });
}

amrex::Real
interface_normal_test_impl(amr_wind::Field& volfrac, const int pdegree)
{

    const int ncoeff = (pdegree + 1) * (pdegree + 1) * (pdegree + 1);

    amrex::Gpu::DeviceVector<amrex::Real> cu(ncoeff, 0.00123);
    amrex::Gpu::DeviceVector<amrex::Real> cv(ncoeff, 0.00213);
    amrex::Gpu::DeviceVector<amrex::Real> cw(ncoeff, 0.00346);

    auto& geom = volfrac.repo().mesh().Geom();

    return 0.0;
}

amrex::Real
interface_curvature_test_impl(amr_wind::Field& volfrac, const int pdegree)
{

    const int ncoeff = (pdegree + 1) * (pdegree + 1) * (pdegree + 1);

    amrex::Gpu::DeviceVector<amrex::Real> coeff(ncoeff, 0.00123);

    auto& geom = volfrac.repo().mesh().Geom();

    return 0.0;
}

} // namespace

TEST_F(VOFOpTest, normal)
{

    constexpr double tol = 1.0e-10;

    populate_parameters();
    {
        amrex::ParmParse pp("geometry");
        amrex::Vector<int> periodic{{0, 0, 0}};
        pp.addarr("is_periodic", periodic);
    }

    initialize_mesh();

    auto& repo = sim().repo();
    const int ncomp = 3;
    const int nghost = 1;
    auto& volfrac = repo.declare_field("volume_fractions", ncomp, nghost);
    const int pdegree = 2;
    auto error_total = interface_normal_test_impl(volfrac, pdegree);

    amrex::ParallelDescriptor::ReduceRealSum(error_total);

    EXPECT_NEAR(error_total, 0.0, tol);
}

} // namespace amr_wind_tests
