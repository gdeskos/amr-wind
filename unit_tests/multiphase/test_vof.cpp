#include "aw_test_utils/MeshTest.H"
#include "amr-wind/core/Physics.H"
#include "amr-wind/multiphase/Multiphase.H"
#include "amr-wind/core/field_ops.H"
#include "amr-wind/utilities/IOManager.H"

namespace amr_wind_tests {

class MultiphaseTest : public MeshTest
{
public:
    void declare_default_fields()
    {
        auto& field_repo = mesh().field_repo();
        auto& rho = field_repo.declare_field("density", 1, 1, 1);
        auto& vel = field_repo.declare_field("velocity", 3, 1, 2);
        auto& umac = field_repo.declare_field(
            "u_mac", 1, 0, 1, amr_wind::FieldLoc::XFACE);
        auto& vmac = field_repo.declare_field(
            "v_mac", 1, 0, 1, amr_wind::FieldLoc::YFACE);
        auto& wmac = field_repo.declare_field(
            "w_mac", 1, 0, 1, amr_wind::FieldLoc::ZFACE);
    }

protected:
    void populate_parameters() override
    {
        MeshTest::populate_parameters();

        {
            amrex::ParmParse pp("time");
            pp.add("max_step", 5);
            pp.add("fixed_dt", 0.001);
        }
        {
            amrex::ParmParse pp("amr");
            amrex::Vector<int> ncell{{32, 32, 32}};
            pp.add("max_level", 0);
            pp.add("max_grid_size", 32);
            pp.add("blocking_factor", 32);
            pp.addarr("n_cell", ncell);
        }

        {
            amrex::ParmParse pp("geometry");
            amrex::Vector<amrex::Real> problo{{0.0, 0.0, 0.0}};
            amrex::Vector<amrex::Real> probhi{{1.0, 1.0, 1.0}};
            pp.addarr("prob_lo", problo);
            pp.addarr("prob_hi", probhi);
        }

        {
            amrex::ParmParse pp("inflo");
            pp.add("multiphase_problem", 1);
        }
    }

};

TEST_F(MultiphaseTest, multiphase_interface)
{
    initialize_mesh();

    declare_default_fields();

    auto& phy_mgr = sim().physics_manager();

    EXPECT_FALSE(phy_mgr.contains(amr_wind::Multiphase::identifier()));
    phy_mgr.create("Multiphase", sim());
    EXPECT_TRUE(phy_mgr.contains(amr_wind::Multiphase::identifier()));
}

TEST_F(MultiphaseTest, Test2)
{
    populate_parameters();
    create_mesh_instance();
    declare_default_fields();
    initialize_mesh();

    // initialize the multiphase_interface
    auto& phy_mgr = sim().physics_manager();
    phy_mgr.create("Multiphase", sim());

    sim().init_physics();

    // Initialize the input/output manager
    auto& iomgr = sim().io_manager();
    iomgr.register_output_var("levelset");
    iomgr.register_output_var("vof");
    iomgr.register_output_var("normal");
    iomgr.register_output_var("u_mac");
    iomgr.register_output_var("v_mac");
    iomgr.register_output_var("w_mac");

    sim().io_manager().initialize_io();

    auto& time = sim().time();

    auto& field_repo = mesh().field_repo();
    const int nlevels = field_repo.num_active_levels();
    // Initialize fields
    for (int lev = 0; lev < nlevels; ++lev) {
        for (auto& pp : sim().physics()) {
            pp->initialize_fields(lev, mesh().Geom(lev));
            pp->post_init_actions();
        }
    }

    // Declare mac velocities
    auto& umac = field_repo.declare_field(
        "u_mac", 1, 0, 1, amr_wind::FieldLoc::XFACE);
    auto& vmac = field_repo.declare_field(
        "v_mac", 1, 0, 1, amr_wind::FieldLoc::YFACE);
    auto& wmac = field_repo.declare_field(
        "w_mac", 1, 0, 1, amr_wind::FieldLoc::ZFACE);
    
    while (time.new_timestep()) {

        // Assign umac, vmac and wmac velocities
        for (int lev = 0; lev < nlevels; ++lev) {
            for (amrex::MFIter mfi(umac(lev)); mfi.isValid(); ++mfi) {
                const auto& vbx = mfi.validbox();
                const auto& dx = mesh().Geom(lev).CellSizeArray();  
                const auto& problo = mesh().Geom(lev).ProbLoArray();
               
                auto ux = umac(lev).array(mfi);
                auto uy = vmac(lev).array(mfi);
                auto uz = wmac(lev).array(mfi);
                amrex::ParallelFor(
                    vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    const amrex::Real xface = problo[0] + i*dx[0];
                    const amrex::Real yface = problo[1] + j*dx[1];
                    const amrex::Real zface = problo[2] + k*dx[2];
                    const amrex::Real xc = problo[0] + (i+0.5)*dx[0];
                    const amrex::Real yc = problo[1] + (j+0.5)*dx[1];
                    const amrex::Real zc = problo[2] + (k+0.5)*dx[2];
                    ux(i,j,k) = 2.0*std::sin(M_PI*xface)*std::sin(M_PI*xface)
                                   *std::sin(2.0*M_PI*yc)*std::sin(2.0*M_PI*zc);
                    uy(i,j,k) =    -std::sin(M_PI*yface)*std::sin(M_PI*yface)
                                   *std::sin(2.0*M_PI*xc)*std::sin(2.0*M_PI*zc);
                    uz(i,j,k) =    -std::sin(M_PI*zface)*std::sin(M_PI*zface)
                                   *std::sin(2.0*M_PI*xc)*std::sin(2.0*M_PI*yc);
                });
            }
        }       

        // output plots
        iomgr.write_plot_file();
    }
}

} // namespace amr_wind_tests
