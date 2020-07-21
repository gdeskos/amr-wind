#include "aw_test_utils/MeshTest.H"
#include "amr-wind/core/Physics.H"
#include "amr-wind/multiphase/Multiphase.H"
#include "amr-wind/core/field_ops.H"
#include "amr-wind/utilities/IOManager.H"

namespace amr_wind_tests {

class  MultiphaseTest: public MeshTest
{
    public:
        void declare_default_fields()
        {
            auto& field_repo = mesh().field_repo();
            auto& rho = field_repo.declare_field("density",1,1,1);
            auto& vel = field_repo.declare_field("velocity",3,1,2);
            auto& umac = field_repo.declare_field("u_mac", 1, 0, 1, amr_wind::FieldLoc::XFACE);
            auto& vmac = field_repo.declare_field("v_mac", 1, 0, 1, amr_wind::FieldLoc::YFACE);
            auto& wmac = field_repo.declare_field("w_mac", 1, 0, 1, amr_wind::FieldLoc::ZFACE);
        }

    protected:    
        void populate_parameters() override
        {
            MeshTest::populate_parameters();

            {
                amrex::ParmParse pp("time");
                pp.add("max_step",5);
                pp.add("fixed_dt",0.001);
            }
            {
                amrex::ParmParse pp("amr");
                amrex::Vector<int> ncell{{32, 32, 32}};
                pp.add("max_level",0);
                pp.add("max_grid_size",32);
                pp.add("blocking_factor",32);
                pp.addarr("n_cell", ncell);
            }

            {
                amrex::ParmParse pp("geometry");
                amrex::Vector<amrex::Real> problo{{0.0, 0.0, 0.0}};
                amrex::Vector<amrex::Real> probhi{{1.0, 1.0, 1.0}};
                pp.addarr("prob_lo",problo);
                pp.addarr("prob_hi",probhi);
            }

            {
                amrex::ParmParse pp("inflo");
                pp.add("multiphase_problem",1); 
            }
        }
};

TEST_F(MultiphaseTest, multiphase_interface)
{
    initialize_mesh();
    
    declare_default_fields();

    auto& phy_mgr = sim().physics_manager();
    
    EXPECT_FALSE(phy_mgr.contains(amr_wind::Multiphase::identifier()));
    phy_mgr.create("Multiphase",sim());
    EXPECT_TRUE(phy_mgr.contains(amr_wind::Multiphase::identifier()));
}

TEST_F(MultiphaseTest, Test2)
{
    populate_parameters();

    initialize_mesh();

    // Declare fields
    declare_default_fields();
    // initialize the multiphase_interface
    auto& phy_mgr = sim().physics_manager();
    phy_mgr.create("Multiphase",sim());

    sim().init_physics();

    // Initialize the input/output manager
    auto& iomgr = sim().io_manager();
    iomgr.register_output_var("vof");
    iomgr.register_output_var("normal");

    auto& time = sim().time();

    while(time.new_timestep()) {

        iomgr.write_plot_file();
    
    }
}

}
