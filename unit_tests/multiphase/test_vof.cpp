#include "aw_test_utils/MeshTest.H"
#include "amr-wind/core/field_ops.H"

namespace amr_wind_tests {

class test_vof : public MeshTest
{
    public:
        void declare_default_fields()
        {
            auto& frepo = mesh().field_repo();
            frepo.declare_field("vel",3,1,2);
        }

};

}
