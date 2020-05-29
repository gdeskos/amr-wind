#ifndef TWOPHASETRANSPORT_H
#define TWOPHASETRANSPORT_H

#include "TransportModel.H"
#include "AMReX_ParmParse.H"
#include "Field.H"

namespace amr_wind {
namespace transport {

class TwoPhaseTransport : public TransportModel
{
public:
    static constexpr bool constant_properties = false;

    static const std::string identifier() { return "TwoPhaseTransport"; }

    TwoPhaseTransport(FieldRepo& repo)
        : m_repo(repo)
        , m_levelset(repo.get_field("levelset"))
        , m_vof(repo.get_field("vof"))
    {
        amrex::ParmParse pp("transport");
        pp.query("viscosity_water", m_mu1);
        pp.query("viscosity_air", m_mu2);
        pp.query("laminar_prandtl",m_Pr);
        pp.query("turbulent_prandtl",m_Prt);
    }
   
    virtual ~TwoPhaseTransport()=default;
    
    inline amrex::Real viscosity() const { return 0.5*(m_mu1+m_mu2); }

    inline amrex::Real thermal_diffusivity() const
    { return 0.5*(m_mu1+m_mu2) / m_Pr; }
     
    inline amrex::Real laminar_prandtl() const { return m_Pr; }

    inline amrex::Real turbulent_prandtl() const { return m_Prt; }

    inline amrex::Real laminar_schmidt(const std::string& scalar_name) const
    {
        amrex::ParmParse pp("transport");
        const std::string key = scalar_name + "_laminar_schmidt";
        amrex::Real lam_schmidt = 1.0;
        pp.query(key.c_str(), lam_schmidt);
        return lam_schmidt;
    }
    
    inline amrex::Real turbulent_schmidt(const std::string& scalar_name) const
    {
        amrex::ParmParse pp("transport");
        const std::string key = scalar_name + "_turbulent_schmidt";
        amrex::Real turb_schmidt = 1.0;
        pp.query(key.c_str(), turb_schmidt);
        return turb_schmidt;
    }

    //! Return the two-phase dynamic viscosity field
    inline std::unique_ptr<ScratchField> mu()
    {
        const int nlevels = m_repo.num_active_levels();

        auto mu = m_repo.create_scratch_field(1, 1);
        
        for (int lev; lev< nlevels; ++lev){
            auto& vof = m_vof(lev);
            auto& visc = (*mu)(lev);
            visc.setVal(0.);

        for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
            const auto& vbx = mfi.tilebox();

            auto F = vof.array(mfi);
            auto viscosity = visc.array(mfi);

            amrex::ParallelFor(
            vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                viscosity(i,j,k)=m_mu1*F(i,j,k)+m_mu2*(1.-F(i,j,k));
            });
        }
        
        }
        return mu;
    }

    //! Return the thermal diffusivity field
    inline std::unique_ptr<ScratchField> alpha()
    {
        auto alpha = mu();
        amrex::Real inv_Pr = 1.0 / m_Pr;
        for (int lev = 0; lev < m_repo.num_active_levels(); ++lev) {
            (*alpha)(lev).mult(inv_Pr);
        }
        return alpha;
    }

    inline std::unique_ptr<ScratchField> scalar_diffusivity(const std::string& scalar_name)
    {
        amrex::Real lam_schmidt = laminar_schmidt(scalar_name);

        amrex::Real inv_schmidt = 1.0 / lam_schmidt;
        auto diff = mu();
        for (int lev = 0; lev < m_repo.num_active_levels(); ++lev) {
            (*diff)(lev).mult(inv_schmidt);
        }

        return diff;
    }

private:
    //! Reference to the field repository (for creating scratch fields)
    FieldRepo& m_repo;
    
    Field& m_levelset;
    Field& m_vof;

    amrex::Real m_mu1{1e-3};
    amrex::Real m_mu2{1.48e-5};

    //! Prandtl number
    amrex::Real m_Pr{1.0};

    //! Turbulent Prandtl number
    amrex::Real m_Prt{1.0};
};

}
}

#endif /* TWOPHASETRANSPORT_H */

