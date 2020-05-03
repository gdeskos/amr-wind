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
    {
        amrex::ParmParse pp("transport");
        pp.query("visc_water", m_mu1);
        pp.query("visc_air", m_mu2);
        std::cout << "hi there" <<std::endl;        
    }
   
    ~TwoPhaseTransport()=default;
    
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

    std::unique_ptr<ScratchField> mu()
    {
        const int nlevels = m_repo.num_active_levels();

        auto mu = m_repo.create_scratch_field(1, 1);
        
        for (int lev; lev< nlevels; ++lev){
            auto& level_set = m_levelset(lev);
            auto& visc = (*mu)(lev);
            const auto& geom = m_repo.mesh().Geom(lev);

            const amrex::Real dx = geom.CellSize()[0];
            const amrex::Real dy = geom.CellSize()[1];
            const amrex::Real dz = geom.CellSize()[2];
            const amrex::Real ds = std::cbrt(dx * dy * dz);
            const amrex::Real epsilon = 2. * ds;

        for (amrex::MFIter mfi(level_set); mfi.isValid(); ++mfi) {
            const auto& vbx = mfi.validbox();

            const auto& dx = geom.CellSizeArray();
            const auto& problo = geom.ProbLoArray();
            auto phi = level_set.array(mfi);
            auto viscosity = visc.array(mfi);

            amrex::ParallelFor(
            vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                if (phi(i, j, k) > epsilon) {
                    const amrex::Real H = 1.0;
                    viscosity(i, j, k) = m_mu1 * H + m_mu2 * (1. - H);
                } else if (phi(i, j, k) < -epsilon) {
                    const amrex::Real H = 0.;
                    viscosity(i, j, k) = m_mu1 * H + m_mu2 * (1. - H);
                } else {
                    const amrex::Real H =
                        0.5 *
                        (1 + phi(i, j, k) / (2 * epsilon) +
                         1. / M_PI * std::sin(phi(i, j, k) * M_PI / epsilon));
                    viscosity(i, j, k) = m_mu1 * H + m_mu2 * (1. - H);
                    std::cout<<viscosity(i,j,k)<<std::endl;
                }
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

