#include <incflo.H>
#include "sgs_model.H"

#ifdef AMREX_USE_EB
#include <AMReX_EB_utils.H>
#endif

using namespace amrex;

void incflo::ReadSGSModelParameters() 
{
    amrex::ParmParse pp("incflo");
    std::string sgs_model_s = "none";
    pp.query("sgs_model",sgs_model_s);

    if(sgs_model_s=="none" || sgs_model_s== "None" || sgs_model_s=="NONE"){
        m_sgs_model=SGSModel::None;
        amrex::Print() << "SGS model = None"<<std::endl;
    }
    else if(sgs_model_s=="smagorinsky" || sgs_model_s== "Smagorinsky" || sgs_model_s=="SMAGORINSKY"){
        m_sgs_model=SGSModel::Smagorinsky;
        m_use_sgs=true;
        amrex::Print() << "SGS model = Smagorinsky"<<std::endl;
	    
        //pp.query("smag_constant", m_C);
	    //pp.query("MT_exp",m_MTexponent); //! Mason & Thomson exponent (m_C) is used as the free-stream value
    }
    else
    {
        amrex::Abort("Unknown SGS model to amr-wind. Sorry, we still use the standard Smagorinsky despite the fact we all know it is wrong!");
    }
}

sgs_model::sgs_model()
{
}

//!########################################
//! Destructor 
//!########################################
sgs_model::~sgs_model()
{
}


/*!
//! Compute eddy viscosity, \f$ \nu_t \f$
*/
void sgs_model::compute_eddy_viscosity(Vector<MultiFab const*> const& rho,
                                       Vector<MultiFab const*> const& vel,
                                       Vector<MultiFab const*> const& tra,
                                       Real time, int nghost)
{

/*    for (int lev = 0; lev <= finest_level; ++lev) {
#ifdef AMREX_USE_EB
    auto const& fact = EBFactory(lev);
    auto const& flags = fact.getMultiEBCellFlagFab();
#endif
    const Geometry& geom_lev=m_incflo->Geom(lev);
    const Real dx = geom_lev.CellSize()[0];
    const Real dy = geom_lev.CellSize()[1];
    const Real dz = geom_lev.CellSize()[2];
    const Real ds = pow(dx*dy*dz,1.0/3.0);
    
    Real idx = 1.0 / dx;
    Real idy = 1.0 / dy;
    Real idz = 1.0 / dz;
    std::cout << dx << dy << dz <<std::endl;
    }
*/
}
