#include <incflo.H>
#include "sgs_model.H"
#include <AMReX_ParmParse.H>

#ifdef AMREX_USE_EB
#include <AMReX_EB_utils.H>
#endif

using namespace amrex;

//!########################################
//! Constructor
//!########################################
sgs_model::sgs_model (incflo* a_incflo)
    : m_incflo(a_incflo)
{
	readParameters();
	
	int finest_level = m_incflo->finestLevel();	
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
void
sgs_model::compute_eddy_viscosity (Vector<MultiFab*> const& velocity,
                                   Vector<MultiFab*> const& density,
                                   Vector<MultiFab const*> const& eddy_viscosity,
                                   Real t, Real dt)
{


}
/*!
//! Read parameters for the SGS model
*/
void sgs_model::readParameters ()
{
    ParmParse pp("sgs");

    pp.query("verbose", m_verbose);
    pp.query("model", m_model);
    if(m_model=="smagorinsky"){
			pp.query("smagorinsky_constant", m_C);
			pp.query("MT_exponent",m_MTexponent); //! Mason & Thomson exponent (m_C) is used as the free-stream value
		}else{
      amrex::Abort("Wrong model, at the moment only Smagorinsky is available");
		}
}

