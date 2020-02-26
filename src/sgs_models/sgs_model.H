#ifndef SGS_MODEL_H_
#define SGS_MODEL_H_

#include <incflo.H>
#include <derive_K.H>

/* G.Deskos
SGS model class 
*/
class sgs_model
{
public:
  sgs_model ();
  ~sgs_model();
 
  void compute_eddy_viscosity (amrex::Vector<amrex::MultiFab const*> const& rho,
                               amrex::Vector<amrex::MultiFab const*> const& vel,
                               amrex::Vector<amrex::MultiFab const*> const& tra,
                               amrex::Real time, int nghost);
			
private: 
	// Function to compute planar averages
  int verbose = 0;

  int finest_level; //define finest level

  // sgs_models verbosity
  int m_verbose = 0;
  amrex::Real m_C=0.135; // This is sooooo arbitrarily chosen 
  amrex::Real m_MTexponent=0.0; //Zero is used by default so no damping is applied
};
#endif
