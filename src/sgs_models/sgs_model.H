#ifndef SGS_MODEL_H_
#define SGS_MODEL_H_

#include <AMReX_AmrCore.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLTensorOp.H>
#include <AMReX_ParmParse.H>

/* G.Deskos
SGS model class 
*/
class incflo;

class sgs_model
{
public:
  sgs_model (incflo* a_incflo);
	~sgs_model();

  void compute_div_sgs (amrex::Vector<amrex::MultiFab*> const& divtau,
                       amrex::Vector<amrex::MultiFab const*> const& velocity,
                       amrex::Vector<amrex::MultiFab const*> const& density,
                       amrex::Vector<amrex::MultiFab const*> const& eta,
                       amrex::Real t);

  void compute_eddy_viscosity(amrex::Vector<amrex::MultiFab*> const& velocity,
                         		  amrex::Vector<amrex::MultiFab*> const& density,
                              amrex::Vector<amrex::MultiFab const*> const& eta,
                              amrex::Real t, amrex::Real dt);
			
private: 
	// Function to compute planar averages
	int verbose = 0;

  void readParameters ();

  incflo* m_incflo;

#ifdef AMREX_USE_EB
  std::unique_ptr<amrex::MLEBTensorOp> m_eb_solve_op;
  std::unique_ptr<amrex::MLEBTensorOp> m_eb_apply_op;
#endif
  std::unique_ptr<amrex::MLTensorOp> m_reg_solve_op;
  std::unique_ptr<amrex::MLTensorOp> m_reg_apply_op;

  // sgs_models verbosity
  int m_verbose = 0;
  std::string m_model = 0;
	amrex::Real m_C=0.165;
	amrex::Real m_MTexponent=1.0;
};
#endif
