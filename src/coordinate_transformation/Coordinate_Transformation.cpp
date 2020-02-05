#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#ifdef AMREX_USE_EB
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EB_utils.H>
#endif

#include <DiffusionOp.H>
#include <diffusion_F.H>

using namespace amrex;

// Define unit vectors to easily convert indices
extern const amrex::IntVect e_x;
extern const amrex::IntVect e_y;
extern const amrex::IntVect e_z;


//
// Constructor:
// We set up everything which doesn't change between timesteps here
//
DiffusionOp::DiffusionOp(AmrCore* _amrcore,
#ifdef AMREX_USE_EB
                         Vector<std::unique_ptr<EBFArrayBoxFactory>>* _ebfactory,
#endif
                         std::array<amrex::LinOpBCType,AMREX_SPACEDIM> a_velbc_lo,
                         std::array<amrex::LinOpBCType,AMREX_SPACEDIM> a_velbc_hi,
                         std::array<amrex::LinOpBCType,AMREX_SPACEDIM> a_scalbc_lo,
                         std::array<amrex::LinOpBCType,AMREX_SPACEDIM> a_scalbc_hi,
                         int _nghost,
                         int _probtype)
{
    if(verbose > 0)
        amrex::Print() << "Constructing DiffusionOp class" << std::endl;
    
    probtype = _probtype;
    nghost = _nghost;

    m_velbc_lo = a_velbc_lo;
    m_velbc_hi = a_velbc_hi;

    m_scalbc_lo = a_scalbc_lo;
    m_scalbc_hi = a_scalbc_hi;

    // Get inputs from ParmParse
    readParameters();

    // Actually do the setup work here
#ifdef AMREX_USE_EB
    setup(_amrcore, _ebfactory);
#else
    setup(_amrcore);
#endif
}

