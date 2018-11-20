#include <AMReX_Array.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_BLassert.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>

#include <incflo.H>
#include <boundary_conditions_F.H>
#include <divergence_F.H>

//
// This subroutines averages component by component
// The assumption is that cc is multicomponent
// 
void
incflo::AverageCcToFc(int lev, 
                                const MultiFab& cc,
                                Array<std::unique_ptr<MultiFab>,AMREX_SPACEDIM>& fc )
{
    AMREX_ASSERT(cc.nComp()==AMREX_SPACEDIM);
    AMREX_ASSERT(AMREX_SPACEDIM==3);
    
    // First allocate fc
    BoxArray x_ba = cc.boxArray();
    x_ba.surroundingNodes(0);
    fc[0].reset(new MultiFab(x_ba,cc.DistributionMap(),1,nghost, MFInfo(), *ebfactory[lev]));
    fc[0]->setVal(1.0e200);

    BoxArray y_ba = cc.boxArray();
    y_ba.surroundingNodes(1);
    fc[1].reset(new MultiFab(y_ba,cc.DistributionMap(),1,nghost, MFInfo(), *ebfactory[lev]));
    fc[1]->setVal(1.0e200);

    BoxArray z_ba = cc.boxArray();
    z_ba.surroundingNodes(2);
    fc[2].reset(new MultiFab(z_ba,cc.DistributionMap(),1,nghost, MFInfo(), *ebfactory[lev]));
    fc[2]->setVal(1.0e200);

    //
    // Average
    // We do not care about EB because faces in covered regions
    // should never get used so we can set them to whatever values
    // we like
    //
#ifdef _OPENMP
#pragma omp parallel 
#endif
    for (MFIter mfi(*vel[lev],true); mfi.isValid(); ++mfi)
    {
        // Boxes for staggered components
        Box bx = mfi.tilebox();

        average_cc_to_fc( BL_TO_FORTRAN_BOX(bx),
                          BL_TO_FORTRAN_ANYD((*fc[0])[mfi]),
                          BL_TO_FORTRAN_ANYD((*fc[1])[mfi]),
                          BL_TO_FORTRAN_ANYD((*fc[2])[mfi]),
                          BL_TO_FORTRAN_ANYD(cc[mfi]));
    }

    // Set BCs for the fac-centered velocities
	fc[0]->FillBoundary(geom[lev].periodicity());
	fc[1]->FillBoundary(geom[lev].periodicity());
	fc[2]->FillBoundary(geom[lev].periodicity());
    // We do not fill BCs and halo regions in this routine
} 
