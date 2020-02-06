#ifndef COORDINATE_TRANSFORMATION_H_
#define COORDINATE_TRANSFORMATION_H_

#include <AMReX_AmrCore.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLTensorOp.H>

/* G.Deskos
// Coordinate transformation class for a curvilinear solver 
*/

class CoordTransform
{
public:

    void readParameters();
	
		
private: 

    // AmrCore data 
    amrex::AmrCore* amrcore;	
    amrex::Vector<amrex::Geometry> geom;
    amrex::Vector<amrex::BoxArray> grids;          
    amrex::Vector<amrex::DistributionMapping> dmap;

    int nghost = 0; 
    int probtype = 0;

    amrex::Vector<std::unique_ptr<amrex::MultiFab>> xi; //--> x
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> eta; // --> y
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> zeta; // --> z
    std::unique_ptr<amrex::MLTensorOp> Jacobian; // --> J

    // Options to control mesh movement
		int damping_coeff=3;
    
		// Coordinate transformation verbosity 
		int verbose = 0;
}
#endif
