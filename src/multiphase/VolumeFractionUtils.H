//
//  VolumeFractionUtils.H
//  amr-wind
//


#ifndef VolumeFractionUtils_H
#define VolumeFractionUtils_H

#include <AMReX_AmrCore.H>
#include "DirectionSelector.H"

class incflo;

class VolumeFractionUtils
{
    
public:
    
    VolumeFractionUtils(amrex::Vector<amrex::Geometry>& Geom,
                   amrex::Vector<amrex::MultiFab*> const& vof);

    void reconstruct();

private:

    void compute_plane_constant_alpha(); //computes the 
};

#endif /* VolumeFractionUtils_H */

