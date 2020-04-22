//
//  VolumeFractionUtils.H
//  amr-wind
//


#ifndef LevelSetUtils_H
#define LevelSetUtils_H

#include <AMReX_AmrCore.H>
#include <AMReX_levelset.H>
#include "DirectionSelector.H"

class incflo;

class LevelSetUtils
{
    
public:
    
    LevelSetUtils(amrex::Vector<amrex::Geometry>& Geom,
                   amrex::Vector<amrex::MultiFab*> const& ls);

private:

    amrex::Vector<amrex::MultiFab*> const& normals
};

#endif /* LevelSetUtils_H */

