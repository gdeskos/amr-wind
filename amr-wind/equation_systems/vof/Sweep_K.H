#ifndef Sweep_K_H_
#define Sweep_K_H_

#include "amr-wind/derive/derive_K.H"
#include "amr-wind/equation_systems/vof/VolumeFractions_K.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

/** Split advection of the interfae along the x,y,z directions 
 *
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void swp(
    int i,
    int j,
    int k,
    int d,
    amrex::Real dt, 
    amrex::Real dx,
    amrex::Real dy,
    amrex::Real dz,
    amrex::Array4<amrex::Real const> const& velocity,
    amrex::Array4<amrex::Real> const& c) noexcept
{
    amrex::Real a1,a2;
    if(d==0){
        a1=velocity(i,j,k)*dt/dx;
        a2=velocity(i+1,j,k)*dt/dx;
    }else if (d==1){
        a1=velocity(i,j,k)*dt/dy;
        a2=velocity(i,j+1,k)*dt/dy;
    }else if (d==2){
        a1=velocity(i,j,k)*dt/dz;
        a2=velocity(i,j,k+1)*dt/dz;
    }
    // 3 cases: 1: DEFAULT (c=0.0 and fluxes=0.0); 2: c=1.0; 3: c>0.0


}
#endif // Sweep_K.H
