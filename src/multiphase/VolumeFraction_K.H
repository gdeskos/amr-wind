#ifndef VolumeFraction_K_H_
#define VolumeFraction_K_H_

#include <AMReX_FArrayBox.H>
#include <cmath>

/* Computes alpha from a volume fraction F and the surface outward normal vector m=(m1,m2,m3)^T
 * 
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real volume_fraction_reconstruct (int i, int j, int k,
                                        amrex::Real x0, amrex::Real y0, amrex::Real z0,
                                        amrex::Real dx, amrex::Real dy, amrex::Real dz,
                                        amrex::Real mx, amrex::Real my, amrex::Real mz,
                                        amrex::Real vof) noexcept
{
    using namespace amrex;
  
    Real cch,c01,c02,c03,np1,np2,np3;
    Real m1,m2,m3,m12,numer,denom,p,pst,q,arc,csarc;

    // Define everything in a unit box (1x1x1)
    np1=std::abs(mx);np2=std::abs(my);np3=std::abs(mz); 
    m1=std::min(np1,np2);m3=std::max(np1,np2);

    if(np3<np1){
        m2=m1; m1=np3;
    }
    else if(np3>=m3){
        m2=m3;m3=np3;
    }
    else{
        m2=np3;
    }

    Real eps0=std::numeric_limits<Real>::epsilon();
    denom = std::max(6.*m1*m2*m3,eps0);                          
    cch = std::min(vof,1-vof); // limit to: 0 < cch < 1/2
    c01 = m1*m1*m1/denom;      // get cch ranges
    c02  = c01 + 0.5*(m2-m1)/m3;
    m12 = m1 + m2;
    if (m12 <= m3){
        c03 = 0.5*m12/m3;
    }else{
        numer = m3*m3*(3.*m12-m3) + m1*m1*(m1-3.*m3) + m2*m2*(m2-3.*m3);
        c03 = numer/denom;
    }

    Real AlphaUnit,Alpha;
    if (cch <= c01){
     AlphaUnit = std::pow(6.*m1*m2*m3,1./3.);  // case 1
    }
 
    //move coordinate system to the origin x0,y0,z0
    Alpha=AlphaUnit-mx*x0-my*y0-mz*z0;
    //
    Alpha=Alpha;
    
    return Alpha;
}

#endif // VolumeFraction_K.H
