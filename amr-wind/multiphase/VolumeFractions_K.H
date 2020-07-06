#ifndef VolumeFraction_K_H_
#define VolumeFraction_K_H_

#include <AMReX_FArrayBox.H>
#include <cmath>

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mixed_Youngs_centered (int i,  
                            int j, 
                            int k, 
                            amrex::Real idx, 
                            amrex::Real idy, 
                            amrex::Real idz, 
                            amrex::Array4<amrex::Real const> const& c,
                            amrex::Array4<amrex::Real> const& mxyz) noexcept

{
    int i0=i-1; int j0=j-1; int k0=k-1; int cn;
    amrex::Real m1,m2,t0,t1,t2;
    amrex::Vector<amrex::Vector<amrex::Real>> m{{0.0, 0.0, 0.0},
                                                {0.0, 0.0, 0.0},
                                                {0.0, 0.0, 0.0},
                                                {0.0, 0.0, 0.0}};

    /** Write the plane as: sgn(mx) X = my Y + mz Z + alpha  
     *                          m00 X = m01 Y + m02 Z + alpha 
     */
    m1 = c(i0,j0+1,k0)   + c(i0,j0+1,k0+2) + c(i0,j0,k0+1) + c(i0,j0+2,k0+1) + c(i0,j0+1,k0+1);
    m2 = c(i0+2,j0+1,k0) + c(i0+2,j0+1,k0+2) + c(i0+2,j0,k0+1)+ c(i0+2,j0+2,k0+1) + c(i0+2,j0+1,k0+1);
    
    if(m1>m2){
        m[0][0]=1.;
    }else{
        m[0][0]=-1.;
    } 
    
    m1 = c(i0,j0,k0+1)    + c(i0+2,j0,k0+1) + c(i0+1,j0,k0+1);
    m2 = c(i0,j0+2,k0+1)  + c(i0+2,j0+2,k0+1) + c(i0+1,j0+2,k0+1);
    
    m[0][1] = 0.5*(m1-m2);

    m1 = c(i0,j0+1,k0)    + c(i0+2,j0+1,k0+0) + c(i0+1,j0+1,k0+0);
    m2 = c(i0,j0+1,k0+2)  + c(i0+2,j0+1,k0+2) + c(i0+1,j0+1,k0+2);
    
    m[0][2] = 0.5*(m1-m2);

    /** Write the plane as: sgn(mx) Y = mx Y + mz Z + alpha  
     *                          m11 Y = m10 Y + m12 Z + alpha 
     */

    m1 = c(i0,j0,k0+1)   + c(i0,j0+2,k0+1)   + c(i0,j0+1,k0+1);
    m2 = c(i0+2,j0,k0+1) + c(i0+2,j0+2,k0+1) + c(i0+2,j0+1,k0+1);

    m[1][0] = 0.5*(m1-m2);
    
    m1 = c(i0+1,j0,k0+0) + c(i0+1,j0,k0+2)   + c(i0+2,j0,k0+1)   + c(i0,j0,k0+1) + c(i0+1,j0,k0+1);
    m2 = c(i0+1,j0+2,k0) + c(i0+1,j0+2,k0+2) + c(i0+2,j0+2,k0+1) + c(i0,j0+2,k0+1) + c(i0+1,j0+2,k0+1);

    if(m1>m2){
        m[1][1] = 1.;
    }else{
        m[1][1] = -1.;
    }

    m1 = c(i0+1,j0,k0)+ c(i0+1,j0+1,k0)+ c(i0+1,j0+2,k0);
    m2 = c(i0+1,j0,k0+2)+ c(i0+1,j0+1,k0+2)+ c(i0+1,j0+2,k0+2);
    m[1][2] = 0.5*(m1-m2);

    /**write the plane as: sgn(mz) Z =  mx X +  my Y + alpha 
     * m22 Z = m20 X + m21 Y + alpha
    */
    m1 = c(i0,j0+1,k0)+ c(i0,j0+1,k0+2)+ c(i0,j0+1,k0+1);
    m2 = c(i0+2,j0+1,k0)+ c(i0+2,j0+1,k0+2)+ c(i0+2,j0+1,k0+1);
    m[2][0] = 0.5*(m1-m2);

    m1 = c(i0+1,j0,k0)  + c(i0+1,j0,k0+2)+ c(i0+1,j0,k0+1);
    m2 = c(i0+1,j0+2,k0)+ c(i0+1,j0+2,k0+2)+ c(i0+1,j0+2,k0+1);
    m[2][1] = 0.5*(m1-m2);

    m1 = c(i0,j0+1,k0)   + c(i0+2,j0+1,k0)   + c(i0+1,j0,k0)   + c(i0+1,j0+2,k0)   + c(i0+1,j0+1,k0);
    m2 = c(i0,j0+1,k0+2) + c(i0+2,j0+1,k0+2) + c(i0+1,j0,k0+2) + c(i0+1,j0+2,k0+2) + c(i0+1,j0+1,k0+2);

    if(m1>m2){
        m[2][2] = 1.;
    }else{
        m[2][2] = -1.;
    }

    // normalize each set (mx,my,mz): |mx|+|my|+|mz| = 1
    t0 = std::abs(m[0][0]) + std::abs(m[0][1]) + std::abs(m[0][2]);
    m[0][0] = m[0][0]/t0;
    m[0][1] = m[0][1]/t0;
    m[0][2] = m[0][2]/t0;

    t0 = std::abs(m[1][0]) + std::abs(m[1][1]) + std::abs(m[1][2]);
    m[1][0] = m[1][0]/t0;
    m[1][1] = m[1][1]/t0;
    m[1][2] = m[1][2]/t0;

    t0 = std::abs(m[2][0]) + std::abs(m[2][1]) + std::abs(m[2][2]);
    m[2][0] = m[2][0]/t0;
    m[2][1] = m[2][1]/t0;
    m[2][2] = m[2][2]/t0;

    // choose among the three central schemes 
    t0 = std::abs(m[0][0]);
    t1 = std::abs(m[1][1]);
    t2 = std::abs(m[2][2]);

    cn = 0;
    if (t1 > t0){
        t0 = t1;
        cn = 1;
    }

    if (t2 > t0) {
        cn=2;
    }

    //Use the Youngs-CIAM scheme to compute the gradients
    // Here we need to use the derive Stencils here 
    m1 =    c(i0,j0,k0)   + c(i0,j0+2,k0)   + c(i0+0,j0,k0+2) + c(i0,j0+2,k0+2) +
        2.*(c(i0,j0,k0+1) + c(i0,j0+2,k0+1) + c(i0,j0+1,k0)   + c(i0,j0+1,k0+2)) +
        4.*c(i0,j0+1,k0+1);

    m2 = c(i0+2,j0,k0)       + c(i0+2,j0+2,k0) + c(i0+2,j0+0,k0+2) + c(i0+2,j0+2,k0+2) +
         2.*(c(i0+2,j0+0,k0+1) + c(i0+2,j0+2,k0+1) + c(i0+2,j0+1,k0)   + c(i0+2,j0+1,k0+2)) +
         4.*c(i0+2,j0+1,k0+1);
    m[3][0] = m1-m2;

    m1 = c(i0,j0,k0) + c(i0,j0,k0+2) + c(i0+2,j0+0,k0) + c(i0+2,j0,k0+2) +
         2.*(c(i0,j0,k0+1) + c(i0+2,j0,k0+1) + c(i0+1,j0,k0) + c(i0+1,j0,k0+2)) +4.*c(i0+1,j0,k0+1);
    m2 =  c(i0,j0+2,k0) + c(i0,j0+2,k0+2) + c(i0+2,j0+2,k0) + c(i0+2,j0+2,k0+2) +2.*(c(i0,j0+2,k0+1) + 
          c(i0+2,j0+2,k0+1) + c(i0+1,j0+2,k0+0) + c(i0+1,j0+2,k0+2)) +4.*c(i0+1,j0+2,k0+1);
    m[3][1] = m1-m2;

    m1 = c(i0,j0,k0) + c(i0,j0+2,k0) + c(i0+2,j0,k0) +    c(i0+2,j0+2,k0) +2.*(c(i0,j0+1,k0) + 
         c(i0+2,j0+1,k0) + c(i0+1,j0,k0) + c(i0+1,j0+2,k0)) +4.*c(i0+1,j0+1,k0);

    m2 = c(i0+0,j0,k0+2) + c(i0+0,j0+2,k0+2) + c(i0+2,j0+0,k0+2)  +   c(i0+2,j0+2,k0+2) +2.*(c(i0,j0+1,k0+2) + 
         c(i0+2,j0+1,k0+2) + c(i0+1,j0+0,k0+2) + c(i0+1,j0+2,k0+2)) +4.*c(i0+1,j0+1,k0+2);

    m[3][2]  = m1-m2;

    // normalize the set (mx,my,mz): |mx|+|my|+|mz| = 1 
    amrex::Real NOT_ZERO =1e-16;
    t0 = std::abs(m[3][0]) + std::abs(m[3][1]) + std::abs(m[3][2]) + NOT_ZERO;
    m[3][0] = m[3][0]/t0;
    m[3][1] = m[3][1]/t0;
    m[3][2] = m[3][2]/t0;
    // choose between the previous choice and Youngs-CIAM 
    t0 = std::abs(m[3][0]);
    t1 = std::abs(m[3][1]);
    t2 = std::abs(m[3][2]);
    if (t1 > t0) {t0 = t1;}
    if (t2 > t0) {t0 = t2;}

    if (std::abs(m[cn][cn]) > t0) {cn = 3;}

    // components of the normal vector */
    mxyz(i,j,k,0) = m[cn][0];
    mxyz(i,j,k,1)=  m[cn][1];
    mxyz(i,j,k,2)=  m[cn][2];
}

/* Computes alpha from a volume fraction F and the surface outward normal vector m=(m1,m2,m3)^T
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real volume_fraction_intercept  (int i,  
                                        int j, 
                                        int k, 
                                        amrex::Real idx, 
                                        amrex::Real idy, 
                                        amrex::Real idz, 
                                        amrex::Array4<amrex::Real const> const& c,
                                        amrex::Array4<amrex::Real> const& mxyz) noexcept
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
    
    return Alpha;
}

/** Computes the volume fraction from the normal m, the intercept alpha
 *  and the hexahedron dimensions and starting point (x0,y0,z0)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_volume_fraction    (int i,  
                                        int j, 
                                        int k, 
                                        amrex::Real x0, 
                                        amrex::Real y0, 
                                        amrex::Real z0, 
                                        amrex::Real idx, 
                                        amrex::Real idy, 
                                        amrex::Real idz, 
                                        amrex::Array4<amrex::Real const> const& mxyz,
                                        amrex::Array4<amrex::Real const> const& alpha,
                                        amrex::Array4<amrex::Real> const& c) noexcept
{
    
}



#endif // VolumeFraction_K.H
