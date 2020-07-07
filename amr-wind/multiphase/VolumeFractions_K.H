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
    m1 = c(i0,j0+1,k0) + c(i0,j0+1,k0+2) + c(i0,j0,k0+1) + c(i0,j0+2,k0+1) + c(i0,j0+1,k0+1);
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

    /** Write the plane as: sgn(mx) Y = mx X + mz Z + alpha  
     *                          m11 Y = m10 X + m12 Z + alpha 
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

    // Compute gradients
  
    m1 =    c(i0,j0,k0)   + c(i0,j0+2,k0)   + c(i0,j0,k0+2) + c(i0,j0+2,k0+2) +
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
                                        amrex::Real dx, 
                                        amrex::Real dy, 
                                        amrex::Real dz, 
                                        amrex::Array4<amrex::Real const> const& c,
                                        amrex::Array4<amrex::Real const> const& mxyz) noexcept
{
    using namespace amrex;

    Real cch,c01,c02,c03,np1,np2,np3;
    Real m1,m2,m3,m12,numer,denom;

    // Define everything in a unit box (1x1x1)
    np1=std::abs(mxyz(i,j,k,0)*dx);
    np2=std::abs(mxyz(i,j,k,1)*dy);
    np3=std::abs(mxyz(i,j,k,2)*dz); 
    m1=std::min(np1,np2);
    m3=std::max(np1,np2);

    if(np3<np1){
        m2=m1; 
        m1=np3;
    }
    else if(np3>=m3){
        m2=m3;
        m3=np3;
    }
    else{
        m2=np3;
    }

    Real eps0=1e-50;
    denom = std::max(6.*m1*m2*m3,eps0);                          
    cch = std::min(c(i,j,k),1.-c(i,j,k));  // limit to: 0 < cch < 1/2
    c01 = m1*m1*m1/denom;                 // get cch ranges
    c02  = c01 + 0.5*(m2-m1)/m3;
    m12 = m1 + m2;
    if (m12 <= m3){
        c03 = 0.5*m12/m3;
    }else{
        numer = m3*m3*(3.*m12-m3) + m1*m1*(m1-3.*m3) + m2*m2*(m2-3.*m3);
        c03 = numer/denom;
    }
    Real alpha,p,q,pst,arc,csarc;
    // 1: C<=C1; 2: C1<=C<=C2; 3: C2<=C<=C3; 4: C3<=C<=1/2 (a: m12<=m3; b: m3<m12)) 
    if (cch <= c01){
       alpha = std::pow(denom*cch,1./3.);  // case (1)
    }else if (cch <= c02){
       alpha = 0.5*(m1 + std::sqrt(m1*m1 + 8.*m2*m3*(cch-c01)));  // case (2)
    }else if (cch <= c03){
       p = 2.*m1*m2;
       q = 1.5*m1*m2*(m12 - 2.*m3*cch);
       pst = std::sqrt(p);
       arc = 1./3.*std::acos(q/(p*pst));
       csarc = std::cos(arc);
       alpha = pst*(std::sqrt(3.*(1.-csarc*csarc)) - csarc) + m12;   //  case (3)

    }else if (m12 <= m3){
     alpha = m3*cch + 0.5*m12;                                       // case (4a)
    }else{                                  
     p = m12*m3 + m1*m2 - 0.25;                                     
     q = 1.5*m1*m2*m3*(0.5-cch);
     pst = std::sqrt(p);
     arc = 1./3.*std::acos(q/(p*pst));
     csarc = std::cos(arc);
     alpha = pst*(std::sqrt(3.*(1.-csarc*csarc)) - csarc) + 0.5;     // case (4b)
    }

    if (c(i,j,k)>0.5){ alpha = 1. - alpha;}
  
    // compute alpha for the given coefficients  
    alpha = alpha + std::min(0.,mxyz(i,j,k,0)) + std::min(0.,mxyz(i,j,k,1)) + std::min(0.,mxyz(i,j,k,2));
    return alpha;
}

/** Computes the volume fraction from the normal m, the intercept alpha
 *  and the hexahedron dimensions and starting point (x0,y0,z0)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void compute_volume_fraction    (int i,  
                                 int j, 
                                 int k, 
                                 amrex::Real x0, 
                                 amrex::Real y0, 
                                 amrex::Real z0, 
                                 amrex::Real dx, 
                                 amrex::Real dy, 
                                 amrex::Real dz, 
                                 amrex::Array4<amrex::Real const> const& mxyz,
                                 amrex::Array4<amrex::Real const> const& alpha,
                                 amrex::Array4<amrex::Real> const& c) noexcept
{
    using namespace amrex;
    
    // move origin to x0 
    Real al = alpha(i,j,k) - mxyz(i,j,k,0)*x0 - mxyz(i,j,k,1)*y0 - mxyz(i,j,k,2)*z0;

    // reflect the figure when negative coefficients
    al = al + std::max(0.,-mxyz(i,j,k,0)*dx) + std::max(0.,-mxyz(i,j,k,1)*dy) + std::max(0.,-mxyz(i,j,k,2)*dz); 
    
    Real np1 = std::abs(mxyz(i,j,k,0));  // need positive coefficients
    Real np2 = std::abs(mxyz(i,j,k,1));
    Real np3 = std::abs(mxyz(i,j,k,2));
    Real almax = np1*dx + np2*dy + np3*dz;                       
    al  = std::max(0.,std::min(1.0,al/almax));         // get new al within safe limits
    Real alh = std::min(al,1.0-al);                         // limit to: 0 < alh < 1/2

    // normalized equation: m1*y1 + m2*y2 + m3*y3 = alh, with 0 <= m1 <= m2 <= m3 
    // the problem is then solved again in the unit cube
    np1 = np1/almax;
    np2 = np2/almax;
    np3 = np3/almax;

    Real  m1,m2,m3;
    m1 = std::min(np1*dx,np2*dy); // order coefficients
    m3 = std::max(np1*dx,np2*dy);
    Real top = np3*dz;
  
    if (top < m1){ 
       m2 = m1;
       m1 = top;
    }else if (top >= m3){
       m2 = m3;
       m3 = top;
    }else{
       m2 = top;
    }

    Real m12 = m1 + m2;
    Real mm = std::min(m12,m3);
    Real eps0=std::numeric_limits<Real>::epsilon();
    Real denom = std::max(6.*m1*m2*m3,eps0);

    Real frac;
    //1: al<=m1; 2: m1<=al<=m2; 3: m2<=al<=mm; 4: mm<=al<=1/2 (a:m12<=m3; b:m3<m12)) 
  if (alh <= m1){
     frac = alh*alh*alh/denom;                                         // case (1)
  }else if (alh <= m2){
     frac = 0.5*alh*(alh-m1)/(m2*m3) +  m1*m1*m1/denom;                // case (2)
  }else if (alh <= mm){
     top = alh*alh*(3.*m12-alh) + m1*m1*(m1-3.*alh);             
     frac = (top + m2*m2*(m2-3.*alh))/denom;                           // case (3)
  }else if (m12 <= m3){
     frac = (alh - 0.5*m12)/m3;                                        // case (4a)
  }else{
     top = alh*alh*(3.-2.*alh) + m1*m1*(m1-3.*alh);             
     frac = (top + m2*m2*(m2-3.*alh) + m3*m3*(m3-3.*alh))/denom;   // case (4b)
  }
  
  top = dx*dy*dz; 
  
  if (al <= 0.5){
     c(i,j,k) = frac*top;
  }else{
     c(i,j,k) = (1.-frac)*top;
  }
}

#endif // VolumeFraction_K.H
