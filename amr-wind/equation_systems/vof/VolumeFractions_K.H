#ifndef VolumeFraction_K_H_
#define VolumeFraction_K_H_

#include "amr-wind/derive/derive_K.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

template <typename Stencil>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mixed_Youngs_centered(
    int i,
    int j,
    int k,
    amrex::Real dx,
    amrex::Real dy,
    amrex::Real dz,
    amrex::Array4<amrex::Real const> const& C,
    amrex::Array4<amrex::Real> const& mxyz) noexcept

{
    amrex::Real c1, c2, c3; // Stencil coefficients for the differentiation
    amrex::Real m1, m2, m3, t0, t1, t2;
    amrex::Vector<amrex::Vector<amrex::Real>> m{
        {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    /**      3 x 3 x 3 BLOCK
     *
     *
     *            o-------o-------o------o
     *           /   X   /   X   /  X   /|
     *          o-------o-------o------o |
     *         /   X   /   X   /  X   /| |
     *        o-------o-------o------o | |
     *       /   X   /   X   /  X   /| | |
     *      o------o-------o-------o | | |
     *      |      |       |       | | | o
     *      |  X   |   X   |   X   | | |/|
     *      |      |       |       | | / |
     *      o------o-------o-------o / | |
     *      |      |       |       | | | /
     *      |  X   |   X   |   X   | | |/|
     *      |      |       |       | |/| |
     *      o------o-------o-------o/| | |
     *      |      |       |       | | | /
     *      |  X   |   X   |   X   | | /
     *      |      |       |       | /
     *      o------o-------o-------o
     *   z  y
     *   | /
     *   |/___ x
     *
     *   Normal vector is computed for the middle cell (i,j,k)
     */

    /** Write the plane as: sgn(mx) X = my Y + mz Z + alpha
     *                          m00 X = m01 Y + m02 Z + alpha
     */
    m1 = C(i - 1, j, k - 1) + C(i - 1, j, k + 1) + C(i - 1, j - 1, k) +
         C(i - 1, j + 1, k) + C(i - 1, j, k);
    m2 = C(i + 1, j, k - 1) + C(i + 1, j, k + 1) + C(i + 1, j - 1, k) +
         C(i + 1, j + 1, k) + C(i + 1, j, k);

    // Check the sign of mx
    if (m1 > m2) {
        m[0][0] = 1.;
    } else {
        m[0][0] = -1.;
    }

    // We compute m01 and m02 based on the height function derivatives (central
    // differences only for interior cells)
    m1 = dx * (C(i - 1, j - 1, k) + C(i, j - 1, k) +
               C(i + 1, j - 1, k)); // height function x_j-1,k
    m2 = dx * (C(i - 1, j, k) + C(i, j, k) +
               C(i + 1, j, k)); // height function x_j,k
    m3 = dx * (C(i - 1, j + 1, k) + C(i, j + 1, k) +
               C(i + 1, j + 1, k)); // height function x_j+1,k

    c3 = Stencil::c10;
    c2 = Stencil::c11;
    c1 = Stencil::c12;
    m[0][1] = (c3 * m3 + c2 * m2 + c1 * m1) / dy; // local gradient my

    m1 = dx * (C(i - 1, j, k - 1) + C(i, j, k - 1) +
               C(i + 1, j, k - 1)); // height function for x_j,k-1
    m2 = dx * (C(i - 1, j, k) + C(i, j, k) +
               C(i + 1, j, k)); // height function for x_j,k
    m3 = dx * (C(i - 1, j, k + 1) + C(i, j, k + 1) +
               C(i + 1, j, k + 1)); // height function for x_j,k+1

    c3 = Stencil::c20;
    c2 = Stencil::c21;
    c1 = Stencil::c22;
    m[0][2] = (c3 * m3 + c2 * m2 + c1 * m1) / dz; // local gradient mz

    /** Write the plane as: sgn(my) Y = mx X + mz Z + alpha
     *                          m11 Y = m10 X + m12 Z + alpha
     */
    m1 = dy * (C(i - 1, j - 1, k) + C(i - 1, j, k) +
               C(i - 1, j + 1, k)); // height function for y_i-1,k
    m2 = dy * (C(i, j - 1, k) + C(i, j, k) +
               C(i, j + 1, k)); // height function for y_i,k
    m3 = dy * (C(i + 1, j - 1, k) + C(i + 1, j, k) +
               C(i + 1, j + 1, k)); // height function for y_i+1,k

    c1 = Stencil::c02;
    c2 = Stencil::c01;
    c3 = Stencil::c00;
    m[1][0] = (c1 * m1 + c2 * m2 + c3 * m3) / dx; // local gradient mx

    m1 = C(i, j - 1, k - 1) + C(i, j - 1, k + 1) + C(i + 1, j - 1, k) +
         C(i - 1, j - 1, k) + C(i, j - 1, k);
    m2 = C(i, j + 1, k - 1) + C(i, j + 1, k + 1) + C(i + 1, j + 1, k) +
         C(i - 1, j + 1, k) + C(i, j + 1, k);

    if (m1 > m2) {
        m[1][1] = 1.;
    } else {
        m[1][1] = -1.;
    }

    m1 = dy * (C(i, j - 1, k - 1) + C(i, j, k - 1) +
               C(i, j + 1, k - 1)); // height function for y_i,k-1
    m3 = dy * (C(i, j - 1, k) + C(i, j, k) +
               C(i, j + 1, k)); // height function for y_i,k
    m3 = dy * (C(i, j - 1, k + 1) + C(i, j, k + 1) +
               C(i, j + 1, k + 1)); // height function for y_i,k+1

    c1 = Stencil::c22;
    c2 = Stencil::c21;
    c3 = Stencil::c20;
    m[1][2] = (c1 * m1 + c2 * m2 + c3 * m3) / dz; // local gradient mz

    /**write the plane as: sgn(mz) Z =  mx X +  my Y + alpha
     * m22 Z = m20 X + m21 Y + alpha
     */
    m1 = dz * (C(i - 1, j, k - 1) + C(i - 1, j, k) +
               C(i - 1, j, k + 1)); // height function for z_i-1,j
    m2 = dz * (C(i, j, k - 1) + C(i, j, k) +
               C(i, j, k + 1)); // height function for z_i,j
    m3 = dz * (C(i + 1, j, k - 1) + C(i + 1, j, k) +
               C(i + 1, j, k + 1)); // height function for z_i+1,j

    c1 = Stencil::c02;
    c2 = Stencil::c01;
    c3 = Stencil::c00;
    m[2][0] = (m1 * c1 + m2 * c2 + m3 * c3) / dx; // local gradient mx

    m1 = dz * (C(i, j - 1, k - 1) + C(i, j - 1, k) +
               C(i, j - 1, k + 1)); // height function for z_i,j-1
    m2 = dz * (C(i, j, k - 1) + C(i, j, k) +
               C(i, j, k + 1)); // height function for z_i,j
    m3 = dz * (C(i, j + 1, k - 1) + C(i, j + 1, k) +
               C(i, j + 1, k + 1)); // height function for z_i,j+1
    c1 = Stencil::c02;
    c2 = Stencil::c01;
    c3 = Stencil::c00;
    m[2][1] = (m1 * c1 + m2 * c2 + m3 * c3) / dy; // local gradient my

    m1 = C(i - 1, j, k - 1) + C(i + 1, j, k - 1) + C(i, j - 1, k - 1) +
         C(i, j + 1, k - 1) + C(i, j, k - 1);
    m2 = C(i - 1, j, k + 1) + C(i + 1, j, k + 1) + C(i, j - 1, k + 1) +
         C(i, j + 1, k + 1) + C(i, j, k + 1);

    if (m1 > m2) {
        m[2][2] = 1.;
    } else {
        m[2][2] = -1.;
    }

    // normalize each set (mx,my,mz): |mx|+|my|+|mz| = 1
    t0 = std::abs(m[0][0]) + std::abs(m[0][1]) + std::abs(m[0][2]);
    m[0][0] = m[0][0] / t0;
    m[0][1] = m[0][1] / t0;
    m[0][2] = m[0][2] / t0;

    t0 = std::abs(m[1][0]) + std::abs(m[1][1]) + std::abs(m[1][2]);
    m[1][0] = m[1][0] / t0;
    m[1][1] = m[1][1] / t0;
    m[1][2] = m[1][2] / t0;

    t0 = std::abs(m[2][0]) + std::abs(m[2][1]) + std::abs(m[2][2]);
    m[2][0] = m[2][0] / t0;
    m[2][1] = m[2][1] / t0;
    m[2][2] = m[2][2] / t0;

    // choose among the three central schemes
    t0 = std::abs(m[0][0]);
    t1 = std::abs(m[1][1]);
    t2 = std::abs(m[2][2]);

    int cn = 0;
    if (t1 > t0) {
        t0 = t1;
        cn = 1;
    }

    if (t2 > t0) {
        cn = 2;
    }

    // Compute gradients using simple upwind differeneces and cell averaging -
    // Young's method
    amrex::Vector<amrex::Vector<amrex::Real>> dCdx{
        {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    amrex::Real Ci, Cip, Cj, Cjp, Ck, Ckp;
    // Vertex 0 (i-1/2,j+1/2,k+1/2)
    Ci = 0.25 * (C(i - 1, j, k) + C(i - 1, j + 1, k) + C(i - 1, j, k + 1) +
                 C(i - 1, j + 1, k + 1));
    Cip = 0.25 *
          (C(i, j, k) + C(i, j + 1, k) + C(i, j, k + 1) + C(i, j + 1, k + 1));
    Cj = 0.25 *
         (C(i, j, k) + C(i - 1, j, k) + C(i, j, k + 1) + C(i - 1, j, k + 1));
    Cjp = 0.25 * (C(i, j + 1, k) + C(i - 1, j + 1, k) + C(i, j + 1, k + 1) +
                  C(i - 1, j + 1, k + 1));
    Ck = 0.25 *
         (C(i, j, k) + C(i - 1, j, k) + C(i, j + 1, k) + C(i - 1, j + 1, k));
    Ckp = 0.25 * (C(i, j, k + 1) + C(i - 1, j, k + 1) + C(i, j + 1, k + 1) +
                  C(i - 1, j + 1, k + 1));
    dCdx[0][0] = -(Cip - Ci) / dx;
    dCdx[0][1] = -(Cjp - Cj) / dy;
    dCdx[0][2] = -(Ckp - Ck) / dz;
    // Vertex 1 (i+1/2,j+1/2,k+1/2)
    Cip = 0.25 * (C(i + 1, j, k) + C(i + 1, j + 1, k) + C(i + 1, j, k + 1) +
                  C(i + 1, j + 1, k + 1));
    Ci = 0.25 *
         (C(i, j, k) + C(i, j + 1, k) + C(i, j, k + 1) + C(i, j + 1, k + 1));
    Cjp = 0.25 *
          (C(i, j, k) + C(i + 1, j, k) + C(i, j, k + 1) + C(i + 1, j, k + 1));
    Cj = 0.25 * (C(i, j + 1, k) + C(i + 1, j + 1, k) + C(i, j + 1, k + 1) +
                 C(i + 1, j + 1, k + 1));
    Ckp = 0.25 *
          (C(i, j, k) + C(i + 1, j, k) + C(i, j + 1, k) + C(i + 1, j + 1, k));
    Ck = 0.25 * (C(i, j, k + 1) + C(i + 1, j, k + 1) + C(i, j + 1, k + 1) +
                 C(i + 1, j + 1, k + 1));
    dCdx[1][0] = -(Cip - Ci) / dx;
    dCdx[1][1] = -(Cjp - Cj) / dy;
    dCdx[1][2] = -(Ckp - Ck) / dz;
    // Vertex 2 (i-1/2,j-1/2,k+1/2)
    Ci = 0.25 * (C(i - 1, j, k) + C(i - 1, j - 1, k) + C(i - 1, j, k + 1) +
                 C(i - 1, j - 1, k + 1));
    Cip = 0.25 *
          (C(i, j, k) + C(i, j - 1, k) + C(i, j, k + 1) + C(i, j - 1, k + 1));
    Cjp = 0.25 *
          (C(i, j, k) + C(i - 1, j, k) + C(i, j, k + 1) + C(i - 1, j, k + 1));
    Cj = 0.25 * (C(i, j - 1, k) + C(i - 1, j - 1, k) + C(i, j - 1, k + 1) +
                 C(i - 1, j - 1, k + 1));
    Ck = 0.25 *
         (C(i, j, k) + C(i - 1, j, k) + C(i, j - 1, k) + C(i - 1, j - 1, k));
    Ckp = 0.25 * (C(i, j, k + 1) + C(i - 1, j, k + 1) + C(i, j - 1, k + 1) +
                  C(i - 1, j - 1, k + 1));
    dCdx[2][0] = -(Cip - Ci) / dx;
    dCdx[2][1] = -(Cjp - Cj) / dy;
    dCdx[2][2] = -(Ckp - Ck) / dz;
    // Vertex 3 (i-1/2,j-1/2,k-1/2)
    Ci = 0.25 * (C(i - 1, j, k) + C(i - 1, j - 1, k) + C(i - 1, j, k - 1) +
                 C(i - 1, j - 1, k - 1));
    Cip = 0.25 *
          (C(i, j, k) + C(i, j - 1, k) + C(i, j, k - 1) + C(i, j - 1, k - 1));
    Cjp = 0.25 *
          (C(i, j, k) + C(i - 1, j, k) + C(i, j, k - 1) + C(i - 1, j, k - 1));
    Cj = 0.25 * (C(i, j - 1, k) + C(i - 1, j - 1, k) + C(i, j - 1, k + 1) +
                 C(i - 1, j - 1, k + 1));
    Ckp = 0.25 *
          (C(i, j, k) + C(i - 1, j, k) + C(i, j - 1, k) + C(i - 1, j - 1, k));
    Ck = 0.25 * (C(i, j, k - 1) + C(i - 1, j, k - 1) + C(i, j - 1, k - 1) +
                 C(i - 1, j - 1, k - 1));
    dCdx[3][0] = -(Cip - Ci) / dx;
    dCdx[3][1] = -(Cjp - Cj) / dy;
    dCdx[3][2] = -(Ckp - Ck) / dz;
    // Vertex 4 (i+1/2,j-1/2,k-1/2)
    Cip = 0.25 * (C(i + 1, j, k) + C(i + 1, j - 1, k) + C(i + 1, j, k - 1) +
                  C(i + 1, j - 1, k - 1));
    Ci = 0.25 *
         (C(i, j, k) + C(i, j - 1, k) + C(i, j, k - 1) + C(i, j - 1, k - 1));
    Cjp = 0.25 *
          (C(i, j, k) + C(i + 1, j, k) + C(i, j, k - 1) + C(i + 1, j, k - 1));
    Cj = 0.25 * (C(i, j - 1, k) + C(i + 1, j - 1, k) + C(i, j - 1, k - 1) +
                 C(i + 1, j - 1, k - 1));
    Ckp = 0.25 *
          (C(i, j, k) + C(i + 1, j, k) + C(i, j - 1, k) + C(i + 1, j - 1, k));
    Ck = 0.25 * (C(i, j, k - 1) + C(i + 1, j, k - 1) + C(i, j - 1, k - 1) +
                 C(i + 1, j - 1, k - 1));
    dCdx[4][0] = -(Cip - Ci) / dx;
    dCdx[4][1] = -(Cjp - Cj) / dy;
    dCdx[4][2] = -(Ckp - Ck) / dz;
    // Vertex 5 (i-1/2,j-1/2,k+1/2)
    Ci = 0.25 * (C(i - 1, j, k) + C(i - 1, j - 1, k) + C(i - 1, j, k + 1) +
                 C(i - 1, j - 1, k + 1));
    Cip = 0.25 *
          (C(i, j, k) + C(i, j - 1, k) + C(i, j, k + 1) + C(i, j - 1, k + 1));
    Cjp = 0.25 *
          (C(i, j, k) + C(i - 1, j, k) + C(i, j, k + 1) + C(i - 1, j, k + 1));
    Cj = 0.25 * (C(i, j - 1, k) + C(i - 1, j - 1, k) + C(i, j - 1, k + 1) +
                 C(i - 1, j - 1, k + 1));
    Ck = 0.25 *
         (C(i, j, k) + C(i - 1, j, k) + C(i, j - 1, k) + C(i - 1, j - 1, k));
    Ckp = 0.25 * (C(i, j, k - 1) + C(i - 1, j, k + 1) + C(i, j - 1, k + 1) +
                  C(i - 1, j - 1, k + 1));
    dCdx[5][0] = -(Cip - Ci) / dx;
    dCdx[5][1] = -(Cjp - Cj) / dy;
    dCdx[5][2] = -(Ckp - Ck) / dz;
    // Vertex 6 (i+1/2,j-1/2,k+1/2)
    Cip = 0.25 * (C(i + 1, j, k) + C(i + 1, j - 1, k) + C(i + 1, j, k + 1) +
                  C(i + 1, j - 1, k + 1));
    Ci = 0.25 *
         (C(i, j, k) + C(i, j - 1, k) + C(i, j, k + 1) + C(i, j - 1, k + 1));
    Cjp = 0.25 *
          (C(i, j, k) + C(i + 1, j, k) + C(i, j, k + 1) + C(i + 1, j, k + 1));
    Cj = 0.25 * (C(i, j - 1, k) + C(i + 1, j - 1, k) + C(i, j - 1, k + 1) +
                 C(i + 1, j - 1, k + 1));
    Ck = 0.25 *
         (C(i, j, k) + C(i + 1, j, k) + C(i, j - 1, k) + C(i + 1, j - 1, k));
    Ckp = 0.25 * (C(i, j, k + 1) + C(i + 1, j, k + 1) + C(i, j - 1, k + 1) +
                  C(i + 1, j - 1, k + 1));
    dCdx[6][0] = -(Cip - Ci) / dx;
    dCdx[6][1] = -(Cjp - Cj) / dy;
    dCdx[6][2] = -(Ckp - Ck) / dz;
    // Vertex 7 (i-1/2,j+1/2,k-1/2)
    Ci = 0.25 * (C(i - 1, j, k) + C(i - 1, j + 1, k) + C(i - 1, j, k - 1) +
                 C(i - 1, j + 1, k - 1));
    Cip = 0.25 *
          (C(i, j, k) + C(i, j + 1, k) + C(i, j, k - 1) + C(i, j + 1, k - 1));
    Cj = 0.25 *
         (C(i, j, k) + C(i - 1, j, k) + C(i, j, k - 1) + C(i - 1, j, k - 1));
    Cjp = 0.25 * (C(i, j + 1, k) + C(i - 1, j + 1, k) + C(i, j + 1, k + 1) +
                  C(i - 1, j + 1, k + 1));
    Ckp = 0.25 *
          (C(i, j, k) + C(i - 1, j, k) + C(i, j + 1, k) + C(i - 1, j + 1, k));
    Ck = 0.25 * (C(i, j, k - 1) + C(i - 1, j, k - 1) + C(i, j + 1, k - 1) +
                 C(i - 1, j + 1, k - 1));
    dCdx[7][0] = -(Cip - Ci) / dx;
    dCdx[7][1] = -(Cjp - Cj) / dy;
    dCdx[7][2] = -(Ckp - Ck) / dz;

    // Average over the 8 vertices !!!
    m[3][0] = 1.0 / 8.0 *
              (dCdx[0][0] + dCdx[1][0] + dCdx[2][0] + dCdx[3][0] + dCdx[4][0] +
               dCdx[5][0] + dCdx[6][0] + dCdx[7][0]);
    m[3][1] = 1.0 / 8.0 *
              (dCdx[0][1] + dCdx[1][1] + dCdx[2][1] + dCdx[3][1] + dCdx[4][1] +
               dCdx[5][1] + dCdx[6][1] + dCdx[7][1]);
    m[3][2] = 1.0 / 8.0 *
              (dCdx[0][2] + dCdx[1][2] + dCdx[2][2] + dCdx[3][2] + dCdx[4][2] +
               dCdx[5][2] + dCdx[6][2] + dCdx[7][2]);

    // normalize the set (mx,my,mz): |mx|+|my|+|mz| = 1
    amrex::Real NOT_ZERO = 1e-16;
    t0 = std::abs(m[3][0]) + std::abs(m[3][1]) + std::abs(m[3][2]) + NOT_ZERO;
    m[3][0] = m[3][0] / t0;
    m[3][1] = m[3][1] / t0;
    m[3][2] = m[3][2] / t0;
    // choose between the previous choice and Youngs-PLIC
    t0 = std::abs(m[3][0]);
    t1 = std::abs(m[3][1]);
    t2 = std::abs(m[3][2]);
    if (t1 > t0) {
        t0 = t1;
    }
    if (t2 > t0) {
        t0 = t2;
    }

    if (std::abs(m[cn][cn]) > t0) {
        cn = 3;
    }

    // components of the normal vector */
    mxyz(i, j, k, 0) = m[cn][0];
    mxyz(i, j, k, 1) = m[cn][1];
    mxyz(i, j, k, 2) = m[cn][2];
}

/* Computes alpha from a volume fraction F and the surface outward normal vector
 * m=(m1,m2,m3)^T
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real volume_fraction_intercept(
    int i,
    int j,
    int k,
    amrex::Real dx,
    amrex::Real dy,
    amrex::Real dz,
    amrex::Array4<amrex::Real const> const& c,
    amrex::Array4<amrex::Real const> const& mxyz) noexcept
{
    using namespace amrex;

    Real cch, c01, c02, c03, np1, np2, np3;
    Real m1, m2, m3, m12, numer, denom;
    Real dx1, dx2, dx3;

    np1 = std::abs(mxyz(i, j, k, 0));
    np2 = std::abs(mxyz(i, j, k, 1));
    np3 = std::abs(mxyz(i, j, k, 2));
    // Sorting out the normals (from the smaller to the larger component).
    // We need to re-arrange them so that m1*dx1<=m2*dx2<=m3*dx3
    m1 = std::min(np1, np2);
    m3 = std::max(np1, np2);
    //if(np1*dx>=np2*dy && np1*dx<np3*dz){
    //    m1=np2; dx1=dy;
    //    m2=np1; dx2=dx;
    //    m3=np3; dx3=dz;
    //}else if(np1*dx>=np3*dz && np1*dx<np2*dy){
    //    m1=np3; dx1=dz;
    //    m2=np1; dx2=dx;
    //    m3=np2; dx3=dy;
    //}else if(np2*dy>=np3*dz && np2*dy<np1*dx){
    //    m1=np3; dx1=dz;
    //    m2=np2; dx2=dy;
    //    m3=np1; dx3=dx;
    //}else if(np2*dy>=np1*dx && np2*dy<np3*dz){
    //    m1=np1; dx1=dx;
    //    m2=np2; dx2=dy;
    //    m3=np3; dx3=dz;    
    //}else if(np3*dz>=np1*dx && np3*dz<np2*dy){
    //    m1=np1; dx1=dx;
    //    m2=np3; dx2=dz;
    //    m3=np2; dx3=dy;
    //}else if(np3*dz>=np2*dy && np3*dz<np1*dx){
    //    m1=np2; dx1=dy;
    //    m2=np3; dx2=dz;
    //    m3=np1; dx3=dx;
    //}else{
    //    m1=np1;m2=np2;m3=np3;
    //    dx1=dx;dx2=dy;dx3=dz;
    //}


    if (np3 < np1) {
        m2 = m1;
        m1 = np3;
    } else if (np3 >= m3) {
        m2 = m3;
        m3 = np3;
    } else {
        m2 = np3;
    }

    Real eps0 = 1e-50;
    denom = std::max(6. * m1 * m2 * m3, eps0);
    cch = std::min(c(i, j, k), 1. - c(i, j, k)); // limit to: 0 < cch < 1/2
    c01 = m1 *m1 * m1 / denom;                  // get cch ranges
    c02 = c01 + (m2*dx2 - m1*dx1) / std::max(2.0*m2*dx3,eps0);
    m12 = m1*dx1 + m2*dx2;
    if (m12 <= m3) {
        c03 = 0.5 * m12 / (m3*dx3);
    } else {
        numer = m3 * m3 * (3. * m12 - m3) + m1 * m1 * (m1 - 3. * m3) +
                m2 * m2 * (m2 - 3. * m3);
        c03 = numer / denom;
    }
    Real alpha, p, q, pst, arc, csarc;
    // 1: C<=C1; 2: C1<=C<=C2; 3: C2<=C<=C3; 4: C3<=C<=1/2 (a: m12<=m3; b:
    // m3<m12))
    if (cch <= c01) {
        alpha = std::pow(denom * cch, 1. / 3.); // case (1)
        // amrex::Print()<<"alpha=  "<<alpha<<", denom =  "<<denom<<", cch =
        // "<<cch<<std::endl;
    } else if (cch <= c02) {
        alpha =
            0.5 *
            (m1*dx1 + std::sqrt(m1 * m1*dx1*dx1 + 8. * m2*dx2 * m3*dx3 * (cch - c01))); // case (2)
    } else if (cch <= c03) {
        p = 2. * m1*dx1 * m2*dx2;
        q = 1.5 * m1 *dx1* m2*dx2 * (m12 - 2. * m3*dx3 * cch);
        pst = std::sqrt(p);
        arc = 1. / 3. * std::acos(q / (p * pst));
        csarc = std::cos(arc);
        alpha = pst * (std::sqrt(3. * (1. - csarc * csarc)) - csarc) +
                m12; //  case (3)

    } else if (m12 <= m3) {
        alpha = m3 *dx3* cch + 0.5 * m12; // case (4a)
    } else {
        p = m12 * m3*dx3 + m1*dx1 * m2*dx2 - 0.25;
        q = 1.5 * m1*dx1 * m2*dx2 * m3*dx3 * (0.5 - cch);
        pst = std::sqrt(p);
        arc = 1. / 3. * std::acos(q / (p * pst));
        csarc = std::cos(arc);
        alpha = pst * (std::sqrt(3. * (1. - csarc * csarc)) - csarc) +
                0.5; // case (4b)
    }

    if (c(i, j, k) > 0.5) {
        alpha = 1.0 - alpha;
    }

    // compute alpha for the given coefficients
    //alpha = alpha + std::min(0., mxyz(i, j, k, 0) * dx) +
    //        std::min(0., mxyz(i, j, k, 1) * dy) +
    //        std::min(0., mxyz(i, j, k, 2) * dz);
    return alpha;
}

/** Computes the "CUT VOLUME" V0 given r0, dr0 and 
*  m_1*x_1 + m_2*x_2 + m_3*x_3 = alpha
*  (1) move origin to r0 along r ;  
* (2) reflect parallelepiped;
* (3) limit alpha (0<= al0 <=0.5); 
* (4) order coefficients: b1<b2<b3;
* (5) calculate volume (NOTE: it is assumed:s0=t0=0; ds0=dt0=1.)
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real FL3D(
    int i,
    int j,
    int k,
    amrex::Array4<amrex::Real const> const& mxyz,
    amrex::Real alpha,
    amrex::Real r0,
    amrex::Real dr0) noexcept
{

    amrex::Real const_tiny = std::numeric_limits<amrex::Real>::epsilon();
    amrex::Real m1,m2,m3; // normal vectors to work with
    amrex::Real al;

    m1=mxyz(i,j,k,0);m2=mxyz(i,j,k,1);m3=mxyz(i,j,k,2);

    // move origin to x0
    al = alpha - m1*r0;

    // reflect the figure when negative coefficients
    al = al + std::max(0.0, -m1 * dr0) +
         std::max(0.0, -m2) +
         std::max(0.0, -m3);

    // normalized equation: m1*y1 + m2*y2 + m3*y3 = alh, with 0 <= m1 <= m2 <=
    // m3 the problem is then solved again in the unit cube
    amrex::Real tmp = std::abs(m1)*dr0+std::abs(m2)+std::abs(m3);
    amrex::Real n1 = std::abs(m1); // need positive coefficients
    amrex::Real n2 = std::abs(m2);
    amrex::Real n3 = std::abs(m3);
    al = std::max(0.0, std::min(1.0, al / tmp)); // get new al within safe limits
    amrex::Real al0 = std::min(al, 1.0 - al);  // limit to: 0 < alh < 1/2

    // Order coefficients
    amrex::Real b1, b2, b3;
    b1 = std::min(n1 * dr0,n2); // order coefficients
    b3 = std::max(n1 * dr0,n2);
    b2 = n3;

    if (b2 < b1) {
        tmp=b1;
        b1 = b2;
        b2 = tmp;
    } else if (b2 > b3) {
        tmp=b3;
        b3 = b2;
        b2 = tmp;
    }
    
    amrex::Real b12 = b1 + b2;
    amrex::Real bm = std::min(b12, b3);
    amrex::Real pr = std::max(6. * m1 * m2 * m3, const_tiny);

    // Compute volume fraction using Aoki Kawano (Computer & Fluids 2016) method
    amrex::Real vm1 = b1;
    amrex::Real vm3 = b3;
    amrex::Real vm2 = b2;
    amrex::Real vm12= b12;
    amrex::Real a = al0;
    amrex::Real v= 0.0;

    if (a > 0.0) {
        if (a<vm1){
            v = a*a*a/(6*vm1*vm2*vm3);
        } else if (a < vm2) {
            v = a * (a - vm1) / (2.0 * vm2 * vm3) +vm1*vm1 / (6.0 * vm2 * vm3 + const_tiny);
        } else if (a < std::min(vm12,vm3)) {
            v = (a*a * (3.0 * vm12 - a)
                + vm1 * vm1 * (vm1 - 3.0 * a) 
                + vm2 * vm2 * (vm2 - 3.0 * a)) /(6.0 * vm1 * vm2 * vm3);
        } else if (vm3 < vm12) {
            v = ( a * a  * (3.0 - 2.0 * a) +
                 vm1 * vm1 * (vm1 - 3.0 * a) +
                 vm2 * vm2 * (vm2 - 3.0 * a) +
                 vm3 * vm3 * (vm3 - 3.0 * a)) / (6.0 * vm1 * vm2 * vm3);
        } else {
            v = (a - 0.5 * vm12) / vm3 ;
        }
    }

    tmp = v;
    amrex::Real FL3D;
    if (al <= 0.5) {
        FL3D = tmp * dr0;
    } else {
        FL3D = (1.0 - tmp) * dr0;
    }

    return FL3D;
}

#endif // VolumeFraction_K.H
