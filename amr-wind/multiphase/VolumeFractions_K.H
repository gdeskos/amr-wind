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

/** Computes the volume fraction from the normal m, the intercept alpha
 *  and the hexahedron dimensions and starting point (x0,y0,z0)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void compute_volume_fraction(
    int i,
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
    Real al = alpha(i, j, k) - mxyz(i, j, k, 0) * x0 - mxyz(i, j, k, 1) * y0 -
              mxyz(i, j, k, 2) * z0;

    // reflect the figure when negative coefficients
    al = al + std::max(0., -mxyz(i, j, k, 0) * dx) +
         std::max(0., -mxyz(i, j, k, 1) * dy) +
         std::max(0., -mxyz(i, j, k, 2) * dz);

    Real np1 = std::abs(mxyz(i, j, k, 0)); // need positive coefficients
    Real np2 = std::abs(mxyz(i, j, k, 1));
    Real np3 = std::abs(mxyz(i, j, k, 2));
    Real almax = np1 * dx + np2 * dy + np3 * dz;
    al = std::max(
        0., std::min(1.0, al / almax)); // get new al within safe limits
    Real alh = std::min(al, 1.0 - al);  // limit to: 0 < alh < 1/2

    // normalized equation: m1*y1 + m2*y2 + m3*y3 = alh, with 0 <= m1 <= m2 <=
    // m3 the problem is then solved again in the unit cube
    np1 = np1 / almax;
    np2 = np2 / almax;
    np3 = np3 / almax;

    Real m1, m2, m3;
    m1 = std::min(np1 * dx, np2 * dy); // order coefficients
    m3 = std::max(np1 * dx, np2 * dy);
    Real top = np3 * dz;

    if (top < m1) {
        m2 = m1;
        m1 = top;
    } else if (top >= m3) {
        m2 = m3;
        m3 = top;
    } else {
        m2 = top;
    }

    Real m12 = m1 + m2;
    Real mm = std::min(m12, m3);
    Real eps0 = std::numeric_limits<Real>::epsilon();
    Real denom = std::max(6. * m1 * m2 * m3, eps0);

    Real frac;
    // 1: al<=m1; 2: m1<=al<=m2; 3: m2<=al<=mm; 4: mm<=al<=1/2 (a:m12<=m3;
    // b:m3<m12))
    if (alh <= m1) {
        frac = alh * alh * alh / denom; // case (1)
    } else if (alh <= m2) {
        frac = 0.5 * alh * (alh - m1) / (m2 * m3) +
               m1 * m1 * m1 / denom; // case (2)
    } else if (alh <= mm) {
        top = alh * alh * (3. * m12 - alh) + m1 * m1 * (m1 - 3. * alh);
        frac = (top + m2 * m2 * (m2 - 3. * alh)) / denom; // case (3)
    } else if (m12 <= m3) {
        frac = (alh - 0.5 * m12) / m3; // case (4a)
    } else {
        top = alh * alh * (3. - 2. * alh) + m1 * m1 * (m1 - 3. * alh);
        frac = (top + m2 * m2 * (m2 - 3. * alh) + m3 * m3 * (m3 - 3. * alh)) /
               denom; // case (4b)
    }

    top = dx * dy * dz;

    if (al <= 0.5) {
        c(i, j, k) = frac * top;
    } else {
        c(i, j, k) = (1. - frac) * top;
    }
}

#endif // VolumeFraction_K.H
