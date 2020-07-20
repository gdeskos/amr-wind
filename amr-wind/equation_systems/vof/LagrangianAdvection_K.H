#ifndef LAGRANGIAN_ADVECTION_K_H_
#define LAGRANGIAN_ADVECTION_K_H_

#include "amr-wind/derive/derive_K.H"
#include "amr-wind/equation_systems/vof/VolumeFractions_K.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

/** Split advection of the interfae along the x,y,z directions
 *
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void lagrangian_advection(
    int i,
    int j,
    int k,
    int d,
    amrex::Real dt,
    amrex::Real dx,
    amrex::Real dy,
    amrex::Real dz,
    amrex::Array4<amrex::Real> const& mxyz,
    amrex::Array4<amrex::Real> const& flux,
    amrex::Array4<amrex::Real const> const& velocity,
    amrex::Array4<amrex::Real> const& cc) noexcept
{
    amrex::Real a1, a2;
    amrex::Real const_tiny = std::numeric_limits<amrex::Real>::epsilon();

    if (d == 0) {
        a1 = velocity(i, j, k) * dt / dx;
        a2 = velocity(i + 1, j, k) * dt / dx;
    } else if (d == 1) {
        a1 = velocity(i, j, k) * dt / dy;
        a2 = velocity(i, j + 1, k) * dt / dy;
    } else if (d == 2) {
        a1 = velocity(i, j, k) * dt / dz;
        a2 = velocity(i, j, k + 1) * dt / dz;
    }
    // 3 cases: 1: DEFAULT (c=0.0 and fluxes=0.0); 2: c=1.0; 3: c>0.0
    flux(i, j, k, 0) = 0.0;
    flux(i, j, k, 1) = 0.0;
    flux(i, j, k, 2) = 0.0;

    if (cc(i, j, k) > 1.0 - const_tiny) {
        flux(i, j, k, 0) = std::max(-a1, 0.0);
        flux(i, j, k, 1) = 1.0 - std::max(a1, 0.0) + std::min(a2, 0.0);
        flux(i, j, k, 2) = std::max(a2, 0.0);
    } else if (cc(i, j, k) > 0.0) {

        // (1) normal vector: mx,my,mz;
        compute_normals(i, j, k, cc, mxyz);

        // (2) mx,my,mz>0. and mx+my+mz = 1.0;
        amrex::Real mx, my, mz, invx, invy, invz;
        mx = mxyz(i, j, k, 0);
        my = mxyz(i, j, k, 1);
        mz = mxyz(i, j, k, 2);

        // (3) Get alpha;
        invx = 1;
        invy = 1;
        invz = 1;
        if (mx < 0.0) {
            mx = -mx;
            invx = -1;
        }
        if (my < 0.0) {
            my = -my;
            invy = -1;
        }
        if (mz < 0.0) {
            mz = -mz;
            invz = -1;
        }
        amrex::Real mm2 = mx + my + mz;
        mx = mx / mm2;
        my = my / mm2;
        mz = mz / mm2;
        mxyz(i, j, k, 0) = mx;
        mxyz(i, j, k, 1) = my;
        mxyz(i, j, k, 2) = mz;
        amrex::Real alpha = AL3D(i, j, k, mxyz, cc);
        // (4) Back to original plane
        mx = invx * mx;
        my = invy * my;
        mz = invz * mz;
        alpha =
            alpha + std::min(0.0, mx) + std::min(0.0, my) + std::min(0.0, mz);

        // (5) Lagrangian Advection
        amrex::Real mm1 = std::max(a1, 0.0);
        mm2 = 1.0 - mm1 + std::min(0.0, a2);
        if (d == 0) {
            mx = mx / (1.0 - a1 + a2);
            alpha = alpha + mx * a1;
            if (a1 < 0.0) {
                flux(i, j, k, 0) = FL3D(i, j, k, mx, my, mz, alpha, a1, -a1);
            }
            if (a2 > 0.0) {
                flux(i, j, k, 2) = FL3D(i, j, k, mx, my, mz, alpha, 1.0, a2);
                flux(i, j, k, 1) = FL3D(i, j, k, mx, my, mz, alpha, mm1, mm2);
            }
        } else if (d == 1) {
            my = my / (1.0 - a1 + a2);
            alpha = alpha + my * a1;
            if (a1 < 0.0) {
                flux(i, j, k, 0) = FL3D(i, j, k, my, mz, mx, alpha, a1, -a1);
            }
            if (a2 > 0.0) {
                flux(i, j, k, 2) = FL3D(i, j, k, my, mz, mx, alpha, 1.0, a2);
                flux(i, j, k, 1) = FL3D(i, j, k, my, mz, mx, alpha, mm1, mm2);
            }
        } else if (d == 2) {
            mz = mz / (1.0 - a1 + a2);
            alpha = alpha + mz * a1;
            if (a1 < 0.0) {
                flux(i, j, k, 0) = FL3D(i, j, k, mz, mx, my, alpha, a1, -a1);
            }
            if (a2 > 0.0) {
                flux(i, j, k, 2) = FL3D(i, j, k, mz, mx, my, alpha, 1.0, a2);
                flux(i, j, k, 1) = FL3D(i, j, k, mz, mx, my, alpha, mm1, mm2);
            }
        }
    }
}

// This applies boundary condition to the volume fluxes
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void apply_fluxes(
    int i,
    int j,
    int k,
    int d,
    amrex::Array4<amrex::Real> const& flux,
    amrex::Array4<amrex::Real> const& cc) noexcept
{
    // Apply BCs to fluxes

    // Compute new values of cc and then do clipping
    if (d == 0) {
        cc(i, j, k) =
            flux(i - 1, j, k, 2) + flux(i, j, k, 1) + flux(i + 1, j, k, 0);
    } else if (d == 1) {
        cc(i, j, k) =
            flux(i, j - 1, k, 2) + flux(i, j, k, 1) + flux(i, j + 1, k, 0);
    } else if (d == 2) {
        cc(i, j, k) =
            flux(i, j, k - 1, 2) + flux(i, j, k, 1) + flux(i, j, k + 1, 0);
    }
    cc(i, j, k) = std::max(0.0, std::min(1.0, cc(i, j, k)));

    // Apply BCs to the volume fractions cc
}

#endif // Sweep_K.H
