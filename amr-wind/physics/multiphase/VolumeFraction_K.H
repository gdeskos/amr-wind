#ifndef VolumeFraction_K_H_
#define VolumeFraction_K_H_

#include <AMReX_FArrayBox.H>
#include <cmath>

/** Computes the "CUT VOLUME" V0 given r0, dr0 and
 *  m_1*x_1 + m_2*x_2 + m_3*x_3 = alpha
 *  (1) move origin to r0 along r ;
 * (2) reflect parallelepiped;
 * (3) limit alpha (0<= al0 <=0.5);
 * (4) order coefficients: b1<b2<b3;
 * (5) calculate volume (NOTE: it is assumed:s0=t0=0; ds0=dt0=1.)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real FL3D(
    amrex::Real m1,
    amrex::Real m2,
    amrex::Real m3,
    amrex::Real alpha,
    amrex::Real r0,
    amrex::Real dr0) noexcept
{

    amrex::Real const_tiny = std::numeric_limits<amrex::Real>::epsilon();
    amrex::Real al;

    // move origin to x0
    al = alpha - m1 * r0;

    // reflect the figure when negative coefficients
    al =
        al + std::max(0.0, -m1 * dr0) + std::max(0.0, -m2) + std::max(0.0, -m3);

    // normalized equation: m1*y1 + m2*y2 + m3*y3 = alh, with 0 <= m1 <= m2 <=
    // m3 the problem is then solved again in the unit cube
    amrex::Real tmp = std::abs(m1) * dr0 + std::abs(m2) + std::abs(m3);
    amrex::Real n1 = std::abs(m1) / tmp; // need positive coefficients
    amrex::Real n2 = std::abs(m2) / tmp;
    amrex::Real n3 = std::abs(m3) / tmp;
    al =
        std::max(0.0, std::min(1.0, al / tmp)); // get new al within safe limits
    amrex::Real al0 = std::min(al, 1.0 - al);   // limit to: 0 < alh < 1/2

    // Order coefficients
    amrex::Real b1, b2, b3;
    b1 = std::min(n1 * dr0, n2); // order coefficients
    b3 = std::max(n1 * dr0, n2);
    b2 = n3;

    if (b2 < b1) {
        tmp = b1;
        b1 = b2;
        b2 = tmp;
    } else if (b2 > b3) {
        tmp = b3;
        b3 = b2;
        b2 = tmp;
    }

    amrex::Real b12 = b1 + b2;

    // Compute volume fraction using Aoki Kawano (Computer & Fluids 2016) method
    amrex::Real vm1 = b1;
    amrex::Real vm3 = b3;
    amrex::Real vm2 = b2;
    amrex::Real vm12 = b12;
    amrex::Real a = al0;
    amrex::Real v = 0.0;

    if (a > 0.0) {
        if (a < vm1) {
            v = a * a * a / (6.0 * vm1 * vm2 * vm3);
        } else if (a < vm2) {
            v = a * (a - vm1) / (2.0 * vm2 * vm3) +
                vm1 * vm1 / (6.0 * vm2 * vm3 + const_tiny);
        } else if (a < std::min(vm12, vm3)) {
            v = (a * a * (3.0 * vm12 - a) + vm1 * vm1 * (vm1 - 3.0 * a) +
                 vm2 * vm2 * (vm2 - 3.0 * a)) /
                (6.0 * vm1 * vm2 * vm3);
        } else if (vm3 < vm12) {
            v = (a * a * (3.0 - 2.0 * a) + vm1 * vm1 * (vm1 - 3.0 * a) +
                 vm2 * vm2 * (vm2 - 3.0 * a) + vm3 * vm3 * (vm3 - 3.0 * a)) /
                (6.0 * vm1 * vm2 * vm3);
        } else {
            v = (a - 0.5 * vm12) / vm3;
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