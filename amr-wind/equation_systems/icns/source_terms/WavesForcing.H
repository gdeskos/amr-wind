#ifndef IBFORCING_H
#define IBFORCING_H

#include "amr-wind/equation_systems/icns/MomentumSource.H"
#include "amr-wind/utilities/trig_ops.H"
#include "amr-wind/core/SimTime.H"

namespace amr_wind {
namespace pde {
namespace icns {

/** Wave-like forcing introduced at the wall to mimick the effect of water waves
 *
 *  \ingroup icns_src ibm
 *
 *  \sa IB
 */
class WavesForcing : public MomentumSource::Register<WavesForcing>
{
public:
    static std::string identifier() { return "WavesForcing"; }

    explicit WavesForcing(const CFDSim& sim);

    virtual ~WavesForcing();

    virtual void operator()(
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Box& bx,
        const FieldState fstate,
        const amrex::Array4<amrex::Real>& src_term) const override;

private:
    const CFDSim& m_sim;

    amrex::Real m_magnitude{1};

    amrex::Real m_lambda{100};

    amrex::Real m_decay_length{25};
};

} // namespace icns
} // namespace pde
} // namespace amr_wind

#endif /* WAVELIKEFORCING_H */
