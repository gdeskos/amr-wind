#ifndef INTERFACETHICKNESSREFINEMENT_H
#define INTERFACETHICKNESSREFINEMENT_H

#include "amr-wind/utilities/tagging/RefinementCriteria.H"

namespace amr_wind {
class Field;

/** AMR refinement using multi-phase interface thickness
 *  \ingroup amr_utils
 *
 *  ```
 *  tagging.labels = it
 *  tagging/it.type = InterfaceThicknessRefinement
 *  # Tagging based on value
 *  tagging/it.values = 10.0 10.0

 *  ```
 */
class InterfaceThicknessRefinement
    : public RefinementCriteria::Register<InterfaceThicknessRefinement>
{
public:
    static std::string identifier() { return "InterfaceThicknessRefinement"; }

    InterfaceThicknessRefinement(const CFDSim& sim);

    virtual ~InterfaceThicknessRefinement() = default;

    //! Read input file and initialize boxarray used to refine each level
    virtual void initialize(const std::string& key) override;

    virtual void
    operator()(int level, amrex::TagBoxArray& tags, amrex::Real time, int ngrow)
        override;

private:
    const CFDSim& m_sim;

    Field* m_vof;

    amrex::Vector<amrex::Real> m_thickness_value;

    int m_max_lev_field;
};

} // namespace amr_wind

#endif /* INTERFACETHICKNESSREFINEMENT_H */
