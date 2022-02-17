#include "amr-wind/equation_systems/DiffusionOps.H"
#include "amr-wind/utilities/console_io.H"

#include "AMReX_MLTensorOp.H"

#include "amr-wind/incflo.H"
#include "amr-wind/incflo_enums.H"

namespace amr_wind {
namespace pde {

template <typename LinOp>
DiffSolverIface<LinOp>::DiffSolverIface(

    PDEFields& fields, const bool has_overset, const std::string& prefix)
    : m_pdefields(fields)
    , m_density(fields.repo.get_field("density"))
    , m_options(prefix, m_pdefields.field.name() + "_" + prefix)
{
    amrex::LPInfo isolve = m_options.lpinfo();
    amrex::LPInfo iapply;

    iapply.setMaxCoarseningLevel(0);

    const auto& mesh = m_pdefields.repo.mesh();
    if (!has_overset) {
        m_solver.reset(new LinOp(
            mesh.Geom(0, mesh.finestLevel()),
            mesh.boxArray(0, mesh.finestLevel()),
            mesh.DistributionMap(0, mesh.finestLevel()), isolve));
        m_applier.reset(new LinOp(
            mesh.Geom(0, mesh.finestLevel()),
            mesh.boxArray(0, mesh.finestLevel()),
            mesh.DistributionMap(0, mesh.finestLevel()), iapply));
    } else {
        auto imask = fields.repo.get_int_field("mask_cell").vec_const_ptrs();
        m_solver.reset(new LinOp(
            mesh.Geom(0, mesh.finestLevel()),
            mesh.boxArray(0, mesh.finestLevel()),
            mesh.DistributionMap(0, mesh.finestLevel()), imask, isolve));
        m_applier.reset(new LinOp(
            mesh.Geom(0, mesh.finestLevel()),
            mesh.boxArray(0, mesh.finestLevel()),
            mesh.DistributionMap(0, mesh.finestLevel()), imask, iapply));
    }

    m_solver->setMaxOrder(m_options.max_order);
    m_applier->setMaxOrder(m_options.max_order);

    // It is the sub-classes responsibility to set the linear solver BC for the
    // operators.
}

/*
template <typename LinOp>
void DiffSolverIface<LinOp>::setup_operator(
    LinOp& linop,
    const amrex::Real alpha,
    const amrex::Real beta,
    const FieldState fstate,
    const ConvectionType convtype)
{
*/

template <typename LinOp>
void DiffSolverIface<LinOp>::setup_operator(
    LinOp& linop,
    const amrex::Real alpha,
    const amrex::Real beta,
    const FieldState fstate)
{
    BL_PROFILE("amr-wind::setup_operator");
    auto& repo = m_pdefields.repo;
    const int nlevels = repo.num_active_levels();

    linop.setScalars(alpha, beta);
    for (int lev = 0; lev < nlevels; ++lev) {
        linop.setLevelBC(lev, &m_pdefields.field(lev));
    }

/*    if (convtype == ConvectionType::Explicit) { 
      this->set_acoeffs(linop, fstate);
    } else {
      this->set_acoeffs_implicit(linop, beta, fstate);
    }
*/
//      this->set_acoeffs(linop, fstate);
 
      this->set_acoeffs_implicit(linop, beta, fstate);
 
    set_bcoeffs(linop);
}

template <typename LinOp>
void DiffSolverIface<LinOp>::set_acoeffs(LinOp& linop, const FieldState fstate)
{
    BL_PROFILE("amr-wind::set_acoeffs");
    auto& repo = m_pdefields.repo;
    const int nlevels = repo.num_active_levels();
    const auto& density = m_density.state(fstate);

    for (int lev = 0; lev < nlevels; ++lev) {
        linop.setACoeffs(lev, density(lev));
    }
}

template <typename LinOp>
void DiffSolverIface<LinOp>::set_acoeffs_implicit(
    LinOp& linop, const amrex::Real dt, const FieldState fstate)
{
    BL_PROFILE("amr-wind::set_acoeffs_implicit");

    auto& repo = m_pdefields.repo;
    auto& field = this->m_pdefields.field;
    auto& geom = repo.mesh().Geom();

    const int nlevels = repo.num_active_levels();
    const auto& density = m_density.state(fstate);

    auto& u_mac = repo.get_field("u_mac");
    auto& v_mac = repo.get_field("v_mac");
    auto& w_mac = repo.get_field("w_mac");

    auto new_diag_ptr = repo.create_scratch_field("new_diag", 1, 0);

    constexpr amrex::Real small_vel = 1.e-10;

    for (int lev = 0; lev < nlevels; ++lev) {
        auto& new_diag = (*new_diag_ptr)(lev);
        const auto dxinv = geom[lev].InvCellSizeArray();

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(new_diag, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& new_diag_a = new_diag.array(mfi);
            const auto& fld = field(lev).const_array(mfi);
            const auto& rho = density(lev).const_array(mfi);

            amrex::Array4<amrex::Real> const& a_umac = u_mac(lev).array(mfi);
            amrex::Array4<amrex::Real> const& a_vmac = v_mac(lev).array(mfi);
            amrex::Array4<amrex::Real> const& a_wmac = w_mac(lev).array(mfi);

            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    amrex::Real delta_pls_umac =
                        (a_umac(i + 1, j, k) > small_vel) ? 1.0 : 0.0;
                    amrex::Real delta_mns_umac =
                        (a_umac(i, j, k) <= -small_vel) ? 1.0 : 0.0;

                    amrex::Real delta_pls_vmac =
                        (a_vmac(i, j + 1, k) > small_vel) ? 1.0 : 0.0;
                    amrex::Real delta_mns_vmac =
                        (a_vmac(i, j, k) <= -small_vel) ? 1.0 : 0.0;

                    amrex::Real delta_pls_wmac =
                        (a_wmac(i, j, k + 1) >= small_vel) ? 1.0 : 0.0;
                    amrex::Real delta_mns_wmac =
                        (a_wmac(i, j, k) < -small_vel) ? 1.0 : 0.0;

                    amrex::Real net_coeff =
                        dxinv[0] * (a_umac(i + 1, j, k) * delta_pls_umac -
                                    a_umac(i, j, k) * delta_mns_umac) +
                        dxinv[1] * (a_vmac(i, j + 1, k) * delta_pls_vmac -
                                    a_vmac(i, j, k) * delta_mns_vmac) +
                        dxinv[2] * (a_wmac(i, j, k + 1) * delta_pls_wmac -
                                    a_wmac(i, j, k) * delta_mns_wmac);

                    new_diag_a(i, j, k) = rho(i, j, k) * (1 + dt * net_coeff);
                });
        }
    }

    for (int lev = 0; lev < nlevels; ++lev) {
        linop.setACoeffs(lev, (*new_diag_ptr)(lev));
    }
}

template <typename LinOp>
void DiffSolverIface<LinOp>::setup_solver(amrex::MLMG& mlmg)
{
    BL_PROFILE("amr-wind::setup_solver");
    // Set all MLMG options
    m_options(mlmg);
}

template <typename LinOp>
void DiffSolverIface<LinOp>::linsys_solve_impl()
{
    FieldState fstate = FieldState::New;
    auto& repo = this->m_pdefields.repo;
    auto& field = this->m_pdefields.field;
    const auto& density = m_density.state(fstate);
    const int nlevels = repo.num_active_levels();
    const int ndim = field.num_comp();
    auto rhs_ptr = repo.create_scratch_field("rhs", field.num_comp(), 0);

    // Always multiply with rho since there is no diffusion term for density
    for (int lev = 0; lev < nlevels; ++lev) {
        auto& rhs = (*rhs_ptr)(lev);

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid();
             ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& rhs_a = rhs.array(mfi);
            const auto& fld = field(lev).const_array(mfi);
            const auto& rho = density(lev).const_array(mfi);

            amrex::ParallelFor(
                bx, ndim,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                    rhs_a(i, j, k, n) = rho(i, j, k) * fld(i, j, k, n);
                });
        }
    }

    amrex::MLMG mlmg(*this->m_solver);
    this->setup_solver(mlmg);

    mlmg.solve(
        field.vec_ptrs(), rhs_ptr->vec_const_ptrs(), this->m_options.rel_tol,
        this->m_options.abs_tol);

    io::print_mlmg_info(field.name() + "_solve", mlmg);
}

template <typename LinOp>
void DiffSolverIface<LinOp>::linsys_solve(const amrex::Real dt)
{
    FieldState fstate = FieldState::New;

    this->setup_operator(*this->m_solver, 1.0, dt, fstate);
    this->linsys_solve_impl();
}

/*template <typename LinOp>
void DiffSolverIface<LinOp>::linsys_solve(const ConvectionType convtype, const amrex::Real dt)
{
    FieldState fstate = FieldState::New;

    this->setup_operator(*this->m_solver, 1.0, dt, fstate, convtype);
    this->linsys_solve_impl();
}
*/

template class DiffSolverIface<amrex::MLABecLaplacian>;
template class DiffSolverIface<amrex::MLTensorOp>;

} // namespace pde
} // namespace amr_wind
