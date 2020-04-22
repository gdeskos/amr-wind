#include <incflo.H>
#include "PlaneAveraging.H"
#include "Physics.H"
#include <cmath>
#include "field_ops.H"
#include "Godunov.H"
#include "MOL.H"
#include "PDEBase.H"
#include "mac_projection.H"
#include "diffusion.H"
#include "TurbulenceModel.H"

using namespace amrex;

void incflo::Advance()
{
    BL_PROFILE("incflo::Advance");

    // Start timing current time step
    Real strt_step = ParallelDescriptor::second();

    // Compute time step size
    bool explicit_diffusion = (m_diff_type == DiffusionType::Explicit);
    ComputeDt(explicit_diffusion);

    velocity().advance_states();
    density().advance_states();
    tracer().advance_states();

    velocity().state(amr_wind::FieldState::Old).fillpatch(m_time.current_time());
    density().state(amr_wind::FieldState::Old).fillpatch(m_time.current_time());
    tracer().state(amr_wind::FieldState::Old).fillpatch(m_time.current_time());

    for (auto& pp: m_physics)
        pp->pre_advance_work();
    
    ApplyPredictor();

    if (!m_use_godunov) {

        velocity().state(amr_wind::FieldState::New).fillpatch(m_time.new_time());
        density().state(amr_wind::FieldState::New).fillpatch(m_time.new_time());
        tracer().state(amr_wind::FieldState::New).fillpatch(m_time.new_time());

        ApplyCorrector();
    }

    if (m_verbose > 1)
    {
        amrex::Print() << "End of time step: " << std::endl;
        PrintMaxValues(m_time.new_time());
    }

#if 0
    if (m_test_tracer_conservation) {
        amrex::Print() << "Sum tracer volume wgt = " << m_time.current_time()+dt << "   " << volWgtSum(0,*tracer[0],0) << std::endl;
    }
#endif

    // Stop timing current time step
    Real end_step = ParallelDescriptor::second() - strt_step;
    ParallelDescriptor::ReduceRealMax(end_step, ParallelDescriptor::IOProcessorNumber());
    if (m_verbose > 0)
    {
        amrex::Print() << "Time per step " << end_step << std::endl;
    }
}

//
// Apply predictor:
//
//  1. Use u = vel_old to compute
//
//      conv_u  = - u grad u
//      conv_r  = - div( u rho  )
//      conv_t  = - div( u trac )
//      eta_old     = visosity at m_time.current_time()
//      if (m_diff_type == DiffusionType::Explicit)
//         divtau _old = div( eta ( (grad u) + (grad u)^T ) ) / rho^n
//         rhs = u + dt * ( conv + divtau_old )
//      else
//         divtau_old  = 0.0
//         rhs = u + dt * conv
//
//      eta     = eta at new_time
//
//  2. Add explicit forcing term i.e. gravity + lagged pressure gradient
//
//      rhs += dt * ( g - grad(p + p0) / rho^nph )
//
//      Note that in order to add the pressure gradient terms divided by rho,
//      we convert the velocity to momentum before adding and then convert them back.
//
//  3. A. If (m_diff_type == DiffusionType::Implicit)
//        solve implicit diffusion equation for u*
//
//     ( 1 - dt / rho^nph * div ( eta grad ) ) u* = u^n + dt * conv_u
//                                                  + dt * ( g - grad(p + p0) / rho^nph )
//
//     B. If (m_diff_type == DiffusionType::Crank-Nicolson)
//        solve semi-implicit diffusion equation for u*
//
//     ( 1 - (dt/2) / rho^nph * div ( eta_old grad ) ) u* = u^n +
//            dt * conv_u + (dt/2) / rho * div (eta_old grad) u^n
//          + dt * ( g - grad(p + p0) / rho^nph )
//
//  4. Apply projection
//
//     Add pressure gradient term back to u*:
//
//      u** = u* + dt * grad p / rho^nph
//
//     Solve Poisson equation for phi:
//
//     div( grad(phi) / rho^nph ) = div( u** )
//
//     Update pressure:
//
//     p = phi / dt
//
//     Update velocity, now divergence free
//
//     vel = u** - dt * grad p / rho^nph
//
// It is assumed that the ghost cels of the old data have been filled and
// the old and new data are the same in valid region.
//
void incflo::ApplyPredictor (bool incremental_projection)
{
    BL_PROFILE("incflo::ApplyPredictor");

    // We use the new time value for things computed on the "*" state
    Real new_time = m_time.new_time();

    if (m_verbose > 2)
    {
        amrex::Print() << "Before predictor step:" << std::endl;
        PrintMaxValues(new_time);
    }

    auto& icns_fields = icns().fields();
    auto& velocity_old = velocity().state(amr_wind::FieldState::Old);
    auto& velocity_new = velocity().state(amr_wind::FieldState::New);
    auto& density_old = density().state(amr_wind::FieldState::Old);
    auto& density_new = density().state(amr_wind::FieldState::New);
    auto& tracer_old = tracer().state(amr_wind::FieldState::Old);
    auto& tracer_new = tracer().state(amr_wind::FieldState::New);

    auto& velocity_forces = icns_fields.src_term;
    // only the old states are used in predictor
    auto& divtau = m_use_godunov
                       ? icns_fields.diff_term
                       : icns_fields.diff_term.state(amr_wind::FieldState::Old);

    // Ensure that density and tracer exists at half time
    auto& density_nph = density_new.create_state(amr_wind::FieldState::NPH);
    auto& tracer_nph = tracer_new.create_state(amr_wind::FieldState::NPH);

    // *************************************************************************************
    // Define the forcing terms to use in the Godunov prediction
    // *************************************************************************************
    if (m_use_godunov)
    {
        compute_vel_forces(velocity_forces.vec_ptrs(),
                           velocity_old.vec_const_ptrs(),
                           density_old.vec_const_ptrs(),
                           tracer_old.vec_const_ptrs());

        for (auto& seqn: scalar_eqns()) {
            seqn->compute_source_term(amr_wind::FieldState::Old);
        }
    }

    // *************************************************************************************
    // Compute viscosity / diffusive coefficients
    // *************************************************************************************
    m_sim.turbulence_model().update_turbulent_viscosity(amr_wind::FieldState::Old);
    icns().compute_mueff(amr_wind::FieldState::Old);
    for (auto& eqns: scalar_eqns())
        eqns->compute_mueff(amr_wind::FieldState::Old);

    // *************************************************************************************
    // Compute explicit viscous term
    // *************************************************************************************
    if (need_divtau()) {
        // Reuse existing buffer to avoid creating new multifabs
        amr_wind::field_ops::copy(velocity_new, velocity_old, 0, 0, velocity_new.num_comp(), 1);
        if (m_wall_model_flag) {
            diffusion::wall_model_bc(velocity_new, m_utau_mean_ground,
                                     m_velocity_mean_ground,
                                     amr_wind::FieldState::Old);
        }
        icns().compute_diffusion_term(amr_wind::FieldState::Old);
        if (m_use_godunov)
            amr_wind::field_ops::add(velocity_forces, divtau, 0, 0, AMREX_SPACEDIM, 0);
    }



    // *************************************************************************************
    // Compute explicit diffusive terms
    // *************************************************************************************
    if (need_divtau()) {
        for (auto& eqn: scalar_eqns()) {
            auto& field = eqn->fields().field;
            // Reuse existing buffer to avoid creating new multifabs
            amr_wind::field_ops::copy(field, field.state(amr_wind::FieldState::Old),
                                      0, 0, field.num_comp(), 1);

            // FIXME: Hard-coded BC
            if (field.name() == "temperature")
                diffusion::heat_flux_bc(field);

            eqn->compute_diffusion_term(amr_wind::FieldState::Old);

            if (m_use_godunov)
                amr_wind::field_ops::add(
                    eqn->fields().src_term,
                    eqn->fields().diff_term, 0, 0,
                    field.num_comp(), 0);
        }
    }

    if (m_use_godunov) {
       IntVect ng(nghost_force());
       icns().fields().src_term.fillpatch(m_time.current_time(), ng);

       for (auto& eqn: scalar_eqns()) {
           eqn->fields().src_term.fillpatch(m_time.current_time(), ng);
       }
    }

    // *************************************************************************************
    // if ( m_use_godunov) Compute the explicit advective terms R_u^(n+1/2), R_s^(n+1/2) and R_t^(n+1/2)
    // if (!m_use_godunov) Compute the explicit advective terms R_u^n      , R_s^n       and R_t^n
    // Note that "get_conv_tracer_old" returns div(rho u tracer)
    // *************************************************************************************

    icns().compute_advection_term(amr_wind::FieldState::Old);

    for (auto& seqn: scalar_eqns()) {
        seqn->compute_advection_term(amr_wind::FieldState::Old);
    }

    // *************************************************************************************
    // Update density first
    // *************************************************************************************
    if (m_constant_density)
    {
        amr_wind::field_ops::copy(density_nph, density_old, 0, 0, 1, 1);
    }

    // Perform scalar update one at a time. This is to allow an updated density
    // at `n+1/2` to be computed before other scalars use it when computing
    // their source terms.
    for (auto& eqn: scalar_eqns()) {
        // Compute (recompute for Godunov) the scalar forcing terms
        eqn->compute_source_term(amr_wind::FieldState::NPH);

        // Update the scalar (if explicit), or the RHS for implicit/CN
        eqn->compute_predictor_rhs(m_diff_type);

        auto& field = eqn->fields().field;
        if (m_diff_type != DiffusionType::Explicit) {
            IntVect ng_diffusion(1);
            field.fillphysbc(new_time, ng_diffusion);

            amrex::Real dt_diff = (m_diff_type == DiffusionType::Implicit)
                ? m_time.deltaT() : 0.5 * m_time.deltaT();

            // FIXME: Hardcoded BC
            if (field.name() == "temperature")
                diffusion::heat_flux_bc(field);

            // Solve diffusion eqn. and update of the scalar field
            eqn->solve(dt_diff);
        }

        // Update scalar at n+1/2
        amr_wind::field_ops::lincomb(
            field.state(amr_wind::FieldState::NPH),
            0.5, field.state(amr_wind::FieldState::Old), 0,
            0.5, field, 0, 0, field.num_comp(), 1);
    }

    // *************************************************************************************
    // Define (or if use_godunov, re-define) the forcing terms, without the viscous terms
    //    and using the half-time density
    // *************************************************************************************
    compute_vel_forces(velocity_forces.vec_ptrs(),
                       velocity_old.vec_const_ptrs(),
                       (density_nph).vec_const_ptrs(),
                       (tracer_nph).vec_const_ptrs());
    
    // *************************************************************************************
    // Update the velocity
    // *************************************************************************************
    icns().compute_predictor_rhs(m_diff_type);

    // *************************************************************************************
    // Solve diffusion equation for u* but using eta_old at old time
    // *************************************************************************************
    if (m_diff_type == DiffusionType::Crank_Nicolson || m_diff_type == DiffusionType::Implicit)
    {
        IntVect ng_diffusion(1);
        velocity_new.fillphysbc(new_time, ng_diffusion);
        density_new.fillphysbc(new_time, ng_diffusion);

        Real dt_diff = (m_diff_type == DiffusionType::Implicit) ? m_time.deltaT() : 0.5*m_time.deltaT();
        if (m_wall_model_flag) {
            diffusion::wall_model_bc(velocity_new, m_utau_mean_ground,
                                     m_velocity_mean_ground,
                                     amr_wind::FieldState::New);
        }
        icns().solve(dt_diff);
    }

    // **********************************************************************************************
    //
    // Project velocity field, update pressure
    //
    // **********************************************************************************************
    ApplyProjection((density_nph).vec_const_ptrs(), new_time, m_time.deltaT(), incremental_projection);

}


//
// Apply corrector:
//
//  Output variables from the predictor are labelled _pred
//
//  1. Use u = vel_pred to compute
//
//      conv_u  = - u grad u
//      conv_r  = - u grad rho
//      conv_t  = - u grad trac
//      eta     = viscosity
//      divtau  = div( eta ( (grad u) + (grad u)^T ) ) / rho
//
//      conv_u  = 0.5 (conv_u + conv_u_pred)
//      conv_r  = 0.5 (conv_r + conv_r_pred)
//      conv_t  = 0.5 (conv_t + conv_t_pred)
//      if (m_diff_type == DiffusionType::Explicit)
//         divtau  = divtau at new_time using (*) state
//      else
//         divtau  = 0.0
//      eta     = eta at new_time
//
//     rhs = u + dt * ( conv + divtau )
//
//  2. Add explicit forcing term i.e. gravity + lagged pressure gradient
//
//      rhs += dt * ( g - grad(p + p0) / rho )
//
//      Note that in order to add the pressure gradient terms divided by rho,
//      we convert the velocity to momentum before adding and then convert them back.
//
//  3. A. If (m_diff_type == DiffusionType::Implicit)
//        solve implicit diffusion equation for u*
//
//     ( 1 - dt / rho * div ( eta grad ) ) u* = u^n + dt * conv_u
//                                                  + dt * ( g - grad(p + p0) / rho )
//
//     B. If (m_diff_type == DiffusionType::Crank-Nicolson)
//        solve semi-implicit diffusion equation for u*
//
//     ( 1 - (dt/2) / rho * div ( eta grad ) ) u* = u^n + dt * conv_u + (dt/2) / rho * div (eta_old grad) u^n
//                                                      + dt * ( g - grad(p + p0) / rho )
//
//  4. Apply projection
//
//     Add pressure gradient term back to u*:
//
//      u** = u* + dt * grad p / rho
//
//     Solve Poisson equation for phi:
//
//     div( grad(phi) / rho ) = div( u** )
//
//     Update pressure:
//
//     p = phi / dt
//
//     Update velocity, now divergence free
//
//     vel = u** - dt * grad p / rho
//
void incflo::ApplyCorrector()
{
    BL_PROFILE("incflo::ApplyCorrector");

    // We use the new time value for things computed on the "*" state
    Real new_time = m_time.new_time();

    if (m_verbose > 2)
    {
        amrex::Print() << "Before corrector step:" << std::endl;
        PrintMaxValues(new_time);
    }

    auto& velocity_new = velocity().state(amr_wind::FieldState::New);
    auto& density_old = density().state(amr_wind::FieldState::Old);
    auto& density_new = density().state(amr_wind::FieldState::New);

    auto& velocity_forces = m_repo.get_field("velocity_src_term");

    // Allocate scratch space for half time density and tracer
    auto& density_nph = density().state(amr_wind::FieldState::NPH);
    auto& tracer_nph = tracer().state(amr_wind::FieldState::NPH);

    // **********************************************************************************************
    // Compute the explicit "new" advective terms R_u^(n+1,*), R_r^(n+1,*) and R_t^(n+1,*)
    // Note that "get_conv_tracer_new" returns div(rho u tracer)
    // We only reach the corrector if !m_use_godunov which means we don't use the forces
    // in constructing the advection term
    // *************************************************************************************

    icns().compute_advection_term(amr_wind::FieldState::New);

    for (auto& seqn: scalar_eqns()) {
        seqn->compute_advection_term(amr_wind::FieldState::New);
    }

    // *************************************************************************************
    // Compute viscosity / diffusive coefficients
    // *************************************************************************************
    m_sim.turbulence_model().update_turbulent_viscosity(amr_wind::FieldState::New);
    icns().compute_mueff(amr_wind::FieldState::New);
    for (auto& eqns: scalar_eqns())
        eqns->compute_mueff(amr_wind::FieldState::New);

    // Here we create divtau of the (n+1,*) state that was computed in the predictor;
    //      we use this laps only if DiffusionType::Explicit
    if (m_diff_type == DiffusionType::Explicit) {
        if (m_wall_model_flag) {
            diffusion::wall_model_bc(velocity_new, m_utau_mean_ground,
                                     m_velocity_mean_ground,
                                     amr_wind::FieldState::New);
        }
        icns().compute_diffusion_term(amr_wind::FieldState::New);

        for (auto& eqns: scalar_eqns()) {
            // FIXME: Hard-coded BC
            if (eqns->fields().field.name() == "temperature")
                diffusion::heat_flux_bc(eqns->fields().field);
            eqns->compute_diffusion_term(amr_wind::FieldState::New);
        }
    }

    // *************************************************************************************
    // Update density first
    // *************************************************************************************
    if (m_constant_density) {
        amr_wind::field_ops::copy(density_nph, density_old, 0, 0, 1, 1);
    }

    // Perform scalar update one at a time. This is to allow an updated density
    // at `n+1/2` to be computed before other scalars use it when computing
    // their source terms.
    for (auto& eqn: scalar_eqns()) {
        // Compute (recompute for Godunov) the scalar forcing terms
        // Note this is (rho * scalar) and not just scalar
        eqn->compute_source_term(amr_wind::FieldState::New);

        // Update (note that dtdt already has rho in it)
        // (rho trac)^new = (rho trac)^old + dt * (
        //                   div(rho trac u) + div (mu grad trac) + rho * f_t
        eqn->compute_corrector_rhs(m_diff_type);

        auto& field = eqn->fields().field;
        if (m_diff_type != DiffusionType::Explicit) {
            IntVect ng_diffusion(1);
            field.fillphysbc(new_time, ng_diffusion);

            amrex::Real dt_diff = (m_diff_type == DiffusionType::Implicit)
                ? m_time.deltaT() : 0.5 * m_time.deltaT();

            // FIXME: Hardcoded BC
            if (field.name() == "temperature")
                diffusion::heat_flux_bc(field);

            // Solve diffusion eqn. and update of the scalar field
            eqn->solve(dt_diff);
        }

        // Update scalar at n+1/2
        amr_wind::field_ops::lincomb(
            field.state(amr_wind::FieldState::NPH),
            0.5, field.state(amr_wind::FieldState::Old), 0,
            0.5, field, 0, 0, field.num_comp(), 1);
    }

    // *************************************************************************************
    // Define the forcing terms to use in the final update (using half-time density)
    // *************************************************************************************
    compute_vel_forces(velocity_forces.vec_ptrs(),velocity_new.vec_const_ptrs(),
                       (density_nph).vec_const_ptrs(),
                       (tracer_nph).vec_const_ptrs());

    // *************************************************************************************
    // Update velocity
    // *************************************************************************************
    icns().compute_corrector_rhs(m_diff_type);

    // **********************************************************************************************
    //
    // Solve diffusion equation for u* at t^{n+1} but using eta at predicted new time
    //
    // **********************************************************************************************

    if (m_diff_type == DiffusionType::Crank_Nicolson || m_diff_type == DiffusionType::Implicit)
    {
        IntVect ng_diffusion(1);
        velocity_new.fillphysbc(new_time, ng_diffusion);
        density_new.fillphysbc(new_time, ng_diffusion);

        Real dt_diff = (m_diff_type == DiffusionType::Implicit) ? m_time.deltaT() : 0.5*m_time.deltaT();
        if (m_wall_model_flag) {
            diffusion::wall_model_bc(velocity_new, m_utau_mean_ground,
                                     m_velocity_mean_ground,
                                     amr_wind::FieldState::New);
        }
        icns().solve(dt_diff);
    }

    // **********************************************************************************************
    //
    // Project velocity field, update pressure
    bool incremental = false;
    ApplyProjection((density_nph).vec_const_ptrs(),new_time, m_time.deltaT(), incremental);

}
