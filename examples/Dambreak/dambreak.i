#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#            SIMULATION STOP            #
#.......................................#
time.stop_time               =   2        # Max (simulated) time to evolve
time.max_step                =   10        # Max number of time steps

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#         TIME STEP COMPUTATION         #
#.......................................#
time.fixed_dt         =   0.001        # Use this constant dt if > 0
time.cfl              =   0.5        # CFL factor

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#            INPUT AND OUTPUT           #
#.......................................#
time.plot_interval  =  10   # Steps between plot files
time.checkpoint_interval =   50  # Steps between checkpoint files
io.restart_file   =   ""  # Checkpoint to restart from 
amrex.throw_exception =1 
amrex.signal_handling = 0
io.outputs = normal intercept vof velocity_src_term velocity_mueff 
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#               PHYSICS                 #
#.......................................#
incflo.use_godunov      = 1
transport.model = TwoPhaseTransport
#transport.viscosity_water= 0.001 
#transport.viscosity_air= 0.001
incflo.rho_air=1.
incflo.rho_water=1000.
turbulence.model = Laminar
ICNS.source_terms = DensityBuoyancy  
incflo.gravity          =   0.  0. -9.81  # Gravitational force (3D)
incflo.surface_tension_coeff=0.0
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#        ADAPTIVE MESH REFINEMENT       #
#.......................................#
amr.n_cell          =   64 16 64    # Grid cells at coarsest AMRlevel
amr.max_level       =   0           # Max AMR level in hierarchy 
amr.blocking_factor = 8
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#              GEOMETRY                 #
#.......................................#
geometry.prob_lo        =   0   0.   0.  # Lo corner coordinates
geometry.prob_hi        =   0.5   0.125  0.5  # Hi corner coordinates
geometry.is_periodic    =   0   1   0   # Periodicity x y z (0/1)

xlo.type =   "slip_wall"
xhi.type =   "slip_wall"
zlo.type =   "slip_wall"
zhi.type =   "slip_wall"


#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#          INITIAL CONDITIONS           #
#.......................................#
incflo.probtype         =   0
incflo.physics = Multiphase
incflo.multiphase_problem = 2
incflo.verbose=3
nodal_proj.mg_rtol=1e-4
nodal_proj.mg_atol=1e-6
nodal_proj.num_pre_smooth = 20
nodal_proj.num_post_smooth = 20
nodal_proj.mg_max_coarsening_level = 10
mac_proj.num_pre_smooth = 20
mac_proj.num_post_smooth = 20
