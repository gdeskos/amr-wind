#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#            SIMULATION STOP            #
#.......................................#
time.stop_time               =   20        # Max (simulated) time to evolve
time.max_step                =   500          # Max number of time steps

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#         TIME STEP COMPUTATION         #
#.......................................#
time.fixed_dt         =   0.001        # Use this constant dt if > 0
time.cfl              =   0.45        # CFL factor

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#            INPUT AND OUTPUT           #
#.......................................#
time.plot_interval  =  10   # Steps between plot files
time.checkpoint_interval =   50  # Steps between checkpoint files
amr.restart   =   ""  # Checkpoint to restart from 
amr.KE_int = 1        # calculate kinetic energy 
amr.plt_volumefraction=1
amr.plt_levelset=1
amrex.throw_exception =1 
amrex.signal_handling = 0
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#               PHYSICS                 #
#.......................................#
incflo.gravity          = 0.  0.  0.  # Gravitational force (3D)
incflo.ro_0             = 1.          # Reference density 
incflo.use_godunov      = 1
incflo.transport = TwoPhaseTransport
transport.visc_water= 0.001 
transport.visc_air= 0.00000148

turbulence.model = Laminar
ICNS.source_terms = SurfaceTension DensityBuoyancy 
incflo.gravity          =   0.  0. -9.81  # Gravitational force (3D)
incflo.surface_tension_coeff=0.07
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#        ADAPTIVE MESH REFINEMENT       #
#.......................................#
amr.n_cell              =   128 16 64   # Grid cells at coarsest AMRlevel
amr.max_level           =   0           # Max AMR level in hierarchy 

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#              GEOMETRY                 #
#.......................................#
geometry.prob_lo        =   0   0.   0.  # Lo corner coordinates
geometry.prob_hi        =   1.168   0.1  0.584  # Hi corner coordinates
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
nodal_proj.verbose=1
