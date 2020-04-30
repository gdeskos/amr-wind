#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#            SIMULATION STOP            #
#.......................................#
time.stop_time               =   20        # Max (simulated) time to evolve
time.max_step                =   200          # Max number of time steps

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#         TIME STEP COMPUTATION         #
#.......................................#
time.fixed_dt         =   0.1        # Use this constant dt if > 0
time.cfl              =   0.45        # CFL factor

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#            INPUT AND OUTPUT           #
#.......................................#
time.plot_interval  =  20   # Steps between plot files
time.checkpoint_interval =   -1  # Steps between checkpoint files
amr.restart   =   ""  # Checkpoint to restart from 
amr.KE_int = 1        # calculate kinetic energy 
amr.plt_volumefraction=1
amr.plt_levelset=1
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#               PHYSICS                 #
#.......................................#
incflo.gravity          = 0.  0.  0.  # Gravitational force (3D)
incflo.ro_0             = 1.          # Reference density 
incflo.use_godunov      = 1
transport.viscosity = 0.00009947183943 
transport.laminar_prandtl = 1.0
turbulence.model = Laminar
ICNS.source_terms = SurfaceTension 

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#        ADAPTIVE MESH REFINEMENT       #
#.......................................#
amr.n_cell              =   64 64 64   # Grid cells at coarsest AMRlevel
amr.max_level           =   0           # Max AMR level in hierarchy 

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#              GEOMETRY                 #
#.......................................#
geometry.prob_lo        =   -5  -5.  -5.  # Lo corner coordinates
geometry.prob_hi        =    5   5.   5.  # Hi corner coordinates
geometry.is_periodic    =   1   1   1   # Periodicity x y z (0/1)

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#          INITIAL CONDITIONS           #
#.......................................#
incflo.probtype         =   0
incflo.physics = Multiphase
inflow.multiphase_problem = 1
