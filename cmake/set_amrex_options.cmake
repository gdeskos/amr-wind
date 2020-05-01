#Set amrex options
set(USE_XSDK_DEFAULTS OFF)
set(DIM 3)
set(ENABLE_PIC OFF)
set(ENABLE_MPI ${AMR_WIND_ENABLE_MPI})
set(ENABLE_OMP ${AMR_WIND_ENABLE_OPENMP})
set(ENABLE_DP ON)
set(ENABLE_EB OFF)
set(ENABLE_FORTRAN_INTERFACES OFF)
set(ENABLE_LINEAR_SOLVERS ON)
set(ENABLE_AMRDATA OFF)
set(ENABLE_PARTICLES OFF)
set(ENABLE_SENSEI_INSITU OFF)
set(ENABLE_CONDUIT OFF)
set(ENABLE_SUNDIALS OFF)
set(ENABLE_FPE OFF)
set(ENABLE_ASSERTIONS OFF)
set(ENABLE_BASE_PROFILE OFF)
set(ENABLE_TINY_PROFILE OFF)
set(ENABLE_TRACE_PROFILE OFF)
set(ENABLE_MEM_PROFILE OFF)
set(ENABLE_COMM_PROFILE OFF)
set(ENABLE_BACKTRACE OFF)
set(ENABLE_PROFPARSER OFF)
set(ENABLE_CUDA ${AMR_WIND_ENABLE_CUDA})
set(ENABLE_ACC OFF)
set(ENABLE_PLOTFILE_TOOLS ${AMR_WIND_ENABLE_FCOMPARE})
set(ENABLE_FORTRAN OFF)
