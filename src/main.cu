#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#include <ctime>
#include "run_gx.h"
#include "version.h"
// #include "reservoir.h"
#include "reductions.h"
#include <fenv.h>
#include <limits.h>

int main(int argc, char* argv[])
{
  // Uncomment the following line to catch NaN and overflow errors at runtime
  //  feenableexcept(FE_INVALID | FE_OVERFLOW);

  MPI_Init(&argc, &argv);
  MPI_Comm mpcom = MPI_COMM_WORLD;
  int iproc, nprocs;
  MPI_Comm_rank(mpcom, &iproc);
  MPI_Comm_size(mpcom, &nprocs);
  
  int nGPUs = 0;
  checkCuda(cudaGetDeviceCount(&nGPUs));
  checkCuda(cudaSetDevice(iproc%nGPUs));
  cudaDeviceSynchronize();

  char run_name[1000];
  if ( argc != 2 ) {
    if(iproc==0)
        fprintf(stderr, "The correct usage is:\n gx <runname>.in\n");
    exit(1);
  } else {
    // if input filename ends in .in, remove .in

    size_t arglen = strnlen( argv[1], NAME_MAX );
    if( arglen > 3 && !strcmp(argv[1] + arglen - 3, ".in")) {
      strncpy(run_name, argv[1], arglen-3);
      run_name[arglen-3] = '\0';
    } else {
      if(iproc==0) fprintf(stderr, "Argument for input filename must include \".in\". Try:\n %s %s.in\n", argv[0], argv[1]);
      exit(1);
    }

    printf(ANSI_COLOR_GREEN);
    if(iproc==0) printf("Running %s \n",run_name);
    std::time_t time = std::time(0);
    if(iproc==0) std::cout << "Start time: " << std::ctime(&time) << std::endl;
    printf(ANSI_COLOR_RESET);
  }
   
  if(iproc==0) printf("Version: %s \t Compiled: %s \n", build_git_sha, build_git_time);
 
  // 
  // Read the input file by instantiating and using a Parameters object
  // 
  Parameters * pars = nullptr;
  pars = new Parameters(iproc, nprocs, mpcom);
  pars->get_nml_vars(run_name);

  //
  // Initialize the computational grid by instantiating and using a Grids object
  //
  Grids * grids = nullptr;
  
  if(iproc==0) DEBUGPRINT("Initializing grids...\n");
  grids = new Grids(pars);
  if(iproc==0) CUDA_DEBUG("Initializing grids: %s \n");

  if(iproc==0) DEBUGPRINT("Local grid dimensions on GPU %d: Nx=%d, Ny=%d, Nz=%d, Nl=%d, Nm=%d, Nspecies=%d\n",
	     grids->iproc, grids->Nx, grids->Ny, grids->Nz, grids->Nl, grids->Nm, grids->Nspecies);

  //
  // Prepare to define the various coefficients that determine the geometry of the simulation
  //
  Geometry    * geo         = nullptr;

  geo = init_geo(pars, grids);

  //
  // Hold here until all threads are ready to continue
  // 
  cudaDeviceSynchronize();

  //
  // Check for a class of Cuda errors
  // 
  checkCuda(cudaGetLastError());

  //
  // Run the calculation
  // 
  run_gx(pars, grids, geo); 

  //
  // This way of measuring runtime is only appropriate for large time intervals.
  // There are more specific ways to get precise timings, especially when there 
  // are kernel calls. 
  // 
  std::time_t time = std::time(0);
  if(iproc==0) std::cout << "End time: " << std::ctime(&time) << std::endl;

  delete pars;
  delete grids;
  delete geo;

  MPI_Finalize();
  cudaDeviceReset();
}
