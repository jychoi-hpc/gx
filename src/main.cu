#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#include "run_gx.h"
#include "version.h"
#include "helper_cuda.h"
// #include "reservoir.h"
#include "reductions.h"
#include <fenv.h>

int main(int argc, char* argv[])
{

  //  feenableexcept(FE_INVALID | FE_OVERFLOW);

  MPI_Init(&argc, &argv);
  MPI_Comm mpcom = MPI_COMM_WORLD;
  int iproc, nprocs;
  MPI_Comm_rank(mpcom, &iproc);
  MPI_Comm_size(mpcom, &nprocs);
  
  int devid = 0; // This should be determined (optionally) on the command line
  int nGPUs = 0;
  checkCuda(cudaGetDeviceCount(&nGPUs));
  checkCuda(cudaSetDevice(iproc%nGPUs));
  cudaDeviceSynchronize();

  char run_name[1000];
  if ( argc < 1) {
    fprintf(stderr, "The correct usage is:\n gx <runname>.in\n");
    exit(1);
  } else {    
    // if input filename ends in .in, remove .in
    if(strlen(argv[1]) > 3 && !strcmp(argv[1] + strlen(argv[1]) - 3, ".in")) {
      strncpy(run_name, argv[1], strlen(argv[1])-3);
      run_name[strlen(argv[1])-3] = '\0';
    } else {
      fprintf(stderr, "Argument for input filename must now include \".in\". Try:\n %s %s.in\n", argv[0], argv[1]);
      exit(1);
    }

    printf("Running %s \n",run_name);
  }
   
  printf("Version: %s \t Compiled: %s \n", build_git_sha, build_git_time);

  Parameters * pars = nullptr;
  pars = new Parameters(iproc, nprocs, mpcom);
  pars->get_nml_vars(run_name);
  
  Grids * grids = nullptr;
  
  DEBUGPRINT("Initializing grids...\n");
  grids = new Grids(pars);
  CUDA_DEBUG("Initializing grids: %s \n");

  DEBUGPRINT("Local grid dimensions on GPU %d: Nx=%d, Ny=%d, Nz=%d, Nl=%d, Nm=%d, Nspecies=%d\n",
	     grids->iproc, grids->Nx, grids->Ny, grids->Nz, grids->Nl, grids->Nm, grids->Nspecies);

  Geometry    * geo         = nullptr;
  Diagnostics * diagnostics = nullptr;

  if (pars->gx) {
    geo = init_geo(pars, grids);


    DEBUGPRINT("Initializing diagnostics...\n");
    diagnostics = new Diagnostics_GK(pars, grids, geo);
    CUDA_DEBUG("Initializing diagnostics: %s \n");    

    //    DEBUGPRINT("Initializing Hermite transforms...\n");
    //    herm = new HermiteTransform(grids, 1); // batch size could ultimately be nspec
    //    CUDA_DEBUG("Initializing Hermite transforms: %s \n");    
  }
  if (pars->krehm) {
    geo = init_geo(pars, grids);
    diagnostics = new Diagnostics_KREHM(pars, grids);
  }

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  
  run_gx(pars, grids, geo, diagnostics); 

  delete pars;
  delete grids;
  delete geo;
  delete diagnostics;

  MPI_Finalize();
  cudaDeviceReset();
}
