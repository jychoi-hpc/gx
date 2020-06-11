#ifdef USE_MPI
#include "mpi.h"
#else
typedef int MPI_Comm;
#endif

#include "geometry.h"
#include "parameters.h"
#include "grids.h"
#include "fields.h"
#include "moments.h"
#include "solver.h"
#include "forcing.h"
#include "timestepper.h"
#include "linear.h"
#include "diagnostics.h"
#include "cuda_constants.h"
#include "get_error.h"

#ifdef GS2_zonal
extern "C" void broadcast_integer(int* a);
#endif

// MM // GPU Function
void getDeviceMemoryUsage() {
  cudaDeviceSynchronize();
  // show memory usage of GPU
  size_t free_byte ;
  size_t total_byte ;
  cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
  if ( cudaSuccess != cuda_status ){
      printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
      exit(1);
  }
  // for some reason, total_byte returned by above call is not correct. 
  cudaDeviceProp prop;
  int dev;
  checkCuda( cudaGetDevice(&dev) );
  checkCuda( cudaGetDeviceProperties(&prop, dev) );
  double free_db = (double) free_byte;
  double total_db = (double) prop.totalGlobalMem;
  double used_db = total_db - free_db ;
  printf("GPU memory usage: used = %f MB (%f %%), free = %f MB (%f %%), total = %f MB\n",
      used_db/1024.0/1024.0, used_db/total_db*100., free_db/1024.0/1024.0, free_db/total_db*100., total_db/1024.0/1024.0);
}

void run_gx(Parameters *pars, Grids* grids, Geometry* geo, Diagnostics* diagnostics)
{
  int iproc = pars->iproc;  

  // MM // I think most if not all of the constructors are CUDA based
  Fields *fields;
  MomentsG *momsG;
  Solver *solver;
  Linear* linear;
  Nonlinear* nonlinear;
  Forcing *forcing;
  Timestepper *stepper; 

  double time = 0;

  // MM ?? curious why the objects are pointers. Is it related to copying info to GPU?
  // I was thinking that, for example, in fields.h since most of the member variables are pointers, the objects themselves might take up less memory and it could be faster to just use the objects themselves.
  if(iproc == 0) {
    DEBUGPRINT("Initializing fields...\n");           fields = new Fields(grids);
    CUDA_DEBUG("Initializing fields: %s \n");

    DEBUGPRINT("Initializing moments...\n");          momsG = new MomentsG(pars, grids);
    CUDA_DEBUG("Initializing moments: %s \n");

    DEBUGPRINT("Setting initial conditions...\n");    momsG -> initialConditions(geo, &time);
    CUDA_DEBUG("Setting initial conditions: %s \n");

    DEBUGPRINT("Initializing field solver...\n");     solver = new Solver(pars, grids, geo);
    CUDA_DEBUG("Initializing field solver: %s \n");

    // initialize fields using field solve
    DEBUGPRINT("Solving for initial fields...\n");    solver -> fieldSolve(momsG, fields);
    CUDA_DEBUG("Solving for initial fields: %s\n");

    DEBUGPRINT("Initializing equations...\n");
    DEBUGPRINT("\tLinear terms...\n");                linear = new Linear(pars, grids, geo);
    CUDA_DEBUG("\t Linear terms: %s \n");

    if(!pars->linear) {
      DEBUGPRINT("\tNonlinear terms...\n");           nonlinear = new Nonlinear(pars, grids, geo);
      CUDA_DEBUG("\tNonlinear terms: %s \n");
    } else {
      nonlinear = NULL;
    }

    if (pars->forcing_init) {
      DEBUGPRINT("Initializing forcing...\n");
      if(strcmp(pars->forcing_type, "Kz") == 0) {                 forcing = new KzForcing(pars);
	CUDA_DEBUG("Initializing Kz Forcing: %s \n");
      } else if (strcmp(pars->forcing_type, "KzImpulse") == 0) {  forcing = new KzForcingImpulse(pars);
	CUDA_DEBUG("Initializing Kz Forcing Impulse: %s \n");
      } else if (strcmp(pars->forcing_type, "general") == 0) {    forcing = new genForcing(pars);
	CUDA_DEBUG("Initializing general Forcing: %s \n");
      } else {
	forcing = NULL;
      }
    } else {
      forcing = NULL;
    }
    checkCuda(cudaGetLastError());
    
    DEBUGPRINT("Initializing timestepper...\n");
    
    if(pars->scheme_opt == K10) {stepper = new Ketcheson10(linear, nonlinear, solver, pars, grids, forcing, pars->dt);
      CUDA_DEBUG("Initalizing timestepper K10: %s\n");
    }
    if(pars->scheme_opt == RK4) {stepper = new RungeKutta4(linear, nonlinear, solver, pars, grids, forcing, pars->dt);
      CUDA_DEBUG("Initalizing timestepper RK4: %s\n");
    }
    if(pars->scheme_opt == RK3) {stepper = new RungeKutta3(linear, nonlinear, solver, pars, grids, forcing, pars->dt);
      CUDA_DEBUG("Initalizing timestepper RK3: %s\n");
    }
    if(pars->scheme_opt == RK2) {stepper = new RungeKutta2(linear, nonlinear, solver, pars, grids, forcing, pars->dt);
      CUDA_DEBUG("Initalizing timestepper RK2: %s\n");
    }
    if(pars->scheme_opt == SSPX2) {stepper = new SSPx2(linear, nonlinear, solver, pars, grids, forcing, pars->dt);
      CUDA_DEBUG("Initalizing timestepper SSPx2: %s\n");
    }
    checkCuda(cudaGetLastError()); // MM // GPU
      
    DEBUGPRINT("After initialization:\n");
    getDeviceMemoryUsage(); // MM // GPU
  
    if (pars->write_moms) diagnostics->write_init(momsG, fields);
  }
  
  // TIMESTEP LOOP
  int counter = 0;

  // MM // GPU
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float timer = 0;
  cudaEventRecord(start,0);

  bool checkstop = false;
  DEBUGPRINT("Running %d timesteps.......\n", pars->nstep);
  DEBUGPRINT("dt = %f\n", stepper->get_dt());

  diagnostics->loop_diagnostics(momsG, fields, stepper->get_dt(), counter, time);

  while(counter<pars->nstep) {
    counter++;
    if(iproc==0) {
      stepper->advance(&time, momsG, fields);
      checkstop = diagnostics->loop_diagnostics(momsG, fields, stepper->get_dt(), counter, time);
      if (checkstop) break;
    }
  }

  if (pars->save_for_restart) {
    momsG->restart_write(&time);
  }
  
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timer,start,stop);

  DEBUGPRINT("Step %d\n", counter);
  DEBUGPRINT("Finished timestep loop\n");
  CUDA_DEBUG("Finishing up: %s \n");

  printf("Total runtime = %f s (%f s / timestep)\n", timer/1000., timer/1000./counter);

  diagnostics->final_diagnostics(momsG, fields);

  DEBUGPRINT("Cleaning up...\n");

  delete fields;
  delete momsG;
  delete solver;
  delete linear;
  delete nonlinear;
  delete forcing;
  delete stepper;
}    




