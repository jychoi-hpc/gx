#include "run_gx.h"

void getDeviceMemoryUsage();

void run_gx(Parameters *pars, Grids *grids, Geometry *geo, Diagnostics *diagnostics)
{
  double time = 0;

  Fields    * fields    = nullptr;
  Solver    * solver    = nullptr;
  Linear    * linear    = nullptr;
  Nonlinear * nonlinear = nullptr;
  MomentsG  ** G = (MomentsG**) malloc(sizeof(void*)*grids->Nspecies);
  for(int is=0; is<grids->Nspecies; is++) {
    G[is] = nullptr;
  }
  Forcing   * forcing   = nullptr;
  
  // set up moments and fields objects
  for(int is=0; is<grids->Nspecies; is++) {
    G[is] = new MomentsG (pars, grids, is);
  }
  fields = new Fields(pars, grids);               
  
  /////////////////////////////////
  //                             //
  // Initialize eqs              // 
  //                             //
  /////////////////////////////////
  if (pars->gx) {
    linear = new Linear_GK(pars, grids, geo);          
    if (!pars->linear) nonlinear = new Nonlinear_GK(pars, grids, geo); 

    solver = new Solver_GK(pars, grids, geo);    

    if (pars->forcing_init) {
      if (pars->forcing_type == "Kz")        forcing = new KzForcing(pars);        
      if (pars->forcing_type == "KzImpulse") forcing = new KzForcingImpulse(pars); 
      if (pars->forcing_type == "general")   forcing = new genForcing(pars);       
    }

    // set up initial conditions
    for(int is=0; is<grids->Nspecies; is++) {
      if(pars->init_electrons_only && pars->species_h[is].type!=1) continue;
      G[is] -> initialConditions(&time);   
    }
    solver -> fieldSolve(G, fields);                
  }

  if (pars->krehm) {
    linear = new Linear_KREHM(pars, grids);          
    if (!pars->linear) nonlinear = new Nonlinear_KREHM(pars, grids);    

    solver = new Solver_KREHM(pars, grids);

    // set up initial conditions
    G[0] -> initialConditions(&time);   
    solver -> fieldSolve(G, fields);                
  }

  //////////////////////////////
  //                          //
  // Kuramoto-Sivashinsky eq  // 
  //                          //
  //////////////////////////////  
  if (pars->ks) {
    linear    = new Linear_KS(pars, grids);    
    if (!pars->linear) nonlinear = new Nonlinear_KS(pars, grids);

    // no field solve for K-S

    // set up initial conditions
    G[0] -> initialConditions(&time);
    //    G -> qvar(grids->Naky);
  }    

  //////////////////////////////
  //                          //
  // Vlasov-Poisson           // 
  //                          //
  //////////////////////////////  
  if (pars->vp) {
    linear    = new Linear_VP(pars, grids);    
    if (!pars->linear) nonlinear = new Nonlinear_VP(pars, grids);

    solver = new Solver_VP(pars, grids);    

    // set up initial conditions
    G[0] -> initVP(&time);
    solver -> fieldSolve(G, fields);
  }    

  Timestepper * timestep;
  switch (pars->scheme_opt)
    {
    case Tmethod::k10   : timestep = new Ketcheson10 (linear, nonlinear, solver, pars, grids, forcing, pars->dt); break;
    case Tmethod::k2    : timestep = new K2          (linear, nonlinear, solver, pars, grids, forcing, pars->dt); break;
    case Tmethod::g3    : timestep = new G3          (linear, nonlinear, solver, pars, grids, forcing, pars->dt); break;
    case Tmethod::rk4   : timestep = new RungeKutta4 (linear, nonlinear, solver, pars, grids, forcing, pars->dt); break;
    case Tmethod::rk2   : timestep = new RungeKutta2 (linear, nonlinear, solver, pars, grids, forcing, pars->dt); break;
    case Tmethod::sspx2 : timestep = new SSPx2       (linear, nonlinear, solver, pars, grids, forcing, pars->dt); break;
    case Tmethod::sspx3 : timestep = new SSPx3       (linear, nonlinear, solver, pars, grids, forcing, pars->dt); break;
    case Tmethod::ssprk3 : timestep = new SSPRK3     (linear, nonlinear, solver, pars, grids, forcing, pars->dt); break;
    }

  getDeviceMemoryUsage();
  
  //  if (pars->write_moms) diagnostics -> write_init(G, fields);
	 
  // TIMESTEP LOOP
  int counter = 0;           float timer = 0;          cudaEvent_t start, stop;    bool checkstop = false;
  cudaEventCreate(&start);   cudaEventCreate(&stop);   cudaEventRecord(start,0);
  bool bvar; 
  bvar = diagnostics -> loop(G, fields, timestep->get_dt(), counter, time);
  
  while(counter<pars->nstep) {
    counter++;

    timestep -> advance(&time, G, fields);
    checkstop = diagnostics -> loop(G, fields, timestep->get_dt(), counter, time);
    if (checkstop) break;
    if (counter % pars->nreal == 0)  { 
      for(int is=0; is<grids->Nspecies; is++) {
        G[is] -> reality(grids->Nl * grids->Nm); 
      }
      solver -> fieldSolve(G, fields);
    }

    for(int is=0; is<grids->Nspecies; is++) {
      if (pars->save_for_restart && counter % pars->nsave == 0) G[is]->restart_write(&time);
    }
  }

  for(int is=0; is<grids->Nspecies; is++) {
    if (pars->save_for_restart) G[is]->restart_write(&time);
  }

  if (pars->eqfix && (
		      (pars->scheme_opt == Tmethod::k10) ||
		      (pars->scheme_opt == Tmethod::g3) ||
		      (pars->scheme_opt == Tmethod::k2))) {
    printf("\n");
    printf("\n");
    printf(ANSI_COLOR_MAGENTA);
    printf("The eqfix option is not compatible with this time-stepping algorithm. \n");
    printf(ANSI_COLOR_GREEN);
    printf("The eqfix option is not compatible with this time-stepping algorithm. \n");
    printf(ANSI_COLOR_RED);
    printf("The eqfix option is not compatible with this time-stepping algorithm. \n");
    printf(ANSI_COLOR_BLUE);
    printf("The eqfix option is not compatible with this time-stepping algorithm. \n");
    printf(ANSI_COLOR_RESET);    
    printf("\n");
    printf("\n");
  }  
  
  cudaEventRecord(stop,0);    cudaEventSynchronize(stop);    cudaEventElapsedTime(&timer,start,stop);
  printf("Total runtime = %f s (%f s / timestep)\n", timer/1000., timer/1000./counter);

  diagnostics->finish(G, fields, time);

  for(int is=0; is<grids->Nspecies; is++) {
    if (G[is])         delete G[is];
  }
  if (linear)    delete linear;
  if (nonlinear) delete nonlinear;
  if (timestep)  delete timestep;

  if (solver)    delete solver;
  if (fields)    delete fields;
  if (forcing)   delete forcing;     
}    

void getDeviceMemoryUsage()
{
  cudaDeviceSynchronize();
  // show memory usage of GPU
  size_t free_byte;  size_t total_byte;  cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte) ;

  if ( cudaSuccess != cuda_status ){
      printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
      exit(1);
  }
  // for some reason, total_byte returned by above call is not correct. 

  int dev;
  cudaDeviceProp prop;
  checkCuda( cudaGetDevice(&dev) );
  checkCuda( cudaGetDeviceProperties(&prop, dev) );
  double free_db = (double) free_byte;
  double total_db = (double) prop.totalGlobalMem;
  double used_db = total_db - free_db ;
  printf("GPU memory usage: used = %f MB (%f %%), free = %f MB (%f %%), total = %f MB\n",
	 used_db /1024.0/1024.0, used_db/total_db*100.,
	 free_db /1024.0/1024.0, free_db/total_db*100.,
	 total_db/1024.0/1024.0);
}
