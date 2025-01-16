#include "run_gx.h"

void run_gx(Parameters *pars, Grids *grids, Geometry *geo)
{
  double time = 0;

  Fields    * fields    = nullptr;
  Solver    * solver    = nullptr;
  Linear    * linear    = nullptr;
  Nonlinear * nonlinear = nullptr;
  ExB       * exb       = nullptr;
  Diagnostics * diagnostics = nullptr;
  MomentsG  ** G = (MomentsG**) malloc(sizeof(void*)*grids->Nspecies);
  for(int is=0; is<grids->Nspecies; is++) {
    G[is] = nullptr;
  }
  Forcing   * forcing   = nullptr;
  
  // set up moments and fields objects
  for(int is=0; is<grids->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G[is] = new MomentsG (pars, grids, is_glob);
  }
  fields = new Fields(pars, grids);               
  
  /////////////////////////////////
  //                             //
  // Initialize eqs              // 
  //                             //
  /////////////////////////////////
  // GX is set up to solve a handful of different equation sets.
  // Some have a geometry associated with them, some do not.
  // Presently the options are "gx", "krehm", "vp", "ks", and "cetg"
  // Most equation sets are undocumented, as they are exploratory or pedagogical in nature
  // 
  if (pars->gx) {
    linear = new Linear_GK(pars, grids, geo);          
    if (!pars->linear) nonlinear = new Nonlinear_GK(pars, grids, geo); 
    if (pars->ExBshear)   exb       = new ExB_GK(pars, grids, geo);
    checkCuda(cudaGetLastError());

    solver = new Solver_GK(pars, grids, geo);    
    checkCuda(cudaGetLastError());

    if (pars->forcing_init) {
      std::cout << "Forcing being ran: " << pars->forcing_type << std::endl;
      if (pars->forcing_type == "Kz")        forcing = new KzForcing(pars);        
      if (pars->forcing_type == "KzImpulse") forcing = new KzForcingImpulse(pars); 
      if (pars->forcing_type == "general")   forcing = new genForcing(pars);       
      if (pars->forcing_type == "HeliInj")   forcing = new HeliInjForcing(pars, grids); 
   }

    // set up initial conditions
    for(int is=0; is<grids->Nspecies; is++) {
      int is_glob = is+grids->is_lo;
      G[is] -> set_zero();
      if(!pars->restart && pars->init_electrons_only && pars->species_h[is_glob].type!=1) continue;
      G[is] -> initialConditions(&time);   
      G[is] -> sync(true);
    }
    solver -> fieldSolve(G, fields);                

    // set up diagnostics
    if(grids->iproc==0) DEBUGPRINT("Initializing diagnostics...\n");
    diagnostics = new Diagnostics_GK(pars, grids, geo, linear, nonlinear);
    if(grids->iproc==0) CUDA_DEBUG("Initializing diagnostics: %s \n");    
    checkCuda(cudaGetLastError());    
  }

  //////////////////////////////
  //                          //
  //     KREHM eq             // 
  //                          //
  //////////////////////////////  
  if (pars->krehm) {
    linear = new Linear_KREHM(pars, grids, geo);          
    if (!pars->linear) nonlinear = new Nonlinear_KREHM(pars, grids);    
    if (pars->forcing_init) {
      std::cout << "Forcing being ran: " << pars->forcing_type << std::endl;
      if (pars->forcing_type == "Kz")        forcing = new KzForcing(pars);
      if (pars->forcing_type == "KzImpulse") forcing = new KzForcingImpulse(pars);
      if (pars->forcing_type == "general")   forcing = new genForcing(pars);
      if (pars->forcing_type == "HeliInj")   forcing = new HeliInjForcing(pars, grids);
   }
    solver = new Solver_KREHM(pars, grids);

    // set up initial conditions
    G[0] -> set_zero();
    G[0] -> initialConditions(&time);   
    if(pars->harris_sheet or pars->periodic_equilibrium or pars->island_coalesce or pars->gaussian_tube or pars-> random_gaussian) solver -> set_equilibrium_current(G[0], fields);
    G[0] -> sync(true);
    solver -> fieldSolve(G, fields);                

    // set up diagnostics
    diagnostics = new Diagnostics_KREHM(pars, grids, geo, linear, nonlinear);
  }
  checkCuda(cudaGetLastError());
  
  //////////////////////////////
  //                          //
  //     cETG eq              // 
  //                          //
  //////////////////////////////  
  if (pars->cetg) {
    linear = new Linear_cetg(pars, grids, geo);          
    if (!pars->linear) nonlinear = new Nonlinear_cetg(pars, grids);    

    solver = new Solver_cetg(pars, grids);

    // set up initial conditions
    G[0] -> set_zero();
    G[0] -> initialConditions(&time);   
    G[0] -> sync();
    solver -> fieldSolve(G, fields);                

    // 
    // Adkins defines tau_bar = Ti/(Te Z). Set value for tau_bar with tau_fac in the Boltzmann section of the input file
    // The default value of tau_bar = 1.0.
    //
    // Separately, one can set Z, which enters into the calculations of the c_(1,2,3) coefficients.
    // Set Z by defining Z_ion in the Boltzmann section of the input file. The default value is 1.0. 
    //
    // Adkins defines a hyperdiffusion model with parameters N_nu and nu_perp.
    // Set nu_perp by defining D_hyper in the Dissipation section of the input file. The default value in GX is 0.1, 
    // which is quite large for the Adkins model. It is important, therefore, to set the value to what you want.
    // With Tony's definitions, a typical value would be 0.0005 or smaller. 
    //
    // Set N_nu by defining nu_hyper in the Dissipation namelist. The default value is nu_hyper = 2
    // Actually, the input variable nu_hyper is deprecated and one should set this using p_hyper = 2
    //
    // IMPORTANT: You must set hyper = true in the Dissipation namelist to turn this operator on.
    //
    // The only remaining parameters to be set are x0, y0, z0, nx, ny, and nz.
    // Note that Adkins' Lz = 2 pi z0, Ly = 2 pi y0, Lx = 2 pi x0.
    //
    // Adkins has no magnetic shear, so set zero_shat = true in the Geometry section of the input file
    // and choose slab = true to get his slab equations.
    //
  }
  checkCuda(cudaGetLastError());

  Timestepper * timestep = nullptr;
  switch (pars->scheme_opt)
    {
    case Tmethod::k10   : timestep = new Ketcheson10 (linear, nonlinear, solver, pars, grids, forcing, exb, pars->dt); break;
    case Tmethod::rk4   : timestep = new RungeKutta4 (linear, nonlinear, solver, pars, grids, forcing, exb, pars->dt); break;
    case Tmethod::rk3   : timestep = new RungeKutta3 (linear, nonlinear, solver, pars, grids, forcing, exb, pars->dt); break;
    case Tmethod::sspx2 : timestep = new SSPx2       (linear, nonlinear, solver, pars, grids, forcing, exb, pars->dt); break;
    case Tmethod::sspx3 : timestep = new SSPx3       (linear, nonlinear, solver, pars, grids, forcing, exb, pars->dt); break;
    default:
      printf("Unknown timestepper id (%u). Aborting.",static_cast<unsigned int>(pars->scheme_opt) );
      exit(1);
    }

  fflush(stdout);
  MPI_Barrier(pars->mpcom);
  printDeviceMemoryUsage(pars->iproc);
  MPI_Barrier(pars->mpcom);
  fflush(stdout);

  //  if (pars->write_moms) diagnostics -> write_init(G, fields);

  // TIMESTEP LOOP
  int counter = 0;           float timer = 0;          cudaEvent_t start, stop;    bool checkstop = false;
  cudaEventCreate(&start);   cudaEventCreate(&stop);   cudaEventRecord(start,0);

  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  while(counter<pars->nstep && time<pars->t_max) {

    checkstop = diagnostics -> loop(G, fields, timestep->get_dt(), counter, time);
    timestep -> advance(&time, G, fields);
    if (checkstop) break;

    if (pars->save_for_restart && counter % pars->nsave == 0) diagnostics -> restart_write(G, &time);

    // this will catch any error in the timestep loop, but it won't be able to identify where the error occurred.
    checkCuda(cudaGetLastError());
    counter++;
    if (counter==pars->nstep || time>=pars->t_max) {
      [[maybe_unused]] auto unused = diagnostics -> loop(G, fields, timestep->get_dt(), counter, time);
    }
  }

  if (pars->save_for_restart) diagnostics -> restart_write(G, &time);

  if (pars->eqfix && (pars->scheme_opt == Tmethod::k10) ) {
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
  printf("Total runtime = %f min (%f s / timestep)\n", timer/1000./60., timer/1000./counter);

  diagnostics->finish(G, fields, time);

  for(int is=0; is<grids->Nspecies; is++) {
    if (G[is])         delete G[is];
  }
  free(G);
  if (diagnostics) delete diagnostics;
  if (linear)    delete linear;
  if (nonlinear) delete nonlinear;
  if (timestep)  delete timestep;
  if (exb)       delete exb;

  if (solver)    delete solver;
  if (fields)    delete fields;
  if (forcing)   delete forcing;     
}    

void uuid_print(cudaUUID_t a){
  std::cout << "GPU ID: ";
  std::vector<std::tuple<int, int> > r = {{0,4}, {4,6}, {6,8}, {8,10}, {10,16}};
  bool first = true;
  for (auto t : r){
    if(!first) std::cout << "-";
    first = false;
    for (int i = std::get<0>(t); i < std::get<1>(t); i++)
      std::cout << std::hex << (unsigned)(unsigned char)a.bytes[i];
  }
}

void printDeviceID()
{
  int dev;
  cudaDeviceProp prop;
  checkCuda( cudaGetDevice(&dev) );
  checkCuda( cudaGetDeviceProperties(&prop, dev) );
  uuid_print(prop.uuid);
}

void printDeviceMemoryUsage(int iproc)
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
  printf(ANSI_COLOR_GREEN);
  printf("Device %d: ", iproc);
  printDeviceID();
  printf(", GPU type: %s, ", prop.name);
  printf("GPU memory usage: used = %f MB (%f %%), free = %f MB (%f %%)\n",
	 used_db /1024.0/1024.0, used_db/total_db*100.,
	 free_db /1024.0/1024.0, free_db/total_db*100.);
  printf(ANSI_COLOR_RESET);
}
