#include "helper_cuda.h"
#include "trinity_interface.h"
#include "run_gx.h"

void gx_get_fluxes_(trin_parameters_struct* tpars, trin_fluxes_struct* tfluxes, char* run_name, int mpcom_f)
{
  Parameters* pars = nullptr;
  MPI_Comm mpcom = MPI_Comm_f2c(mpcom_f);
  int iproc;
  MPI_Comm_rank(mpcom, &iproc);
  pars = new Parameters(iproc);
  // get default values from namelist
  pars->get_nml_vars(run_name);

  // overwrite parameters based on values from trinity
  set_from_trinity(pars, tpars);

  // initialize grids, geometry, and diagnostics
  Grids       * grids       = nullptr;
  
  DEBUGPRINT("Initializing grids...\n");
  grids = new Grids(pars);
  CUDA_DEBUG("Initializing grids: %s \n");

  DEBUGPRINT("Grid dimensions: Nx=%d, Ny=%d, Nz=%d, Nl=%d, Nm=%d, Nspecies=%d\n",
	     grids->Nx, grids->Ny, grids->Nz, grids->Nl, grids->Nm, grids->Nspecies);

  Geometry    * geo         = nullptr;
  Diagnostics_GK * diagnostics = nullptr;

  int igeo = pars->igeo;
  DEBUGPRINT("Initializing geometry...\n");
  if(igeo==0) {
    geo = new S_alpha_geo(pars, grids);
    CUDA_DEBUG("Initializing geometry s_alpha: %s \n");
  }
  else if(igeo==1) {
    geo = new File_geo(pars, grids);
    printf("************************* \n \n \n");
    printf("Warning: may have assumed grho = 1 \n \n \n");
    printf("************************* \n");
    CUDA_DEBUG("Initializing geometry from file: %s \n");
  } 
  else if(igeo==2) {
    geo = new geo_nc(pars, grids);
    CUDA_DEBUG("Initializing geometry from NetCDF file: %s \n");
  } 
  else if(igeo==3) {
    DEBUGPRINT("igeo = 3 not yet implemented!\n");
    exit(1);
    //geo = new Gs2_geo();
  }

  DEBUGPRINT("Initializing diagnostics...\n");
  diagnostics = new Diagnostics_GK(pars, grids, geo);
  CUDA_DEBUG("Initializing diagnostics: %s \n");    
  
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // run gx calculation using (updated) parameters
  run_gx(pars, grids, geo, diagnostics);

  // copy time-averaged fluxes to trinity
  copy_fluxes_to_trinity(pars, tfluxes);

  delete pars;
  delete grids;
  delete geo;
  delete diagnostics;
}

void set_from_trinity(Parameters *pars, trin_parameters_struct *tpars)
{
   pars->equilibrium_type = tpars->equilibrium_type;
   if(tpars->restart>0) pars->restart = true;

   if (tpars->nstep > pars->nstep) {
     printf("ERROR: nstep has been increased above the default value. nstep must be less than or equal to what is in the input file\n");
     exit(1);
   }
   pars->trinity_timestep = tpars->trinity_timestep;
   pars->trinity_iteration = tpars->trinity_iteration;
   pars->trinity_conv_count = tpars->trinity_conv_count;
   pars->nstep = tpars->nstep;
   pars->navg = tpars->navg;
   pars->end_time = tpars->end_time;
   pars->irho = tpars->irho ;
   pars->rhoc = tpars->rhoc ;
   pars->eps = tpars->eps;
   pars->bishop = tpars->bishop ;
   pars->nperiod = tpars->nperiod ;
   pars->nz_in = tpars->ntheta ;

 /* Miller parameters*/
   pars->rmaj = tpars->rgeo_local ;
   pars->r_geo = tpars->rgeo_lcfs ;
   pars->akappa  = tpars->akappa ;
   pars->akappri = tpars->akappri ;
   pars->tri = tpars->tri ;
   pars->tripri = tpars->tripri ;
   pars->shift = tpars->shift ;
   pars->qsf = tpars->qinp ;
   pars->shat = tpars->shat ;

  /* Other geometry parameters - Bishop/Greene & Chance*/
   pars->beta_prime_input = tpars->beta_prime_input ;
   pars->s_hat_input = tpars->s_hat_input ;

  /*Flow shear*/
   pars->g_exb = tpars->g_exb ;

  /* Species parameters... I think allowing 20 species should be enough!*/
  int oldnSpecies = pars->nspec;
  pars->nspec = tpars->ntspec ;

  if (pars->nspec!=oldnSpecies){
          printf("oldnSpecies=%d,  nSpecies=%d\n", oldnSpecies, pars->nspec);
          printf("Number of species set in get_fluxes must equal number of species in gx input file\n");
          exit(1);
  }
  if (pars->debug) printf("nSpecies was set to %d\n", pars->nspec);
  for (int i=0;i<pars->nspec;i++){
           pars->species_h[i].dens = tpars->dens[i] ;
           pars->species_h[i].temp = tpars->temp[i] ;
           pars->species_h[i].fprim = tpars->fprim[i] ;
           pars->species_h[i].tprim = tpars->tprim[i] ;
           pars->species_h[i].nu_ss = tpars->nu[i] ;
  }
  pars->init_species(pars->species_h);

  //jtwist should never be < 0. If we set jtwist < 0 in the input file,
  // this triggers the use of jtwist_square... i.e. jtwist is 
  // set to what it needs to make the box square at the outboard midplane
  if (pars->jtwist < 0) {
    int jtwist_square;
    // determine value of jtwist needed to make X0~Y0
    jtwist_square = (int) round(2*M_PI*abs(pars->shat)*pars->Zp);
    if (jtwist_square == 0) jtwist_square = 1;
    // as currently implemented, there is no way to manually set jtwist from input file
    // there could be some switch here where we choose whether to use
    // jtwist_in or jtwist_square
    pars->jtwist = jtwist_square*2;
    //else use what is set in input file 
  }
  if(pars->jtwist!=0 && abs(pars->shat)>1.e-6) pars->x0 = pars->y0*pars->jtwist/(2*M_PI*pars->Zp*abs(pars->shat));
  //if(abs(pars->shat)<1.e-6) pars->x0 = pars->y0;
}

void copy_fluxes_to_trinity(Parameters *pars_, trin_fluxes_struct *tfluxes)
{
  int id_ns, id_time, id_fluxes, id_Q;
  int ncres, retval;
  char strb[263];
  strcpy(strb, pars_->run_name); 
  strcat(strb, ".nc");
  // open file and get handle ncres
  if (retval = nc_open(strb, NC_NOWRITE, &ncres)) { printf("file: %s \n",strb); ERR(retval);}
  // get handle for time dimension
  if (retval = nc_inq_dimid(ncres, "time", &id_time)) ERR(retval);
  // get handle for Fluxes group
  if (retval = nc_inq_grp_ncid(ncres, "Fluxes", &id_fluxes))    ERR(retval);
  // get handle for qflux
  if (retval = nc_inq_varid(id_fluxes, "qflux", &id_Q)) ERR(retval);

  // get length of time output
  size_t tlen;
  if (retval = nc_inq_dimlen(ncres, id_time, &tlen)) ERR(retval);

  // allocate arrays for time and qflux history
  double *time = (double*) malloc(sizeof(double) * tlen);
  float *qflux = (float*) malloc(sizeof(float) * tlen);

  // read time and qflux history
  if (retval = nc_inq_varid(ncres, "time", &id_time)) ERR(retval);
  if (retval = nc_get_var(ncres, id_time, time)) ERR(retval);
  
  for(int s=0; s<pars_->nspec_in; s++) {
    size_t qstart[] = {0, s};
    size_t qcount[] = {tlen, 1};
    if (retval = nc_get_vara(id_fluxes, id_Q, qstart, qcount, qflux)) ERR(retval);

    // compute time average
    float qflux_sum = 0.; 
    float t_sum = 0.;
    float dt = 0.;
    for(int i=tlen - pars_->navg/pars_->nwrite; i<tlen; i++) {
      dt = time[i] - time[i-1];
      qflux_sum += qflux[i]*dt;
      t_sum += dt;
    }
    tfluxes->qflux[s] = qflux_sum / t_sum;
  }

  // these are placeholders for gx-computed quantities
  float pflux = 0.;
  float heat = 0.;

  for(int s=0; s<pars_->nspec_in; s++) {
    tfluxes->pflux[s] = pflux;
    tfluxes->heat[s] = heat;
  }

  nc_close(ncres);
}
