#include "trinity_interface.h"
#include "run_gx.h"

void gx_get_fluxes_(trin_parameters_struct* tpars, trin_fluxes_struct* tfluxes, char* run_name, int mpcom_f)
{
  Parameters* pars = nullptr;
  MPI_Comm mpcom = MPI_Comm_f2c(mpcom_f);
  int iproc;
  MPI_Comm_rank(mpcom, &iproc);
  printf("Running %s on proc %d\n", run_name, iproc); 
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

  geo = init_geo(pars, grids);

  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  // run gx calculation using (updated) parameters
  run_gx(pars, grids, geo);

  // copy time-averaged fluxes to trinity
  copy_fluxes_to_trinity(pars, geo, tfluxes);

  delete pars;
  delete grids;
  delete geo;
}

void set_from_trinity(Parameters *pars, trin_parameters_struct *tpars)
{
   pars->equilibrium_type = tpars->equilibrium_type;
   if(tpars->restart>0) pars->restart = true;

   //if (tpars->nstep > pars->nstep) {
   //  printf("ERROR: nstep has been increased above the default value. nstep must be less than or equal to what is in the input file\n");
   //  exit(1);
   //}
   pars->trinity_timestep = tpars->trinity_timestep;
   pars->trinity_iteration = tpars->trinity_iteration;
   pars->trinity_conv_count = tpars->trinity_conv_count;
   pars->nstep = tpars->nstep;
   pars->navg = tpars->navg;
   //pars->end_time = tpars->end_time;
   //pars->irho = tpars->irho ;
   pars->rhoc = tpars->rhoc ;
   //pars->eps = tpars->eps;
   //pars->bishop = tpars->bishop ;
   //pars->nperiod = tpars->nperiod ;
   //pars->nz_in = tpars->ntheta ;

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
   //pars->s_hat_input = tpars->s_hat_input ;

  /*Flow shear*/
   pars->g_exb = tpars->g_exb ;

  /* Species parameters... I think allowing 20 species should be enough!*/
  int oldnSpecies = pars->nspec;
  // read nspecies from trinity
  pars->nspec = tpars->ntspec ;
  // trinity always assumes electrons are one of the evolved species
  // if GX is using Boltzmann electrons, decrease number of species by 1
  if(pars->add_Boltzmann_species && pars->Boltzmann_opt == BOLTZMANN_ELECTRONS) {
    pars->nspec = pars->nspec - 1;
  }

  if (pars->nspec!=oldnSpecies){
          printf("oldnSpecies=%d,  nSpecies=%d\n", oldnSpecies, pars->nspec);
          printf("Number of species set in get_fluxes must equal number of species in gx input file\n");
          exit(1);
  }
  if (pars->debug) printf("nSpecies was set to %d\n", pars->nspec);

  if(pars->add_Boltzmann_species && pars->Boltzmann_opt == BOLTZMANN_ELECTRONS) {
    for (int s=0;s<pars->nspec;s++){
      // trinity assumes first species is electrons,
      // so ions require s+1
      pars->species_h[s].z = tpars->z[s+1] ;
      pars->species_h[s].mass = tpars->mass[s+1] ;
      pars->species_h[s].dens = tpars->dens[s+1] ;
      pars->species_h[s].temp = tpars->temp[s+1] ;
      pars->species_h[s].fprim = tpars->fprim[s+1] ;
      pars->species_h[s].tprim = tpars->tprim[s+1] ;
      pars->species_h[s].nu_ss = tpars->nu[s+1] ;
      pars->species_h[s].type = 0;  // all gx species will be ions
    }
  } else {
    for (int s=0;s<pars->nspec;s++){
      pars->species_h[s].z = tpars->z[s] ;
      pars->species_h[s].mass = tpars->mass[s] ;
      pars->species_h[s].dens = tpars->dens[s] ;
      pars->species_h[s].temp = tpars->temp[s] ;
      pars->species_h[s].fprim = tpars->fprim[s] ;
      pars->species_h[s].tprim = tpars->tprim[s] ;
      pars->species_h[s].nu_ss = tpars->nu[s] ;
      pars->species_h[s].type = s == 0 ? 1 : 0; // 0th trinity species is electron, others are ions
    }
  }
  pars->init_species(pars->species_h);

  // write a toml input file with the parameters that trinity changed
  char fname[1500]; // Size needs to be big enough to handle whatever sprintf produces
  sprintf(fname, "%s.trinpars_t%d_i%d", pars->run_name, pars->trinity_timestep, pars->trinity_iteration);
  strcpy(pars->run_name, fname); 

  FILE *fptr;
  fptr = fopen(fname, "w");
  fprintf(fptr, "[Dimensions]\n");
  fprintf(fptr, " ntheta = %d\n", pars->nz_in);
  fprintf(fptr, " nperiod = %d\n", pars->nperiod);
  fprintf(fptr, "\n[Geometry]\n");
  fprintf(fptr, " rhoc = %.9e\n", pars->rhoc);
  fprintf(fptr, " qinp = %.9e\n", pars->qsf);
  fprintf(fptr, " shat = %.9e\n", pars->shat);
  fprintf(fptr, " Rmaj = %.9e\n", pars->rmaj);
  fprintf(fptr, " R_geo = %.9e\n", pars->r_geo);
  fprintf(fptr, " shift = %.9e\n", pars->shift);
  fprintf(fptr, " akappa = %.9e\n", pars->akappa);
  fprintf(fptr, " akappri = %.9e\n", pars->akappri);
  fprintf(fptr, " tri = %.9e\n", pars->tri);
  fprintf(fptr, " tripri = %.9e\n", pars->tripri);
  fprintf(fptr, " betaprim = %.9e\n", pars->beta_prime_input);
  fprintf(fptr, "\n[species]\n");
  fprintf(fptr, " z = [ ");
  for (int i=0;i<pars->nspec;i++){
    fprintf(fptr, "%.9e,\t", pars->species_h[i].z);
  }
  fprintf(fptr, "]\n");
  fprintf(fptr, " mass = [ ");
  for (int i=0;i<pars->nspec;i++){
    fprintf(fptr, "%.9e,\t", pars->species_h[i].mass);
  }
  fprintf(fptr, "]\n");
  fprintf(fptr, " dens = [ ");
  for (int i=0;i<pars->nspec;i++){
    fprintf(fptr, "%.9e,\t", pars->species_h[i].dens);
  }
  fprintf(fptr, "]\n");
  fprintf(fptr, " temp = [ ");
  for (int i=0;i<pars->nspec;i++){
    fprintf(fptr, "%.9e,\t", pars->species_h[i].temp);
  }
  fprintf(fptr, "]\n");
  fprintf(fptr, " fprim = [ ");
  for (int i=0;i<pars->nspec;i++){
    fprintf(fptr, "%.9e,\t", pars->species_h[i].fprim);
  }
  fprintf(fptr, "]\n");
  fprintf(fptr, " tprim = [ ");
  for (int i=0;i<pars->nspec;i++){
    fprintf(fptr, "%.9e,\t", pars->species_h[i].tprim);
  }
  fprintf(fptr, "]\n");
  fprintf(fptr, " vnewk = [ ");
  for (int i=0;i<pars->nspec;i++){
    fprintf(fptr, "%.9e,\t", pars->species_h[i].nu_ss);
  }
  fprintf(fptr, "]\n");
  fclose(fptr);

  if(pars->igeo==1) {
    char command[3750]; // Again, need a long buffer so we never overflow (but this is getting silly, we should restrict run_name to a smaller value)
    // call python geometry module using toml we just created to write the eik.out geo file
    // GX_PATH is defined at compile time via a -D flag
    sprintf(command, "python %s/geometry_modules/miller/gx_geo.py %s %s.eik.out", GX_PATH, fname, fname);
    printf("%s\n", command);
    pars->geofilename = std::string(fname) + ".eik.out";
    system(command);
  }
}

void copy_fluxes_to_trinity(Parameters *pars_, Geometry *geo_, trin_fluxes_struct *tfluxes)
{
  int id_time, id_fluxes, id_Q, id_P;
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
  if (retval = nc_inq_varid(id_fluxes, "pflux", &id_P)) ERR(retval);

  // get length of time output
  size_t tlen;
  if (retval = nc_inq_dimlen(ncres, id_time, &tlen)) ERR(retval);

  // allocate arrays for time and qflux history
  double *time = (double*) malloc(sizeof(double) * tlen);
  float *qflux = (float*) malloc(sizeof(float) * tlen);
  float *pflux = (float*) malloc(sizeof(float) * tlen);

  // read time and qflux history
  if (retval = nc_inq_varid(ncres, "time", &id_time)) ERR(retval);
  if (retval = nc_get_var(ncres, id_time, time)) ERR(retval);

  // compute the surface area from A=Int(J |grad rho| dtheta dalpha)
  // and dV/drhon = Int(J dtheta)
  double surfarea = 0.;
  double dvdrhon = 0.;
  double dz = 2.*M_PI/pars_->nz_in;
  for (int i=0; i<pars_->nz_in; i++) {
    surfarea += 2.*M_PI*geo_->grho_h[i]*geo_->jacobian_h[i]*dz;
    dvdrhon += geo_->jacobian_h[i]*dz;
  }
  // compute surface-averaged grho and dvdrho
  double grhoavg = surfarea/dvdrhon;
  tfluxes->grho = grhoavg;
  tfluxes->dvdrho = dvdrhon;

  
  int is = 1; // counter for trinity ion species
  for(size_t s = 0; s < static_cast<size_t>(pars_->nspec_in); s++) {
    size_t qstart[] = {0, s};
    size_t qcount[] = {tlen, 1};
    if (retval = nc_get_vara(id_fluxes, id_Q, qstart, qcount, qflux)) ERR(retval);
    if (retval = nc_get_vara(id_fluxes, id_P, qstart, qcount, pflux)) ERR(retval);

    // compute time average
    float qflux_sum = 0.; 
    float pflux_sum = 0.; 
    float t_sum = 0.;
    float dt = 0.;
    for(size_t i = tlen - pars_->navg/pars_->nwrite; i < tlen; i++) {
      dt = time[i] - time[i-1];
      qflux_sum += qflux[i]*dt;
      pflux_sum += pflux[i]*dt;
      t_sum += dt;
    }

    // Trinity orders species with electrons first, then ions
    if(pars_->add_Boltzmann_species && pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS) {
      // no electron heat flux or particle flux
      tfluxes->qflux[0] = 0.;
      tfluxes->pflux[0] = 0.;
      tfluxes->heat[0] = 0.;

      // ion heat and particle fluxes
      tfluxes->qflux[is] = qflux_sum / t_sum; 
      tfluxes->pflux[is] = pflux_sum / t_sum; 
      tfluxes->heat[is] = 0.;
      is++;
    } else {
      if(pars_->species_h[s].type==1) { // electrons
        tfluxes->qflux[0] = qflux_sum / t_sum; // are species 0 in trinity
        tfluxes->pflux[0] = pflux_sum / t_sum; 
        tfluxes->heat[0] = 0.;
      }
      else {
        tfluxes->qflux[is] = qflux_sum / t_sum; 
        tfluxes->pflux[is] = pflux_sum / t_sum; 
        tfluxes->heat[is] = 0.;
        is++;
      }
    }
  }

  // these are placeholders for gx-computed quantities
  // float heat = 0.;

  for(int s=0; s<pars_->nspec_in; s++) {
    if(pars_->add_Boltzmann_species && pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS) {
      printf("%s: Species %d: qflux = %g, pflux = %g, heat = %g, surfarea = %g\n", pars_->run_name, s, tfluxes->qflux[s+1], tfluxes->pflux[s+1], tfluxes->heat[s+1], surfarea);
    } else {
      printf("%s: Species %d: qflux = %g, pflux = %g, heat = %g\n", pars_->run_name, s, tfluxes->qflux[s], tfluxes->pflux[s], tfluxes->heat[s]);
    }
  }

  nc_close(ncres);
}
