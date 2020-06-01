#include "parameters.h"
#include <netcdf.h>
#include <iostream>
using namespace std;

extern "C" {void read_nml(char *runname);}

Parameters::Parameters() {
  // MM // I think that a similar pointer-type thing should be done for the constructor/destructor here, since the CPU version shouldn't need to know anything about cuda keywords

  initialized = false;

  // some cuda parameters (not from input file)
  int dev; 
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  maxThreadsPerBlock = prop.maxThreadsPerBlock;
}

Parameters::~Parameters() {
  cudaDeviceSynchronize();
  if(initialized) {
    cudaFree(species);
    cudaFreeHost(species_h);
  }
}

void Parameters::get_nml_vars(char* filename)
{
  strcpy (run_name, filename);
  read_nml(run_name);
  
  char strb[263];
  strcpy(strb, run_name); 
  strcat(strb, ".nc");

  int ncid, retval;
  if (retval = nc_open(strb, NC_WRITE, &ncid)) ERR(retval); 
  nz_in = getint (ncid, "ntheta");
  ny_in = getint (ncid, "ny"); 
  nx_in = getint (ncid, "nx");
  nm_in = getint (ncid, "nhermite"); 
  nl_in = getint (ncid, "nlaguerre");
  nspec_in = getint (ncid, "nspecies");
  
  dt = getfloat (ncid, "dt");
  y0 = getfloat (ncid, "y0");
  x0 = getfloat (ncid, "x0");
  Zp = getfloat (ncid, "zp");

  nperiod = getint (ncid, "nperiod");
  nstep   = getint (ncid, "nstep");
  jtwist  = getint (ncid, "jtwist");
  nwrite  = getint (ncid, "nwrite");
  navg    = getint (ncid, "navg");
  nsave   = getint (ncid, "nsave");
  i_share = getint (ncid, "i_share");

  ikx_fixed = getint (ncid, "ikx_fixed");
  iky_fixed = getint (ncid, "iky_fixed");

  eqfix             = getbool (ncid, "eqfix");
  debug             = getbool (ncid, "debug");
  restart           = getbool (ncid, "restart");
  save_for_restart  = getbool (ncid, "save_for_restart");
  secondary         = getbool (ncid, "secondary");
  write_omega       = getbool (ncid, "write_omega");
  write_fluxes      = getbool (ncid, "write_fluxes");
  write_moms        = getbool (ncid, "write_moms");
  write_rh          = getbool (ncid, "write_rh");
  write_pzt         = getbool (ncid, "write_pzt");
  write_phi         = getbool (ncid, "write_phi");
  write_phi_kpar    = getbool (ncid, "write_phi_kpar");
  write_h_spectrum  = getbool (ncid, "write_h_spectrum");
  write_l_spectrum  = getbool (ncid, "write_l_spectrum");
  write_lh_spectrum = getbool (ncid, "write_lh_spectrum");
  write_spec_v_time = getbool (ncid, "write_spec_v_time");
  init_single       = getbool (ncid, "init_single");
  //  hermite_spectrum_avg_cutoff = getint (ncid, "hermite_spectrum_avg_cutoff");

  cfl = getfloat (ncid, "cfl");

  init_amp   = getfloat (ncid, "init_amp");
  D_hyper    = getfloat (ncid, "d_hyper");
  nu_hyper   = getfloat (ncid, "nu_hyper");
  nu_hyper_l = getfloat (ncid, "nu_hyper_l");
  nu_hyper_m = getfloat (ncid, "nu_hyper_m");

  p_hyper    = getint (ncid, "p_hyper");
  p_hyper_l  = getint (ncid, "p_hyper_l");
  p_hyper_m  = getint (ncid, "p_hyper_m");

  int idum;
  if (retval = nc_inq_varid (ncid, "scheme_dum", &idum))   ERR(retval);
  if (retval = nc_get_att_text (ncid, idum, "value", scheme)) ERR(retval);
  
  if (retval = nc_inq_varid (ncid, "forcing_type_dum", &idum))   ERR(retval);
  if (retval = nc_get_att_text (ncid, idum, "value", forcing_type)) ERR(retval);

  if (retval = nc_inq_varid (ncid, "init_field_dum", &idum))   ERR(retval);
  if (retval = nc_get_att_text (ncid, idum, "value", init_field)) ERR(retval);

  if (retval = nc_inq_varid (ncid, "stir_field_dum", &idum))   ERR(retval);
  if (retval = nc_get_att_text (ncid, idum, "value", stir_field)) ERR(retval);

  forcing_amp = getfloat (ncid, "forcing_amp");
  scale = getfloat (ncid, "scale");
  
  forcing_index = getint (ncid, "forcing_index");
  forcing_init  = getbool (ncid, "forcing_init");

  phi_ext    = getfloat (ncid, "phi_ext");
  kpar_init  = getfloat (ncid, "kpar_init");
  ikx_single = getint (ncid, "ikx_single");
  iky_single = getint (ncid, "iky_single");

  nonlinear_mode = getbool (ncid, "nonlinear_mode");  linear = !nonlinear_mode;
  
  if (retval = nc_inq_varid (ncid, "closure_model_dum", &idum))   ERR(retval);
  if (retval = nc_get_att_text (ncid, idum, "value", closure_model)) ERR(retval);

  if (retval = nc_inq_varid (ncid, "boundary_dum", &idum))   ERR(retval);
  if (retval = nc_get_att_text (ncid, idum, "value", boundary)) ERR(retval);

  if (retval = nc_inq_varid (ncid, "source_dum", &idum))   ERR(retval);
  if (retval = nc_get_att_text (ncid, idum, "value", source)) ERR(retval);

  smith_par_q     = getint (ncid, "smith_par_q");
  smith_perp_q    = getint (ncid, "smith_perp_q");
  iphi00          = getint (ncid, "iphi00");
  hyper           = getbool (ncid, "hyper");
  hypercollisions = getbool (ncid, "hypercollisions");

  if (retval = nc_inq_varid (ncid, "restart_to_file_dum", &idum))   ERR(retval);
  if (retval = nc_get_att_text (ncid, idum, "value", restart_to_file)) ERR(retval);

  if (retval = nc_inq_varid (ncid, "restart_from_file_dum", &idum))   ERR(retval);
  if (retval = nc_get_att_text (ncid, idum, "value", restart_from_file)) ERR(retval);

  if (retval = nc_inq_varid (ncid, "geofile_dum", &idum))   ERR(retval);
  if (retval = nc_get_att_text (ncid, idum, "value", geofilename)) ERR(retval);

  slab       = getbool (ncid, "slab");
  const_curv = getbool (ncid, "const_curv");
  //  snyder_electrons = getbool (ncid, "snyder_electrons");

  igeo     = getint (ncid, "igeo");
  drhodpsi = getfloat (ncid, "drhodpsi");
  kxfac    = getfloat (ncid, "kxfac");
  rmaj     = getfloat (ncid, "Rmaj");
  shift    = getfloat (ncid, "shift");
  eps      = getfloat (ncid, "eps");
  rhoc     = getfloat (ncid, "rhoc");
  qsf      = getfloat (ncid, "q");
  shat     = getfloat (ncid, "shat");
  akappa   = getfloat (ncid, "kappa");
  akappri  = getfloat (ncid, "kappa_prime");
  tri      = getfloat (ncid, "tri");
  tripri   = getfloat (ncid, "tri_prime");
  beta     = getfloat (ncid, "beta");

  beta_prime_input = getfloat (ncid, "beta_prime_input");
  s_hat_input      = getfloat (ncid, "s_hat_input");

  //  if (retval = nc_inq_varid (ncid, "spec_type_dum", &idum))   ERR(retval);
  //  if (retval = nc_get_att_text (ncid, idum, "value", type)) ERR(retval);

  //  getcharc (ncid, "type", type);
  ion_z     = getfloat (ncid, "z");
  ion_mass  = getfloat (ncid, "m");
  ion_dens  = getfloat (ncid, "n0");
  ion_fprim = getfloat (ncid, "n0_prime");
  ion_uprim = getfloat (ncid, "u0_prime");
  ion_temp  = getfloat (ncid, "T0");
  ion_tprim = getfloat (ncid, "T0_prime");
  ion_vnewk = getfloat (ncid, "nu");
  ti_ov_te  = getfloat (ncid, "tite");

  fphi  = getfloat (ncid, "fphi");
  fapar = getfloat (ncid, "fapar");
  fbpar = getfloat (ncid, "fbpar");

  if (retval = nc_close(ncid)) ERR(retval); 
  
  if(nz_in != 1) {
    int ntgrid = nz_in/2 + (nperiod-1)*nz_in; 
    nz_in = 2*ntgrid; // force even
  }
  
  Zp = 2*nperiod - 1; // BD This needs updating
  
  // BD  This is messy. Prefer to go back to original method
  // before, jtwist_old assumed Zp=1
  // now, redefining jtwist = jtwist_old*Zp
  // set default jtwist to 2pi*shat so that X0~Y0
  if (jtwist == -1) {
    //    printf("(A) jtwist = %i \n \n", jtwist);
    jtwist = (int) round(2*M_PI*shat*Zp);  // Use Zp or Z0 here?
    //    printf("(B) jtwist = %i \n \n", jtwist);
  }  
  
  // BD I believe the problem can be fixed using character fill, perhaps?

  if(strcmp(closure_model, "beer4+2")==0) {
    printf("\nUsing Beer 4+2 closure model. Overriding nm=4, nl=2\n\n");
    nm_in = 4;
    nl_in = 2;
    closure_model_opt = BEER42;
  } else if (strcmp(closure_model, "smith_perp")==0) { closure_model_opt = SMITHPERP;
  } else if (strcmp(closure_model, "smith_par")==0)  { closure_model_opt = SMITHPAR;
  }

  if( strcmp(boundary,"periodic") == 0) { boundary_option_periodic = true;
  } else { boundary_option_periodic = false; }
  
  local_limit = false;
  if(qsf < 0  &&  nz_in == 1) { local_limit=true; }

  if     ( strcmp(init_field,"density") == 0) { init = DENS;  }
  //  else if( strcmp(init_field,"phi"    ) == 0) { init = PHI;   }
  else if( strcmp(init_field,"force"  ) == 0) { init = FORCE; }
  else if( strcmp(init_field,"qperp"  ) == 0) { init = QPRP;  }
  else if( strcmp(init_field,"tperp"  ) == 0) { init = TPRP;  }
  else if( strcmp(init_field,"tpar"   ) == 0) { init = TPAR;  }
  else if( strcmp(init_field,"qpar"   ) == 0) { init = QPAR;  }
  else if( strcmp(init_field,"upar"   ) == 0) { init = UPAR;  }
  else if( strcmp(init_field,"ppar"   ) == 0) { init = PPAR;  }
  else if( strcmp(init_field,"pperp"  ) == 0) { init = PPRP;  }
  //  else if( strcmp(init_field,"odd"    ) == 0) { init = ODD;   }
  //  else if( strcmp(init_field,"RH_eq"  ) == 0) { init = RH_equilibrium; new_varenna = true; }
  if     ( strcmp(stir_field,"density") == 0) { stirf = DENS; }
  else if( strcmp(stir_field,"qperp"  ) == 0) { stirf = QPRP;  }
  else if( strcmp(stir_field,"tperp"  ) == 0) { stirf = TPRP;  }
  else if( strcmp(stir_field,"tpar"   ) == 0) { stirf = TPAR;  }
  else if( strcmp(stir_field,"qpar"   ) == 0) { stirf = QPAR;  }
  else if( strcmp(stir_field,"upar"   ) == 0) { stirf = UPAR;  }
  else if( strcmp(stir_field,"ppar"   ) == 0) { stirf = PPAR;  }
  else if( strcmp(stir_field,"pperp"  ) == 0) { stirf = PPRP;  }
  
  if( strcmp(scheme, "sspx2") == 0) scheme_opt = SSPX2;  
  if( strcmp(scheme, "rk2")   == 0) scheme_opt = RK2;  
  if( strcmp(scheme, "rk3")   == 0) scheme_opt = RK3;  
  if( strcmp(scheme, "rk4")   == 0) scheme_opt = RK4;  
  if( strcmp(scheme, "k10")   == 0) scheme_opt = K10;  
  
  if( strcmp(source, "phiext_full")==0 ) {
    source_option = PHIEXT;
    printf("Running Rosenbluth-Hinton zonal flow calculation\n");
  }

  if(hypercollisions) printf("Using hypercollisions.\n");

  adiabatic_electrons = true;
  if(debug) printf("nspec_in = %i \n",nspec_in);

  cudaMalloc((void**)     &species,   sizeof(specie)*nspec_in);
  cudaMallocHost((void**) &species_h, sizeof(specie)*nspec_in);

  nspec = nspec_in;
  /*
  int ionspec = 0;   
  int ispec = 1;
  float mass;
  bool main_ion_species_found = false;
  for(int s=1; s<nspec_in+1; s++) {    
    
    char namelist[100];
    sprintf(namelist,"species_parameters_%d",s); 
    char* type;
    fnr_get_string(&namelist_struct, namelist, "type", &type); 
    
    // main ion mass assumed to be 1. main ion indexed 0.
    if(strcmp(type,"ion") == 0) {
      fnr_get_float(&namelist_struct, namelist, "mass", &mass);
      if((mass == 1. && !main_ion_species_found) || nspec_in==1) {ionspec=0; main_ion_species_found=true;} 
      else {ionspec = ispec; ispec++;}
  */
      int ionspec = 0; // hack
      species_h[ionspec].mass    = ion_mass;
      species_h[ionspec].z       = ion_z;
      species_h[ionspec].dens    = ion_dens;
      species_h[ionspec].temp    = ion_temp;
      species_h[ionspec].tprim   = ion_tprim;
      species_h[ionspec].fprim   = ion_fprim;
      species_h[ionspec].uprim   = ion_uprim;
      species_h[ionspec].nu_ss   = ion_vnewk;
      //      fnr_get_float(&namelist_struct, namelist, "z",     &species_h[ionspec].z);
      //      fnr_get_float(&namelist_struct, namelist, "dens",  &species_h[ionspec].dens);
      //      fnr_get_float(&namelist_struct, namelist, "temp",  &species_h[ionspec].temp);
      //      fnr_get_float(&namelist_struct, namelist, "tprim", &species_h[ionspec].tprim); 
      //      fnr_get_float(&namelist_struct, namelist, "fprim", &species_h[ionspec].fprim); 
      //      fnr_get_float(&namelist_struct, namelist, "uprim", &species_h[ionspec].uprim);
      //      fnr_get_float(&namelist_struct, namelist, "vnewk", &species_h[ionspec].nu_ss);
      /*
}

    if(strcmp(type,"electron") == 0) {

      // kinetic electrons will always be last indexed species
      
      fnr_get_float(&namelist_struct, namelist, "z",     &species_h[nspec_in-1].z);
      fnr_get_float(&namelist_struct, namelist, "mass",  &species_h[nspec_in-1].mass);
      fnr_get_float(&namelist_struct, namelist, "dens",  &species_h[nspec_in-1].dens);
      fnr_get_float(&namelist_struct, namelist, "temp",  &species_h[nspec_in-1].temp);
      fnr_get_float(&namelist_struct, namelist, "tprim", &species_h[nspec_in-1].tprim);
      fnr_get_float(&namelist_struct, namelist, "fprim", &species_h[nspec_in-1].fprim);
      fnr_get_float(&namelist_struct, namelist, "uprim", &species_h[nspec_in-1].uprim);
      fnr_get_float(&namelist_struct, namelist, "vnewk", &species_h[nspec_in-1].nu_ss);
      adiabatic_electrons = false;
    }   
  }
  cudaDeviceSynchronize();
*/
  init_species(species_h);
  CP_TO_GPU (species, species_h, sizeof(specie)*nspec_in);
  
  initialized = true;
  cudaDeviceSynchronize();
}

void Parameters::init_species(specie* species)
{
  for(int s=0; s<nspec_in; s++) {
    species[s].vt   = sqrt(species[s].temp / species[s].mass);
    species[s].zstm = species[s].z / sqrt(species[s].temp * species[s].mass);
    species[s].tz   = species[s].temp / species[s].z;
    species[s].zt   = species[s].z / species[s].temp;
    species[s].rho  = sqrt(species[s].temp * species[s].mass) / species[s].z;
    species[s].rho2 = species[s].rho * species[s].rho;
  }
}

// this function copies elements of parameters object into external_parameters_struct externalpars
int Parameters::set_externalpars(external_parameters_struct* externalpars) {
  externalpars->equilibrium_type = equilibrium_type;
  
  //Defaults if we are not using Trinity
  externalpars->trinity_timestep = -1;
  externalpars->trinity_iteration = -1;
  externalpars->trinity_conv_count = -1;
  
  if (restart) externalpars->restart  = 1;
  else externalpars->restart = 0;
  
  externalpars->nstep = nstep;
  externalpars->navg = navg;
  // We increase the margin_cpu_time to make it stricter than gs2
  externalpars->end_time = time(NULL) + avail_cpu_time - margin_cpu_time*1.2;
  
  externalpars->irho    = irho;
  externalpars->rhoc    = rhoc;
  externalpars->eps     = eps;
  externalpars->bishop  = bishop;
  externalpars->nperiod = nperiod;
  externalpars->ntheta  = nz_in;
  //    printf("nz_in is %d\n", externalpars->ntheta);
  
  /* Miller parameters*/
  externalpars->rgeo_local = rmaj;
  externalpars->rgeo_lcfs  = rmaj;
  externalpars->akappa     = akappa;
  externalpars->akappri    = akappri;
  externalpars->tri        = tri;
  externalpars->tripri     = tripri;
  externalpars->shift      = shift;
  externalpars->qinp       = qsf;
  externalpars->shat       = shat;
  
  // EGH These appear to be redundant
  //externalpars->asym = asym;
  //externalpars->asympri = asympri;
  
  /* Other geometry parameters - Bishop/Greene & Chance*/
  externalpars->beta_prime_input  = beta_prime_input;
  externalpars->s_hat_input       = s_hat_input;
  
  /*Flow shear*/
  externalpars->g_exb = g_exb;
  
  /* Species parameters... I think allowing 20 species should be enough!*/
  
  externalpars->ntspec = nspec_in;
  
  for (int i=0;i<nspec_in;i++){
    externalpars->dens[i]  = species_h[i].dens;
    externalpars->temp[i]  = species_h[i].temp;
    externalpars->fprim[i] = species_h[i].fprim;
    externalpars->tprim[i] = species_h[i].tprim;
    externalpars->nu[i]    = species_h[i].nu_ss;
  }
  return 0;
}

// this function copies elements of external_parameters_struct externalpars into parameters object
int Parameters::import_externalpars(external_parameters_struct* externalpars) {
  equilibrium_type = externalpars->equilibrium_type ;
  if (externalpars->restart==1) restart  = true;
  else if (externalpars->restart==2){
    restart  = true;
    zero_restart_avg = true;
  }
  else restart = false;
  
  if (externalpars->nstep > nstep) {
    printf("ERROR: nstep has been increased above the default value. nstep must be less than or equal to what is in the input file");
    abort();
  }
  trinity_timestep   = externalpars->trinity_timestep;
  trinity_iteration  = externalpars->trinity_iteration;
  trinity_conv_count = externalpars->trinity_conv_count;

  nstep    = externalpars->nstep;
  navg     = externalpars->navg;
  end_time = externalpars->end_time;
  irho     = externalpars->irho ;
  rhoc     = externalpars->rhoc ;
  eps      = externalpars->eps;
  bishop   = externalpars->bishop ;
  nperiod  = externalpars->nperiod ;
  nz_in    = externalpars->ntheta ;
  // NB NEED TO SET EPS IN TRINITY!!!
  //eps = rhoc/rmaj;
  
  /* Miller parameters*/
  rmaj     = externalpars->rgeo_local ;
  r_geo    = externalpars->rgeo_lcfs ;
  akappa   = externalpars->akappa ;
  akappri  = externalpars->akappri ;
  tri      = externalpars->tri ;
  tripri   = externalpars->tripri ;
  shift    = externalpars->shift ;
  qsf      = externalpars->qinp ;
  shat     = externalpars->shat ;

  // EGH These appear to be redundant
  //asym = externalpars->asym ;
  //asympri = externalpars->asympri ;
  
  /* Other geometry parameters - Bishop/Greene & Chance*/
  beta_prime_input = externalpars->beta_prime_input ;
  s_hat_input      = externalpars->s_hat_input ;
  
  /*Flow shear*/
  g_exb = externalpars->g_exb ;
  
  /* Species parameters... I think allowing 20 species should be enough!*/
  int oldnSpecies = nspec_in;
  nspec_in = externalpars->ntspec ;
  
  if (nspec_in!=oldnSpecies){
    printf("oldnSpecies=%d,  nSpecies=%d\n", oldnSpecies, nspec_in);
    printf("Number of species set in get_fluxes must equal number of species in gx input file\n");
    exit(1);
  }
  if (debug) printf("nSpecies was set to %d \n", nspec_in);
  for (int i=0;i<nspec_in;i++){
    species_h[i].dens = externalpars->dens[i] ;
    species_h[i].temp = externalpars->temp[i] ;
    species_h[i].fprim = externalpars->fprim[i] ;
    species_h[i].tprim = externalpars->tprim[i] ;
    species_h[i].nu_ss = externalpars->nu[i] ;
  }
  cudaMemcpy(species, species_h, sizeof(specie)*nspec_in, cudaMemcpyHostToDevice);
  
  //jtwist should never be < 0. If we set jtwist < 0 in the input file,
  // this triggers the use of jtwist_square... i.e. jtwist is 
  // set to what it needs to make the box square at the outboard midplane
  //  printf("jtwist = %i \n \n", jtwist);
  if (jtwist < 0) {
    int jtwist_square;
    // determine value of jtwist needed to make X0~Y0
    jtwist_square = (int) round(2*M_PI*abs(shat)*Zp); // Use Zp here or Z0?
    if (jtwist_square == 0) jtwist_square = 1;
    // as currently implemented, there is no way to manually set jtwist from input file
    // there could be some switch here where we choose whether to use
    // jtwist_in or jtwist_square
    jtwist = jtwist_square*2;
  }
  //  printf("(1) x0 = %f \n \n", x0);
  //  printf("jtwist = %i \n \n", jtwist);
  // BD This is where jtwist is set: 
  if (jtwist!=0 && abs(shat)>1.e-6) x0 = y0*jtwist/(2*M_PI*Zp*abs(shat));  
  //  printf("(2) x0 = %f \n \n", x0);
  
  return 0;
}

int Parameters::getint (int const ncid, const char varname[]) {
  int idum, retval, res;
  if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
  if (retval = nc_get_var  (ncid, idum, &res)) ERR(retval);
  if (debug) printf("%s = %i \n",varname, res);
  return res;
}

bool Parameters::getbool (int const ncid, const char varname[]) {
  int idum, ires, retval;
  bool res;
  if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
  if (retval = nc_get_var  (ncid, idum, &ires)) ERR(retval);
  res = (ires!=0) ? true : false ;
  if (debug) printf("%s = %i \n", varname, ires);
  return res;
}

float Parameters::getfloat (int const ncid, const char varname[]) {
  int idum, retval;
  float res;
  if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
  if (retval = nc_get_var  (ncid, idum, &res)) ERR(retval);
  if (debug) printf("%s = %f \n",varname, res);
  return res;
}


