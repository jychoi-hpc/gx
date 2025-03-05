#include "parameters.h"
#include "ncdf.h"
#include <netcdf.h>
#include <netcdf_par.h>
#include <iostream>
#include "version.h"
#include <unistd.h>
using namespace std;

Parameters::Parameters(char* filename, int iproc_in, int nprocs_in, MPI_Comm mpcom_in) {
  initialized = false;

  iproc = iproc_in;
  nprocs = nprocs_in;
  mpcom = mpcom_in;

  // some cuda parameters (not from input file)
  int dev; 
  cudaGetDevice(&dev);
  if (false) printf("device id = %d \n",dev);

  strcpy (run_name, filename);
  strcpy(nml_file, run_name);
  strcat(nml_file, ".in");
  
  const auto nml = toml::parse(nml_file);
  auto tnml = nml;
  if (nml.contains("Restart")) tnml = toml::find(nml, "Restart");
  restart           = toml::find_or <bool>   (tnml, "restart", false  );
  append_on_restart = toml::find_or <bool> (tnml, "append_on_restart", true);
  strcpy(default_restart_filename, filename);
  strcat(default_restart_filename, ".restart.nc");
}

Parameters::~Parameters() {
  cudaDeviceSynchronize();
  if(initialized) {
    free(species_h);
  }
}

void Parameters::get_nml_vars(NetCDF* ncdf)
{
  nc_inputs_ = ncdf->nc_inputs;

  const auto nml = toml::parse(nml_file);

  debug  = toml::find_or <bool> (nml, "debug",   false);

  // read parameters from various TOML groups in input file
  get_Dimensions(nml);
  get_Domain(nml);
  get_Time(nml);
  get_Initialization(nml);
  get_Restart(nml);
  get_Dissipation(nml);
  get_KREHM(nml);
  get_Expert(nml);
  get_Diagnostics(nml);
  get_Resize(nml);
  get_Forcing(nml);
  get_Boltzmann(nml);
  get_Geometry(nml);
  get_Physics(nml);
  get_species(nml);
  
  initialized = true;
  printf(ANSI_COLOR_RESET);    
}

template <typename T>
T Parameters::find_or(const toml::value nml, int ncid, const char varname[], T default_val){
  T val;
  val = toml::find_or (nml, toml::key(varname), default_val);
 
  // store in netcdf
  nc_inputs_->put_var (ncid, varname, val, debug);

  return val;
}

void Parameters::get_Dimensions(const toml::value nml)
{
  auto tnml = nml;
  if (nml.contains("Dimensions")) tnml = toml::find(nml, "Dimensions");
  int ncid = nc_inputs_->put_group("Dimensions");

  nz_in    = find_or (tnml, ncid, "ntheta",    32);
  ny_in    = find_or <int> (tnml, ncid, "ny",        0);
  nx_in    = find_or <int> (tnml, ncid, "nx",        0);
  int nky_in   = find_or <int> (tnml, ncid, "nky",       0);
  int nkx_in   = find_or <int> (tnml, ncid, "nkx",       0);
  nm_in    = find_or <int> (tnml, ncid, "nhermite",  4);
  nl_in    = find_or <int> (tnml, ncid, "nlaguerre", 2);
  nspec_in = find_or <int> (tnml, ncid, "nspecies",  1);
  nperiod  = find_or <int> (tnml, ncid, "nperiod",   1);
  if(nz_in != 1) {
    int ntgrid = nz_in/2 + (nperiod-1)*nz_in; 
    nz_in = 2*ntgrid; // force even
    nc_inputs_->mod_var<int> (ncid, "ntheta", nz_in);
  }

  assert((ny_in > 0 || nky_in > 0) && "must set ny or nky");
  if(nky_in > 0 && ny_in > 0) assert((nky_in == 1 + (ny_in-1)/3) && "nky and ny have been set inconsistenly. only one of these needs to be set.");
  else if(nky_in > 0) {
    ny_in = 3*(nky_in-1) + 1;
    nc_inputs_->mod_var<int> (ncid, "ny", ny_in);
  }

  assert((nx_in > 0 || nkx_in > 0) && "must set nx or nkx");
  if(nkx_in > 0 && nx_in > 0) assert((nkx_in == 1 + 2*((nx_in-1)/3)) && "nkx and nx have been set inconsistenly. only one of these needs to be set.");
  else if(nkx_in > 0) {
    nx_in = ((nkx_in - 1) /  2) * 3 + 1;
    nc_inputs_->mod_var<int> (ncid, "nx", nx_in);
  }
}

void Parameters::get_Domain(const toml::value nml)
{
  auto tnml = nml;
  if (nml.contains("Domain")) tnml = toml::find(nml, "Domain");
  int ncid = nc_inputs_->put_group("Domain");

  y0       = find_or <float>       (tnml, ncid, "y0",          10.0  );
  x0       = find_or <float>       (tnml, ncid, "x0",          -1.0  );
  z0       = find_or <float>       (tnml, ncid, "z0",           1.0  );
  jtwist   = find_or <int>         (tnml, ncid, "jtwist",      -1000 );
  Zp       = find_or <int>         (tnml, ncid, "zp",           2*nperiod-1    );
  // possible values of boundary are:
  // "linked": twist-and-shift BC, with generalization for non-axisymmetric geometry (Martin et al 2018)
  // "forced periodic" (or simply "periodic"): use periodic BCs with no cutting of flux tube
  // "exact periodic": cut flux tube at a location where gds21 = 0, and then use periodic BCs (currently for VMEC geometry only)
  // "continuous drifts": cut flux tube at a location where gbdrift0 = 0, and then use (generalized) twist-and-shift BC (currently for VMEC geometry only)
  // "fix aspect": cut flux tube at a location where y0/x0 takes the desired value, and then use (generalized) twist-and-shift BC (VMEC geometry only)
  boundary = find_or <std::string> (tnml, ncid, "boundary", "linked" );
  nonTwist = find_or <bool>        (tnml, ncid, "nonTwist", false);
  long_wavelength_GK = find_or <bool>   (tnml, ncid, "long_wavelength_GK",   false ); // JFP, long wavelength GK limit where bs = 0, except in quasineutrality where 1 - Gamma0(b) --> b.
  zero_shat_threshold = find_or <float>   (tnml, ncid, "zero_shat_threshold", 1e-5);

  if( boundary == "periodic" || boundary == "exact periodic" || boundary == "forced periodic") { 
    boundary_option_periodic = true;
  } else { 
    boundary_option_periodic = false; 
  }
}

void Parameters::get_Time(const toml::value nml)
{
  auto tnml = nml;
  if (nml.contains("Time")) tnml = toml::find(nml, "Time");
  int ncid = nc_inputs_->put_group("Time");

  dt      = find_or <float> (tnml, ncid, "dt",       0.05 );
  dt_max  = find_or <float> (tnml, ncid, "dt_max",   static_cast<float>(dt) );
  dt_min  = find_or <float> (tnml, ncid, "dt_min",   1e-7 );
  fixed_dt = find_or <bool> (tnml, ncid, "fixed_dt", false );
  nstep   = find_or <int>   (tnml, ncid, "nstep",   2e9 );
  nstep_restart   = find_or <int>   (tnml, ncid, "nstep_restart",   -1 );
  scheme = find_or <string> (tnml, ncid, "scheme",    "rk3"   );
  cfl = find_or <float> (tnml, ncid, "cfl", 0.9);
  stages = find_or <int>    (tnml, ncid, "stages",  10   );
  t_max = find_or <float> (tnml, ncid, "t_max", 1.e20);
  t_add = find_or <float> (tnml, ncid, "t_add", -1.0);

  if (scheme == "sspx3") scheme_opt = Tmethod::sspx3;
  if (scheme == "k10")   scheme_opt = Tmethod::k10;
  if (scheme == "rk4")   scheme_opt = Tmethod::rk4;
  if (scheme == "rk3")   scheme_opt = Tmethod::rk3;
  if (scheme == "sspx2") scheme_opt = Tmethod::sspx2;
}

void Parameters::get_Initialization(const toml::value nml)
{
  auto tnml = nml;
  if (nml.contains("Initialization")) tnml = toml::find(nml, "Initialization");
  int ncid = nc_inputs_->put_group("Initialization");

  init_field = find_or <string> (tnml, ncid, "init_field", "density");
  init_amp   = find_or <float>  (tnml, ncid, "init_amp",   1.0e-5   );
  kpar_init  = find_or <float>  (tnml, ncid, "kpar_init",     0.0   );
  ikpar_init  = find_or <int>  (tnml, ncid, "ikpar_init",     static_cast<int>(kpar_init)  );
  random_init     = find_or <bool> (tnml, ncid, "random_init",     false);
  gaussian_init = find_or <bool> (tnml, ncid, "gaussian_init", false);
  if( tnml.contains("gaussian_envelope_constant_coefficient") && !gaussian_init )
  {
      std::cerr << "WARNING: gaussian_envelope_constant_coefficient was specified, but gaussian_init was not, so this will be ignored." << std::endl;
  }
  gauss_env_const_coeff = find_or <float> (tnml, ncid, "gaussian_envelope_constant_coefficient", 1.0);
  if( tnml.contains("gaussian_envelope_sine_coefficient") && !gaussian_init )
  {
      std::cerr << "WARNING: gaussian_envelope_sine_coefficient was specified, but gaussian_init was not, so this will be ignored." << std::endl;
  }
  gauss_env_sin_coeff = find_or <float> (tnml, ncid, "gaussian_envelope_sine_coefficient", 0.0);
  if( tnml.contains("gaussian_width") && !gaussian_init )
  {
      std::cerr << "WARNING: gaussian_width was specified, but gaussian_init was not, so this will be ignored." << std::endl;
  }
  gaussian_width  = find_or <float>  (tnml, ncid, "gaussian_width",     0.5   );
  init_electrons_only     = find_or <bool> (tnml, ncid, "init_electrons_only",     false);
  densfac = find_or <float> (tnml, ncid, "densfac", 1.0);
  uparfac = find_or <float> (tnml, ncid, "uparfac", 1.0);
  tparfac = find_or <float> (tnml, ncid, "tparfac", 1.0);
  tprpfac = find_or <float> (tnml, ncid, "tprpfac", 1.0);
  qparfac = find_or <float> (tnml, ncid, "qparfac", 1.0);
  qprpfac = find_or <float> (tnml, ncid, "qprpfac", 1.0);
  random_seed = find_or <unsigned int> (tnml, ncid, "random_seed", 22);
  if (random_init) {
    ikpar_init = 0; 
    nc_inputs_->mod_var<int> (ncid, "ikpar_init", ikpar_init);
  }

  if     ( init_field == "density") { initf = inits::density; }
  else if( init_field == "upar"   ) { initf = inits::upar   ; }
  else if( init_field == "tpar"   ) { initf = inits::tpar   ; }
  else if( init_field == "tperp"  ) { initf = inits::tperp  ; }
  else if( init_field == "qpar"   ) { initf = inits::qpar   ; }
  else if( init_field == "qperp"  ) { initf = inits::qperp  ; }
  else if( init_field == "all"  ) { initf = inits::all  ; }
}

void Parameters::get_Restart(const toml::value nml)
{
  auto tnml = nml;
  if (nml.contains("Restart")) tnml = toml::find(nml, "Restart");
  int ncid = nc_inputs_->put_group("Restart");

  restart           = find_or <bool>   (tnml, ncid, "restart",                 false  );
  restart_if_exists = find_or <bool>   (tnml, ncid, "restart_if_exists",       false  );
  save_for_restart  = find_or <bool>   (tnml, ncid, "save_for_restart",         true  );
  restart_to_file   = find_or <string> (tnml, ncid, "restart_to_file", default_restart_filename);
  restart_from_file = find_or <string> (tnml, ncid, "restart_from_file", default_restart_filename);  
  restart_with_perturb = find_or <bool> (tnml, ncid, "restart_with_perturb", false  );
  append_on_restart = find_or <bool> (tnml, ncid, "append_on_restart", true);
  scale             = find_or <float>  (tnml, ncid, "scale",                      1.0 );
  nsave   = find_or <int>   (tnml, ncid, "nsave", 10000 );
  nsave = max(1, nsave);
  if (restart_if_exists) {
    if (access(restart_from_file.c_str(), F_OK) == 0) restart = true;
    else restart = false;
  }
}

void Parameters::get_Dissipation(const toml::value nml)
{
  auto tnml = nml;
  if (nml.contains("Dissipation")) tnml = toml::find(nml, "Dissipation");
  int ncid = nc_inputs_->put_group("Dissipation");

  closure_model  = find_or <string> (tnml, ncid, "closure_model", "none" );
  smith_par_q    = find_or <int>    (tnml, ncid, "smith_par_q",        3 );
  smith_perp_q   = find_or <int>    (tnml, ncid, "smith_perp_q",       3 );
  D_HB       = find_or <float>  (tnml, ncid, "D_HB",          1.0   ); 
  w_osc      = find_or <float>  (tnml, ncid, "w_osc",         0.0   ); 
  D_hyper    = find_or <float>  (tnml, ncid, "D_hyper",       0.1   ); 
  nu_hyper_z = find_or <float>  (tnml, ncid, "nu_hyper_z",    0.1 ); 
  nu_hyper_l = find_or <float>  (tnml, ncid, "nu_hyper_l",    0.0   ); 
  nu_hyper_m = find_or <float>  (tnml, ncid, "nu_hyper_m",    1.0   ); 
  nu_hyper_lm = find_or <float> (tnml, ncid, "nu_hyper_lm",   0.0   ); 
  p_hyper    = find_or <int>    (tnml, ncid, "p_hyper",         2   ); 
  p_hyper_z  = find_or <int>    (tnml, ncid, "p_hyper_z",       6   ); 
  p_hyper_l  = find_or <int>    (tnml, ncid, "p_hyper_l",       6   ); 
  p_hyper_m  = find_or <int>    (tnml, ncid, "p_hyper_m",       fmin(20, nm_in/2) ); 
  p_hyper_lm = find_or <int>    (tnml, ncid, "p_hyper_lm",      6   ); 
  p_HB       = find_or <int>    (tnml, ncid, "p_HB",            2   ); 
  hyper      = find_or <bool>   (tnml, ncid, "hyper",         false ); 
  HB_hyper   = find_or <bool>   (tnml, ncid, "HB_hyper",      false ); 
  hypercollisions_const = find_or <bool> (tnml, ncid, "hypercollisions_const", false);
  hypercollisions_kz = find_or <bool> (tnml, ncid, "hypercollisions_kz", false);
  hypercollisions_kz = find_or <bool> (tnml, ncid, "hypercollisions", hypercollisions_kz); // "hypercollisions" now gives hypercollisions_kz
  hyperz = find_or <bool> (tnml, ncid, "hyperz", false);

  closure_model_opt = Closure::none   ;
  if( closure_model == "beer4+2") {
    if(iproc==0) printf("\nUsing Beer 4+2 closure model. Overriding nm=4, nl=2\n\n");
    nm_in = 4;
    nl_in = 2;
    closure_model_opt = Closure::beer42;
  } else if (closure_model == "smith_perp") { closure_model_opt = Closure::smithperp;
  } else if (closure_model == "smith_par")  { closure_model_opt = Closure::smithpar; 
  }

  if(iproc==0) {
    if(hypercollisions_kz) printf("Using hypercollisions with coefficient proportional to kz (default).\n");
    if(hypercollisions_const) printf("Using hypercollisions with const coefficient.\n");
    if(hyper) printf("Using perpendicular hyperdiffusion.\n");
    if(hyperz) printf("Using parallel hyperdiffusion.\n");
  }
}

void Parameters::get_KREHM(const toml::value nml)
{
  auto tnml = nml;
  if (nml.contains("KREHM")) tnml = toml::find(nml, "KREHM");
  int ncid = nc_inputs_->put_group("KREHM");

  krehm             = find_or <bool>  (tnml, ncid, "krehm",     false );
  gx = !krehm;
  rho_i             = find_or <float> (tnml, ncid, "rho_i",       1.0 );
  d_e               = find_or <float> (tnml, ncid, "d_e",         1.0 );
  nu_ei             = find_or <float> (tnml, ncid, "nu_ei",       0.0 );
  eta               = find_or <float> (tnml, ncid, "eta",         0.0 );
  zt                = find_or <float> (tnml, ncid, "zt",          1.0 );
  harris_sheet      = find_or <bool>  (tnml, ncid, "harris_sheet", false);
  periodic_equilibrium = find_or <bool> (tnml, ncid, "periodic_equilibrium", false);
  island_coalesce = find_or <bool> (tnml, ncid, "island_coalesce", false);
  k0                = find_or <float> (tnml, ncid, "k0", 10.0);
  gaussian_tube     = find_or <bool> (tnml, ncid, "gaussian_tube", false);
  random_gaussian   = find_or <bool> (tnml, ncid, "random_gaussian", false);
  kc                = find_or <float> (tnml, ncid, "kc", 25.0);
  rho_s = rho_i*sqrtf(zt/2);
  if(eta>0.0) nu_ei = eta/d_e/d_e;
  // allow hypercollisions = true to give correct behavior for KREHM (which always uses const option)
  if(krehm && hypercollisions_kz) {hypercollisions_const = true; hypercollisions_kz = false;}
}

void Parameters::get_Expert(const toml::value nml)
{
  auto tnml = nml;
  if (nml.contains("Expert")) tnml = toml::find(nml, "Expert");
  int ncid = nc_inputs_->put_group("Expert");

  i_share     = find_or <int>    (tnml, ncid, "i_share",         8 );
  int dev;
  cudaDeviceProp prop;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  size_t maxSharedSize;
  maxSharedSize = prop.sharedMemPerBlockOptin > 0 ? prop.sharedMemPerBlockOptin : prop.sharedMemPerBlock ;
  int i_share_max = maxSharedSize/((nl_in+2)*(nm_in+4)*sizeof(cuComplex));
  if(i_share > i_share_max) {
    if(iproc==0) printf("Using i_share = %d would exceed shared memory limits. Setting i_share = %d instead.\n", i_share, i_share_max);
    i_share = i_share_max;
  }
  
  dealias_kz  = find_or <bool>   (tnml, ncid, "dealias_kz",  false );
  nreal       = find_or <int>    (tnml, ncid, "nreal",           1 );  
  local_limit = find_or <bool>   (tnml, ncid, "local_limit", false );
  init_single = find_or <bool>   (tnml, ncid, "init_single", false );
  ikx_single  = find_or <int>    (tnml, ncid, "ikx_single",      0 );
  iky_single  = find_or <int>    (tnml, ncid, "iky_single",      1 );
  ikx_fixed   = find_or <int>    (tnml, ncid, "ikx_fixed",      -1 );
  iky_fixed   = find_or <int>    (tnml, ncid, "iky_fixed",      -1 );
  eqfix       = find_or <bool>   (tnml, ncid, "eqfix",       false );
  secondary   = find_or <bool>   (tnml, ncid, "secondary",   false );
  phi_ext     = find_or <float>  (tnml, ncid, "phi_ext",       0.0 );
  source      = find_or <string> (tnml, ncid, "source",  "default" );
  tp_t0       = find_or <float>  (tnml, ncid, "t0",           -1.0 );
  tp_tf       = find_or <float>  (tnml, ncid, "tf",           -1.0 );
  tprim0      = find_or <float>  (tnml, ncid, "tprim0",       -1.0 );
  tprimf      = find_or <float>  (tnml, ncid, "tprimf",       -1.0 );
  use_NCCL    = find_or <bool>   (tnml, ncid, "use_NCCL",    true );
  use_fft_callbacks    = find_or <bool>   (tnml, ncid, "use_fft_callbacks",    false );
  damp_ends_widthfrac = find_or <float> (tnml, ncid, "damp_ends_widthfrac", 1./8.);
  damp_ends_amp = find_or <float> (tnml, ncid, "damp_ends_amp", 0.1);

  if (eqfix && iproc==0 && scheme_opt == Tmethod::k10) {
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

  if( source == "phiext_full") {
    source_option = PHIEXT;
    if(iproc==0) printf("Running Rosenbluth-Hinton zonal flow calculation\n");
  }
}

void Parameters::get_Diagnostics(const toml::value nml)
{
  auto tnml = nml;
  if (nml.contains("Diagnostics")) tnml = toml::find(nml, "Diagnostics");
  int ncid = nc_inputs_->put_group("Diagnostics");

  nwrite  = find_or <int>   (tnml, ncid, "nwrite", 1000);
  navg    = find_or <int>   (tnml, ncid, "navg",   1000);
  nwrite_big  = find_or <int>   (tnml, ncid, "nwrite_big", (long)  nwrite*100 );
  fixed_amplitude   = find_or <bool> (tnml, ncid, "fixed_amplitude", false);
  write_omega       = find_or <bool> (tnml, ncid, "omega",          false );
  write_free_energy = find_or <bool> (tnml, ncid, "free_energy",    true  );
  write_fluxes      = find_or <bool> (tnml, ncid, "fluxes",         false );
  write_moms        = find_or <bool> (tnml, ncid, "moments",           false );
  write_fields      = find_or <bool> (tnml, ncid, "fields",         false );

  if (write_omega && fixed_amplitude) {
    if (nonlinear_mode || nwrite < 3) fixed_amplitude = false;
  }
}

void Parameters::get_Resize(const toml::value nml)
{
  auto tnml = nml;
  if (nml.contains("Resize")) tnml = find(nml, "Resize");
  int ncid = nc_inputs_->put_group("Resize");

  domain_change = find_or <bool> (tnml, ncid, "domain_change", false);
  z0_mult = find_or <int> (tnml, ncid, "z0_mult", 1);  assert( (z0_mult > 0) && "z0_mult must be an integer >= 1");
  y0_mult = find_or <int> (tnml, ncid, "y0_mult", 1);  assert( (y0_mult > 0) && "y0_mult must be an integer >= 1");
  x0_mult = find_or <int> (tnml, ncid, "x0_mult", 1);  assert( (x0_mult > 0) && "x0_mult must be an integer >= 1");
  nx_mult = find_or <int> (tnml, ncid, "nx_mult", 1);  assert( (nx_mult > 0) && "nx_mult must be an integer >= 1");
  ny_mult = find_or <int> (tnml, ncid, "ny_mult", 1);  assert( (ny_mult > 0) && "ny_mult must be an integer >= 1");
  nm_add  = find_or <int> (tnml, ncid, "nm_add" , 0);  
  nl_add  = find_or <int> (tnml, ncid, "nl_add" , 0);  
  ns_add  = find_or <int> (tnml, ncid, "ns_add" , 0);  assert( (ns_add >= 0) && "ns_add must be an integer >= 0");
  
  ntheta_mult = find_or <int> (tnml, ncid, "nz_mult", 1);
  assert( (ntheta_mult > 0) && "ntheta_mult must be an integer >= 1");

  if (!domain_change) {
    assert ((nx_mult == 1) && "When domain_change is false, nx_mult must be 1");
    assert ((ny_mult == 1) && "When domain_change is false, ny_mult must be 1");
    assert ((ntheta_mult == 1) && "When domain_change is false, ntheta_mult must be 1");
    assert ((x0_mult == 1) && "When domain_change is false, x0_mult must be 1");
    assert ((y0_mult == 1) && "When domain_change is false, y0_mult must be 1");
    assert ((z0_mult == 1) && "When domain_change is false, z0_mult must be 1");
    assert ((nl_add == 0) && "When domain_change is false, nl_add must be 0");
    assert ((nm_add == 0) && "When domain_change is false, nm_add must be 0");
  }

  if (domain_change) {
    printf( "You are changing the simulation domain with this input file. \n");
    if (x0_mult > 1) printf("Compared to the restart file, you have increased x0 by a factor of %d \n",x0_mult);
    if (y0_mult > 1) printf("Compared to the restart file, you have increased y0 by a factor of %d \n",y0_mult);
    if (z0_mult > 1) printf("Compared to the restart file, you have increased z0 by a factor of %d \n",z0_mult);
    if (nx_mult > 1) printf("Compared to the restart file, you have increased nx by a factor of %d \n",nx_mult);
    if (ny_mult > 1) printf("Compared to the restart file, you have increased ny by a factor of %d \n",ny_mult);
    if (ntheta_mult > 1) printf("Compared to the restart file, you have increased nx ntheta a factor of %d \n",ntheta_mult);
    if (nl_add > 0) printf("Compared to the restart file, you have added %d Laguerre basis elements. \n",nl_add);
    if (nl_add < 0) printf("Compared to the restart file, you have removed %d Laguerre basis elements. \n",-nl_add);
    if (nm_add > 0) printf("Compared to the restart file, you have added %d Hermite basis elements. \n",nm_add);
    if (nm_add < 0) printf("Compared to the restart file, you have removed %d Hermite basis elements. \n",-nm_add);
    if (ns_add > 0) printf("Compared to the restart file, you have added %d species. \n",ns_add);
  }    
}

void Parameters::get_Forcing(const toml::value nml)
{
  auto tnml = nml;
  if (nml.contains("Forcing")) tnml = find(nml, "Forcing");
  int ncid = nc_inputs_->put_group("Forcing");

  forcing_type  = find_or <string> (tnml, ncid, "forcing_type",    "Kz" ); //Needed for Helicity Injection; forcing_type=HeliInj
  stir_field    = find_or <string> (tnml, ncid, "stir_field", "density" ); //Needed for Helicity Injection - Field to be perturbed
  forcing_amp   = find_or <float>  (tnml, ncid, "forcing_amp",      1.0 );
  pos_forcing_amp = find_or <float> (tnml, ncid, "pos_forcing_amp",  1.0); //Needed for Helicity Injection - Positive Amplitude
  neg_forcing_amp = find_or <float> (tnml, ncid, "neg_forcing_amp",  1.0); //Needed for Helicity Injection - Negative Amplitude
  forcing_index = find_or <int>    (tnml, ncid, "forcing_index",    1   );
  forcing_init  = find_or <bool>   (tnml, ncid, "forcing_init",   false ); 
  no_fields     = find_or <bool>   (tnml, ncid, "no_fields",      false );
  forcing_kz    = find_or <int>    (tnml, ncid, "forcing_kz",         0 ); //Needed for Helicity Injection - Mode of kz perturbed
  forcing_k2min = find_or <int>    (tnml, ncid, "forcing_k2min",      0 ); //Needed for Helicity Injection - Minimum kperp that can be perturbed
  forcing_k2max = find_or <int>    (tnml, ncid, "forcing_k2max",      0 ); //Needed for Helicity Injection - Maximum mode of kperp that can be perturbed

  if     ( stir_field == "density") { stirf = stirs::density; }
  else if( stir_field == "upar"   ) { stirf = stirs::upar   ; }
  else if( stir_field == "tpar"   ) { stirf = stirs::tpar   ; }
  else if( stir_field == "tperp"  ) { stirf = stirs::tperp  ; }
  else if( stir_field == "qpar"   ) { stirf = stirs::qpar   ; }
  else if( stir_field == "qperp"  ) { stirf = stirs::qperp  ; }
  else if( stir_field == "ppar"   ) { stirf = stirs::ppar   ; }
  else if( stir_field == "pperp"  ) { stirf = stirs::pperp  ; }
}

void Parameters::get_Boltzmann(const toml::value nml)
{
  auto tnml = nml;
  if (nml.contains("Boltzmann")) tnml = find(nml, "Boltzmann");
  int ncid = nc_inputs_->put_group("Boltzmann");

  add_Boltzmann_species = find_or <bool>   (tnml, ncid, "add_Boltzmann_species", false);
  Btype                 = find_or <string> (tnml, ncid, "Boltzmann_type", "electrons" );
  // allow some sloppiness here:
  if (Btype == "Electrons") Boltzmann_opt = BOLTZMANN_ELECTRONS;
  if (Btype == "Electron" ) Boltzmann_opt = BOLTZMANN_ELECTRONS;
  if (Btype == "Ions")      Boltzmann_opt = BOLTZMANN_IONS;
  if (Btype == "Ion" )      Boltzmann_opt = BOLTZMANN_IONS;
  if (Btype == "electrons") Boltzmann_opt = BOLTZMANN_ELECTRONS;
  if (Btype == "electron" ) Boltzmann_opt = BOLTZMANN_ELECTRONS;
  if (Btype == "ions")      Boltzmann_opt = BOLTZMANN_IONS;
  if (Btype == "ion" )      Boltzmann_opt = BOLTZMANN_IONS;
  // "tau_fac" is the multiplier for the Boltzmann response. it is defined as T_ref/T_Boltzmann
  tau_fac  = find_or <float> (tnml, ncid, "tau_fac", 1.0);

  ///////////////////////////////////////////////////////////////////////
  //                                                                   //
  // Testing that we have working options                              //
  //                                                                   //
  ///////////////////////////////////////////////////////////////////////

  all_kinetic = true;
  if (add_Boltzmann_species) all_kinetic = false;

  if (all_kinetic) {
    assert( (iphi00 <= 0)
	    && "Specifying all species are kinetic and also iphi00 > 0 is not allowed");
    assert( !add_Boltzmann_species
	    && "Specifying all species are kinetic and also add_Boltzmann_species is not allowed");
  }

  if (!all_kinetic) {
    assert( (tau_fac >= 0.)
	    && "Specifying all_kinetic == false and also tau_fac < 0. is not allowed");
    assert( ( (Boltzmann_opt==BOLTZMANN_ELECTRONS) || (Boltzmann_opt==BOLTZMANN_IONS) )
	    && "If all_kinetic == false then a legal Boltzmann_opt must be specified");
    assert( add_Boltzmann_species
	    && "If all_kinetic == false then add_Boltzmann_species should be true");
  }
}

void Parameters::get_Physics(const toml::value nml)
{
  auto tnml = nml;
  if (nml.contains("Physics")) tnml = find(nml, "Physics");
  int ncid = nc_inputs_->put_group("Physics");

  beta = find_or <float> (tnml, ncid, "beta",    0.0 );
  nonlinear_mode = find_or <bool>   (tnml, ncid, "nonlinear_mode",    nonlinear_mode );  linear = !nonlinear_mode;

  g_exb    = find_or <float> (tnml, ncid, "g_exb",       0.0 );
  // Default to ExB shear on if g_exb is nonzero
  ExBshear = find_or <bool> (tnml, ncid, "ExBshear", ( g_exb != 0.0 )  );
  // Default to including the phase factor
  ExBshear_phase = find_or <bool> (tnml, ncid, "ExBshear_phase",  true); // If false, neglect phase correction in FFT. Only relevant for nonlinear simulations.

  if (!ExBshear) ExBshear_phase = false; 
  fphi     = find_or <float> (tnml, ncid, "fphi",        1.0);
  fapar    = find_or <float> (tnml, ncid, "fapar",       beta > 0.0? 1.0 : 0.0);
  fbpar    = find_or <float> (tnml, ncid, "fbpar",       beta > 0.0? 1.0 : 0.0);
  // electromagnetic doesn't make sense with adiabatic electrons
  if (!all_kinetic && Boltzmann_opt == BOLTZMANN_ELECTRONS && beta > 0.0 ) {
    std::cerr << "Boltzmann Electrons specified with non-zero beta. This is invalid. Setting beta = 0 instead." << std::endl;
    beta = 0.0; 
    fapar = 0.0;
    fbpar = 0.0;
    nc_inputs_->mod_var<float>(ncid, "beta", 0.0);
    nc_inputs_->mod_var<float>(ncid, "fapar", 0.0);
    nc_inputs_->mod_var<float>(ncid, "fbpar", 0.0);
  }

  if (!all_kinetic && Boltzmann_opt == BOLTZMANN_IONS && fbpar > 0.0 && beta > 0.0 ) {
    std::cerr << "Boltzmann Ions specified with non-zero fbpar. This is currently unsupported. Setting fbpar = 0 instead." << std::endl;
    nc_inputs_->mod_var<float>(ncid, "fbpar", 0.0);
  }

  ei_colls = find_or <bool> (tnml, ncid, "ei_colls", true);
  coll_conservation = find_or <bool> (tnml, ncid, "coll_conservation", true);

  if(iproc==0 && all_kinetic && beta == 0.0 && !krehm) {
    printf("Warning: you are using kinetic electrons in a purely electrostatic calculation (beta==0.0).\n");
    printf("This will require a very small dt to resolve the high-frequency electrostatic shear Alfven wave (omega_H mode).\n");
    printf("It is recommended to instead use a small but finite value of beta to alleviate the timestep restriction.\n");
  }
}

void Parameters::get_Geometry(const toml::value nml)
{
  auto tnml = nml;
  if (nml.contains("Geometry")) tnml = find(nml, "Geometry");
  int ncid = nc_inputs_->put_group("Geometry");

  geo_option  = find_or <string> (tnml, ncid, "geo_option", "none");
  geofilename = find_or <string> (tnml, ncid, "geo_file", geofilename );  

  drhodpsi    = find_or <float> (tnml, ncid, "drhodpsi", 1.0 );
  kxfac       = find_or <float> (tnml, ncid, "kxfac",    1.0 );
  rmaj        = find_or <float> (tnml, ncid, "Rmaj",     1.0 );
  rmaj        = find_or <float> (tnml, ncid, "rmaj",    (double) rmaj );
  r_geo       = find_or <float> (tnml, ncid, "R_geo",     1.0 );
  shift       = find_or <float> (tnml, ncid, "shift",    0.0 );
  eps         = find_or <float> (tnml, ncid, "eps",    0.167 );
  qsf         = find_or <float> (tnml, ncid, "qinp",     1.4 );
  akappa      = find_or <float> (tnml, ncid, "akappa",     1.0 );
  akappri     = find_or <float> (tnml, ncid, "akappri",    0.0 );
  tri         = find_or <float> (tnml, ncid, "tri",        1.0 );
  tripri      = find_or <float> (tnml, ncid, "tripri",     0.0 );
  beta_prime_input    = find_or <float> (tnml, ncid, "beta_prime_input", -1.0 );
  beta_prime_input    = find_or <float> (tnml, ncid, "betaprim", (double) beta_prime_input );
  zero_shat   = find_or <bool>  (tnml, ncid, "zero_shat", false); 
  // Set zero_shat = true in the input file when the actual magnetic shear is inconveniently low. 
  if(geo_option=="s-alpha" || geo_option=="slab" || geo_option=="const-curv" || igeo==0) {
    shat        = find_or <float> (tnml, ncid, "shat",     0.8 );
  } else {
    // shat will be taken from geometry file; do nothing here
  }

  RBzeta_override = find_or<float>( tnml, ncid, "RBzeta", 0.0 ); // For explicitly setting I(psi) for flying-slab simulations
  if( geo_option != "slab" && RBzeta_override != 0.0 )
  {
    printf("ERROR: R B_zeta has been set explicitly, but this is only legal in a slab! Detected geo_option is %s (not 'slab')", geo_option.c_str() );
  }
  
  // the following parameters are exclusively for interfacing 
  // with the GS2 geometry module via eiktest
  // for parameter definitions, see src/geo/geometry.f90 in the GS2 repo
  // the defaults are the same as the defaults in gs2's eiktest.f90
  rhoc = find_or <float> (tnml, ncid, "rhoc", 0.5); 
  geoType = find_or <int> (tnml, ncid, "geoType", 0); 
  iflux = find_or <int> (tnml, ncid, "iflux", 0); 
  delrho = find_or <float> (tnml, ncid, "delrho", 0.01); 
  bishop = find_or <int> (tnml, ncid, "bishop", 0); 
  irho = find_or <int> (tnml, ncid, "irho", 2); 
  isym = find_or <int> (tnml, ncid, "isym", 0); 
  eqfile = find_or <string> (tnml, ncid, "eqfile", "none" );  
  s_hat_input = find_or <float> (tnml, ncid, "s_hat_input", 1.0 );
  if(abs(s_hat_input) < zero_shat_threshold) {
    s_hat_input = 1e-8;
  }
  p_prime_input = find_or <float> (tnml, ncid, "p_prime_input", -2.0 );
  invLp_input = find_or <float> (tnml, ncid, "invLp_input", 5.0 );
  alpha_input = find_or <float> (tnml, ncid, "alpha_input", 0.0 );
  efit_eq = find_or <bool> (tnml, ncid, "efit_eq", false);
  dfit_eq = find_or <bool> (tnml, ncid, "dfit_eq", false);
  gen_eq = find_or <bool> (tnml, ncid, "gen_eq", false);
  ppl_eq = find_or <bool> (tnml, ncid, "ppl_eq", false);
  local_eq = find_or <bool> (tnml, ncid, "local_eq", false);
  idfit_eq = find_or <bool> (tnml, ncid, "idfit_eq", false);
  chs_eq = find_or <bool> (tnml, ncid, "chs_eq", false);
  transp_eq = find_or <bool> (tnml, ncid, "transp_eq", false);
  gs2d_eq = find_or <bool> (tnml, ncid, "gs2d_eq", false);
}

void Parameters::get_species(const toml::value nml)
{
  // note: no writing to netcdf in this method.
  // species parameters are written to netcdf via the NcSpecies class
  auto tnml = nml;

  species_h = (specie *) malloc(nspec_in*sizeof(specie));
  if (nml.contains("species")) {
    for (int is=0; is < nspec_in; is++) {
      species_h[is].nu_ss = 0.;
      species_h[is].temp = 1.;
      
      species_h[is].z     = toml::find <float>  (nml, "species", "z",     is);
      species_h[is].mass  = toml::find <float>  (nml, "species", "mass",  is);
      species_h[is].dens  = toml::find <float>  (nml, "species", "dens",  is);
      if(nml.at("species").count("temp")>0) species_h[is].temp  = toml::find <float>  (nml, "species", "temp",  is);
      species_h[is].tprim = toml::find <float>  (nml, "species", "tprim", is);
      species_h[is].fprim = toml::find <float>  (nml, "species", "fprim", is);
      if(nml.at("species").count("vnewk")>0) species_h[is].nu_ss = toml::find <float>  (nml, "species", "vnewk", is);
      string stype        = toml::find <string> (nml, "species", "type",  is);
      species_h[is].type = stype == "ion" ? 0 : 1;
    }
  } else if(krehm) {
    species_h[0].temp = 1.0;
    species_h[0].mass = 1.0;
    species_h[0].type = 1;
  }

  // initialize derived species parameters
  vtmax = -1.;
  vtmin = 1000000000;
  tzmax = -1.;
  etamax = -1.;
  float numax = -1.;
  for(int s=0; s<nspec_in; s++) {
    species_h[s].vt   = sqrt(species_h[s].temp / species_h[s].mass);
    species_h[s].tz   = species_h[s].temp / species_h[s].z;
    species_h[s].zt   = species_h[s].z / species_h[s].temp;
    species_h[s].rho2 = species_h[s].temp * species_h[s].mass / (species_h[s].z * species_h[s].z); // note this does not have a factor of 1/B**2
    species_h[s].nt    = species_h[s].dens * species_h[s].temp;
    species_h[s].nz    = species_h[s].dens * species_h[s].z;
    species_h[s].jparfac = species_h[s].nz * species_h[s].vt * beta / 2.;
    species_h[s].jperpfac = -species_h[s].dens * species_h[s].temp * beta / 2.; 
    if (long_wavelength_GK) {
      species_h[s].rho2  = 0; // setting rho2 = 0.
      species_h[s].rho2_long_wavelength_GK  = species_h[s].temp * species_h[s].mass / (species_h[s].z * species_h[s].z); // note this does not have a factor of 1/B**2. This rho2 is used for quasineutrality 1-Gam0 --> b_s approximation, whereas rho2 = 0 elsewhere for long_wavelength_GK.
      if(iproc==0) printf("You are running GX with the long wavelength approximation.");
    }
    if (debug) {
      printf("species = %d \n",s);
      printf("mass, z, temp, dens = %f, %f, %f, %f \n",
	     species_h[s].mass, species_h[s].z, species_h[s].temp, species_h[s].dens);
      printf("vt, tz, zt = %f, %f, %f \n",
	     species_h[s].vt, species_h[s].tz, species_h[s].zt);
      printf("rho2, nt, nz = %f, %f, %f \n",
	     species_h[s].rho2, species_h[s].nt, species_h[s].nz);
      printf("jparfac, jperpfac = %f, %f \n", 
             species_h[s].jparfac, species_h[s].jperpfac);
      printf("nu_ss = %f, tprim = %f, fprim = %f\n\n", species_h[s].nu_ss, species_h[s].tprim, species_h[s].fprim);
    }      
    vtmax = fmax(vtmax, species_h[s].vt);
    vtmin = fmin(vtmin, species_h[s].vt);
    tzmax = fmax(tzmax, abs(species_h[s].tz));
    etamax = fmax(etamax, species_h[s].tprim/species_h[s].fprim);
    numax = fmax(numax, species_h[s].nu_ss);

    if(species_h[s].type == 1) {
      ne = species_h[s].dens;
      Te = species_h[s].temp;
    }
  }

  collisions = false;
  if (numax > 0.) collisions = true;
  if(debug && iproc==0) printf("nspec_in = %i \n",nspec_in);
}

void Parameters::set_jtwist_x0(float *shat_in, float *gds21, float *gds22)
{
  float shat = *shat_in;
  // note: twist_shift_geo_fac reduces to 2*pi*shat*(2*nPeriod - 1) in the axisymmetric limit
  // as gds21[0] is gds21 evaluated at the most negative value of theta in the flux tube.
  float twist_shift_geo_fac = 2.*shat*gds21[0]/gds22[0];

  if(iproc==0) {
    if(debug) printf("set_jtwist_x0: shat = %f, twist_shift_geo_fac = %f\n", shat, twist_shift_geo_fac);

    // check consistency of boundary and geo_option
    printf(ANSI_COLOR_RED);
    if(boundary == "continuous drifts" || boundary == "fix aspect") {
      if(geo_option != "vmec" && geo_option != "pyvmec" && geo_option != "vmec_c" && geo_option != "desc") printf("Warning: boundary option \"%s\" is not available with the requested geometry module. Using standard twist-shift BCs (boundary = \"linked\")\n", boundary.c_str()); 
    }
    if(boundary == "exact periodic") {
      if(geo_option != "vmec" && geo_option != "pyvmec" && geo_option != "vmec_c" && geo_option != "desc") printf("Warning: boundary option \"%s\" is not available with the requested geometry module. Using standard periodic BCs (boundary = \"periodic\")\n", boundary.c_str()); 
    }
    printf(ANSI_COLOR_RESET);
  }

  if (jtwist==0) {
    // this is an error
    if(iproc==0) {
      printf(ANSI_COLOR_RED);
      printf("************************** \n");
      printf("************************** \n");
      printf("jtwist = 0 is not allowed! \n");
      printf("************************** \n");
      printf("************************** \n");
      printf(ANSI_COLOR_RESET);
    }
  }
  if (abs(shat) < zero_shat_threshold) {
    if(iproc==0) {
      printf("Magnetic shear shat is smaller than threshold value. Setting shat = 1e-8.\n");
    }
    shat = 1e-8;
    zero_shat = true;
    boundary_option_periodic = true;
  }

  if (boundary_option_periodic) {
    // for periodic BCs, set jtwist=2*nx_in, which will give a separate periodic domain for each mode (no linking).
    // just need to make sure x0 is set
    // either take x0 from input file, or if it was not set
    // (indicated by x0 = -1) then set it to y0 by default
    jtwist = 2*nx_in;
    if (x0 == -1) {
      x0 = y0;
    }
    if (geo_option=="slab" && iproc==0) {
      printf("Parallel box size is 2 * pi * z0 = %f \n",2*M_PI*z0);
      if(zero_shat) printf("And regardless of other messages, the magnetic shear is zero.\n");      
    }
    if(iproc==0) {
      printf(ANSI_COLOR_MAGENTA);
      printf("Using periodic BCs with x0 = %f, y0 = %f\n", x0, y0);
      printf(ANSI_COLOR_RESET);
    }
  } else { // use twist-and-shift BCs
    // if both jtwist and x0 were not set in input file
    if (jtwist == -1000 && x0 < 0.0) {
      // set jtwist so that x0~y0
      jtwist = (int) round(twist_shift_geo_fac);
      if(jtwist == 0) {
	//
	// Per the discussion in April, 2023, we want to change the logic in this section.
	// Instead of setting zero_shat = true, we want to force jtwist = 1 and
	// then take the x0 that that gives.
	//
	// We could calculate kx_max and advise the user to set nx such that
	// kx_max ~ ky_max (for example). For small values of magnetic shear,
	// this will produce recommendations for nx that can be quite large.
	// But that is probably the best thing to do.
	//      
        if(iproc==0) {
          printf(ANSI_COLOR_RED);
          printf("Warning: twist_shift_geo_fac is so small that it was giving jtwist=0, but the minimum possible value is jtwist = 1.\n");
          printf("Setting jtwist = 1 results in x0 = %f, so that kx_max = %f for your grid with Nx = %d.\n", y0/abs(twist_shift_geo_fac), ((int)(nx_in-1)/3)/y0*abs(twist_shift_geo_fac), nx_in);
          printf("Consider using an alternative boundary option.\n");
          printf(ANSI_COLOR_RESET);
        }

        jtwist = 1;
      } 

      x0 = y0 * abs(jtwist)/abs(twist_shift_geo_fac);
    } 
    // if jtwist was set in input file but x0 was not
    else if (x0 < 0.0) {
      x0 = y0 * abs(jtwist)/abs(twist_shift_geo_fac);
    } 
    // if x0 was set in input file 
    else {
      // compute jtwist that will give x0 ~ the input value
      int jtwist_0 = (int) round(twist_shift_geo_fac/y0*x0);
      
      // if both jtwist and x0 were set in input file, make sure the input jtwist is consistent with the input x0,
      // and print warning if not.
      if (jtwist != -1000) {
        if (jtwist_0 != jtwist) {
          if(iproc==0) printf("Warning: x0 and jtwist set inconsistently. Resetting jtwist = %d\n", jtwist_0);
        }
      }
      if(jtwist_0 == 0) {
        if(iproc==0) {
          printf(ANSI_COLOR_RED);
          printf("Warning: twist_shift_geo_fac is so small that it was giving jtwist=0, but the minimum possible value is jtwist = 1.\n");
          printf("Setting jtwist = 1 results in x0 = %f, so that kx_max = %f for your grid with Nx = %d.\n", y0/(abs(twist_shift_geo_fac)), ((int)(nx_in-1)/3)/y0*(abs(twist_shift_geo_fac)), nx_in);
          printf("Consider using an alternative boundary option.\n");
          printf(ANSI_COLOR_RESET);
        }

        jtwist_0 = 1;
      } 
      jtwist = jtwist_0;
      // reset x0 to be consistent with the integer jtwist we just computed
      x0 = y0 * abs(jtwist)/abs(twist_shift_geo_fac);
    }
    printf(ANSI_COLOR_MAGENTA);
    if(iproc==0) printf("Using (generalized) twist-and-shift BCs. Final values are jtwist = %d, shat = %f, x0 = %f, y0 = %f\n", jtwist, shat, x0, y0);
  
    printf(ANSI_COLOR_RESET);
  }

}

