#include "parameters.h"
#include "ncdf.h"
#include <netcdf.h>
#include <netcdf_par.h>
#include "toml.hpp"
#include <iostream>
#include "version.h"
using namespace std;

Parameters::Parameters(int iproc_in, int nprocs_in, MPI_Comm mpcom_in) {
  initialized = false;

  iproc = iproc_in;
  nprocs = nprocs_in;
  mpcom = mpcom_in;

  // some cuda parameters (not from input file)
  int dev; 
  cudaGetDevice(&dev);
  if (false) printf("device id = %d \n",dev);
}

Parameters::~Parameters() {
  cudaDeviceSynchronize();
  if(initialized) {
    free(species_h);
  }
}


void Parameters::get_nml_vars(char* filename)
{
  strcpy (run_name, filename);
  char nml_file[1000];
  
  strcpy(nml_file, run_name);
  strcat(nml_file, ".in");

  //printf(ANSI_COLOR_MAGENTA);

  const auto nml = toml::parse(nml_file);

  //repeat = toml::find_or <bool> (nml, "repeat",  false);  
  debug  = toml::find_or <bool> (nml, "debug",   false);

  char default_restart_filename[1000];
  strcpy(default_restart_filename, filename);
  strcat(default_restart_filename, ".restart.nc");

  auto tnml = nml;

  //
  // This next line seems to be completely superfluous!
  //
  tnml = nml;
  if (nml.contains("Dimensions")) tnml = toml::find(nml, "Dimensions");
  
  nz_in    = toml::find_or <int> (tnml, "ntheta",    32);
  ny_in    = toml::find_or <int> (tnml, "ny",        0);
  nx_in    = toml::find_or <int> (tnml, "nx",        0);
  int nky_in   = toml::find_or <int> (tnml, "nky",       0);
  int nkx_in   = toml::find_or <int> (tnml, "nkx",       0);
  nm_in    = toml::find_or <int> (tnml, "nhermite",  4);
  nl_in    = toml::find_or <int> (tnml, "nlaguerre", 2);
  nspec_in = toml::find_or <int> (tnml, "nspecies",  1);
  nperiod  = toml::find_or <int> (tnml, "nperiod",   1);

  assert((ny_in > 0 || nky_in > 0) && "must set ny or nky");
  if(nky_in > 0 && ny_in > 0) assert((nky_in == 1 + (ny_in-1)/3) && "nky and ny have been set inconsistenly. only one of these needs to be set.");
  else if(nky_in > 0) ny_in = 3*(nky_in-1) + 1;

  assert((nx_in > 0 || nkx_in > 0) && "must set nx or nkx");
  if(nkx_in > 0 && nx_in > 0) assert((nkx_in == 1 + 2*((nx_in-1)/3)) && "nkx and nx have been set inconsistenly. only one of these needs to be set.");
  else if(nkx_in > 0) nx_in = ((nkx_in - 1) /  2) * 3 + 1;
  
  // NRM: are these warnings still needed?
  //if (nx_in<4) {printf("Warning: Behavior is not guaranteed for nx = %d \n",nx_in);}
  //if (ny_in<4) {printf("Warning: Behavior is not guaranteed for ny = %d \n",ny_in);}

  tnml = nml;
  if (nml.contains("Domain")) tnml = toml::find(nml, "Domain");
  y0       = toml::find_or <float>       (tnml, "y0",          10.0  );
  x0       = toml::find_or <float>       (tnml, "x0",          -1.0  );
  z0       = toml::find_or <float>       (tnml, "z0",           1.0  );
  jtwist   = toml::find_or <int>         (tnml, "jtwist",      -1000 );
  Zp       = toml::find_or <int>         (tnml, "zp",           2*nperiod-1    );
  // possible values of boundary are:
  // "linked": twist-and-shift BC, with generalization for non-axisymmetric geometry (Martin et al 2018)
  // "forced periodic" (or simply "periodic"): use periodic BCs with no cutting of flux tube
  // "exact periodic": cut flux tube at a location where gds21 = 0, and then use periodic BCs (currently for VMEC geometry only)
  // "continuous drifts": cut flux tube at a location where gbdrift0 = 0, and then use (generalized) twist-and-shift BC (currently for VMEC geometry only)
  // "fix aspect": cut flux tube at a location where y0/x0 takes the desired value, and then use (generalized) twist-and-shift BC (VMEC geometry only)
  boundary = toml::find_or <std::string> (tnml, "boundary", "linked" );
  long_wavelength_GK = toml::find_or <bool>   (tnml, "long_wavelength_GK",   false ); // JFP, long wavelength GK limit where bs = 0, except in quasineutrality where 1 - Gamma0(b) --> b.
  zero_shat_threshold = toml::find_or <float>   (tnml, "zero_shat_threshold", 1e-5);
  bool ExBshear_domain = toml::find_or <bool>        (tnml, "ExBshear",    false ); // included for backwards-compat. ExBshear now specified in Physics
  float g_exb_domain    = toml::find_or <float>       (tnml, "g_exb",        0.0  ); // included for backwards-compat. g_exb now specified in Physics
  
  tnml = nml;  
  if (nml.contains("Time")) tnml = toml::find (nml, "Time");
  dt      = toml::find_or <float> (tnml, "dt",       0.05 );
  nstep   = toml::find_or <int>   (tnml, "nstep",   2e9 );
  scheme = toml::find_or <string> (tnml, "scheme",    "rk3"   );
  cfl = toml::find_or <float> (tnml, "cfl", 0.9);
  stages = toml::find_or <int>    (tnml, "stages",  10   );
  t_max = toml::find_or <float> (tnml, "t_max", 1.e20);
  t_add = toml::find_or <float> (tnml, "t_add", -1.0);
  int nwrite_time  = toml::find_or <int>   (tnml, "nwrite",   1000    ); // included for backwards-compat. nwrite now specified in Diagnostics
  int navg_time    = toml::find_or <int>   (tnml, "navg",       10    ); // included for backwards-compat. navg now specified in Diagnostics

  if (nml.contains("Initialization")) tnml = toml::find(nml, "Initialization");
  init_field = toml::find_or <string> (tnml, "init_field", "density");
  init_amp   = toml::find_or <float>  (tnml, "init_amp",   1.0e-5   );
  kpar_init  = toml::find_or <float>  (tnml, "kpar_init",     0.0   );
  ikpar_init  = toml::find_or <int>  (tnml, "ikpar_init",     (long) kpar_init  );
  random_init     = toml::find_or <bool> (tnml, "random_init",     false);
  init_electrons_only     = toml::find_or <bool> (tnml, "init_electrons_only",     false);
  densfac = toml::find_or <float> (tnml, "densfac", 1.0);
  uparfac = toml::find_or <float> (tnml, "uparfac", 1.0);
  tparfac = toml::find_or <float> (tnml, "tparfac", 1.0);
  tprpfac = toml::find_or <float> (tnml, "tprpfac", 1.0);
  qparfac = toml::find_or <float> (tnml, "qparfac", 1.0);
  qprpfac = toml::find_or <float> (tnml, "qprpfac", 1.0);
  if (random_init) ikpar_init = 0; 

  if (nml.contains("Restart")) tnml = toml::find(nml, "Restart");
  restart           = toml::find_or <bool>   (tnml, "restart",                 false  );
  save_for_restart  = toml::find_or <bool>   (tnml, "save_for_restart",         true  );
  restart_to_file   = toml::find_or <string> (tnml, "restart_to_file", default_restart_filename);
  restart_from_file = toml::find_or <string> (tnml, "restart_from_file", default_restart_filename);  
  scale             = toml::find_or <float>  (tnml, "scale",                      1.0 );
  nsave   = toml::find_or <int>   (tnml, "nsave", 10000 );
  nsave = max(1, nsave);

  if (nml.contains("Dissipation")) tnml = toml::find(nml, "Dissipation");
  closure_model  = toml::find_or <string> (tnml, "closure_model", "none" );
  smith_par_q    = toml::find_or <int>    (tnml, "smith_par_q",        3 );
  smith_perp_q   = toml::find_or <int>    (tnml, "smith_perp_q",       3 );
  D_HB       = toml::find_or <float>  (tnml, "D_HB",          1.0   ); 
  w_osc      = toml::find_or <float>  (tnml, "w_osc",         0.0   ); 
  D_hyper    = toml::find_or <float>  (tnml, "D_hyper",       0.1   ); 
  nu_hyper_z = toml::find_or <float>  (tnml, "nu_hyper_z",    0.0   ); 
  nu_hyper_l = toml::find_or <float>  (tnml, "nu_hyper_l",    0.0   ); 
  nu_hyper_m = toml::find_or <float>  (tnml, "nu_hyper_m",    1.0   ); 
  nu_hyper_lm = toml::find_or <float> (tnml, "nu_hyper_lm",   0.0   ); 
  p_hyper    = toml::find_or <int>    (tnml, "p_hyper",         2   ); 
  p_hyper_z  = toml::find_or <int>    (tnml, "p_hyper_z",       6   ); 
  p_hyper_l  = toml::find_or <int>    (tnml, "p_hyper_l",       6   ); 
  p_hyper_m  = toml::find_or <int>    (tnml, "p_hyper_m",       min(20, nm_in/2) ); 
  p_hyper_lm = toml::find_or <int>    (tnml, "p_hyper_lm",      6   ); 
  p_HB       = toml::find_or <int>    (tnml, "p_HB",            2   ); 
  hyper      = toml::find_or <bool>   (tnml, "hyper",         false ); 
  HB_hyper   = toml::find_or <bool>   (tnml, "HB_hyper",      false ); 
  hypercollisions_const = toml::find_or <bool> (tnml, "hypercollisions_const", false);
  hypercollisions_kz = toml::find_or <bool> (tnml, "hypercollisions_kz", false);
  hypercollisions_kz = toml::find_or <bool> (tnml, "hypercollisions", hypercollisions_kz); // "hypercollisions" now gives hypercollisions_kz
  hyperz = toml::find_or <bool> (tnml, "hyperz", false);

  tnml = nml;
  if (nml.contains("Collisional_slab_ETG")) tnml = toml::find (nml, "Collisional_slab_ETG"); 

  cetg              = toml::find_or <bool>  (tnml, "cetg",         false );

  if (cetg) gx = false;
  if (cetg) nm_in = 1;
  if (cetg) nl_in = 2;
  
  tnml = nml;
  if (nml.contains("Vlasov_Poisson")) tnml = toml::find (nml, "Vlasov_Poisson");
  
  vp                = toml::find_or <bool>  (tnml, "vp",         false );
  vp_closure        = toml::find_or <bool>  (tnml, "vp_closure",  true );
  vp_nu             = toml::find_or <float> (tnml, "vp_nu",       -1.0 );
  vp_nuh            = toml::find_or <float> (tnml, "vp_nuh",      -1.0 );
  vp_alpha          = toml::find_or <int>   (tnml, "vp_alpha",       1 );
  vp_alpha_h        = toml::find_or <int>   (tnml, "vp_alpha_h",     2 );
  if (vp) gx = false;
  
  tnml = nml;
  if (nml.contains("KS")) tnml = toml::find (nml, "KS");
  
  ks                = toml::find_or <bool>  (tnml, "ks",         false );
  write_ks          = toml::find_or <bool>  (tnml, "write_ks",   false );
  eps_ks            = toml::find_or <float> (tnml, "eps_ks",       0.0 );
  ks_t0             = toml::find_or <float> (tnml, "ks_t0",       -1.0 );
  ks_tf             = toml::find_or <float> (tnml, "ks_tf",       -1.0 );
  ks_eps0           = toml::find_or <float> (tnml, "ks_eps0",     -1.0 );
  ks_epsf           = toml::find_or <float> (tnml, "ks_epsf",     -1.0 );
  if (ks) gx = false;
  
  tnml = nml;
  if (nml.contains("KREHM")) tnml = toml::find (nml, "KREHM");
  
  krehm             = toml::find_or <bool>  (tnml, "krehm",     false );
  if(krehm) gx = false;
  rho_i             = toml::find_or <float> (tnml, "rho_i",       1.0 );
  d_e               = toml::find_or <float> (tnml, "d_e",         1.0 );
  nu_ei             = toml::find_or <float> (tnml, "nu_ei",       0.0 );
  eta               = toml::find_or <float> (tnml, "eta",         0.0 );
  zt                = toml::find_or <float> (tnml, "zt",          1.0 );
  harris_sheet      = toml::find_or <bool>  (tnml, "harris_sheet", false);
  periodic_equilibrium = toml::find_or <bool> (tnml, "periodic_equilibrium", false);
  k0                = toml::find_or <float> (tnml, "k0", 10.0);
  gaussian_tube     = toml::find_or <bool> (tnml, "gaussian_tube", false);
  rho_s = rho_i*sqrtf(zt/2);
  if(eta>0.0) nu_ei = eta/d_e/d_e;

  tnml = nml;
  if (nml.contains("Expert")) tnml = toml::find (nml, "Expert");

  i_share     = toml::find_or <int>    (tnml, "i_share",         8 );
  int dev;
  cudaDeviceProp prop;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  size_t maxSharedSize = prop.sharedMemPerBlockOptin;
  int i_share_max = maxSharedSize/((nl_in+2)*(nm_in+4)*sizeof(cuComplex));
  if(i_share > i_share_max) {
    if(iproc==0) printf("Using i_share = %d would exceed shared memory limits. Setting i_share = %d instead.\n", i_share, i_share_max);
    i_share = i_share_max;
  }
  
  dealias_kz  = toml::find_or <bool>   (tnml, "dealias_kz",  false );
  nreal       = toml::find_or <int>    (tnml, "nreal",           1 );  
  local_limit = toml::find_or <bool>   (tnml, "local_limit", false );
  init_single = toml::find_or <bool>   (tnml, "init_single", false );
  ikx_single  = toml::find_or <int>    (tnml, "ikx_single",      0 );
  iky_single  = toml::find_or <int>    (tnml, "iky_single",      1 );
  ikx_fixed   = toml::find_or <int>    (tnml, "ikx_fixed",      -1 );
  iky_fixed   = toml::find_or <int>    (tnml, "iky_fixed",      -1 );
  eqfix       = toml::find_or <bool>   (tnml, "eqfix",       false );
  secondary   = toml::find_or <bool>   (tnml, "secondary",   false );
  phi_ext     = toml::find_or <float>  (tnml, "phi_ext",       0.0 );
  source      = toml::find_or <string> (tnml, "source",  "default" );
  tp_t0       = toml::find_or <float>  (tnml, "t0",           -1.0 );
  tp_tf       = toml::find_or <float>  (tnml, "tf",           -1.0 );
  tprim0      = toml::find_or <float>  (tnml, "tprim0",       -1.0 );
  tprimf      = toml::find_or <float>  (tnml, "tprimf",       -1.0 );
  hegna       = toml::find_or <bool>   (tnml, "hegna",       false );
  use_NCCL    = toml::find_or <bool>   (tnml, "use_NCCL",    true );
  damp_ends_widthfrac = toml::find_or <float> (tnml, "damp_ends_widthfrac", 1./8.);
  damp_ends_amp = toml::find_or <float> (tnml, "damp_ends_amp", 0.1);

  if( hegna ){
    printf("\nIn order to recover the Hegna model, setting nm=4, nl=2.\n");
    printf("For consistency, vnewk values should be relatively high.\n");
    nm_in = 4;
    nl_in = 2;
  }
  
  tnml = nml;
  if (nml.contains("Diagnostics")) tnml = toml::find (nml, "Diagnostics");

  nwrite  = toml::find_or <int>   (tnml, "nwrite", (long)  nwrite_time    );
  navg    = toml::find_or <int>   (tnml, "navg",   (long)    navg_time    );
  nwrite_big  = toml::find_or <int>   (tnml, "nwrite_big", (long)  nwrite*100 );
  fixed_amplitude   = toml::find_or <bool> (tnml, "fixed_amplitude", false);
  write_omega       = toml::find_or <bool> (tnml, "omega",          false );
  write_free_energy = toml::find_or <bool> (tnml, "free_energy",    true  ); if (ks) write_free_energy = false;
  write_fluxes      = toml::find_or <bool> (tnml, "fluxes",         false );
  write_moms        = toml::find_or <bool> (tnml, "moments",           false );
  write_fields      = toml::find_or <bool> (tnml, "fields",         false );
  
  tnml = nml;
  if (nml.contains("Resize")) tnml = toml::find (nml, "Resize");

  domain_change = toml::find_or <bool> (tnml, "domain_change", false);
  z0_mult = toml::find_or <int> (tnml, "z0_mult", 1);  assert( (z0_mult > 0) && "z0_mult must be an integer >= 1");
  y0_mult = toml::find_or <int> (tnml, "y0_mult", 1);  assert( (y0_mult > 0) && "y0_mult must be an integer >= 1");
  x0_mult = toml::find_or <int> (tnml, "x0_mult", 1);  assert( (x0_mult > 0) && "x0_mult must be an integer >= 1");
  nx_mult = toml::find_or <int> (tnml, "nx_mult", 1);  assert( (nx_mult > 0) && "nx_mult must be an integer >= 1");
  ny_mult = toml::find_or <int> (tnml, "ny_mult", 1);  assert( (ny_mult > 0) && "ny_mult must be an integer >= 1");
  nm_add  = toml::find_or <int> (tnml, "nm_add" , 0);  
  nl_add  = toml::find_or <int> (tnml, "nl_add" , 0);  
  ns_add  = toml::find_or <int> (tnml, "ns_add" , 0);  assert( (ns_add >= 0) && "ns_add must be an integer >= 0");
  
  ntheta_mult = toml::find_or <int> (tnml, "nz_mult", 1);
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
  
  tnml = nml;
  if (nml.contains("Controls")) tnml = toml::find (nml, "Controls");
  dealias_kz = toml::find_or <bool>   (tnml, "dealias_kz",  dealias_kz   ); // included for backwards-compat. now specified in expert
  nonlinear_mode = toml::find_or <bool>   (tnml, "nonlinear_mode",    false );  linear = !nonlinear_mode; // included for backwards-compat. nonlinear_mode now specified in Physics
  closure_model  = toml::find_or <string> (tnml, "closure_model", closure_model ); // included for backwards-compat. closure_model now specified in Dissipation
  smith_par_q    = toml::find_or <int>    (tnml, "smith_par_q",   (long) smith_par_q );  // included for backwards-compat. smith_par_q now specified in Dissipation
  smith_perp_q   = toml::find_or <int>    (tnml, "smith_perp_q",  (long) smith_perp_q ); // included for backwards-compat. smith_perp_q now specified in Dissipation

  scheme = toml::find_or <string> (tnml, "scheme",    scheme   ); // included for backwards-compat. scheme now specified in Time
  stages     = toml::find_or <int>    (tnml, "stages",    (long)  stages  ); // included for backwards-compat. stages now specified in Time
  cfl        = toml::find_or <float>  (tnml, "cfl",      (double)     cfl   ); // included for backwards-compat. cfl now specified in Time
  init_field = toml::find_or <string> (tnml, "init_field", init_field); // included for backwards-compat. init_field now specified in Initialization
  init_amp   = toml::find_or <float>  (tnml, "init_amp", (double)  init_amp  ); // included for backwards-compat. init_amp now specified in Initialization
  kpar_init  = toml::find_or <float>  (tnml, "kpar_init",    (double) kpar_init   ); // included for backwards-compat. kpar_init now specified in Initialization
  ikpar_init  = toml::find_or <int>  (tnml, "ikpar_init", (long) ikpar_init  ); // included for backwards-compat. ikpar_init now specified in Initialization
  D_HB       = toml::find_or <float>  (tnml, "D_HB",        (double)  D_HB   ); // included for backwards-compat. now specified in Dissipation 
  w_osc      = toml::find_or <float>  (tnml, "w_osc",       (double)  w_osc   ); // included for backwards-compat. now specified in Dissipation
  D_hyper    = toml::find_or <float>  (tnml, "D_hyper",     (double)  D_hyper   ); // included for backwards-compat. now specified in Dissipation
  nu_hyper_l = toml::find_or <float>  (tnml, "nu_hyper_l",  (double)  nu_hyper_l   ); // included for backwards-compat. now specified in Dissipation
  nu_hyper_m = toml::find_or <float>  (tnml, "nu_hyper_m",  (double)  nu_hyper_m   ); // included for backwards-compat. now specified in Dissipation
  nu_hyper   = toml::find_or <int>    (tnml, "nu_hyper",      (long)  nu_hyper   ); // included for backwards-compat. now specified in Dissipation
  p_hyper    = toml::find_or <int>    (tnml, "p_hyper",       (long)  p_hyper   ); // included for backwards-compat. now specified in Dissipation
  p_hyper_l  = toml::find_or <int>    (tnml, "p_hyper_l",     (long)  p_hyper_l   ); // included for backwards-compat. now specified in Dissipation
  p_hyper_m  = toml::find_or <int>    (tnml, "p_hyper_m",     (long)  p_hyper_m   ); // included for backwards-compat. now specified in Dissipation  
  p_HB       = toml::find_or <int>    (tnml, "p_HB",          (long)  p_HB   ); // included for backwards-compat. now specified in Dissipation
  hyper      = toml::find_or <bool>   (tnml, "hyper",         hyper ); // included for backwards-compat. now specified in Dissipation
  HB_hyper   = toml::find_or <bool>   (tnml, "HB_hyper",      HB_hyper ); // included for backwards-compat. now specified in Dissipation
  random_init     = toml::find_or <bool> (tnml, "random_init",     random_init); // include for backwards-compat. now specified in Initialization
  init_electrons_only     = toml::find_or <bool> (tnml, "init_electrons_only",     init_electrons_only); // include for backwards-compat. now specified in Initialization
  if (random_init) ikpar_init = 0; 
  
  if (write_omega && fixed_amplitude) {
    if (nonlinear_mode || nwrite < 3) fixed_amplitude = false;
  }

  tnml = nml;
  if (nml.contains("Forcing")) tnml = toml::find (nml, "Forcing");  

  forcing_type  = toml::find_or <string> (tnml, "forcing_type",    "Kz" ); //Needed for Helicity Injection; forcing_type=HeliInj
  stir_field    = toml::find_or <string> (tnml, "stir_field", "density" ); //Needed for Helicity Injection - Field to be perturbed
  forcing_amp   = toml::find_or <float>  (tnml, "forcing_amp",      1.0 );
  pos_forcing_amp = toml::find_or <float> (tnml, "pos_forcing_amp",  1.0); //Needed for Helicity Injection - Positive Amplitude
  neg_forcing_amp = toml::find_or <float> (tnml, "neg_forcing_amp",  1.0); //Needed for Helicity Injection - Negative Amplitude
  forcing_index = toml::find_or <int>    (tnml, "forcing_index",    1   );
  forcing_init  = toml::find_or <bool>   (tnml, "forcing_init",   false ); 
  no_fields     = toml::find_or <bool>   (tnml, "no_fields",      false );
  forcing_kz    = toml::find_or <int>    (tnml, "forcing_kz",         0 ); //Needed for Helicity Injection - Mode of kz perturbed
  forcing_k2min = toml::find_or <int>    (tnml, "forcing_k2min",      0 ); //Needed for Helicity Injection - Minimum kperp that can be perturbed
  forcing_k2max = toml::find_or <int>    (tnml, "forcing_k2max",      0 ); //Needed for Helicity Injection - Maximum mode of kperp that can be perturbed

  ///////////////////////////////////////////////////////////////////////
  //
  // New way to handle Boltzmann response. 
  // 
  // Preferred: 
  //     All kinetic? Leave out iphi00 entirely; leave out add_Boltzmann_species or set add_Boltzmann_species = false
  //
  //     Include a Boltzmann species? Set add_Boltzmann_species = true, and set Boltzmann_type 
  //            as either "electrons" (to recover iphi00=2 in the old way)
  //            or "ions"             (to recover iphi00=1 in the old way)
  //     Use "tau_fac" instead of ti_ov_te as the multiplier for the Boltzmann response
  // 
  // Backward compatibility possibilities (not preferred)
  //     
  //     Assume there is always a Boltzmann species unless electromagnetic
  //     and control the response with iphi00 and ti_ov_te
  //
  //     This mode of operation is deprecated, and should be removed in early 2021.
  //
  ///////////////////////////////////////////////////////////////////////
  
  tnml = nml;
  if (nml.contains("Boltzmann")) tnml = toml::find (nml, "Boltzmann");  

  add_Boltzmann_species = toml::find_or <bool>   (tnml, "add_Boltzmann_species", false);
  Btype                 = toml::find_or <string> (tnml, "Boltzmann_type", "electrons" );
  iphi00                = toml::find_or <int>    (tnml, "iphi00",                  -2 );

  // Get the value of Z when running the Adkins collisional ETG equations

  ion_z                 = toml::find_or <float>   (tnml, "Z_ion",                   1. );
  
  // For backward compatibility, check if iphi00 was specified and act accordingly
  if (iphi00 > 0) {
    if (iphi00 == 1) Btype = "Ions";
    if (iphi00 == 2) Btype = "Electrons";
    add_Boltzmann_species = true;
  }
  
  // allow some sloppiness here:
  
  if (Btype == "Electrons") Boltzmann_opt = BOLTZMANN_ELECTRONS;
  if (Btype == "Electron" ) Boltzmann_opt = BOLTZMANN_ELECTRONS;
  if (Btype == "Ions")      Boltzmann_opt = BOLTZMANN_IONS;
  if (Btype == "Ion" )      Boltzmann_opt = BOLTZMANN_IONS;

  if (Btype == "electrons") Boltzmann_opt = BOLTZMANN_ELECTRONS;
  if (Btype == "electron" ) Boltzmann_opt = BOLTZMANN_ELECTRONS;
  if (Btype == "ions")      Boltzmann_opt = BOLTZMANN_IONS;
  if (Btype == "ion" )      Boltzmann_opt = BOLTZMANN_IONS;

  // backward compatibility, sets default overall as tau_fac = unity
  ti_ov_te = toml::find_or <float> (tnml, "TiTe", 1.0);     
  // check for new, physically sensible value in the input file
  tau_fac  = toml::find_or <float> (tnml, "tau_fac", -1.0);
  
  if (tau_fac > 0.) ti_ov_te = tau_fac;                 // new definition has priority if it was provided
  tau_fac = ti_ov_te;                                   // In the body of the code, use tau_fac instead of ti_ov_te
  
  // For the Adkins collisional ETG model, tau_fac should be set to Ti/(Te Z) = tau_bar
  
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

  tnml = nml;
  if (nml.contains("Geometry")) tnml = toml::find (nml, "Geometry");  

  geo_option  = toml::find_or <string> (tnml, "geo_option", "none");
  geofilename = toml::find_or <string> (tnml, "geofile",  "eik.out" ); // included for backwards-compat. use geo_file instead. 
  geofilename = toml::find_or <string> (tnml, "geo_file", geofilename );  
  slab        = toml::find_or <bool>   (tnml, "slab",         false ); // included for backwards-compat. use geo_option = "slab" instead.
  const_curv  = toml::find_or <bool>   (tnml, "const_curv",   false ); // included for backwards-compat. use geo_option = "const-curv" instead.

  igeo        = toml::find_or <int>   (tnml, "igeo",       -1 ); // included for backwards-compat. use geo_option instead.
  float beta_geo = toml::find_or <float> (tnml, "beta", 0.0 ); // included for backwards-compat. beta now set in Physics
  drhodpsi    = toml::find_or <float> (tnml, "drhodpsi", 1.0 );
  kxfac       = toml::find_or <float> (tnml, "kxfac",    1.0 );
  rmaj        = toml::find_or <float> (tnml, "Rmaj",     1.0 );
  rmaj        = toml::find_or <float> (tnml, "rmaj",    (double) rmaj );
  r_geo       = toml::find_or <float> (tnml, "R_geo",     1.0 );
  shift       = toml::find_or <float> (tnml, "shift",    0.0 );
  eps         = toml::find_or <float> (tnml, "eps",    0.167 );
  qsf         = toml::find_or <float> (tnml, "qinp",     1.4 );
  akappa      = toml::find_or <float> (tnml, "akappa",     1.0 );
  akappri     = toml::find_or <float> (tnml, "akappri",    0.0 );
  tri         = toml::find_or <float> (tnml, "tri",        1.0 );
  tripri      = toml::find_or <float> (tnml, "tripri",     0.0 );
  beta_prime_input    = toml::find_or <float> (tnml, "beta_prime_input", -1.0 );
  beta_prime_input    = toml::find_or <float> (tnml, "betaprim", (double) beta_prime_input );
  zero_shat   = toml::find_or <bool>  (tnml, "zero_shat", false); 
  // Set zero_shat = true in the input file when the actual magnetic shear is inconveniently low. 
  if(geo_option=="s-alpha" || geo_option=="slab" || geo_option=="const-curv" || igeo==0) {
    shat        = toml::find_or <float> (tnml, "shat",     0.8 );
  } else {
    // shat will be taken from geometry file; do nothing here
    //shat        = toml::find <float> (tnml, "shat");
    //printf("Using the value of shat that appears in the .in file. \n");
    //printf("Be sure it is consistent with the value in the geometry file. \n");
    //printf("Using shat = %f \n",shat);
  }
  
  // the following parameters are exclusively for interfacing 
  // with the GS2 geometry module via eiktest
  // for parameter definitions, see src/geo/geometry.f90 in the GS2 repo
  // the defaults are the same as the defaults in gs2's eiktest.f90
  rhoc = toml::find_or <float> (tnml, "rhoc", 0.5); 
  geoType = toml::find_or <int> (tnml, "geoType", 0); 
  iflux = toml::find_or <int> (tnml, "iflux", 0); 
  delrho = toml::find_or <float> (tnml, "delrho", 0.01); 
  bishop = toml::find_or <int> (tnml, "bishop", 0); 
  irho = toml::find_or <int> (tnml, "irho", 2); 
  isym = toml::find_or <int> (tnml, "isym", 0); 
  eqfile = toml::find_or <string> (tnml, "eqfile", "none" );  
  s_hat_input = toml::find_or <float> (tnml, "s_hat_input", 1.0 );
  if(abs(s_hat_input) < zero_shat_threshold) {
    s_hat_input = 1e-8;
  }
  p_prime_input = toml::find_or <float> (tnml, "p_prime_input", -2.0 );
  invLp_input = toml::find_or <float> (tnml, "invLp_input", 5.0 );
  alpha_input = toml::find_or <float> (tnml, "alpha_input", 0.0 );
  efit_eq = toml::find_or <bool> (tnml, "efit_eq", false);
  dfit_eq = toml::find_or <bool> (tnml, "dfit_eq", false);
  gen_eq = toml::find_or <bool> (tnml, "gen_eq", false);
  ppl_eq = toml::find_or <bool> (tnml, "ppl_eq", false);
  local_eq = toml::find_or <bool> (tnml, "local_eq", false);
  idfit_eq = toml::find_or <bool> (tnml, "idfit_eq", false);
  chs_eq = toml::find_or <bool> (tnml, "chs_eq", false);
  transp_eq = toml::find_or <bool> (tnml, "transp_eq", false);
  gs2d_eq = toml::find_or <bool> (tnml, "gs2d_eq", false);

  tnml = nml;
  if (nml.contains("Physics")) tnml = toml::find(nml, "Physics");
  beta = toml::find_or <float> (tnml, "beta",    0.0 );
  if (beta == 0.0 && beta_geo > 0.0) beta = beta_geo; 
  nonlinear_mode = toml::find_or <bool>   (tnml, "nonlinear_mode",    nonlinear_mode );  linear = !nonlinear_mode;
  ExBshear = toml::find_or <bool> (tnml, "ExBshear",    ExBshear_domain );
  g_exb    = toml::find_or <float> (tnml, "g_exb",       (double) g_exb_domain  );
  fphi     = toml::find_or <float> (tnml, "fphi",        1.0);
  fapar    = toml::find_or <float> (tnml, "fapar",       beta > 0.0? 1.0 : 0.0);
  fbpar    = toml::find_or <float> (tnml, "fbpar",       beta > 0.0? 1.0 : 0.0);
  // electromagnetic doesn't make sense with adiabatic species
  if (!all_kinetic) {
    beta = 0.0; 
    fapar = 0.0;
    fbpar = 0.0;
  }
  ei_colls = toml::find_or <bool> (tnml, "ei_colls", true);
  coll_conservation = toml::find_or <bool> (tnml, "coll_conservation", true);

  gx = (!ks && !vp && !krehm && !cetg);
  assert (!(ks && vp));
  assert (ks || vp || gx || krehm || cetg);
  
//  wspectra.resize(nw_spectra);
//  pspectra.resize(np_spectra);
//  aspectra.resize(na_spectra);
//  qspectra.resize(nq_spectra);
//
//  wspectra.assign(nw_spectra, 0);
//  pspectra.assign(np_spectra, 0);
//  aspectra.assign(na_spectra, 0);
//  qspectra.assign(nq_spectra, 0);
//
//  tnml = nml;
//  if (nml.contains("Wspectra")) tnml = toml::find (nml, "Wspectra");  
//
//  wspectra [WSPECTRA_species] = (toml::find_or <bool> (tnml, "species",          false)) == true ? 1 : 0;
//  wspectra [WSPECTRA_kx]      = (toml::find_or <bool> (tnml, "kx",               false)) == true ? 1 : 0;
//  wspectra [WSPECTRA_ky]      = (toml::find_or <bool> (tnml, "ky",               false)) == true ? 1 : 0;
//  wspectra [WSPECTRA_kz]      = (toml::find_or <bool> (tnml, "kz",               false)) == true ? 1 : 0;
//  wspectra [WSPECTRA_z]       = (toml::find_or <bool> (tnml, "z",                false)) == true ? 1 : 0;
//  wspectra [WSPECTRA_l]       = (toml::find_or <bool> (tnml, "laguerre",         false)) == true ? 1 : 0;
//  wspectra [WSPECTRA_m]       = (toml::find_or <bool> (tnml, "hermite",          false)) == true ? 1 : 0;
//  wspectra [WSPECTRA_lm]      = (toml::find_or <bool> (tnml, "hermite_laguerre", false)) == true ? 1 : 0;
//  wspectra [WSPECTRA_kperp]   = (toml::find_or <bool> (tnml, "kperp",            false)) == true ? 1 : 0;
//  wspectra [WSPECTRA_kxky]    = (toml::find_or <bool> (tnml, "kxky",             false)) == true ? 1 : 0;
//
//  tnml = nml;
//  if (nml.contains("Pspectra")) tnml = toml::find (nml, "Pspectra");  
//
//  pspectra [PSPECTRA_species] = (toml::find_or <bool> (tnml, "species",          false)) == true ? 1 : 0;
//  pspectra [PSPECTRA_kx]      = (toml::find_or <bool> (tnml, "kx",               false)) == true ? 1 : 0;
//  pspectra [PSPECTRA_ky]      = (toml::find_or <bool> (tnml, "ky",               false)) == true ? 1 : 0;
//  pspectra [PSPECTRA_kz]      = (toml::find_or <bool> (tnml, "kz",               false)) == true ? 1 : 0;
//  pspectra [PSPECTRA_z]       = (toml::find_or <bool> (tnml, "z",                false)) == true ? 1 : 0;
//  pspectra [PSPECTRA_kperp]   = (toml::find_or <bool> (tnml, "kperp",            false)) == true ? 1 : 0;
//  pspectra [PSPECTRA_kxky]    = (toml::find_or <bool> (tnml, "kxky",             false)) == true ? 1 : 0;
//
//  tnml = nml;
//  if (nml.contains("Qspectra")) tnml = toml::find (nml, "Qspectra");  
//
//  qspectra [QSPECTRA_species] = (toml::find_or <bool> (tnml, "species",          false)) == true ? 1 : 0;
//  qspectra [QSPECTRA_kx]      = (toml::find_or <bool> (tnml, "kx",               false)) == true ? 1 : 0;
//  qspectra [QSPECTRA_ky]      = (toml::find_or <bool> (tnml, "ky",               false)) == true ? 1 : 0;
//  qspectra [QSPECTRA_kz]      = (toml::find_or <bool> (tnml, "kz",               false)) == true ? 1 : 0;
//  qspectra [QSPECTRA_z]       = (toml::find_or <bool> (tnml, "z",                false)) == true ? 1 : 0;
//  qspectra [QSPECTRA_kperp]   = (toml::find_or <bool> (tnml, "kperp",            false)) == true ? 1 : 0;
//  qspectra [QSPECTRA_kxky]    = (toml::find_or <bool> (tnml, "kxky",             false)) == true ? 1 : 0;
//
//  // if we have adiabatic ions, slave the aspectra to the wspectra as appropriate
//  if (!all_kinetic) {
//    aspectra [ ASPECTRA_species ] = wspectra [ WSPECTRA_species ];
//    aspectra [ ASPECTRA_kx      ] = wspectra [ WSPECTRA_kx      ];
//    aspectra [ ASPECTRA_ky      ] = wspectra [ WSPECTRA_ky      ];
//    aspectra [ ASPECTRA_kz      ] = wspectra [ WSPECTRA_kz      ];
//    aspectra [ ASPECTRA_z       ] = wspectra [ WSPECTRA_z       ];
//    aspectra [ ASPECTRA_kperp   ] = wspectra [ WSPECTRA_kperp   ];
//    aspectra [ ASPECTRA_kxky    ] = wspectra [ WSPECTRA_kxky    ];
//  }
//  // for backwards compatibility
//  if (write_l_spectrum)  wspectra[WSPECTRA_l] = 1;
//  if (write_h_spectrum)  wspectra[WSPECTRA_m] = 1;
//  if (write_lh_spectrum) wspectra[WSPECTRA_lm] = 1;
//  
//  // Some diagnostics are not yet available:
//  wspectra[ WSPECTRA_kperp] = 0;
//  pspectra[ PSPECTRA_kperp] = 0;
//  aspectra[ ASPECTRA_kperp] = 0;
//  qspectra[ QSPECTRA_kperp] = 0;
//  
//  // If Wtot is requested, turn Ws, Ps, Phi2 on:
//  if (write_free_energy) {
//    wspectra[WSPECTRA_species] = 1;
//    pspectra[PSPECTRA_species] = 1;
//    if ( add_Boltzmann_species ) aspectra[ASPECTRA_species] = 1;
//  }
//
//
//  int ksize = 0;
//  for (int k=0; k<pspectra.size(); k++) ksize = max(ksize, pspectra[k]);
//  for (int k=0; k<wspectra.size(); k++) ksize = max(ksize, wspectra[k]);
//  for (int k=0; k<aspectra.size(); k++) ksize = max(ksize, aspectra[k]);
//  for (int k=0; k<qspectra.size(); k++) ksize = max(ksize, qspectra[k]);
//
//  tnml = nml;
//  if (nml.contains("PZT")) tnml = toml::find (nml, "PZT");  
//
//  diagnosing_pzt = write_pzt;
//  
//  diagnosing_spectra = false;
//  if (ksize > 0) diagnosing_spectra = true;
//
//  diagnosing_kzspec = false;
//  if ((wspectra[ WSPECTRA_kz ] == 1) || (pspectra[ PSPECTRA_kz ] == 1) || (aspectra[ ASPECTRA_kz ] == 1)) {
//    diagnosing_kzspec = true;
//  }
//  
//  diagnosing_moments = false;
//  if (write_moms || write_phi || write_phi_kpar) diagnosing_moments = true;
  
  species_h = (specie *) malloc(nspec_in*sizeof(specie));
  if (nml.contains("species")) {
    for (int is=0; is < nspec_in; is++) {
      species_h[is].uprim = 0.;
      species_h[is].nu_ss = 0.;
      species_h[is].temp = 1.;
      
      species_h[is].z     = toml::find <float>  (nml, "species", "z",     is);
      species_h[is].mass  = toml::find <float>  (nml, "species", "mass",  is);
      species_h[is].dens  = toml::find <float>  (nml, "species", "dens",  is);
      if(nml.at("species").count("temp")>0) species_h[is].temp  = toml::find <float>  (nml, "species", "temp",  is);
      species_h[is].tprim = toml::find <float>  (nml, "species", "tprim", is);
      species_h[is].fprim = toml::find <float>  (nml, "species", "fprim", is);
      if(nml.at("species").count("uprim")>0) species_h[is].uprim = toml::find <float>  (nml, "species", "uprim", is);
      if(nml.at("species").count("vnewk")>0) species_h[is].nu_ss = toml::find <float>  (nml, "species", "vnewk", is);
      string stype        = toml::find <string> (nml, "species", "type",  is);
      species_h[is].type = stype == "ion" ? 0 : 1;
    }
  } else if(krehm) {
    species_h[0].temp = 1.0;
    species_h[0].mass = 1.0;
    species_h[0].type = 1;
  } else if(cetg) {
    species_h[0].mass = 1.0;
    species_h[0].z    = 1.0;
    species_h[0].dens = 1.0;
    species_h[0].temp = 1.0;
    species_h[0].type = 1; 
  }
  
  float numax = -1.;
  collisions = false;
  for (int i=0; i<nspec_in; i++) {numax = max(numax, species_h[i].nu_ss);}
  if (numax > 0.) collisions = true;
  
  Reservoir = false;
  add_noise = false;
  ResFakeData = false;
  ResWrite = false;
  ResBatch = false;
  
  tnml = nml;
  if (nml.contains("Reservoir")) tnml = toml::find (nml, "Reservoir");  

  Reservoir          = toml::find_or <bool>  (tnml, "Use_reservoir",  false  );
  ResQ               = toml::find_or <int>   (tnml, "Q",                 20  );
  ResK               = toml::find_or <int>   (tnml, "K",                  3  );
  ResTrainingSteps   = toml::find_or <int>   (tnml, "training_steps",     0  );
  ResPredict_Steps   = toml::find_or <int>   (tnml, "prediction_steps", 200  );
  ResTrainingDelta   = toml::find_or <int>   (tnml, "training_delta",     0  );
  ResSpectralRadius  = toml::find_or <float> (tnml, "spectral_radius",  0.6  );
  ResReg             = toml::find_or <float> (tnml, "regularization", 1.0e-4 );
  ResSigma           = toml::find_or <float> (tnml, "input_sigma",      0.5  );
  ResSigmaNoise      = toml::find_or <float> (tnml, "noise",           -1.0  );
  ResFakeData        = toml::find_or <bool>  (tnml, "fake_data",      false  );
  ResWrite           = toml::find_or <bool>  (tnml, "write",          false  );
  ResBatch           = toml::find_or <bool>  (tnml, "batch",          false  );
  
  if (ResTrainingSteps == 0) ResTrainingSteps = nstep/nwrite;
  if (ResTrainingDelta == 0) ResTrainingDelta = nwrite;
  if (ResSigmaNoise > 0.) add_noise = true;

  
  if(nz_in != 1) {
    int ntgrid = nz_in/2 + (nperiod-1)*nz_in; 
    nz_in = 2*ntgrid; // force even
  }
  
  //Zp = 2*nperiod - 1; // BD This needs updating
  
  // BD  This is messy. Prefer to go back to original method
  // before, jtwist_old assumed Zp=1
  // now, redefining jtwist = jtwist_old*Zp

  //  if(strcmp(closure_model, "beer4+2")==0) {
  closure_model_opt = Closure::none   ;
  if( closure_model == "beer4+2") {
    if(iproc==0) printf("\nUsing Beer 4+2 closure model. Overriding nm=4, nl=2\n\n");
    nm_in = 4;
    nl_in = 2;
    closure_model_opt = Closure::beer42;
  } else if (closure_model == "smith_perp") { closure_model_opt = Closure::smithperp;
  } else if (closure_model == "smith_par")  { closure_model_opt = Closure::smithpar; 
  }

  if( boundary == "periodic" || boundary == "exact periodic" || boundary == "forced periodic") { boundary_option_periodic = true;
  } else { boundary_option_periodic = false; }

  if     ( init_field == "density") { initf = inits::density; }
  else if( init_field == "upar"   ) { initf = inits::upar   ; }
  else if( init_field == "tpar"   ) { initf = inits::tpar   ; }
  else if( init_field == "tperp"  ) { initf = inits::tperp  ; }
  else if( init_field == "qpar"   ) { initf = inits::qpar   ; }
  else if( init_field == "qperp"  ) { initf = inits::qperp  ; }
  else if( init_field == "all"  ) { initf = inits::all  ; }
  
  if     ( stir_field == "density") { stirf = stirs::density; }
  else if( stir_field == "upar"   ) { stirf = stirs::upar   ; }
  else if( stir_field == "tpar"   ) { stirf = stirs::tpar   ; }
  else if( stir_field == "tperp"  ) { stirf = stirs::tperp  ; }
  else if( stir_field == "qpar"   ) { stirf = stirs::qpar   ; }
  else if( stir_field == "qperp"  ) { stirf = stirs::qperp  ; }
  else if( stir_field == "ppar"   ) { stirf = stirs::ppar   ; }
  else if( stir_field == "pperp"  ) { stirf = stirs::pperp  ; }
  
  if (scheme == "sspx3") scheme_opt = Tmethod::sspx3;
  if (scheme == "g3")    scheme_opt = Tmethod::g3;
  if (scheme == "k10")   scheme_opt = Tmethod::k10;
  if (scheme == "k2")    scheme_opt = Tmethod::k2;
  if (scheme == "rk4")   scheme_opt = Tmethod::rk4;
  if (scheme == "rk3")   scheme_opt = Tmethod::rk3;
  if (scheme == "sspx2") scheme_opt = Tmethod::sspx2;
  if (scheme == "rk2")   scheme_opt = Tmethod::rk2;

  if (eqfix && iproc==0 && ((scheme_opt == Tmethod::k10) || (scheme_opt == Tmethod::g3)  || (scheme_opt == Tmethod::k2))) {
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
  //  printf("scheme_opt = %d \n",scheme_opt);
    
  if( source == "phiext_full") {
    source_option = PHIEXT;
    if(iproc==0) printf("Running Rosenbluth-Hinton zonal flow calculation\n");
  }

  if(iproc==0) {
    if(hypercollisions_kz) printf("Using hypercollisions with coefficient proportional to kz (default).\n");
    if(hypercollisions_const) printf("Using hypercollisions with const coefficient.\n");
    if(hyper) printf("Using hyperdiffusion.\n");

    if(debug) printf("nspec_in = %i \n",nspec_in);

    if(all_kinetic && beta == 0.0 && !krehm) {
      printf("Warning: you are using kinetic electrons in a purely electrostatic calculation (beta==0.0).\n");
      printf("This will require a very small dt to resolve the high-frequency electrostatic shear Alfven wave (omega_H mode).\n");
      printf("It is recommended to instead use a small but finite value of beta to alleviate the timestep restriction.\n");
    }
  }

  nspec = nspec_in;
  init_species(species_h);
  initialized = true;
  printf(ANSI_COLOR_RESET);    
}

void Parameters::store_ncdf(int ncid, NcDims *nc_dims) {
  // open the netcdf4 file for this run
  // store all inputs for future reference
  int retval, idim, sdim, wdim, pdim, adim, qdim, gamdim, phi2dim, nc_out, nc_inputs, nc_diss;
  if (retval = nc_def_grp(ncid,      "Inputs",         &nc_inputs)) ERR(retval);
  if (retval = nc_def_grp(nc_inputs, "Domain",         &nc_dom))    ERR(retval);  
  if (retval = nc_def_grp(nc_inputs, "Time",           &nc_time))   ERR(retval);  
  if (retval = nc_def_grp(nc_inputs, "KS",             &nc_ks))     ERR(retval);  
  if (retval = nc_def_grp(nc_inputs, "Vlasov_Poisson", &nc_vp))     ERR(retval);  
  if (retval = nc_def_grp(nc_inputs, "KREHM",          &nc_krehm))  ERR(retval);  
  if (retval = nc_def_grp(nc_inputs, "collisionalETG", &nc_cetg))   ERR(retval);
  if (retval = nc_def_grp(nc_inputs, "Restart",        &nc_rst))    ERR(retval);  
  if (retval = nc_def_grp(nc_inputs, "Controls",       &nc_con))    ERR(retval);
  if (retval = nc_def_grp(nc_con,    "Numerical_Diss", &nc_diss))   ERR(retval);
  if (retval = nc_def_grp(nc_con,    "Forcing",        &nc_frc))    ERR(retval);  
  if (retval = nc_def_grp(nc_inputs, "Expert",         &nc_expert)) ERR(retval);  
  if (retval = nc_def_grp(nc_inputs, "Diagnostics",    &nc_diag))   ERR(retval);  
  if (retval = nc_def_grp(nc_inputs, "Resize",         &nc_resize)) ERR(retval);  
  if (retval = nc_def_grp(nc_inputs, "Reservoir",      &nc_ml))     ERR(retval);
  if (retval = nc_def_grp(nc_inputs, "Species",        &nc_spec))   ERR(retval);
  if (retval = nc_def_grp(nc_spec,   "Boltzmann",      &nc_bz))     ERR(retval);  

  static char file_header[] = "GX simulation data";
  if (retval = nc_put_att_text (ncid, NC_GLOBAL, "Title", strlen(file_header), file_header)) ERR(retval);

  int ivar; 
  if (retval = nc_def_var (ncid, "ny",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nx",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "ntheta",      NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nhermite",    NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nlaguerre",   NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nspecies",    NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nperiod",     NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "debug",       NC_INT,   0, NULL, &ivar)) ERR(retval);
  //if (retval = nc_def_var (ncid, "repeat",      NC_INT,   0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (nc_time, "dt",       NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_time, "nstep",    NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_time, "nwrite",   NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_time, "navg",     NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_time, "nsave",    NC_INT,   0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (nc_ks, "ks",         NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ks, "write_ks",   NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ks, "eps_ks",     NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ks, "ks_t0",      NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ks, "ks_tf",      NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ks, "ks_eps0",    NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ks, "ks_epsf",    NC_FLOAT, 0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (nc_vp, "vp",         NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_vp, "vp_closure", NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_vp, "vp_nu",      NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_vp, "vp_nuh",     NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_vp, "vp_alpha",   NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_vp, "vp_alpha_h", NC_INT,   0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (nc_krehm, "krehm",   NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_krehm, "rho_i",   NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_krehm, "d_e",     NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_krehm, "nu_ei",   NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_krehm, "zt",      NC_FLOAT, 0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (nc_cetg, "cetg",     NC_INT,   0, NULL, &ivar)) ERR(retval);
  
  specs[0] = nc_dims->species;
  if (retval = nc_def_var (nc_spec, "species_type", NC_INT,   1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_spec, "z",            NC_FLOAT, 1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_spec, "m",            NC_FLOAT, 1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_spec, "n0",           NC_FLOAT, 1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_spec, "n0_prime",     NC_FLOAT, 1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_spec, "u0_prime",     NC_FLOAT, 1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_spec, "T0",           NC_FLOAT, 1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_spec, "T0_prime",     NC_FLOAT, 1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_spec, "nu",           NC_FLOAT, 1, specs, &ivar)) ERR(retval);

  if (retval = nc_def_var (nc_dom, "y0",            NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_dom, "x0",            NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_dom, "z0",            NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_dom, "zp",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_dom, "jtwist",        NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_dom, "boundary_dum",  NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (nc_dom, ivar, "value", boundary.size(), boundary.c_str())) ERR(retval);
  if (retval = nc_def_var (nc_dom, "long_wavelength_GK",      NC_INT,   0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (nc_ml, "Use_reservoir",  NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ml, "Q",              NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ml, "K",              NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ml, "TrainingSteps",  NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ml, "PredictSteps",   NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ml, "TrainingDelta",  NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ml, "SpecRadius",     NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ml, "Regularizer",    NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ml, "InputSigma",     NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ml, "SigmaNoise",     NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ml, "FakeData",       NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ml, "ResWrite",       NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_ml, "ResBatch",       NC_INT,   0, NULL, &ivar)) ERR(retval);
  
  if (retval = nc_def_var (nc_rst, "scale",            NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_rst, "restart",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_rst, "save_for_restart", NC_INT,   0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (nc_rst, "restart_from_file_dum", NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (nc_rst, ivar, "value", restart_from_file.size(), restart_from_file.c_str())) ERR(retval);

  if (retval = nc_def_var (nc_rst, "restart_to_file_dum",   NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (nc_rst, ivar, "value", restart_to_file.size(), restart_to_file.c_str())) ERR(retval);

  if (retval = nc_def_var (nc_diag, "fixed_amp",       NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "omega",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "fluxes",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "fields",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "eigenfunctions",  NC_INT,   0, NULL, &ivar)) ERR(retval);
  
  if (retval = nc_def_var (nc_diag, "all_zonal_scalars", NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "avg_zvE",         NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "avg_zkxvEy",      NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "avg_zkden",       NC_INT,   0, NULL, &ivar)) ERR(retval); 
  if (retval = nc_def_var (nc_diag, "avg_zkUpar",      NC_INT,   0, NULL, &ivar)) ERR(retval); 
  if (retval = nc_def_var (nc_diag, "avg_zkTpar",      NC_INT,   0, NULL, &ivar)) ERR(retval); 
  if (retval = nc_def_var (nc_diag, "avg_zkTperp",     NC_INT,   0, NULL, &ivar)) ERR(retval); 
  if (retval = nc_def_var (nc_diag, "avg_zkqpar",      NC_INT,   0, NULL, &ivar)) ERR(retval); 

  if (retval = nc_def_var (nc_diag, "all_zonal",       NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "vEy",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "kxvEy",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "kden",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "kUpar",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "kTpar",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "kqpar",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "kTperp",          NC_INT,   0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (nc_diag, "all_non_zonal",   NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "xyPhi" ,          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "xyApar" ,          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "xyvEx",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "xyvEy",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "xykxvEy",         NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "xyden",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "xyUpar",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "xyTpar",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "xyTperp",         NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "xyqpar",          NC_INT,   0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (nc_diag, "moms",            NC_INT,   0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (nc_diag, "free_energy",     NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "phi",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "phi_kpar",        NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "rh",              NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diag, "pzt",             NC_INT,   0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (nc_con, "scheme_dum",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (nc_con, ivar, "value", scheme.size(), scheme.c_str())) ERR(retval);
  if (retval = nc_def_var (nc_con, "stages",                NC_INT,   0, NULL, &ivar)) ERR(retval);
  
  if (retval = nc_def_var (nc_con, "dealias_kz",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_con, "nonlinear_mode",        NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_con, "closure_model_dum",     NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (nc_con, ivar, "value", closure_model.size(), closure_model.c_str())) ERR(retval);
  if (retval = nc_def_var (nc_con, "smith_par_q",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_con, "smith_perp_q",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_con, "cfl",                   NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_con, "fphi",                  NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_con, "fapar",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_con, "fbpar",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_con, "init_amp",              NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_con, "collisions",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_con, "init_field_dum",        NC_INT,   0, NULL, &ivar)) ERR(retval);  
  if (retval = nc_put_att_text (nc_con, ivar, "value", init_field.size(), init_field.c_str())) ERR(retval);
  if (retval = nc_def_var (nc_con, "init_electrons_only",   NC_INT,   0, NULL, &ivar)) ERR(retval);  

  if (retval = nc_def_var (nc_con, "ikpar_init",             NC_INT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_con, "random_init",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  
  if (retval = nc_def_var (nc_diss, "hyper",                 NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diss, "D_hyper",               NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diss, "nu_hyper",              NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diss, "hypercollisions_const",       NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diss, "hypercollisions_kz",       NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diss, "nu_hyper_l",            NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diss, "nu_hyper_m",            NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diss, "p_hyper",               NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diss, "p_hyper_l",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diss, "p_hyper_m",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diss, "w_osc",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diss, "D_HB",                  NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diss, "p_HB",                  NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_diss, "HB_hyper",              NC_INT,   0, NULL, &ivar)) ERR(retval);
  
  if (retval = nc_def_var (nc_frc, "forcing_init",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_frc, "forcing_type_dum",      NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (nc_frc, ivar, "value", forcing_type.size(), forcing_type.c_str())) ERR(retval);
  if (retval = nc_def_var (nc_frc, "no_fields",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_frc, "forcing_amp",           NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_frc, "forcing_index",         NC_INT,   0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (nc_frc, "stir_field_dum",        NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (nc_frc, ivar, "value", stir_field.size(), stir_field.c_str())) ERR(retval);


  if (retval = nc_def_var (nc_expert, "i_share",               NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_expert, "nreal",                 NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_expert, "init_single",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_expert, "ikx_single",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_expert, "iky_single",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_expert, "ikx_fixed",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_expert, "iky_fixed",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_expert, "eqfix",                 NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_expert, "secondary",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_expert, "phi_ext",               NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_expert, "t0",                    NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_expert, "tf",                    NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_expert, "tprim0",                NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_expert, "tprimf",                NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_expert, "source_dum",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (nc_expert, ivar, "value", source.size(), source.c_str())) ERR(retval);
  if (retval = nc_def_var (nc_expert, "hegna",                 NC_INT,   0, NULL, &ivar)) ERR(retval);  // bb6126 - hegna test

  // for boltzmann opts need attribute BD bug

  if (retval = nc_def_var (nc_bz, "Z_ion",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_bz, "tau_fac",               NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_bz, "add_Boltzmann_species", NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_bz, "all_kinetic",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_bz, "Boltzmann_type_dum",    NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (nc_bz, ivar, "value", Btype.size(), Btype.c_str() ) ) ERR(retval);
  
  // geometry
  if (retval = nc_def_var (nc_inputs, "igeo",                  NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "slab",                  NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "const_curv",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "geofile_dum",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (nc_inputs, ivar, "value", geofilename.size(), geofilename.c_str())) ERR(retval);

  if (retval = nc_def_var (nc_inputs, "drhodpsi",              NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "kxfac",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "Rmaj",                  NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "shift",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "eps",                   NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "q",                     NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "shat",                  NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "kappa",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "kappa_prime",           NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "tri",                   NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "tri_prime",             NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  
  if (retval = nc_def_var (nc_inputs, "beta",                  NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "zero_shat",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "B_ref",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "a_ref",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "grhoavg",               NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_inputs, "surfarea",              NC_FLOAT, 0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (nc_resize, "domain_change",      NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_resize, "x0_mult",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_resize, "y0_mult",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_resize, "z0_mult",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_resize, "nx_mult",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_resize, "ny_mult",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_resize, "nz_mult",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_resize, "nl_add",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_resize, "nm_add",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (nc_resize, "ns_add",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  
  if (retval = nc_def_var (ncid, "code_info",             NC_INT,   0, NULL, &ivar)) ERR(retval);

  std::string hash(build_git_sha);                       
  if (retval = nc_put_att_text (ncid, ivar, "Hash",      hash.size(), hash.c_str() ) ) ERR(retval);
  std::string compiled(build_git_time);                  
  if (retval = nc_put_att_text (ncid, ivar, "BuildDate", compiled.size(), compiled.c_str() ) ) ERR(retval);
  std::string builder(build_user);                       
  if (retval = nc_put_att_text (ncid, ivar, "BuildUser", builder.size(), builder.c_str() ) ) ERR(retval);
  std::string build_host(build_hostname);                
  if (retval = nc_put_att_text (ncid, ivar, "BuildHost", build_host.size(), build_host.c_str() ) ) ERR(retval);

  putbool  (ncid, "debug",     debug);
  putint   (ncid, "ntheta",    nz_in);
  putint   (ncid, "nx",        nx_in);
  putint   (ncid, "ny",        ny_in);
  putint   (ncid, "nhermite",  nm_in);
  putint   (ncid, "nlaguerre", nl_in);
  putint   (ncid, "nspecies",  nspec_in);
  putint   (ncid, "nperiod",   nperiod);
  
  putbool (nc_resize, "domain_change", domain_change );
  putint  (nc_resize, "x0_mult"      , x0_mult       );
  putint  (nc_resize, "y0_mult"      , y0_mult       );
  putint  (nc_resize, "z0_mult"      , z0_mult       );
  putint  (nc_resize, "nx_mult"      , nx_mult       );
  putint  (nc_resize, "ny_mult"      , ny_mult       );
  putint  (nc_resize, "nz_mult"      , ntheta_mult   );
  putint  (nc_resize, "nl_add"       , nl_add        );
  putint  (nc_resize, "nm_add"       , nm_add        );  
  putint  (nc_resize, "ns_add"       , ns_add        );  
  
  putbool  (nc_ml, "Use_reservoir", Reservoir         );
  putint   (nc_ml, "Q"            , ResQ              );
  putint   (nc_ml, "K"            , ResK              );
  putint   (nc_ml, "TrainingSteps", ResTrainingSteps  );
  putint   (nc_ml, "PredictSteps" , ResPredict_Steps  );
  putint   (nc_ml, "TrainingDelta", ResTrainingDelta  );
  put_real (nc_ml, "SpecRadius"   , ResSpectralRadius );
  put_real (nc_ml, "Regularizer"  , ResReg            );
  put_real (nc_ml, "InputSigma"   , ResSigma          );
  put_real (nc_ml, "SigmaNoise"   , ResSigmaNoise     );
  putbool  (nc_ml, "FakeData"     , ResFakeData       );
  putbool  (nc_ml, "ResWrite"     , ResWrite          );
  putbool  (nc_ml, "ResBatch"     , ResBatch          );
  
  putbool  (nc_bz, "all_kinetic",           all_kinetic           );
  putbool  (nc_bz, "add_Boltzmann_species", add_Boltzmann_species );
  put_real (nc_bz, "tau_fac",               tau_fac               );
  put_real (nc_bz, "Z_ion",                 ion_z                 );

  put_real (nc_dom, "y0",      y0      );
  put_real (nc_dom, "x0",      x0      );
  if (geo_option=="slab") put_real (nc_dom, "z0",      z0      );
  putint   (nc_dom, "zp",      Zp      );
  putint   (nc_dom, "jtwist",  jtwist  );
  putbool  (nc_dom, "long_wavelength_GK", long_wavelength_GK);

  put_real (nc_time, "dt",      dt      );
  putint   (nc_time, "nstep",   nstep   );
  putint   (nc_time, "navg",    navg    );
  putint   (nc_time, "nsave",   nsave   );
  putint   (nc_time, "nwrite",  nwrite  );
  
  putbool  (nc_ks, "ks",       ks       );
  putbool  (nc_ks, "write_ks", write_ks );
  put_real (nc_ks, "eps_ks",   eps_ks   );
  put_real (nc_ks, "ks_t0",    ks_t0    );
  put_real (nc_ks, "ks_tf",    ks_tf    );
  put_real (nc_ks, "ks_eps0",  ks_eps0  );
  put_real (nc_ks, "ks_epsf",  ks_epsf  );

  putbool  (nc_vp, "vp",         vp          );
  putbool  (nc_vp, "vp_closure", vp_closure  );
  putint   (nc_vp, "vp_alpha",   vp_alpha    );
  putint   (nc_vp, "vp_alpha_h", vp_alpha_h  );
  put_real (nc_vp, "vp_nu",      vp_nu       );
  put_real (nc_vp, "vp_nuh",     vp_nuh      );

  putbool  (nc_cetg,  "cetg", cetg);
  putbool  (nc_krehm, "krehm", krehm);
  put_real (nc_krehm, "rho_i", rho_i);
  put_real (nc_krehm, "d_e", d_e);
  put_real (nc_krehm, "nu_ei", nu_ei);
  put_real (nc_krehm, "zt", zt);
  
  putbool  (nc_rst, "restart",           restart          );
  putbool  (nc_rst, "save_for_restart",  save_for_restart );  
  put_real (nc_rst, "scale",             scale            );

  putbool  (nc_diag, "all_zonal",    write_all_kmom    );
  putbool  (nc_diag, "vEy",          write_vEy         );
  putbool  (nc_diag, "kxvEy",        write_kxvEy       );
  putbool  (nc_diag, "kden",         write_kden        );
  putbool  (nc_diag, "kUpar",        write_kUpar       );
  putbool  (nc_diag, "kTpar",        write_kTpar       );
  putbool  (nc_diag, "kTperp",       write_kTperp      );
  putbool  (nc_diag, "kqpar",        write_kqpar       );

  putbool  (nc_diag, "all_non_zonal", write_all_xymom  );
  putbool  (nc_diag, "xyPhi",        write_xyPhi       );
  putbool  (nc_diag, "xyApar",       write_xyApar       );
  putbool  (nc_diag, "xyvEx",        write_xyvEx       );
  putbool  (nc_diag, "xyvEy",        write_xyvEy       );
  putbool  (nc_diag, "xykxvEy",      write_xykxvEy     );
  putbool  (nc_diag, "xyden",        write_xyden       );
  putbool  (nc_diag, "xyUpar",       write_xyUpar      );
  putbool  (nc_diag, "xyTpar",       write_xyTpar      );
  putbool  (nc_diag, "xyTperp",      write_xyTperp     );
  putbool  (nc_diag, "xyqpar",       write_xyqpar      );

  putbool  (nc_diag, "all_zonal_scalars", write_all_avgz);
  putbool  (nc_diag, "avg_zvE",      write_avg_zvE     );
  putbool  (nc_diag, "avg_zkxvEy",   write_avg_zkxvEy  );
  putbool  (nc_diag, "avg_zkden",    write_avg_zkden   );
  putbool  (nc_diag, "avg_zkUpar",   write_avg_zkUpar  );
  putbool  (nc_diag, "avg_zkTpar",   write_avg_zkTpar  );
  putbool  (nc_diag, "avg_zkTperp",  write_avg_zkTperp );
  putbool  (nc_diag, "avg_zkqpar",   write_avg_zkqpar  );

  putbool  (nc_diag, "eigenfunctions", write_eigenfuncs   );
  putbool  (nc_diag, "fields",         write_fields       );
  putbool  (nc_diag, "omega",          write_omega        );
  putbool  (nc_diag, "fixed_amp",      fixed_amplitude    );
  putbool  (nc_diag, "free_energy",    write_free_energy  );
  putbool  (nc_diag, "fluxes",         write_fluxes       );
  putbool  (nc_diag, "moms",           write_moms         );
  putbool  (nc_diag, "rh",             write_rh           );
  putbool  (nc_diag, "pzt",            write_pzt          );
  putbool  (nc_diag, "phi",            write_phi          );
  putbool  (nc_diag, "phi_kpar",       write_phi_kpar     );

  putint   (nc_expert, "nreal",   nreal);
  putint   (nc_expert, "i_share", i_share);
  putint   (nc_expert, "ikx_fixed",  ikx_fixed  );
  putint   (nc_expert, "iky_fixed",  iky_fixed  );
  putbool  (nc_expert, "eqfix",      eqfix      );
  putbool  (nc_expert, "init_single", init_single  );
  putbool  (nc_expert, "secondary",   secondary    );
  putbool  (nc_expert, "hegna",       hegna        );
  put_real (nc_expert, "phi_ext",     phi_ext      );
  putint   (nc_expert, "ikx_single",  ikx_single   );
  putint   (nc_expert, "iky_single",  iky_single   );
  put_real (nc_expert, "t0",          tp_t0        );
  put_real (nc_expert, "tf",          tp_tf        );
  put_real (nc_expert, "tprim0",      tprim0       );
  put_real (nc_expert, "tprimf",      tprimf       );

  putint   (nc_con,  "stages",          stages          );
  put_real (nc_con,  "cfl",             cfl             );
  put_real (nc_con,  "init_amp",        init_amp        );
  put_real (nc_con,  "ikpar_init",      ikpar_init       );
  putbool  (nc_con,  "random_init",     random_init     );
  putbool  (nc_con,  "dealias_kz",      dealias_kz      );
  putbool  (nc_con,  "nonlinear_mode",  nonlinear_mode  );   
  putint   (nc_con,  "smith_par_q",     smith_par_q     );
  putint   (nc_con,  "smith_perp_q",    smith_perp_q    );
  putbool  (nc_con,  "collisions",      collisions      );
  put_real (nc_con,  "fphi",            fphi            );
  put_real (nc_con,  "fapar",           fapar           );
  put_real (nc_con,  "fbpar",           fbpar           );
  
  putbool  (nc_diss, "hyper",           hyper           );
  put_real (nc_diss, "D_hyper",         D_hyper         );
  putbool  (nc_diss, "hypercollisions_const", hypercollisions_const );
  putbool  (nc_diss, "hypercollisions_kz", hypercollisions_kz );
  put_real (nc_diss, "nu_hyper_l",      nu_hyper_l      );
  put_real (nc_diss, "nu_hyper_m",      nu_hyper_m      );
  putint   (nc_diss, "nu_hyper",        nu_hyper        );
  putint   (nc_diss, "p_hyper",         p_hyper         );
  putint   (nc_diss, "p_hyper_l",       p_hyper_l       );
  putint   (nc_diss, "p_hyper_m",       p_hyper_m       );
  put_real (nc_diss, "D_HB",            D_HB            );
  putint   (nc_diss, "p_HB",            p_HB            );
  putbool  (nc_diss, "HB_hyper",        HB_hyper        );
  put_real (nc_diss, "w_osc",           w_osc           );
  
  put_real (nc_frc, "forcing_amp",      forcing_amp     );
  putint   (nc_frc, "forcing_index",    forcing_index   );
  putbool  (nc_frc, "forcing_init",     forcing_init    );
  putbool  (nc_frc, "no_fields",        no_fields       );
      
  //  putbool  (ncid, "snyder_electrons", snyder_electrons);

  putbool  (nc_inputs, "slab",        slab       );
  putbool  (nc_inputs, "const_curv",  const_curv );
  putint   (nc_inputs, "igeo",        igeo       );
  put_real (nc_inputs, "drhodpsi",    drhodpsi   );
  put_real (nc_inputs, "kxfac",       kxfac      );
  if (igeo == 0) {
    put_real (nc_inputs, "Rmaj",        rmaj       );
    put_real (nc_inputs, "shift",       shift      );
    put_real (nc_inputs, "eps",         eps        );
    put_real (nc_inputs, "q",           qsf        );
    put_real (nc_inputs, "shat",        shat       );
  }
  put_real (nc_inputs, "beta",        beta       );
  putbool  (nc_inputs, "zero_shat",   zero_shat  );
  put_real (nc_inputs, "B_ref", B_ref);
  put_real (nc_inputs, "a_ref", a_ref);
  put_real (nc_inputs, "grhoavg", grhoavg);
  put_real (nc_inputs, "surfarea", surfarea);

  // record the values of jtwist and x0 used in runname.nc
  putint (nc_dom, "jtwist", jtwist);
  put_real (nc_dom, "x0", x0);

  putspec (nc_spec, nspec_in, species_h);
}

void Parameters::init_species(specie* species)
{
  vtmax = -1.;
  vtmin = 1000000000;
  tzmax = -1.;
  etamax = -1.;
  for(int s=0; s<nspec_in; s++) {
    species[s].vt   = sqrt(species[s].temp / species[s].mass);
    species[s].tz   = species[s].temp / species[s].z;
    species[s].zt   = species[s].z / species[s].temp;
    species[s].rho2 = species[s].temp * species[s].mass / (species[s].z * species[s].z); // note this does not have a factor of 1/B**2
    species[s].nt    = species[s].dens * species[s].temp;
    species[s].nz    = species[s].dens * species[s].z;
    species[s].jparfac = species[s].nz * species[s].vt * beta / 2.;
    species[s].jperpfac = -species[s].dens * species[s].temp * beta / 2.; 
    if (long_wavelength_GK) {
      species[s].rho2  = 0; // setting rho2 = 0.
      species[s].rho2_long_wavelength_GK  = species[s].temp * species[s].mass / (species[s].z * species[s].z); // note this does not have a factor of 1/B**2. This rho2 is used for quasineutrality 1-Gam0 --> b_s approximation, whereas rho2 = 0 elsewhere for long_wavelength_GK.
      if(iproc==0) printf("You are running GX with the long wavelength approximation.");
    }
    if (debug) {
      printf("species = %d \n",s);
      printf("mass, z, temp, dens = %f, %f, %f, %f \n",
	     species[s].mass, species[s].z, species[s].temp, species[s].dens);
      printf("vt, tz, zt = %f, %f, %f \n",
	     species[s].vt, species[s].tz, species[s].zt);
      printf("rho2, nt, nz = %f, %f, %f \n",
	     species[s].rho2, species[s].nt, species[s].nz);
      printf("jparfac, jperpfac = %f, %f \n", 
             species[s].jparfac, species[s].jperpfac);
      printf("nu_ss = %f, tprim = %f, fprim = %f, uprim = %f\n\n", species[s].nu_ss, species[s].tprim, species[s].fprim, species[s].uprim);
    }      
    vtmax = max(vtmax, species[s].vt);
    vtmin = min(vtmin, species[s].vt);
    tzmax = max(tzmax, abs(species[s].tz));
    etamax = max(etamax, species[s].tprim/species[s].fprim);

    if(species[s].type == 1) {
      ne = species[s].dens;
      Te = species[s].temp;
    }
  }
}

int Parameters::getint (int ncid, const char varname[]) {
  int idum, retval, res;
  if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
  if (retval = nc_get_var  (ncid, idum, &res)) ERR(retval);
  //  if (debug) printf("%s = %i \n",varname, res);
  return res;
}

void Parameters::putint (int ncid, const char varname[], int val) {
  int idum, retval;
  if (debug) printf("%s = %i \n",varname, val);
  if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
  if (retval = nc_put_var  (ncid, idum, &val)) ERR(retval);
}

void Parameters::putbool (int ncid, const char varname[], bool val) {
  int idum, retval;
  int b = val==true ? 1 : 0 ; 
  if (debug) printf("%s = %d \n",varname, b);
  if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
  if (retval = nc_put_var  (ncid, idum, &b))       ERR(retval);
}

bool Parameters::getbool (int ncid, const char varname[]) {
  int idum, ires, retval;
  bool res;
  if (debug) printf("%s = %i \n", varname, ires);
  if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
  if (retval = nc_get_var  (ncid, idum, &ires)) ERR(retval);
  res = (ires!=0) ? true : false ;
  return res;
}

float Parameters::get_real (int ncid, const char varname[]) {
  int idum, retval;
  float res;
  if (debug) printf("%s = %f \n",varname, res);
  if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
  if (retval = nc_get_var  (ncid, idum, &res)) ERR(retval);
  return res;
}

void Parameters::put_real (int ncid, const char varname[], float val) {
  int idum, retval;
  if (debug) printf("%s = %f \n",varname, val);
  if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
  if (retval = nc_put_var  (ncid, idum, &val)) ERR(retval);
}

void Parameters::put_wspectra (int ncid, std::vector<int> s) {

  int idum, retval;
  wspectra_start[0] = 0;
  wspectra_count[0] = nw_spectra;

  if (retval = nc_inq_varid(ncid, "wspectra", &idum))     ERR(retval);
  if (retval = nc_put_vara (ncid, idum, wspectra_start, wspectra_count, s.data())) ERR(retval);
}

void Parameters::put_pspectra (int ncid, std::vector<int> s) {

  int idum, retval;
  pspectra_start[0] = 0;
  pspectra_count[0] = np_spectra;

  if (retval = nc_inq_varid(ncid, "pspectra", &idum))     ERR(retval);
  if (retval = nc_put_vara (ncid, idum, pspectra_start, pspectra_count, s.data())) ERR(retval);
}

void Parameters::put_aspectra (int ncid, std::vector<int> s) {

  int idum, retval;
  aspectra_start[0] = 0;
  aspectra_count[0] = na_spectra;

  if (retval = nc_inq_varid(ncid, "aspectra", &idum))     ERR(retval);
  if (retval = nc_put_vara (ncid, idum, aspectra_start, aspectra_count, s.data())) ERR(retval);
}

void Parameters::put_qspectra (int ncid, std::vector<int> s) {

  int idum, retval;
  qspectra_start[0] = 0;
  qspectra_count[0] = nq_spectra;

  if (retval = nc_inq_varid(ncid, "qspectra", &idum))     ERR(retval);
  if (retval = nc_put_vara (ncid, idum, qspectra_start, qspectra_count, s.data())) ERR(retval);
}

void Parameters::put_gamspectra (int ncid, std::vector<int> s) {

  int idum, retval;
  gamspectra_start[0] = 0;
  gamspectra_count[0] = ngam_spectra;

  if (retval = nc_inq_varid(ncid, "gamspectra", &idum))     ERR(retval);
  if (retval = nc_put_vara (ncid, idum, gamspectra_start, gamspectra_count, s.data())) ERR(retval);
}

void Parameters::put_phi2spectra (int ncid, std::vector<int> s) {

  int idum, retval;
  phi2spectra_start[0] = 0;
  phi2spectra_count[0] = nphi2_spectra;

  if (retval = nc_inq_varid(ncid, "phi2spectra", &idum))     ERR(retval);
  if (retval = nc_put_vara (ncid, idum, phi2spectra_start, phi2spectra_count, s.data())) ERR(retval);
}

void Parameters::putspec (int  ncid, int nspec, specie* spec) {
  int idum, retval;
  
  is_start[0] = 0;
  is_count[0] = nspec;

  // this stuff should all be in species itself!
  // reason for all this is basically legacy + cuda does not support <vector>
  
  std::vector <float> zs, ms, ns, Ts, Tps, nps, ups, nus;
  std::vector <int> types;
  
  for (int is=0; is<nspec; is++) {
    zs.push_back(spec[is].z);
    ms.push_back(spec[is].mass);
    ns.push_back(spec[is].dens);
    Ts.push_back(spec[is].temp);
    Tps.push_back(spec[is].tprim);
    nps.push_back(spec[is].fprim);
    ups.push_back(spec[is].uprim);
    nus.push_back(spec[is].nu_ss);
    types.push_back(spec[is].type);
  }
  float *z = &zs[0];
  float *m = &ms[0];
  float *n0 = &ns[0];
  float *T0 = &Ts[0];
  float *Tp = &Tps[0];
  float *np = &nps[0];
  float *up = &ups[0];
  float *nu = &nus[0];
  int *st = &types[0];
  
  if (retval = nc_inq_varid(ncid, "z", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, z))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "m", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, m))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "n0", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, n0))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "n0_prime", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, np))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "u0_prime", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, up))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "T0", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, T0))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "T0_prime", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, Tp))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "nu", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, nu))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "species_type", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, st))  ERR(retval);
}

void Parameters::set_jtwist_x0(float *shat_in, float *gds21, float *gds22)
{
  float shat = *shat_in;
  // note: twist_shift_geo_fac reduces to 2*pi*shat in the axisymmetric limit
  float twist_shift_geo_fac = 2.*shat*gds21[0]/gds22[0];

  if(iproc==0) {
    if(debug) printf("set_jtwist_x0: shat = %f, twist_shift_geo_fac = %f\n", shat, twist_shift_geo_fac);

    // check consistency of boundary and geo_option
    printf(ANSI_COLOR_RED);
    if(boundary == "continuous drifts" || boundary == "fix aspect") {
      if(geo_option != "vmec") printf("Warning: boundary option \"%s\" is only available with the VMEC geometry module. Using standard twist-shift BCs (boundary = \"linked\")\n", boundary.c_str()); 
    }
    if(boundary == "exact periodic") {
      if(geo_option != "vmec") printf("Warning: boundary option \"%s\" is only available with the VMEC geometry module. Using standard periodic BCs (boundary = \"periodic\")\n", boundary.c_str()); 
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

