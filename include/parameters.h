#pragma once
#include "gpu_defs.h"
#include "get_error.h"

#define DEBUGPRINT(_fmt, ...)  if (pars->debug) fprintf(stderr, "[file %s, line %d]: " _fmt, __FILE__, __LINE__, ##__VA_ARGS__)
#define DEBUG_PRINT(_fmt, ...)  if (pars_->debug) fprintf(stderr, "[file %s, line %d]: " _fmt, __FILE__, __LINE__, ##__VA_ARGS__)

#define CP_ON_GPU(to, from, isize) checkCuda(cudaMemcpy(to, from, isize, cudaMemcpyDeviceToDevice))
#define CP_ON_GPU_ASYNC(to, from, isize, stream) checkCuda(cudaMemcpyAsync(to, from, isize, cudaMemcpyDeviceToDevice, stream))
#define CP_TO_GPU(gpu, cpu, isize) checkCuda(cudaMemcpy(gpu, cpu, isize, cudaMemcpyHostToDevice))
#define CP_TO_CPU(cpu, gpu, isize) checkCuda(cudaMemcpy(cpu, gpu, isize, cudaMemcpyDeviceToHost))

#define CUDA_DEBUG(_fmt, ...) if (pars->debug) fprintf(stderr, "[file %s, line %d]: " _fmt, __FILE__, __LINE__, ##__VA_ARGS__, cudaGetErrorString(cudaGetLastError()))

#define ERR(e) {printf("Error: %s. See file: %s, line %d\n", nc_strerror(e),__FILE__,__LINE__); exit(2);}

#define NC_SUCCESS 0
#define NC_ERR( expr ) {\
  int retval = (expr);\
  if ( retval != NC_SUCCESS ) {\
    fprintf(stderr, "NetCDF Error: %s (retval = %d) in \"%s\" at %s:%d \n", nc_strerror(retval), static_cast<unsigned int>(retval), #expr, __FILE__, __LINE__);\
    exit(2);\
  }\
}

#include "species.h"
// #include <cufft.h>
#include <string>
#include <vector>
#include <mpi.h>
#include "toml.hpp"
//#include "ncdf.h"

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

enum class inits {density, upar, tpar, tperp, qpar, qperp, all};
enum class stirs {density, upar, tpar, tperp, qpar, qperp, ppar, pperp};
enum class Tmethod {sspx2, sspx3, rk3, rk4, k10};
enum class Closure {none, beer42, smithperp, smithpar};
	       
#define RH_equilibrium 3
#define PHIEXT 1

#define BOLTZMANN_IONS 1
#define BOLTZMANN_ELECTRONS 2

class NetCDF;
class NcDims;
class NcInputs;

class Parameters {

 public:
  Parameters(char* filename, int iproc=0, int nprocs=1, MPI_Comm mpcom=MPI_COMM_WORLD);
  ~Parameters(void);
  
  int iproc, nprocs;
  MPI_Comm mpcom;
  void get_nml_vars(NetCDF* ncdf);
  void get_Dimensions(const toml::value nml);
  void get_Domain(const toml::value nml);
  void get_Time(const toml::value nml);
  void get_Initialization(const toml::value nml);
  void get_Restart(const toml::value nml);
  void get_Dissipation(const toml::value nml);
  void get_KREHM(const toml::value nml);
  void get_Expert(const toml::value nml);
  void get_Diagnostics(const toml::value nml);
  void get_Resize(const toml::value nml);
  void get_Forcing(const toml::value nml);
  void get_Boltzmann(const toml::value nml);
  void get_Geometry(const toml::value nml);
  void get_Physics(const toml::value nml);
  void get_species(const toml::value nml);

  template <typename T>
  T find_or(const toml::value nml, int ncid, const char varname[], T val);

  void set_jtwist_x0(float* shat, float *gds21, float *gds22);

  int p_HB, p_hyper_l, p_hyper_m, p_hyper_lm, irho, nwrite, nwrite_big, navg, nsave, igeo, nreal;
  int p_hyper_z;
  int nz_in, nperiod, Zp, bishop, scan_number, icovering;
  int nx_in, ny_in, jtwist, nm_in, nl_in, nstep, nstep_restart, nspec_in, nspec;
  int x0_mult, y0_mult, z0_mult, nx_mult, ny_mult, ntheta_mult;
  int nm_add, nl_add, ns_add;
  int forcing_index, smith_par_q, smith_perp_q, forcing_kz, forcing_k2min, forcing_k2max;
  int equilibrium_type, source_option, inlpm, p_hyper, iphi00;
  int dorland_phase_ifac, ivarenna, iflr, i_share;
  int iky_single, ikx_single, iky_fixed, ikx_fixed;
  int Boltzmann_opt;
  int stages;
  int geoType, iflux, isym;
  //  int lh_ikx, lh_iky;
  int zonal_dens_switch, q0_dens_switch;
  // formerly part of time struct
  int trinity_timestep, trinity_iteration, trinity_conv_count, end_time;   
  int ResQ, ResK, ResTrainingSteps, ResTrainingDelta, ResPredict_Steps; 
  
  inits initf;
  stirs stirf;
  Tmethod scheme_opt;
  Closure closure_model_opt;
  
  float rhoc, eps, shat, qsf, rmaj, r_geo, shift, akappa, akappri, RBzeta_override;
  float tri, tripri, drhodpsi, epsl, kxfac, cfl, phi_ext, scale, tau_fac;
  float ti_ov_te, beta, g_exb, s_hat_input, beta_prime_input, init_amp;
  float x0, y0, z0, dt, dt_max, dt_min, fixed_dt;
  float fphi, fapar, fbpar, kpar_init, shaping_ps;
  int ikpar_init;
  float densfac, uparfac, tparfac, tprpfac, qparfac, qprpfac;
  float forcing_amp, pos_forcing_amp, neg_forcing_amp, me_ov_mi, nu_ei, eta, nu_hyper, D_hyper;
  float dnlpm, dnlpm_dens, dnlpm_tprp, nu_hyper_l, nu_hyper_m, nu_hyper_lm;
  float nu_hyper_z;
  float D_HB, w_osc;
  float low_cutoff, high_cutoff, nlpm_max, tau_nlpm;
  float ion_z, ion_mass, ion_dens, ion_fprim, ion_temp, ion_tprim, ion_vnewk;
  float avail_cpu_time, margin_cpu_time;
  float tp_t0, tp_tf, tprim0, tprimf;
  float ks_t0, ks_tf, ks_eps0, ks_epsf;
  float ResSpectralRadius, ResReg, ResSigma, ResSigmaNoise; 
  float eps_ks;
  float vp_nu, vp_nuh;
  int vp_alpha, vp_alpha_h;
  float vtmax, tzmax, etamax, vtmin;
  float delrho, p_prime_input, invLp_input, alpha_input;
  float B_ref, a_ref, grhoavg, surfarea;
  float t_max, t_add;
  float zero_shat_threshold;

  unsigned int random_seed;

  // parameters for KREHM system
  bool krehm;
  float rho_s, rho_i, d_e, zt;
  bool harris_sheet;
  bool periodic_equilibrium;
  bool island_coalesce;
  float k0; 
  bool gaussian_tube;
  float kc; 
  bool random_gaussian;
  cuComplex phi_test, smith_perp_w0;

  specie *species_h;
  float ne, Te;

  bool adiabatic_electrons, snyder_electrons, stationary_ions, dorland_qneut;
  bool all_kinetic, ks, gx, add_Boltzmann_species, write_ks, random_init;
  bool gaussian_init;
  float gauss_env_const_coeff, gauss_env_sin_coeff;
  float gaussian_width;
  bool write_all_kmom, write_kmom, write_xymom, write_all_xymom, write_avgz, write_all_avgz;
  bool zero_shat;
  bool nonTwist;
  
  bool write_avg_zvE, write_avg_zkxvEy, write_avg_zkden, write_avg_zkUpar;
  bool write_avg_zkTpar, write_avg_zkTperp, write_avg_zkqpar;

  bool write_vEy, write_kxvEy, write_kden, write_kUpar, write_kTpar, write_kTperp, write_kqpar;

  bool write_xyvEx, write_xyvEy, write_xykxvEy, write_xyden, write_xyUpar;
  bool write_xyTpar, write_xyTperp, write_xyqpar;
  bool write_xyPhi; 
  bool write_xyApar; 

  bool nonlinear_mode, linear, iso_shear, secondary, local_limit, hyper, HB_hyper;
  bool hyperz;
  bool no_landau_damping, turn_off_gradients_test, slab, hypercollisions_const, hypercollisions_kz;
  bool write_netcdf, write_omega, write_rh, write_phi, restart, restart_if_exists, save_for_restart, restart_with_perturb, append_on_restart;
  bool fixed_amplitude, write_fields, write_eigenfuncs; 
  bool append_old, no_omegad, eqfix, write_pzt, collisions, domain_change;
  bool const_curv, varenna, varenna_fsa, dorland_phase_complex, add_noise;
  bool new_varenna, new_catto, nlpm, dorland_nlpm, dorland_nlpm_phase, ExBshear;
  bool nlpm_kxdep, nlpm_nlps, nlpm_cutoff_avg, nlpm_zonal_kx1_only, smagorinsky;
  bool debug, init_single, boundary_option_periodic, forcing_init, no_fields; 
  bool init_electrons_only;
  bool nlpm_test, new_nlpm, hammett_nlpm_interference, nlpm_abs_sgn, nlpm_hilbert;
  bool low_b, low_b_all, higher_order_moments, nlpm_zonal_only, nlpm_vol_avg;
  bool no_nonlin_flr, no_nonlin_cross_terms, no_nonlin_dens_cross_term;
  bool zero_order_nonlin_flr_only, no_zonal_nlpm, diagnosing_kzspec;
  bool write_l_spectrum, write_h_spectrum, write_lh_spectrum, repeat;
  bool new_style;
  bool write_phi_kpar, write_moms, write_fluxes, diagnosing_spectra;
  bool write_free_energy, diagnosing_moments, diagnosing_pzt;
  bool ostem_rname, new_varenna_fsa, qpar0_switch, qprp0_switch;
  bool zero_restart_avg, no_zderiv_covering, no_zderiv, zderiv_loop;
  bool Reservoir, ResFakeData, ResWrite, ResBatch;
  bool dealias_kz;
  bool hegna;  // bb6126 - hegna test
  bool ei_colls, coll_conservation;
  bool efit_eq, dfit_eq, gen_eq, ppl_eq, local_eq, idfit_eq, chs_eq, transp_eq, gs2d_eq;
  //  bool tpar_omegad_corrections, tperp_omegad_corrections, qpar_gradpar_corrections ;
  //  bool qpar_bgrad_corrections, qperp_gradpar_corrections, qperp_bgrad_corrections ;
  bool use_NCCL;
  bool use_fft_callbacks;
  bool long_wavelength_GK;
  bool ExBshear_phase;
  float damp_ends_widthfrac, damp_ends_amp;

  char *scan_type;
  char *equilibrium_option, *nlpm_option;
  char run_name[1255];
  char nml_file[1255];

  std::string Btype;
  std::string code_info;
  
  std::string restart_from_file, restart_to_file;
  char default_restart_filename[1000];
  
  std::string scheme, forcing_type, init_field, stir_field;
  std::string closure_model, boundary, source;
  
  std::string geo_option;
  std::string geofilename;
  std::string eqfile;
  
  cudaDeviceProp prop;
  int maxThreadsPerBlock;

 private:
  NcInputs* nc_inputs_;
  bool initialized;
};

