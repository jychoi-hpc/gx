#pragma once
#include "grids.h"
#include "parameters.h"
#include "geometry.h"

class NetCDF_ids {

 public: 
  NetCDF_ids(Grids* grids, Parameters* pars, Geometry* geo);
  ~NetCDF_ids();

  void close_nc_file();
  
  int nx, ny, nz, kx_dim, ky_dim, nhermite, nlaguerre, kx, ky;
  int m_dim, l_dim, s_dim;
  int theta, theta_x, bmag, bgrad, gbdrift, gbdrift0, periodic;
  int cvdrift, cvdrift0, gds2, gds21, gds22, grho, jacobian;
  int nstep, dt, restart, time_dim, nspec, char16_dim;
  int cfl, init, init_amp, init_single, iky_single, ikx_single, kpar_init;
  int nu_hyper_l, nu_hyper_m, p_hyper_l, p_hyper_m, scheme_opt;
  int closure_model_opt, file, gpu, write_rh, phi_rh, forcing_index; 
  int iphi00, local_limit, linear, forcing, forcing_type, forcing_amp;
  int hypercollisions, snyder_electrons, nlpm, no_zonal_nlpm;
  int no_nl_flr, no_nl_cross_terms, no_nl_dens_cross_term;  
  int phi_ext, time, nwrite, navg, nsave, debug, omega, gamma;
  int density, upar, phi, apar, density0, phi0, qflux, omega_t, gamma_t;
  int write_omega, write_phi, write_apar, write_h_spectrum, write_l_spectrum;
  int write_lh_spectrum, lh_ikx, lh_iky, lhspec, lspec, hspec, source_opt;
  int write_spec_v_time, lhspec_t, lspec_t, hspec_t, density_kpar, phi_kpar;
  int eqfix, ikx_fixed, iky_fixed, write_pzt, prim, sec, tert;
  int hmom_real, hmom_real_t, hmom_imag, hmom_imag_t;

  //  char closure_model[32], scheme[32], source[32];
  
  int v_kx[1];            // dims for a real scalar as a function of kx 
  int v_ky[1];            // dims for a real scalar as a function of ky 
  int geo_v_theta[1];     // dims for a real scalar as a function of theta
  int scalar_v_time[1];   // dims for a real scalar as function of time
  int complex_v_time[2];  // dims for a complex scalar as function of time
  int pzt_v_time[1];      // dims for real scalar v time
  
  int g_v_l[1];          // dims for a real quantity vs l
  int g_v_lt[2];         // dims for a real quantity vs l, time

  int g_v_m[1];          // dims for a real quantity vs m
  int g_v_mt[2];         // dims for a real quantity vs m, time

  int g_v_lm[2];         // dims for a real quantity vs l, m
  int g_v_lmt[3];        // dims for a real quantity vs l, m, time

  int final_field[2];    // dims for a real slice at outboard midplane at end of run (x, y)

  int zkxky[3];          // dims for a real quantity vs kx, ky, z
  int omega_end[3];      // dims for a complex quantity vs kx, ky
  int omega_v_time[4];   // dims for a complex quantity vs kx, ky, time
  int moments_out[4];    // dims for a complex quantity vs ky, kx, z
  
  size_t rh_start[2], rh_count[2];

  size_t time_start[1], time_count[1];

  size_t pzt_start[1], pzt_count[1];
  
  // BD eventually needs [grids_->Nspecies] instead of [1]
  size_t flux_start[1], flux_count[1];

  size_t l_start[1],  l_count[1];
  size_t lt_start[2], lt_count[2];

  size_t m_start[1],  m_count[1];
  size_t mt_start[2], mt_count[2];

  size_t lh_start[2],  lh_count[2];
  size_t lht_start[3], lht_count[3];

  size_t om_start[3],  om_count[3];
  size_t omt_start[4], omt_count[4];

  size_t mom_start[4], mom_count[4];
  size_t geo_start[1], geo_count[1];
  size_t zkxky_start[3], zkxky_count[3];
  size_t ky_start[1], ky_count[1];
  size_t kx_start[1], kx_count[1];
  
  float *theta_extended;

 private:   
  const Parameters* pars_;
  const Grids* grids_;
  const Geometry* geo_;
};
