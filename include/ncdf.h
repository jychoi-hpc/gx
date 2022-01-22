#pragma once
#include "grids.h"
#include "parameters.h"
#include "geometry.h"
#include "reductions.h"
#include "nca.h"
#include "device_funcs.h"
#include "grad_perp.h"

class NetCDF_ids {

 public: 
  NetCDF_ids(Grids* grids, Parameters* pars, Geometry* geo = nullptr);
  ~NetCDF_ids();

  void close_nc_file();

  void reduce2k    (float * fk, cuComplex * f);

  void write_Wm    (float * G2, bool endrun = false);
  void write_Wl    (float * G2, bool endrun = false);
  void write_Wlm   (float * G2, bool endrun = false);

  void write_Ws    (float * G2, bool endrun = false);
  void write_Wz    (float * G2, bool endrun = false);
  void write_Wkz   (float * G2, bool endrun = false);
  void write_Wky   (float * G2, bool endrun = false);
  void write_Wkx   (float * G2, bool endrun = false);
  void write_Wkxky (float * G2, bool endrun = false);
  
  void write_Ps    (float * P2, bool endrun = false);
  void write_Pz    (float * P2, bool endrun = false);
  void write_Pkz   (float * P2, bool endrun = false);
  void write_Pky   (float * P2, bool endrun = false);
  void write_Pkx   (float * P2, bool endrun = false);  
  void write_Pkxky (float * P2, bool endrun = false);
  
  void write_As    (float * P2, bool endrun = false);
  void write_Az    (float * P2, bool endrun = false);
  void write_Akz   (float * G2, bool endrun = false);
  void write_Aky   (float * P2, bool endrun = false);
  void write_Akx   (float * P2, bool endrun = false);
  void write_Akxky (float * P2, bool endrun = false);

  void write_Q     (float * Q,   bool endrun = false);
  void write_omg   (cuComplex *W, bool endrun = false);
  void write_moment(nca *D, cuComplex *f, float* vol_fac);

  void write_gy    (float * gy_d,     bool endrun = false);
  
  void write_zonal(nca *D, cuComplex* f, bool shear, float adj);
  void write_zonal_nc(nca *D, bool endrun = false);
  void write_nc(nca *D, bool endrun = false);
  void write_nc(nca *D, double data, bool endrun = false);
  void write_nc(nca *D, float data, bool endrun = false);
  void write_ks_data(nca *D, cuComplex *G);
  void write_ks_data(nca *D, float *G);
  void write_Wtot();
  
  nca *rh, *omg, *den, *wphi, *denk, *wphik, *den0, *wphi0, *qs; 
  nca *Wm, *Wl, *Wlm, *Pzt, *pZt, *pzT, *Wtot;
  nca *Ps, *Pky, *Pkx, *Pkxky, *Pz, *Pkz;
  nca *Ws, *Wky, *Wkx, *Wkxky, *Wz, *Wkz;
  nca *As, *Aky, *Akx, *Akxky, *Az, *Akz;
  nca *g_y;
  nca *r_y; 

  nca *vEy,    *xyvEx,    *xyvEy,    *avg_zvE;
  nca *kxvEy,  *xykxvEy,  *avg_zkxvEy;
  //  nca *kyvE,   *xykyvE,   *avg_zkyvE;
  nca *kden,   *xyden,    *avg_zkden;
  nca *kUpar,  *xyUpar,   *avg_zkUpar;
  nca *kTpar,  *xyTpar,   *avg_zkTpar;
  nca *kTperp, *xyTperp,  *avg_zkTperp;
  nca *kqpar,  *xyqpar,   *avg_zkqpar;

  nca *time, *z_time, *nz_time;
  nca *r_time; 
  
  int nx, ny, nz, nkz, kx_dim, ky_dim, kx, ky, kz;
  int m_dim, l_dim, s_dim, y, y_dim, x, x_dim;
  int zy, zx, nzy, nzx;
  int state; 
  int theta, theta_x, bmag, bgrad, gbdrift, gbdrift0, periodic;
  int cvdrift, cvdrift0, gds2, gds21, gds22, grho, jacobian;
  int nstep, dt, restart, time_dim, nspec, char16_dim;
  int cfl, init, init_amp, init_single, iky_single, ikx_single, kpar_init;
  int nu_hyper_l, nu_hyper_m, p_hyper_l, p_hyper_m, scheme_opt;
  int closure_model_opt, file, gpu, forcing_index; 
  int Boltzmann_opt, local_limit, linear, forcing, forcing_type, forcing_amp;
  int hypercollisions, snyder_electrons;
  int phi_ext, nwrite, navg, nsave, debug, nreal;
  int density, upar, phi, apar, density0, phi0, qflux;
  int write_apar, collisions;
  int source_opt;
  int density_kpar, phi_kpar;
  int eqfix, ikx_fixed, iky_fixed, prim, sec, tert;
  int z_file, zx_dim, zy_dim, ztime_dim;
  int r_file, res_dim, rtime_dim; 
  int nz_file, nzx_dim, nzy_dim, nztime_dim;
  
  int v_z[1];            // dims for a scalar as a function of z
  int v_kz[1];           // dims for a scalar as a function of kz 
  int v_kx[1];           // dims for a scalar as a function of kx 
  int v_ky[1];           // dims for a scalar as a function of ky 
  int geo_v_theta[1];    // dims for a scalar as a function of theta
  int zkxky[3];          // dims for a real quantity vs kx, ky, z
  size_t zkxky_start[3], zkxky_count[3];

  size_t geo_start[1], geo_count[1];
  size_t ky_start[1], ky_count[1];
  size_t kx_start[1], kx_count[1];
  size_t kz_start[1], kz_count[1];
  size_t z_start[1], z_count[1];
  size_t y_start[1], y_count[1];
  size_t x_start[1], x_count[1];
  
  float * theta_extended ;
  
 private:   

  Parameters * pars_   ;
  Grids      * grids_  ;
  Geometry   * geo_    ;
  GradPerp   * grad_phi;
  GradPerp   * grad_perp; 
  Red        * red     ;
  Red        * pot     ;
  Red        * ph2     ;
  Red        * all_red ;
  
  float *primary, *secondary, *tertiary;
  cuComplex * t_bar     ;
  cuComplex * amom      ;
  cuComplex * df        ;
  cuComplex * favg      ; 
  float totW;

  dim3 dgx, dbx, dgxy, dbxy, dGr, dBr, dbp, dgp, dbfla, dgfla, dball, dgall; 
};
