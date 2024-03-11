#pragma once
#include "grids.h"
#include "parameters.h"
#include "geometry.h"
#include "reductions.h"
#include "device_funcs.h"
#include "grad_perp.h"
#include "nca.h"
#include "netcdf.h"
#include "netcdf_par.h"
#include <string>

using namespace std;

class NetCDF_ids {

 public: 
  NetCDF_ids(Grids* grids, Parameters* pars, Geometry* geo = nullptr);
  ~NetCDF_ids();

  int fileid;

  void close_nc_file();

  void reduce2k    (float * fk, cuComplex * f);
  void reduce2zk   (float * fk, cuComplex * f);
  
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
 
  void write_Ms (float * P2, bool endrun = false);
  void write_Mky (float * P2, bool endrun = false);
  void write_Mkx (float * P2, bool endrun = false);
  void write_Mkxky (float * P2, bool endrun = false);
  void write_Mkperp (float * P2, bool endrun = false);

  void write_As    (float * P2, bool endrun = false);
  void write_Az    (float * P2, bool endrun = false);
  void write_Akz   (float * G2, bool endrun = false);
  void write_Aky   (float * P2, bool endrun = false);
  void write_Akx   (float * P2, bool endrun = false);
  void write_Akxky (float * P2, bool endrun = false);

  void write_Phi2t    (float * P2, bool endrun = false);
  void write_Phi2z    (float * P2, bool endrun = false);
  void write_Phi2kz   (float * P2, bool endrun = false);
  void write_Phi2ky   (float * P2, bool endrun = false);
  void write_Phi2kx   (float * P2, bool endrun = false);
  void write_Phi2kxky (float * P2, bool endrun = false);

  void write_Gam     (float * Gam,   bool endrun = false);
  void write_Gamz   (float * Gam,   bool endrun = false);
  void write_Gamky   (float * Gam,   bool endrun = false);
  void write_Gamkx   (float * Gam,   bool endrun = false);
  void write_Gamkxky (float * Gam,   bool endrun = false);
  void write_Q     (float * Q,   bool endrun = false);
  void write_Qz   (float * Q,   bool endrun = false);
  void write_Qky   (float * Q,   bool endrun = false);
  void write_Qkx   (float * Q,   bool endrun = false);
  void write_Qkxky (float * Q,   bool endrun = false);
  void write_omg   (cuComplex *W, bool endrun = false);
  void write_moment(nca *D, cuComplex *f, float* vol_fac);
  void write_fields(nca *D, cuComplex *a, bool endrun = true);
  void write_fields_realspace(nca *D, cuComplex *a, bool endrun=true);
  void write_gy    (float * gy_d,     bool endrun = false);
  
  void write_zonal(nca *D, cuComplex* f, bool shear, float adj);
  void write_zonal_nc(nca *D, bool endrun = false);
  void write_nc(nca *D, bool endrun = false);
  void write_nc(nca *D, double data, bool endrun = false);
  void write_nc(nca *D, float data, bool endrun = false);
  void write_ks_data(nca *D, cuComplex *G);
  void write_ks_data(nca *D, float *G);
  void write_Wtot();
  
  nca *rh, *omg, *den, *wphi, *denk, *wphik, *den0, *wphi0, *qs, *ps; 
  nca *Wm, *Wl, *Wlm, *Pzt, *pZt, *pzT, *Wtot;
  nca *Ps, *Pky, *Pkx, *Pkxky, *Pz, *Pkz;
  nca *Ms, *Mky, *Mkx, *Mkxky, *Mkperp;
  nca *Ws, *Wky, *Wkx, *Wkxky, *Wz, *Wkz;
  nca *As, *Aky, *Akx, *Akxky, *Az, *Akz;
  nca *Qs, *Qky, *Qkx, *Qkxky, *Qz, *Qkz;
  nca *Gams, *Gamky, *Gamkx, *Gamkxky, *Gamz, *Gamkz;
  nca *Phi2t, *Phi2ky, *Phi2kx, *Phi2kxky, *Phi2z, *Phi2kz;
  nca *fields_phi, *fields_apar, *fields_apar_realspace, *fields_bpar;
  nca *g_y;
  nca *r_y; 

  nca *vEy,    *xyvEx,    *xyvEy,    *avg_zvE;
  nca *kxvEy,  *xykxvEy,  *avg_zkxvEy;
  //  nca *kyvE,   *xykyvE,   *avg_zkyvE;
  nca *xyPhi; 
  nca *xyApar; 
  nca *kden,   *xyden,    *avg_zkden;
  nca *kUpar,  *xyUpar,   *avg_zkUpar;
  nca *kTpar,  *xyTpar,   *avg_zkTpar;
  nca *kTperp, *xyTperp,  *avg_zkTperp;
  nca *kqpar,  *xyqpar,   *avg_zkqpar;

  nca *time, *z_time, *nz_time;
  nca *r_time; 

  int nx, ny, nz, nkz, kx_dim, ky_dim, kx, ky, kz;
  int kperp_dim;
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
  int hegna;  // bb6126 - hegna test
  
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
  
  float primary[1], secondary[1], tertiary[1];
  cuComplex * t_bar     ;
  cuComplex * amom      ;
  cuComplex * df        ;
  cuComplex * favg      ; 
  float totW;

  dim3 dgx, dbx, dgxy, dbxy, dGr, dBr, dbp, dgp, dbfla, dgfla, dball, dgall; 
};

class NcDims;
class NcGrids;
class NcGeo;
class NcDiagnostics;

class NetCDF {
 public:
  NetCDF(Parameters* pars, Grids* grids, Geometry* geo = nullptr, string suffix = ".out.nc");
  ~NetCDF();

  int fileid;
  NcDims *nc_dims;
  NcGrids *nc_grids;
  NcGeo *nc_geo;
  NcDiagnostics *nc_diagnostics;
  void sync();
 private:
  void close_nc_file();
  Parameters *pars_;
  Grids *grids_;
  Geometry *geo_;
};

class NcDims {
 public:
  NcDims(Parameters *pars, Grids *grids, int fileid) {
    int retval;
    if (retval = nc_def_dim (fileid, "ri",      2,                &ri)) ERR(retval);
    if (retval = nc_def_dim (fileid, "x",       pars->nx_in,     &x)) ERR(retval);
    if (retval = nc_def_dim (fileid, "y",       pars->ny_in,     &y)) ERR(retval);
    if (retval = nc_def_dim (fileid, "theta",   grids->Nz,       &z)) ERR(retval);  
    if (retval = nc_def_dim (fileid, "kx",      grids->Nakx,     &kx)) ERR(retval);
    if (retval = nc_def_dim (fileid, "ky",      grids->Naky,     &ky)) ERR(retval);
    if (retval = nc_def_dim (fileid, "kz",      grids->Nz,       &kz)) ERR(retval);  
    if (retval = nc_def_dim (fileid, "m",       pars->nm_in,     &m)) ERR(retval);
    if (retval = nc_def_dim (fileid, "l",       pars->nl_in,     &l)) ERR(retval);
    if (retval = nc_def_dim (fileid, "s",       pars->nspec_in,  &species)) ERR(retval);
    if (retval = nc_def_dim (fileid, "time",    NC_UNLIMITED,     &time)) ERR(retval);
  };
  ~NcDims() {};

  int time, species, kx, ky, kz, x, y, z, l, m, ri;
};

class NcGrids {
 public:
  NcGrids(Grids* grids, NcDims* nc_dims, int fileid) {
    int retval;
    // define Grids group in ncdf
    if (retval = nc_def_grp(fileid, "Grids", &grid_id)) ERR(retval);

    if (retval = nc_def_var(grid_id, "time", NC_DOUBLE, 1, &nc_dims->time, &time)) ERR(retval);
    if (retval = nc_def_var(grid_id, "kx", NC_FLOAT, 1, &nc_dims->kx, &kx)) ERR(retval);
    if (retval = nc_def_var(grid_id, "ky", NC_FLOAT, 1, &nc_dims->ky, &ky)) ERR(retval);
    if (retval = nc_def_var(grid_id, "kz", NC_FLOAT, 1, &nc_dims->kz, &kz)) ERR(retval);
    if (retval = nc_def_var(grid_id, "x",  NC_FLOAT, 1, &nc_dims->x, &x))  ERR(retval);  
    if (retval = nc_def_var(grid_id, "y",  NC_FLOAT, 1, &nc_dims->y, &y))  ERR(retval);  
    if (retval = nc_def_var(grid_id, "theta",  NC_FLOAT, 1, &nc_dims->z, &z))  ERR(retval);  
 
    if (retval = nc_put_var(grid_id, kx, grids->kx_outh)) ERR(retval);
    if (retval = nc_put_var(grid_id, ky, grids->ky_h)) ERR(retval);
    if (retval = nc_put_var(grid_id, kz, grids->kz_outh)) ERR(retval);
    if (retval = nc_put_var(grid_id, x, grids->x_h)) ERR(retval);
    if (retval = nc_put_var(grid_id, y, grids->y_h)) ERR(retval);
    if (retval = nc_put_var(grid_id, z, grids->z_h)) ERR(retval);

    if (retval = nc_var_par_access(grid_id, time, NC_COLLECTIVE)) ERR(retval);
  };
  ~NcGrids() {};
  void write_time(double time_val) {
    size_t count = 1;
    int retval;
    if (retval = nc_put_vara(grid_id, time, &time_index, &count, &time_val)) ERR(retval);
    time_index += 1;
  }

  int grid_id; // ncdf id for geo group
  // ncdf ids for grid variables
  int time, kx, ky, kz, x, y, z;

  size_t time_index = 0;
};

class NcGeo {
 public:
  NcGeo(Grids *grids, Geometry *geo, NcDims* nc_dims, int fileid) {
    int retval;
    // define Geometry group in ncdf
    if (retval = nc_def_grp(fileid, "Geometry", &geo_id)) ERR(retval);

    // define Geometry variables
    if (retval = nc_def_var (geo_id, "bmag",     NC_FLOAT, 1, &nc_dims->z, &bmag))     ERR(retval);
    if (retval = nc_def_var (geo_id, "bgrad",    NC_FLOAT, 1, &nc_dims->z, &bgrad))    ERR(retval);
    if (retval = nc_def_var (geo_id, "gbdrift",  NC_FLOAT, 1, &nc_dims->z, &gbdrift))  ERR(retval);
    if (retval = nc_def_var (geo_id, "gbdrift0", NC_FLOAT, 1, &nc_dims->z, &gbdrift0)) ERR(retval);
    if (retval = nc_def_var (geo_id, "cvdrift",  NC_FLOAT, 1, &nc_dims->z, &cvdrift))  ERR(retval);
    if (retval = nc_def_var (geo_id, "cvdrift0", NC_FLOAT, 1, &nc_dims->z, &cvdrift0)) ERR(retval);
    if (retval = nc_def_var (geo_id, "gds2",     NC_FLOAT, 1, &nc_dims->z, &gds2))     ERR(retval);
    if (retval = nc_def_var (geo_id, "gds21",    NC_FLOAT, 1, &nc_dims->z, &gds21))    ERR(retval);
    if (retval = nc_def_var (geo_id, "gds22",    NC_FLOAT, 1, &nc_dims->z, &gds22))    ERR(retval);
    if (retval = nc_def_var (geo_id, "grho",     NC_FLOAT, 1, &nc_dims->z, &grho))     ERR(retval);
    if (retval = nc_def_var (geo_id, "jacobian", NC_FLOAT, 1, &nc_dims->z, &jacobian)) ERR(retval);
    if (retval = nc_def_var (geo_id, "gradpar",  NC_FLOAT, 0, NULL, &gradpar))     ERR(retval);
    if (retval = nc_def_var (geo_id, "nperiod",  NC_INT, 0, NULL, &nperiod))     ERR(retval);
    if (retval = nc_def_var (geo_id, "q",  NC_FLOAT, 0, NULL, &q))     ERR(retval);
    if (retval = nc_def_var (geo_id, "shat",  NC_FLOAT, 0, NULL, &shat))     ERR(retval);
    if (retval = nc_def_var (geo_id, "shift",  NC_FLOAT, 0, NULL, &shift))     ERR(retval);
    if (retval = nc_def_var (geo_id, "rmaj",  NC_FLOAT, 0, NULL, &rmaj))     ERR(retval);
    if (retval = nc_def_var (geo_id, "aminor",  NC_FLOAT, 0, NULL, &aminor))     ERR(retval);
    if (retval = nc_def_var (geo_id, "kxfac",  NC_FLOAT, 0, NULL, &kxfac))     ERR(retval);
    if (retval = nc_def_var (geo_id, "drhodpsi",  NC_FLOAT, 0, NULL, &drhodpsi))     ERR(retval);
    if (retval = nc_def_var (geo_id, "theta_scale",  NC_FLOAT, 0, NULL, &theta_scale))     ERR(retval);
    if (retval = nc_def_var (geo_id, "nfp",  NC_INT, 0, NULL, &nfp))     ERR(retval);
    if (retval = nc_def_var (geo_id, "alpha",  NC_FLOAT, 0, NULL, &alpha))     ERR(retval);
    if (retval = nc_def_var (geo_id, "zeta_center",  NC_FLOAT, 0, NULL, &zeta_center))     ERR(retval);

    // write variables
    if (retval = nc_put_var(geo_id, bmag,     geo->bmag_h))     ERR(retval);
    if (retval = nc_put_var(geo_id, bgrad,    geo->bgrad_h))    ERR(retval);
    if (retval = nc_put_var(geo_id, gbdrift,  geo->gbdrift_h))  ERR(retval);
    if (retval = nc_put_var(geo_id, gbdrift0, geo->gbdrift0_h)) ERR(retval);
    if (retval = nc_put_var(geo_id, cvdrift,  geo->cvdrift_h))  ERR(retval);
    if (retval = nc_put_var(geo_id, cvdrift0, geo->cvdrift0_h)) ERR(retval);
    if (retval = nc_put_var(geo_id, gds2,     geo->gds2_h))     ERR(retval);
    if (retval = nc_put_var(geo_id, gds21,    geo->gds21_h))    ERR(retval);  
    if (retval = nc_put_var(geo_id, gds22,    geo->gds22_h))    ERR(retval);
    if (retval = nc_put_var(geo_id, grho,     geo->grho_h))     ERR(retval);
    if (retval = nc_put_var(geo_id, jacobian, geo->jacobian_h)) ERR(retval);
    if (retval = nc_put_var(geo_id, nperiod, &geo->nperiod))   ERR(retval);
    if (retval = nc_put_var(geo_id, gradpar, &geo->gradpar))   ERR(retval);
    if (retval = nc_put_var(geo_id, q, &geo->qsf))   ERR(retval);
    if (retval = nc_put_var(geo_id, shat, &geo->shat))   ERR(retval);
    if (retval = nc_put_var(geo_id, shift, &geo->shift))   ERR(retval);
    if (retval = nc_put_var(geo_id, rmaj, &geo->rmaj))   ERR(retval);
    if (retval = nc_put_var(geo_id, aminor, &geo->aminor))   ERR(retval);
    if (retval = nc_put_var(geo_id, kxfac, &geo->kxfac))   ERR(retval);
    if (retval = nc_put_var(geo_id, drhodpsi, &geo->drhodpsi))   ERR(retval);
    if (retval = nc_put_var(geo_id, theta_scale, &geo->theta_scale))   ERR(retval);
    if (retval = nc_put_var(geo_id, nfp, &geo->nfp))   ERR(retval);
    if (retval = nc_put_var(geo_id, alpha, &geo->alpha))   ERR(retval);
    if (retval = nc_put_var(geo_id, zeta_center, &geo->zeta_center))   ERR(retval);
  }
  ~NcGeo() {};
  int geo_id; // ncdf id for geo group
  // ncdf ids for geo variables
  int bmag, bgrad, gbdrift, gbdrift0, cvdrift, cvdrift0;
  int gds2, gds21, gds22, grho, jacobian, gradpar;
  int q, shat, shift, kxfac, rmaj, aminor, drhodpsi, theta_scale, nperiod, nfp;
  int alpha, zeta_center;
};

class NcDiagnostics {
 public:
  NcDiagnostics(int fileid) {
    int retval;
    // create ncdf group id for Diagnostics
    if (retval = nc_def_grp(fileid, "Diagnostics", &diagnostics_id)) ERR(retval);
  };
  ~NcDiagnostics() {};
  int diagnostics_id; // ncdf id for diagnostics group
};
