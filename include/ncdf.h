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
#include "unistd.h"
#include "version.h"

using namespace std;

class NcInputs;
class NcDims;
class NcGrids;
class NcGeo;
class NcSpecies;
class NcDiagnostics;

class NetCDF {
 public:
  NetCDF(Parameters* pars, string suffix = ".out.nc");
  ~NetCDF();

  int fileid;
  NcDims *nc_dims;
  NcGrids *nc_grids;
  NcGeo *nc_geo;
  NcDiagnostics *nc_diagnostics;
  NcInputs *nc_inputs;
  NcSpecies *nc_species;
  void setup(Parameters* pars, Grids* grids, Geometry* geo);
  void sync();
 private:
  void close_nc_file();
  bool append_;
};

class NcInputs {
 public:
  NcInputs(int fileid, bool append) :
    append_(append)
  {
    int retval;
    if (!append_) {
      if (retval = nc_def_grp(fileid, "Inputs", &inputs_id)) ERR(retval);

      int ivar;
      if (retval = nc_def_var (inputs_id, "code_info",             NC_INT,   0, NULL, &ivar)) ERR(retval);
      std::string hash(build_git_sha);                       
      if (retval = nc_put_att_text (inputs_id, ivar, "Hash",      hash.size(), hash.c_str() ) ) ERR(retval);
      std::string compiled(build_git_time);                  
      if (retval = nc_put_att_text (inputs_id, ivar, "BuildDate", compiled.size(), compiled.c_str() ) ) ERR(retval);
      std::string builder(build_user);                       
      if (retval = nc_put_att_text (inputs_id, ivar, "BuildUser", builder.size(), builder.c_str() ) ) ERR(retval);
      std::string build_host(build_hostname);                
      if (retval = nc_put_att_text (inputs_id, ivar, "BuildHost", build_host.size(), build_host.c_str() ) ) ERR(retval);
    }
  };

  ~NcInputs() {};

  int put_group(const char groupname[]) {
    if (append_) return -1; // don't modify inputs when appending

    int retval, group_id;
    if (retval = nc_def_grp(inputs_id, groupname, &group_id)) ERR(retval);
    return group_id;
  };

  template<typename T>
  void put_var(int ncid, const char varname[], T val, bool debug=false) {
    if (append_) return; // don't modify inputs when appending

    int idum, retval;
    if constexpr(std::is_same_v<T, bool>) {
      int b = val ? 1 : 0;
      if (debug) printf("%s = %d \n", varname, b);
      if (retval = nc_def_var(ncid, varname, NC_INT, 0, NULL, &idum)) ERR(retval);
      if (retval = nc_put_var(ncid, idum, &b)) ERR(retval);
    } else if constexpr(std::is_same_v<T, int>) {
      if (debug) printf("%s = %d \n", varname, val);
      if (retval = nc_def_var(ncid, varname, NC_INT, 0, NULL, &idum)) ERR(retval);
      if (retval = nc_put_var(ncid, idum, &val)) ERR(retval);
    } else if constexpr(std::is_same_v<T, unsigned int>) {
      if (debug) printf("%s = %u \n", varname, val);
      if (retval = nc_def_var(ncid, varname, NC_INT, 0, NULL, &idum)) ERR(retval);
      int signed_val = static_cast<int>(val);
      if (retval = nc_put_var(ncid, idum, &signed_val)) ERR(retval);
    } else if constexpr(std::is_same_v<T, float>) {
      if (debug) printf("%s = %f \n", varname, val);
      if (retval = nc_def_var(ncid, varname, NC_FLOAT, 0, NULL, &idum)) ERR(retval);
      if (retval = nc_put_var(ncid, idum, &val)) ERR(retval);
    } else {
      //static_assert(false, "put_var only supports bool, int, unsigned int, and float types");
    }
  };

  template<typename T>
  void mod_var(int ncid, const char varname[], T val, bool debug=false) {
    if (append_) return; // don't modify inputs when appending
    int idum, retval;
    if (debug) printf("modifying %s = %d \n", varname, val);
    if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
    if (retval = nc_put_var(ncid, idum, &val)) ERR(retval);
  };

  private:
   const bool append_;
   int inputs_id;
  
};

class NcDims {
 public:
  NcDims(Parameters *pars, Grids *grids, int fileid, bool append) {
    int retval;
    if (append) {
      if (retval = nc_inq_dimid (fileid, "ri",      &ri)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "x",       &x)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "y",       &y)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "theta",   &z)) ERR(retval);  
      if (retval = nc_inq_dimid (fileid, "kx",      &kx)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "ky",      &ky)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "kz",      &kz)) ERR(retval);  
      if (retval = nc_inq_dimid (fileid, "m",       &m)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "l",       &l)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "s",       &species)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "time",    &time)) ERR(retval);
    } else {
      if (retval = nc_def_dim (fileid, "ri",      2,               &ri)) ERR(retval);
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
    }
  };
  ~NcDims() {};

  int time, species, kx, ky, kz, x, y, z, l, m, ri;
};

class NcGrids {
 public:
  NcGrids(Grids* grids, NcDims* nc_dims, int fileid, bool append) {
    int retval;
    if (append) {
      if (retval = nc_inq_grp_ncid(fileid, "Grids", &grid_id)) ERR(retval);

      if (retval = nc_inq_varid(grid_id, "time", &time)) ERR(retval);
      if (retval = nc_inq_varid(grid_id, "kx", &kx)) ERR(retval);
      if (retval = nc_inq_varid(grid_id, "ky", &ky)) ERR(retval);
      if (retval = nc_inq_varid(grid_id, "kz", &kz)) ERR(retval);
      if (retval = nc_inq_varid(grid_id, "x", &x))  ERR(retval);  
      if (retval = nc_inq_varid(grid_id, "y", &y))  ERR(retval);  
      if (retval = nc_inq_varid(grid_id, "theta", &z))  ERR(retval);  

      if (retval = nc_var_par_access(grid_id, time, NC_COLLECTIVE)) ERR(retval);

      // read time_index
      if (retval = nc_inq_dimlen(fileid, nc_dims->time, &time_index)) ERR(retval);
    } else { 
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
    }
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

class NcSpecies {
 public:
  NcSpecies(Parameters *pars, NcDims* nc_dims, int fileid) {
    int retval;
    // define Geometry group in ncdf
    if (retval = nc_def_grp(fileid, "Species", &spec_id)) ERR(retval);

    if (retval = nc_def_var (spec_id, "species_type", NC_INT,   1, &nc_dims->species, &species_type_id)) ERR(retval);
    if (retval = nc_def_var (spec_id, "z",            NC_FLOAT, 1, &nc_dims->species, &z_id)) ERR(retval);
    if (retval = nc_def_var (spec_id, "m",            NC_FLOAT, 1, &nc_dims->species, &m_id)) ERR(retval);
    if (retval = nc_def_var (spec_id, "n0",           NC_FLOAT, 1, &nc_dims->species, &n0_id)) ERR(retval);
    if (retval = nc_def_var (spec_id, "n0_prime",     NC_FLOAT, 1, &nc_dims->species, &n0_prime_id)) ERR(retval);
    if (retval = nc_def_var (spec_id, "u0_prime",     NC_FLOAT, 1, &nc_dims->species, &u0_prime_id)) ERR(retval);
    if (retval = nc_def_var (spec_id, "T0",           NC_FLOAT, 1, &nc_dims->species, &T0_id)) ERR(retval);
    if (retval = nc_def_var (spec_id, "T0_prime",     NC_FLOAT, 1, &nc_dims->species, &T0_prime_id)) ERR(retval);
    if (retval = nc_def_var (spec_id, "nu",           NC_FLOAT, 1, &nc_dims->species, &nu_id)) ERR(retval);

    is_start[0] = 0;
    int nspec = pars->nspec_in;
    is_count[0] = nspec;
  
    // this stuff should all be in species itself!
    // reason for all this is basically legacy + cuda does not support <vector>
    
    std::vector <float> zs, ms, ns, Ts, Tps, nps, nus;
    std::vector <int> types;
    
    for (int is=0; is<nspec; is++) {
      zs.push_back(pars->species_h[is].z);
      ms.push_back(pars->species_h[is].mass);
      ns.push_back(pars->species_h[is].dens);
      Ts.push_back(pars->species_h[is].temp);
      Tps.push_back(pars->species_h[is].tprim);
      nps.push_back(pars->species_h[is].fprim);
      nus.push_back(pars->species_h[is].nu_ss);
      types.push_back(pars->species_h[is].type);
    }
    float *z = &zs[0];
    float *m = &ms[0];
    float *n0 = &ns[0];
    float *T0 = &Ts[0];
    float *Tp = &Tps[0];
    float *np = &nps[0];
    float *nu = &nus[0];
    int *st = &types[0];
    
    if (retval = nc_put_vara (spec_id, z_id, is_start, is_count, z))  ERR(retval);
    if (retval = nc_put_vara (spec_id, m_id, is_start, is_count, m))  ERR(retval);
    if (retval = nc_put_vara (spec_id, n0_id, is_start, is_count, n0))  ERR(retval);
    if (retval = nc_put_vara (spec_id, n0_prime_id, is_start, is_count, np))  ERR(retval);
    if (retval = nc_put_vara (spec_id, T0_id, is_start, is_count, T0))  ERR(retval);
    if (retval = nc_put_vara (spec_id, T0_prime_id, is_start, is_count, Tp))  ERR(retval);
    if (retval = nc_put_vara (spec_id, nu_id, is_start, is_count, nu))  ERR(retval);
    if (retval = nc_put_vara (spec_id, species_type_id, is_start, is_count, st))  ERR(retval);
  };

  ~NcSpecies() {};
  int spec_id;
  int species_type_id, z_id, m_id, n0_id, n0_prime_id, u0_prime_id, T0_id, T0_prime_id, nu_id;
  size_t is_start[1], is_count[1]; 
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
  NcDiagnostics(int fileid, bool append) {
    int retval;
    if (append) {
      if (retval = nc_inq_grp_ncid(fileid, "Diagnostics", &diagnostics_id)) ERR(retval);
    } else {
      // create ncdf group id for Diagnostics
      if (retval = nc_def_grp(fileid, "Diagnostics", &diagnostics_id)) ERR(retval);
    }
  };
  ~NcDiagnostics() {};
  int diagnostics_id; // ncdf id for diagnostics group
};
