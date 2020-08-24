#include "netcdf.h"
#include "ncdf.h"

NetCDF_ids::NetCDF_ids(Grids* grids, Parameters* pars, Geometry* geo) :
  grids_(grids), pars_(pars), geo_(geo)
{

  char strb[263];
  strcpy(strb, pars_->run_name); 
  strcat(strb, ".nc");

  //  if (pars_->write_omega) {printf("In ncdf. pars_->write_omega is true. \n");}

  int retval, idum;
  
  if (retval = nc_open(strb, NC_WRITE, &file)) ERR(retval);
  if (retval = nc_redef(file));
  
  int ri;
  // Get handles for the dimensions
  if (retval = nc_inq_dimid(file, "ri", &ri))  ERR(retval);
  
  if (retval = nc_def_dim(file, "ky",        grids_->Naky,  &ky_dim))    ERR(retval);
  if (retval = nc_def_dim(file, "kx",        grids_->Nakx,  &kx_dim))    ERR(retval);
  if (retval = nc_def_dim(file, "z",         grids_->Nz,    &nz))        ERR(retval);  

  if (retval = nc_inq_dimid(file, "m",       &m_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "l",       &l_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "s",       &s_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "time",    &time_dim)) ERR(retval);

  //  if (retval = nc_def_var(file, "m",          NC_INT,    1, &
  if (retval = nc_def_var(file, "time",       NC_DOUBLE, 1, &time_dim, &time))    ERR(retval);

  // BD need to decide whether to keep this around
  //  if (retval = nc_def_var(file, "linear",       NC_INT,   0, 0, &linear))          ERR(retval);

  if (retval = nc_def_var(file, "periodic",       NC_INT, 0, 0, &periodic))        ERR(retval);
  if (retval = nc_def_var(file, "local_limit",    NC_INT, 0, 0, &local_limit))     ERR(retval);

  v_ky[0] = ky_dim;
  if (retval = nc_def_var(file, "ky",       NC_FLOAT, 1, v_ky, &ky))              ERR(retval);

  v_kx[0] = kx_dim;
  if (retval = nc_def_var(file, "kx",       NC_FLOAT, 1, v_kx, &kx))              ERR(retval);

  geo_v_theta[0] = nz;
  if (retval = nc_def_var(file, "theta",    NC_FLOAT, 1, geo_v_theta, &theta))    ERR(retval);
  if (retval = nc_def_var(file, "bmag",     NC_FLOAT, 1, geo_v_theta, &bmag))     ERR(retval);
  if (retval = nc_def_var(file, "bgrad",    NC_FLOAT, 1, geo_v_theta, &bgrad))    ERR(retval);
  if (retval = nc_def_var(file, "gbdrift",  NC_FLOAT, 1, geo_v_theta, &gbdrift))  ERR(retval);
  if (retval = nc_def_var(file, "gbdrift0", NC_FLOAT, 1, geo_v_theta, &gbdrift0)) ERR(retval);
  if (retval = nc_def_var(file, "cvdrift",  NC_FLOAT, 1, geo_v_theta, &cvdrift))  ERR(retval);
  if (retval = nc_def_var(file, "cvdrift0", NC_FLOAT, 1, geo_v_theta, &cvdrift0)) ERR(retval);
  if (retval = nc_def_var(file, "gds2",     NC_FLOAT, 1, geo_v_theta, &gds2))     ERR(retval);
  if (retval = nc_def_var(file, "gds21",    NC_FLOAT, 1, geo_v_theta, &gds21))    ERR(retval);
  if (retval = nc_def_var(file, "gds22",    NC_FLOAT, 1, geo_v_theta, &gds22))    ERR(retval);
  if (retval = nc_def_var(file, "grho",     NC_FLOAT, 1, geo_v_theta, &grho))     ERR(retval);
  if (retval = nc_def_var(file, "jacobian", NC_FLOAT, 1, geo_v_theta, &jacobian)) ERR(retval);
  
  
  zkxky[0] = nz;
  zkxky[1] = kx_dim; 
  zkxky[2] = ky_dim;
  
  if (retval = nc_def_var(file, "theta_x",  NC_FLOAT, 3, zkxky, &theta_x))  ERR(retval);

  // define the dependent variables now, starting with their dimensions

  //  final_field[0] = nx;
  //  final_field[1] = ny;

  if (pars_->write_moms || pars_->write_phi_kpar) {
    moments_out[0] = nz; 
    moments_out[1] = kx_dim; 
    moments_out[2] = ky_dim;
    moments_out[3] = ri;
  }

  if (pars_->write_moms) {
    if (grids_->Nm>1) {
      if (retval = nc_def_var(file, "upar",    NC_FLOAT, 4, moments_out, &upar))    ERR(retval);
    }
    if (retval = nc_def_var(file, "phi",      NC_FLOAT, 4, moments_out, &phi))      ERR(retval);
    if (retval = nc_def_var(file, "phi0",     NC_FLOAT, 4, moments_out, &phi0))     ERR(retval);
    if (retval = nc_def_var(file, "density",  NC_FLOAT, 4, moments_out, &density))  ERR(retval);
    if (retval = nc_def_var(file, "density0", NC_FLOAT, 4, moments_out, &density0)) ERR(retval);
  }

  if (pars_->write_phi_kpar) {
    if (retval = nc_def_var(file, "density_kpar", NC_FLOAT,
			    4, moments_out, &density_kpar)) ERR(retval);
    if (retval = nc_def_var(file, "phi_kpar",    NC_FLOAT,
			    4, moments_out, &phi_kpar))     ERR(retval);
  }

  if (pars_->write_omega) {
    omega_v_time[0] = time_dim;
    omega_v_time[1] = kx_dim;  
    omega_v_time[2] = ky_dim; 
    omega_v_time[3] = ri;
    
    if (retval = nc_def_var(file, "omega_v_time", NC_FLOAT, 4,
			    omega_v_time, &omega_t)) ERR(retval);

    omt_start[0] = 1;
    omt_start[1] = 0;
    omt_start[2] = 0;
    omt_start[3] = 0; 
    
    omt_count[0]= 1; 
    omt_count[1] = grids_->Nakx;
    omt_count[2] = grids_->Naky;
    omt_count[3] = 2; 
  }

  if (pars_->write_rh) {
    complex_v_time[0] = time_dim;
    complex_v_time[1] = ri;
    
    if (retval = nc_def_var(file, "phi_rh", NC_FLOAT, 2, complex_v_time, &phi_rh)) ERR(retval);

    rh_start[0] = 0;
    rh_start[1] = 0;

    rh_count[0] = 1;
    rh_count[1] = 2;
  }

  if (pars_->write_pzt) {
    pzt_v_time[0] = time_dim;
    if (retval = nc_def_var(file, "prim", NC_FLOAT, 1, pzt_v_time, &prim)) ERR(retval);
    if (retval = nc_def_var(file, "tert", NC_FLOAT, 1, pzt_v_time, &tert)) ERR(retval);
    if (retval = nc_def_var(file, "sec",  NC_FLOAT, 1, pzt_v_time, &sec))  ERR(retval);
  }
  
  if (pars_->write_fluxes) {
    scalar_v_time[0] = time_dim;
  
    // ultimately this should have a species index so for now we will need to copy qflux down to a scalar
    if (retval = nc_def_var(file, "qflux", NC_FLOAT, 1, scalar_v_time, &qflux)) ERR(retval);
  }

  if (pars_->write_lh_spectrum) {
    g_v_lm[0] = m_dim; 
    g_v_lm[1] = l_dim;
  
    if (retval = nc_def_var(file, "lh_spec", NC_FLOAT, 2, g_v_lm, &lhspec))  ERR(retval);
    lh_start[0] = 0;
    lh_start[1] = 0;

    lh_count[0] = grids_->Nm;
    lh_count[1] = grids_->Nl;

    if (pars_->write_spec_v_time) {
      g_v_lmt[0] = time_dim;
      g_v_lmt[1] = m_dim;
      g_v_lmt[2] = l_dim;

      if (retval = nc_def_var(file, "lh_spec_t", NC_FLOAT, 3, g_v_lmt, &lhspec_t))  ERR(retval);
      lht_start[0] = 0;
      lht_start[1] = 0;
      lht_start[2] = 0;

      lht_count[0] = 1;
      lht_count[1] = grids_->Nm;
      lht_count[2] = grids_->Nl;
    }
  }
    
  if (pars_->write_l_spectrum) {
    g_v_l[0] = l_dim;
    if (retval = nc_def_var(file, "l_spec", NC_FLOAT, 1, g_v_l, &lspec))  ERR(retval);
    l_start[0] = 0;
    l_count[0] = grids_->Nl;    

    if (pars_->write_spec_v_time) {
      g_v_lt[0] = time_dim;
      g_v_lt[1] = l_dim;

      if (retval = nc_def_var(file, "l_spec_t", NC_FLOAT, 2, g_v_lt, &lspec_t))  ERR(retval);
      lt_start[0] = 0;
      lt_start[1] = 0;

      lt_count[0] = 1;    
      lt_count[1] = grids_->Nl;          
    }
  }
  
  
  if (pars_->write_h_spectrum) {
    g_v_m[0] = m_dim;
    if (retval = nc_def_var(file, "h_spec", NC_FLOAT, 1, g_v_m, &hspec))  ERR(retval);
    if (retval = nc_def_var(file, "hmom_real", NC_FLOAT, 1, g_v_m, &hmom_real)) ERR(retval);
    if (retval = nc_def_var(file, "hmom_imag", NC_FLOAT, 1, g_v_m, &hmom_imag)) ERR(retval);

    m_start[0] = 0; 
    m_count[0] = grids_->Nm;

   if (pars_->write_spec_v_time) {
      g_v_mt[0] = time_dim;
      g_v_mt[1] = m_dim;
      if (retval = nc_def_var(file, "h_spec_t", NC_FLOAT, 2, g_v_mt, &hspec_t))  ERR(retval);
      if (retval = nc_def_var(file, "hmom_real_t", NC_FLOAT, 2, g_v_mt, &hmom_real_t))  ERR(retval);
      if (retval = nc_def_var(file, "hmom_imag_t", NC_FLOAT, 2, g_v_mt, &hmom_imag_t)) ERR(retval);
      mt_start[0] = 0;
      mt_start[1] = 0;

      mt_count[0] = 1;    
      mt_count[1] = grids_->Nm;          
    }
  }

  pzt_start[0] = 0;
  pzt_count[0] = 1;
  
  time_start[0] = 0;
  time_count[0] = 1;

  flux_start[0] = 0;
  flux_count[0] = 1;

  mom_start[0] = 0;
  mom_start[1] = 0;
  mom_start[2] = 0;
  mom_start[3] = 0; 
  
  mom_count[0] = grids_->Nz;
  mom_count[1] = grids_->Nakx;
  mom_count[2] = grids_->Naky;
  mom_count[3] = 2; 

  DEBUGPRINT("ncdf:  ending definition mode for NetCDF \n");
  
  if (retval = nc_enddef(file)) ERR(retval);

  ///////////////////////////////////
  /// write parameters of this run //
  ///////////////////////////////////
  ky_start[0] = 0;
  ky_count[0] = grids_->Naky;

  if (retval = nc_put_vara(file, ky, ky_start, ky_count, grids_->ky_h))         ERR(retval);

  kx_start[0] = 0;
  kx_count[0] = grids_->Nakx;

  if (retval = nc_put_vara(file, kx, kx_start, kx_count, grids_->kx_h))         ERR(retval);

  idum = pars_->boundary_option_periodic ? 1 : 0;
  if (retval = nc_put_var(file, periodic,      &idum))                   ERR(retval);

  idum = pars_->local_limit ? 1 : 0;
  if (retval = nc_put_var(file, local_limit,   &pars_->local_limit))     ERR(retval);

  geo_start[0] = 0;
  geo_count[0] = grids_->Nz;
  
  if (retval = nc_put_vara(file, theta,    geo_start, geo_count, geo_->z_h))         ERR(retval);

  if (geo_->shat != 0.) {
    
    int Nx = grids_->Nx;
    int Ny = grids_->Ny;
    int Nz = grids_->Nz;
    
    int Naky = grids_->Naky;
    
    zkxky_count[0] = Nz;
    zkxky_count[1] = 1;
    zkxky_count[2] = 1;
    
    size_t size = sizeof(float)*Nz;
    cudaMallocHost((void**) &theta_extended, size);
    float th0;
    for (int i=0; i<(Nx-1)/3+1; i++) {
      for (int j=0; j<(Ny-1)/3+1; j++) {
	if (j==0) {th0 = 0.;} else {th0 = grids_->kx_h[i]/(grids_->ky_h[j]*pars_->shat);}
	for (int k=0; k<Nz; k++) {
	  theta_extended[k] = geo_->z_h[k] - th0;
	}
	zkxky_start[0] = 0;
	zkxky_start[1] = i;
	zkxky_start[2] = j;
	if (retval = nc_put_vara(file, theta_x, zkxky_start, zkxky_count, theta_extended)) ERR(retval);
      }
    }
    for(int i=2*Nx/3+1; i<Nx; i++) {
      for(int j=0; j<Naky; j++) {
	if (j==0) {th0 = 0.;} else {th0 = grids_->kx_h[i]/(grids_->ky_h[j]*pars_->shat);}
	for (int k=0; k<Nz; k++) {
	  theta_extended[k] = geo_->z_h[k] - th0;
	}
	zkxky_start[0] = 0;
	zkxky_start[1] = i-2*Nx/3+(Nx-1)/3;
	zkxky_start[2] = j;
	if (retval = nc_put_vara(file, theta_x, zkxky_start, zkxky_count, theta_extended)) ERR(retval);
      }
    }
    cudaFreeHost(theta_extended);
  }

  if (retval = nc_put_vara(file, theta,    geo_start, geo_count, geo_->z_h))         ERR(retval);
  if (retval = nc_put_vara(file, bmag,     geo_start, geo_count, geo_->bmag_h))      ERR(retval);
  if (retval = nc_put_vara(file, bgrad,    geo_start, geo_count, geo_->bgrad_h))     ERR(retval);
  if (retval = nc_put_vara(file, gbdrift,  geo_start, geo_count, geo_->gbdrift_h))   ERR(retval);
  if (retval = nc_put_vara(file, gbdrift0, geo_start, geo_count, geo_->gbdrift0_h))  ERR(retval);
  if (retval = nc_put_vara(file, cvdrift,  geo_start, geo_count, geo_->cvdrift_h))   ERR(retval);
  if (retval = nc_put_vara(file, cvdrift0, geo_start, geo_count, geo_->cvdrift0_h))  ERR(retval);
  if (retval = nc_put_vara(file, gds2,     geo_start, geo_count, geo_->gds2_h))      ERR(retval);
  if (retval = nc_put_vara(file, gds21,    geo_start, geo_count, geo_->gds21_h))     ERR(retval);  
  if (retval = nc_put_vara(file, gds22,    geo_start, geo_count, geo_->gds22_h))     ERR(retval);
  if (retval = nc_put_vara(file, grho,     geo_start, geo_count, geo_->grho_h))      ERR(retval);
  if (retval = nc_put_vara(file, jacobian, geo_start, geo_count, geo_->jacobian_h))  ERR(retval);
  
}

NetCDF_ids::~NetCDF_ids() {
}

void NetCDF_ids::close_nc_file() {
  int retval;
  if (retval = nc_close(file)) ERR(retval);
}


