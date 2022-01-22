#include "netcdf.h"
#include "ncdf.h"
#define GFLA    <<< dgfla, dbfla >>> 
#define loop_R  <<< dGr,  dBr  >>>
#define loop_xy <<< dgxy, dbxy >>> 
#define loop_x  <<< dgx,  dbx  >>> 
#define loop_y  <<< dgp,  dbp  >>> 
#define Gmom    <<< dgall, dball >>> 

NetCDF_ids::NetCDF_ids(Grids* grids, Parameters* pars, Geometry* geo) :
  grids_(grids), pars_(pars), geo_(geo),
  red(nullptr), pot(nullptr), ph2(nullptr), all_red(nullptr), grad_phi(nullptr), grad_perp(nullptr)
{

  primary     = nullptr;  secondary   = nullptr;  tertiary    = nullptr;   amom = nullptr;
  df          = nullptr;  favg        = nullptr;

  if (pars_->diagnosing_spectra || pars_->diagnosing_kzspec) {
    float dum = 1.0;
    red = new          All_Reduce(grids_, pars_->wspectra); CUDA_DEBUG("Reductions: %s \n"); // G**2
    pot = new Grid_Species_Reduce(grids_, pars_->aspectra); CUDA_DEBUG("Reductions: %s \n"); // (1-G0) Phi**2 keeping track of species
    ph2 = new         Grid_Reduce(grids_, pars_->aspectra); CUDA_DEBUG("Reductions: %s \n"); // Phi**2
  }

  int nS  = grids_->Nspecies;
  int nM  = grids_->Nm;
  int nL  = grids_->Nl;
  int nY  = grids_->Nyc;
  int nYk = grids_->Naky;
  int nX  = grids_->Nx;
  int nXk = grids_->Nakx;
  int nZ  = grids_->Nz;
  int nR  = nX  * nY  * nZ;
  int nK  = nXk * nYk * nZ;
  int nG  = nR * grids_->Nmoms * nS;

  theta_extended = nullptr;
  char strb[263];
  strcpy(strb, pars_->run_name); 
  strcat(strb, ".nc");

  int retval, idum;

  // Loop over full real-space grid
  int nt1 = min(grids_->NxNyNz, 1024);
  int nb1 = 1 + (grids_->NxNyNz-1)/nt1;

  dBr = dim3(nt1, 1, 1);
  dGr = dim3(nb1, 1, 1);
  
  // Loop over x-space grid
  nt1 = min(grids_->Nx, 512);
  nb1 = 1 + (grids_->Nx-1)/nt1;

  dbx = dim3(nt1, 1, 1);
  dgx = dim3(nb1, 1, 1);

  // Loop over y-space grid
  nt1 = min(grids_->Ny, 512);
  nb1 = 1 + (grids_->Ny-1)/nt1;

  dbp = dim3(nt1, 1, 1);
  dgp = dim3(nb1, 1, 1);
  
  // Double loop, over y-space and x-space grids
  nt1 = min(32, grids_->Ny);
  nb1 = 1 + (grids_->Ny-1)/nt1;

  int nt2 = min(32, grids_->Nx);
  int nb2 = 1 + (grids_->Nx-1)/nt2;
  
  dbxy = dim3(nt1, nt2, 1);
  dgxy = dim3(nb1, nb2, 1);

  // Single loop, over Nx*Nyc elements
  nt1 = min(128, grids_->Nx);
  nb1 = 1 + (grids_->Nx*grids_->Nyc-1)/nt1;

  dbfla = dim3(nt1, 1, 1);
  dgfla = dim3(nb1, 1, 1);

  // Triple loop, native elements
  int nt3, nb3;
  nt1 = min(16, grids_->Nyc);  nb1 = 1 + (grids_->Nyc-1)/nt1;
  nt2 = min(16, grids_->Nx);   nb2 = 1 + (grids_->Nx -1)/nt2;
  nt3 = min(4,  grids_->Nz);   nb3 = 1 + (grids_->Nz -1)/nt3;                   

  dball = dim3(nt1, nt2, nt3);
  dgall = dim3(nb1, nb2, nb3);
  
  if (pars_->write_kmom || pars_->write_xymom || pars_->write_avgz) {
    int nbatch = grids_->Nz;
    grad_phi = new GradPerp(grids_, nbatch, grids_->NxNycNz);

    cudaMalloc (&df,     sizeof(cuComplex)*grids_->NxNycNz);
    cudaMalloc (&favg,   sizeof(cuComplex)*grids_->Nx);
    cudaMalloc (&amom,   sizeof(cuComplex)*grids_->NxNycNz);
  } 
  
  if (pars_->ResWrite) {
    r_file = pars_->ncresid;
    if (retval = nc_redef(r_file));
    if (retval = nc_inq_dimid(r_file, "r",    &res_dim))   ERR(retval);
    if (retval = nc_inq_dimid(r_file, "time", &rtime_dim)) ERR(retval);

    //    v_ky[0] = res_dim;
    //    if (retval = nc_def_var(r_file, "r",  NC_INT, 1, v_ky, &state)) ERR(retval);
  }
  
  if (pars_->write_xymom) {
    z_file = pars_->nczid;
    if (retval = nc_redef(z_file)); 
    if (retval = nc_inq_dimid(z_file, "x",    &zx_dim))    ERR(retval);
    if (retval = nc_inq_dimid(z_file, "y",    &zy_dim))    ERR(retval);
    if (retval = nc_inq_dimid(z_file, "time", &ztime_dim)) ERR(retval);
    
    v_ky[0] = zy_dim;
    if (retval = nc_def_var(z_file, "y",  NC_FLOAT, 1, v_ky, &zy)) ERR(retval);  
    
    v_kx[0] = zx_dim;
    if (retval = nc_def_var(z_file, "x",  NC_FLOAT, 1, v_kx, &zx)) ERR(retval);  
  }
    
  file = pars_->ncid;
  if (retval = nc_redef(file));
  
  int ri;
  // Get handles for the dimensions
  if (retval = nc_inq_dimid(file, "ri", &ri))  ERR(retval);
  
  if (retval = nc_def_dim(file, "kz",        grids_->Nz,    &nkz))       ERR(retval);  
  if (retval = nc_def_dim(file, "ky",        grids_->Naky,  &ky_dim))    ERR(retval);
  if (retval = nc_def_dim(file, "kx",        grids_->Nakx,  &kx_dim))    ERR(retval);
  if (retval = nc_def_dim(file, "theta",     grids_->Nz,    &nz))        ERR(retval);  
  
  if (retval = nc_inq_dimid(file, "x",       &x_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "y",       &y_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "m",       &m_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "l",       &l_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "s",       &s_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "time",    &time_dim)) ERR(retval);

  if (retval = nc_def_var(file, "periodic",       NC_INT, 0, 0, &periodic))        ERR(retval);
  if (retval = nc_def_var(file, "local_limit",    NC_INT, 0, 0, &local_limit))     ERR(retval);

  v_ky[0] = ky_dim;
  if (retval = nc_def_var(file, "ky",       NC_FLOAT, 1, v_ky, &ky))              ERR(retval);

  v_kx[0] = kx_dim;
  if (retval = nc_def_var(file, "kx",       NC_FLOAT, 1, v_kx, &kx))              ERR(retval);

  v_kz[0] = nkz;
  if (retval = nc_def_var(file, "kz",       NC_FLOAT, 1, v_kz, &kz))              ERR(retval);

  v_ky[0] = y_dim;
  if (retval = nc_def_var(file, "y",        NC_FLOAT, 1, v_ky, &y))               ERR(retval);  

  v_kx[0] = x_dim;
  if (retval = nc_def_var(file, "x",        NC_FLOAT, 1, v_kx, &x))               ERR(retval);  

  //  v_z[0] = nz;
  //  if (retval = nc_def_var(file, "z",        NC_FLOAT, 1, v_z, &z_h))              ERR(retval);
  
  // z_h needs to be defined.
  // Z0 would typically be q R
  // and then z_h would run from - (2 pi q R)/2 : + (2 pi q R)/2
  // but there are complications to get right:
  // normalization of R?
  // Allow for Z0 to be specified directly
  // Allow nperiod > 1 

  int nc_sp;
  if (retval = nc_inq_grp_ncid(file, "Spectra", &nc_sp)) ERR(retval);

  int nc_flux;
  if (retval = nc_inq_grp_ncid(file, "Fluxes", &nc_flux)) ERR(retval);

  int nc_special;
  if (retval = nc_inq_grp_ncid(file, "Special", &nc_special)) ERR(retval);

  int nc_zonal;
  if (retval = nc_inq_grp_ncid(file, "Zonal_x", &nc_zonal)) ERR(retval);

  int nc_geo;
  if (retval = nc_inq_grp_ncid(file, "Geometry", &nc_geo)) ERR(retval);

  int ivar;
  if (geo_) {
    geo_v_theta[0] = nz; 
    if (retval = nc_def_var (file,   "theta",    NC_FLOAT, 1, geo_v_theta, &theta))    ERR(retval);
    if (retval = nc_def_var (nc_geo, "bmag",     NC_FLOAT, 1, geo_v_theta, &bmag))     ERR(retval);
    if (retval = nc_def_var (nc_geo, "bgrad",    NC_FLOAT, 1, geo_v_theta, &bgrad))    ERR(retval);
    if (retval = nc_def_var (nc_geo, "gbdrift",  NC_FLOAT, 1, geo_v_theta, &gbdrift))  ERR(retval);
    if (retval = nc_def_var (nc_geo, "gbdrift0", NC_FLOAT, 1, geo_v_theta, &gbdrift0)) ERR(retval);
    if (retval = nc_def_var (nc_geo, "cvdrift",  NC_FLOAT, 1, geo_v_theta, &cvdrift))  ERR(retval);
    if (retval = nc_def_var (nc_geo, "cvdrift0", NC_FLOAT, 1, geo_v_theta, &cvdrift0)) ERR(retval);
    if (retval = nc_def_var (nc_geo, "gds2",     NC_FLOAT, 1, geo_v_theta, &gds2))     ERR(retval);
    if (retval = nc_def_var (nc_geo, "gds21",    NC_FLOAT, 1, geo_v_theta, &gds21))    ERR(retval);
    if (retval = nc_def_var (nc_geo, "gds22",    NC_FLOAT, 1, geo_v_theta, &gds22))    ERR(retval);
    if (retval = nc_def_var (nc_geo, "grho",     NC_FLOAT, 1, geo_v_theta, &grho))     ERR(retval);
    if (retval = nc_def_var (nc_geo, "jacobian", NC_FLOAT, 1, geo_v_theta, &jacobian)) ERR(retval);
    if (retval = nc_def_var (nc_geo, "gradpar",  NC_FLOAT, 0, NULL,        &ivar))     ERR(retval);
  }
  
  ////////////////////////////
  //                        //
  //       TIME             //
  //                        //
  ////////////////////////////

  if (pars_->ResWrite) {
    r_time = new nca(0);
    r_time -> write_v_time = true;

    r_time -> file = r_file;
    r_time -> time_dims[0] = rtime_dim;
    if (retval = nc_def_var(r_file, "time", NC_DOUBLE, 1, r_time -> time_dims, &r_time -> time))   ERR(retval);
  }
  
  if (pars_->write_xymom) {
    z_time = new nca(0); 
    z_time -> write_v_time = true;
    
    z_time -> file = z_file;
    z_time -> time_dims[0] = ztime_dim;
    if (retval = nc_def_var(z_file, "time", NC_DOUBLE, 1, z_time -> time_dims, &z_time -> time))   ERR(retval);
  }
  
  time = new nca(0); 
  time -> write_v_time = true;

  time -> file = file;
  time -> time_dims[0] = time_dim;
  if (retval = nc_def_var(file, "time",     NC_DOUBLE, 1, time -> time_dims, &time -> time))    ERR(retval);
  
  ////////////////////////////
  //                        //
  //       DENSITY          //
  //                        //
  ////////////////////////////

  if (pars_->write_moms) {
    den = new nca(0);
    den -> write = true;

    den -> dims[0] = s_dim;
    den -> dims[1] = nz;
    den -> dims[2] = kx_dim;
    den -> dims[3] = ky_dim;
    den -> dims[4] = ri;

    den -> file = nc_special;
    if (retval = nc_def_var(nc_special, "density",  NC_FLOAT, 5, den->dims, &den->idx )) ERR(retval);
    
    den -> start[0] = 0;
    den -> start[1] = 0;
    den -> start[2] = 0;
    den -> start[3] = 0; 
    den -> start[4] = 0; 
    
    den -> count[0] = grids_->Nspecies;
    den -> count[1] = grids_->Nz;
    den -> count[2] = grids_->Nakx;
    den -> count[3] = grids_->Naky;
    den -> count[4] = 2;

    den -> ns = grids_->Nspecies;
    
  } else {
    den = new nca(0);
  }
  
  ////////////////////////////
  //                        //
  //       DENSITY(t=0)     //
  //                        //
  ////////////////////////////

  if (pars_->write_moms) {
    
    den0 = new nca(0); 
    den0 -> write = true;

    den0 -> dims[0] = s_dim;
    den0 -> dims[1] = nz;
    den0 -> dims[2] = kx_dim;
    den0 -> dims[3] = ky_dim;
    den0 -> dims[4] = ri;

    den0 -> file = nc_special;
    if (retval = nc_def_var(nc_special, "density0",  NC_FLOAT, 5, den0 -> dims, &den0 -> idx )) ERR(retval);
    
    den0 -> start[0] = 0;
    den0 -> start[1] = 0;
    den0 -> start[2] = 0;
    den0 -> start[3] = 0; 
    den0 -> start[4] = 0; 
    
    den0 -> count[0] = grids_->Nspecies;
    den0 -> count[1] = grids_->Nz;
    den0 -> count[2] = grids_->Nakx;
    den0 -> count[3] = grids_->Naky;
    den0 -> count[4] = 2;

    den0 -> ns = grids_->Nspecies;
  } else {
    den0 = new nca(0);     
  }
  
  ////////////////////////////
  //                        //
  //       Phi              //
  //                        //
  ////////////////////////////

  if (pars_->write_phi) {
    wphi = new nca(0); 
    wphi -> write = true;

    wphi -> dims[0] = nz;
    wphi -> dims[1] = kx_dim;
    wphi -> dims[2] = ky_dim;
    wphi -> dims[3] = ri;

    wphi -> file = nc_special;
    if (retval = nc_def_var(nc_special, "phi",      NC_FLOAT, 4, wphi -> dims, &wphi -> idx ))      ERR(retval);

    wphi -> start[0] = 0;
    wphi -> start[1] = 0;
    wphi -> start[2] = 0;
    wphi -> start[3] = 0; 
    
    wphi -> count[0] = grids_->Nz;
    wphi -> count[1] = grids_->Nakx;
    wphi -> count[2] = grids_->Naky;
    wphi -> count[3] = 2;

    wphi -> ns = 1;    
  } else {
    wphi = new nca(0); 
  }

  ////////////////////////////
  //                        //
  //       Phi(t=0)         //
  //                        //
  ////////////////////////////

  if (pars_->write_phi) {
    wphi0 = new nca(0); 
    wphi0 -> write = true;

    wphi0 -> dims[0] = nz;
    wphi0 -> dims[1] = kx_dim;
    wphi0 -> dims[2] = ky_dim;
    wphi0 -> dims[3] = ri;
    
    wphi0 -> file = nc_special;
    if (retval = nc_def_var(nc_special, "phi0",      NC_FLOAT, 4, wphi0 -> dims, &wphi0 -> idx ))      ERR(retval);

    wphi0 -> start[0] = 0;
    wphi0 -> start[1] = 0;
    wphi0 -> start[2] = 0;
    wphi0 -> start[3] = 0; 
    
    wphi0 -> count[0] = grids_->Nz;
    wphi0 -> count[1] = grids_->Nakx;
    wphi0 -> count[2] = grids_->Naky;
    wphi0 -> count[3] = 2;

    wphi0 -> ns = 1;
  } else {
    wphi0 = new nca(0); 
  }

  ////////////////////////////
  //                        //
  //   DENSITY(kpar)        //
  //                        //
  ////////////////////////////

  if (pars_->write_phi_kpar and pars_->write_moms) {      
    denk = new nca(0); 
    denk -> write = true;

    denk -> dims[0] = s_dim;
    denk -> dims[1] = nkz;
    denk -> dims[2] = kx_dim;
    denk -> dims[3] = ky_dim;
    denk -> dims[4] = ri;

    denk -> file = nc_special;
    if (retval = nc_def_var(nc_special, "density_kpar", NC_FLOAT, 5, denk -> dims, &denk -> idx)) ERR(retval);    

    denk -> start[0] = 0;
    denk -> start[1] = 0;
    denk -> start[2] = 0;
    denk -> start[3] = 0; 
    denk -> start[4] = 0; 
    
    denk -> count[0] = grids_->Nspecies;
    denk -> count[1] = grids_->Nz;
    denk -> count[2] = grids_->Nakx;
    denk -> count[3] = grids_->Naky;
    denk -> count[4] = 2;

    denk -> ns = 1;
  } else {
    denk = new nca(0); 
  }

  ////////////////////////////
  //                        //
  //   Phi(kpar)            //
  //                        //
  ////////////////////////////

  if (pars_->write_phi_kpar) {
    wphik = new nca(0); 
    wphik -> write = true;

    wphik -> dims[0] = nkz;
    wphik -> dims[1] = kx_dim;
    wphik -> dims[2] = ky_dim;
    wphik -> dims[3] = ri;
    
    wphik -> file = nc_special;
    if (retval = nc_def_var(nc_special, "phi2_kz",    NC_FLOAT, 4, wphik -> dims, &wphik -> idx))     ERR(retval);

    wphik -> start[0] = 0;
    wphik -> start[1] = 0;
    wphik -> start[2] = 0;
    wphik -> start[3] = 0; 
    
    wphik -> count[0] = grids_->Nz;
    wphik -> count[1] = grids_->Nakx;
    wphik -> count[2] = grids_->Naky;
    wphik -> count[3] = 2;

    wphik -> ns = 1;
  } else {
    wphik = new nca(0);     
  }

  ////////////////////////////
  //                        //
  //   Frequencies          //
  //                        //
  ////////////////////////////

  if (pars_->write_omega) {
    omg = new nca(-nX * nY, 2 * nXk * nYk);
    omg -> write_v_time = true;
  
    omg -> time_dims[0] = time_dim; 
    omg -> time_dims[1] = ky_dim; 
    omg -> time_dims[2] = kx_dim;
    omg -> time_dims[3] = ri;
    
    omg -> file = nc_special;
    if (retval = nc_def_var(nc_special, "omega_v_time", NC_FLOAT, 4, omg -> time_dims, &omg -> time)) ERR(retval);

    omg -> time_start[0] = 1;
    
    omg -> time_count[1] = grids_->Naky;
    omg -> time_count[2] = grids_->Nakx;
    omg -> time_count[3] = 2;

    for (int i=0; i < nXk * nYk * 2; i++) omg->cpu[i] = 0.;
  } else {
    omg = new nca(0);
  }

  ////////////////////////////
  //                        //
  // Rosenbluth-Hinton      //
  //                        //
  ////////////////////////////

  if (pars_->write_rh) {
    rh = new nca(0); 
    rh -> write = true;

    rh -> time_dims[0] = time_dim;
    rh -> time_dims[1] = ri;
    
    if (retval = nc_def_var(nc_special, "phi_rh", NC_FLOAT, 2, rh -> time_dims, &rh -> time)) ERR(retval);

    rh -> time_count[1] = 2;
  } else {
    rh = new nca(0); 
  }
    
  ////////////////////////////
  //                        //
  //     PZT estimates      //
  //                        //
  ////////////////////////////

  if (pars_->write_pzt) {
    Pzt = new nca(0);
    pZt = new nca(0);
    pzT = new nca(0);
    Pzt -> write_v_time = true;
  
    Pzt -> time_dims[0] = time_dim;
    pZt -> time_dims[0] = time_dim;
    pzT -> time_dims[0] = time_dim;

    Pzt -> file = nc_special;
    pZt -> file = nc_special;
    pzT -> file = nc_special;
    if (retval = nc_def_var(nc_special, "prim", NC_FLOAT, 1, Pzt -> time_dims, &Pzt -> idx)) ERR(retval);
    if (retval = nc_def_var(nc_special, "sec",  NC_FLOAT, 1, pZt -> time_dims, &pZt -> idx)) ERR(retval);
    if (retval = nc_def_var(nc_special, "tert", NC_FLOAT, 1, pzT -> time_dims, &pzT -> idx)) ERR(retval);

    cudaMallocHost (&primary,   sizeof(float));    primary[0] = 0.;  
    cudaMallocHost (&secondary, sizeof(float));    secondary[0] = 0.;
    cudaMallocHost (&tertiary,  sizeof(float));    tertiary[0] = 0.;
    cudaMalloc     (&t_bar,     sizeof(cuComplex) * nR * nS);
  } else {
    Pzt = new nca(0);
    pZt = new nca(0);
    pzT = new nca(0);
  }

  ////////////////////////////
  //                        //
  // (1-G0)phi**2 (species)  //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_species] > 0) {
    Ps = new nca(nS); 
    Ps -> write_v_time = true;

    Ps -> time_dims[0] = time_dim;
    Ps -> time_dims[1] = s_dim;
    
    Ps -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Pst", NC_FLOAT, 2, Ps -> time_dims, &Ps -> time))  ERR(retval);
    
    Ps -> time_count[1] = grids_->Nspecies;
  } else {
    Ps = new nca(0);
  }
  
  ////////////////////////////
  //                        //
  //   P (kx, species)      //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_kx] > 0) {
    Pkx = new nca(nX*nS, nXk*nS); 
    Pkx -> write_v_time = true;

    Pkx -> time_dims[0] = time_dim;
    Pkx -> time_dims[1] = s_dim;
    Pkx -> time_dims[2] = kx_dim;
    
    Pkx -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Pkxst", NC_FLOAT, 3, Pkx -> time_dims, &Pkx -> time))  ERR(retval);

    Pkx -> time_count[1] = grids_->Nspecies;
    Pkx -> time_count[2] = grids_->Nakx;      
  } else {
    Pkx = new nca(0);
  }
  
  ////////////////////////////
  //                        //
  //   P (ky, species)      //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_ky] > 0) {
    Pky = new nca(nY*nS, nYk*nS);
    Pky -> write_v_time = true;
    
    Pky -> time_dims[0] = time_dim;
    Pky -> time_dims[1] = s_dim;
    Pky -> time_dims[2] = ky_dim;

    Pky -> file = nc_sp;     
    if (retval = nc_def_var(nc_sp, "Pkyst", NC_FLOAT, 3, Pky->time_dims, &Pky->time))  ERR(retval);

    Pky -> time_count[1] = grids_->Nspecies;
    Pky -> time_count[2] = grids_->Naky;
    
  } else {
    Pky = new nca(0);
  }
  
  ////////////////////////////
  //                        //
  //   P (kz, species)      //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_kz] > 0) {
    Pkz = new nca(nZ*nS, nZ*nS); 
    Pkz -> write_v_time = true;

    Pkz -> time_dims[0] = time_dim;
    Pkz -> time_dims[1] = s_dim;
    Pkz -> time_dims[2] = nkz;
    
    Pkz -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Pkzst", NC_FLOAT, 3, Pkz -> time_dims, &Pkz -> time))  ERR(retval);

    Pkz -> time_count[1] = grids_->Nspecies;
    Pkz -> time_count[2] = grids_->Nz;
  } else {
    Pkz = new nca(0); 
  }
  
  ////////////////////////////
  //                        //
  //   P (z, species)       //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_z] > 0) {
    Pz = new nca(nZ*nS); 
    Pz -> write_v_time = true;

    Pz -> time_dims[0] = time_dim;
    Pz -> time_dims[1] = s_dim;
    Pz -> time_dims[2] = nz;
    
    Pz -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Pzst", NC_FLOAT, 3, Pz -> time_dims, &Pz -> time))  ERR(retval);
    
    Pz -> time_count[1] = grids_->Nspecies;
    Pz -> time_count[2] = grids_->Nz;
  } else {
    Pz = new nca(0); 
  }
  
  ////////////////////////////
  //                        //
  //   P (kx,ky,  species)  //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_kxky] > 0) {
    Pkxky = new nca(nX * nY * nS, nXk * nYk * nS); 
    
    Pkxky -> write_v_time = true;

    Pkxky -> time_dims[0] = time_dim;
    Pkxky -> time_dims[1] = s_dim;
    Pkxky -> time_dims[2] = ky_dim;
    Pkxky -> time_dims[3] = kx_dim;
    
    Pkxky -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Pkxkyst", NC_FLOAT, 4, Pkxky -> time_dims, &Pkxky -> time))  ERR(retval);
    
    Pkxky -> time_count[1] = grids_->Nspecies;
    Pkxky -> time_count[2] = grids_->Naky;
    Pkxky -> time_count[3] = grids_->Nakx;
  } else {
    Pkxky = new nca(0); 
  }

  ////////////////////////////
  //                        //
  //   W (species)          //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_species] > 0) {
    Ws = new nca(nS); 
    Ws -> write_v_time = true;

    Ws -> time_dims[0] = time_dim;
    Ws -> time_dims[1] = s_dim;
    
    Ws -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Wst", NC_FLOAT, 2, Ws -> time_dims, &Ws -> time))  ERR(retval);
    
    Ws -> time_count[1] = grids_->Nspecies;
  } else {
    Ws = new nca(0); 
  }

  ////////////////////////////
  //                        //
  //   W (kx, species)      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_kx] > 0) {
    Wkx = new nca(nX*nS, nXk*nS); 
    Wkx -> write_v_time = true;

    Wkx -> time_dims[0] = time_dim;
    Wkx -> time_dims[1] = s_dim;
    Wkx -> time_dims[2] = kx_dim;
    
    Wkx -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Wkxst", NC_FLOAT, 3, Wkx -> time_dims, &Wkx -> time))  ERR(retval);
    
    Wkx -> time_count[1] = grids_->Nspecies;
    Wkx -> time_count[2] = grids_->Nakx;      
  } else {
    Wkx = new nca(0);
  }
  
  ////////////////////////////
  //                        //
  //   W (ky, species)      //
  //                        //
  ////////////////////////////
  
  if (pars_->wspectra[WSPECTRA_ky] > 0) {
    Wky = new nca(nY*nS, nYk*nS); 
    Wky -> write_v_time = true;

    Wky -> time_dims[0] = time_dim;
    Wky -> time_dims[1] = s_dim;
    Wky -> time_dims[2] = ky_dim;
    
    Wky -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Wkyst", NC_FLOAT, 3, Wky -> time_dims, &Wky -> time))  ERR(retval);
    
    Wky -> time_count[1] = grids_->Nspecies;
    Wky -> time_count[2] = grids_->Naky;      
  } else {
    Wky = new nca(0); 
  }
  
  ////////////////////////////
  //                        //
  //   W (kz, species)      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_kz] > 0) {
    Wkz = new nca(nZ * nS, nZ * nS); 
    Wkz -> write_v_time = true;

    Wkz -> time_dims[0] = time_dim;
    Wkz -> time_dims[1] = s_dim;
    Wkz -> time_dims[2] = nkz;
    
    Wkz -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Wkzst", NC_FLOAT, 3, Wkz -> time_dims, &Wkz -> time))  ERR(retval);
    
    Wkz -> time_count[1] = grids_->Nspecies;
    Wkz -> time_count[2] = grids_->Nz;
  } else {
    Wkz = new nca(0); 
  }

  ////////////////////////////
  //                        //
  //   W (z, species)       //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_z] > 0) {
    Wz = new nca(nZ*nS); 
    Wz -> write_v_time = true;

    Wz -> time_dims[0] = time_dim;
    Wz -> time_dims[1] = s_dim;
    Wz -> time_dims[2] = nz;
    
    Wz -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Wzst", NC_FLOAT, 3, Wz -> time_dims, &Wz -> time))  ERR(retval);
    
    Wz -> time_count[1] = grids_->Nspecies;
    Wz -> time_count[2] = grids_->Nz;
  } else {
    Wz = new nca(0); 
  }
  
  ////////////////////////////
  //                        //
  //   W (kx,ky,  species)  //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_kxky] > 0) {
    Wkxky = new nca(nX * nY * nS, nXk * nYk * nS); 
    Wkxky -> write_v_time = true;

    Wkxky -> time_dims[0] = time_dim;
    Wkxky -> time_dims[1] = s_dim;
    Wkxky -> time_dims[2] = ky_dim;
    Wkxky -> time_dims[3] = kx_dim;
    
    Wkxky -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Wkxkyst", NC_FLOAT, 4, Wkxky -> time_dims, &Wkxky -> time))  ERR(retval);
    
    Wkxky -> time_count[1] = grids_->Nspecies;
    Wkxky -> time_count[2] = grids_->Naky;      
    Wkxky -> time_count[3] = grids_->Nakx;
  } else {
    Wkxky = new nca(0); 
  }

  ////////////////////////////
  //                        //
  // W (adiabatic species)  //
  //                        //
  ////////////////////////////

  if (pars_->aspectra[ASPECTRA_species] > 0) {
    As = new nca(1); 
    As -> write_v_time;

    As -> time_dims[0] = time_dim;
    
    As -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "At", NC_FLOAT, 1, As -> time_dims, &As -> time))  ERR(retval);

  } else {
    As = new nca(0); 
  }

  ////////////////////////////
  //                        //
  //   W (kx) adiabatic     //
  //                        //
  ////////////////////////////

  if (pars_->aspectra[ASPECTRA_kx] > 0) {
    Akx = new nca(nX, nXk); 
    Akx -> write_v_time = true;

    Akx -> time_dims[0] = time_dim;
    Akx -> time_dims[1] = kx_dim;
    
    Akx -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Akxst", NC_FLOAT, 2, Akx -> time_dims, &Akx -> time))  ERR(retval);
    
    Akx -> time_count[1] = grids_->Nakx;      
  } else {
    Akx = new nca(0); 
  }
  
  ////////////////////////////
  //                        //
  //   W (ky) adiabatic     //
  //                        //
  ////////////////////////////

  if (pars_->aspectra[ASPECTRA_ky] > 0) {
    Aky = new nca(nY, nYk); 
    Aky -> write_v_time = true;

    Aky -> time_dims[0] = time_dim;
    Aky -> time_dims[1] = ky_dim;
    
    Aky -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Akyst", NC_FLOAT, 2, Aky -> time_dims, &Aky -> time))  ERR(retval);

    Aky -> time_count[1] = grids_->Naky;      
  } else {
    Aky = new nca(0); 
  }
  
  ////////////////////////////
  //                        //
  //   A (kz)  adiabatic    //
  //                        //
  ////////////////////////////

  if (pars_->aspectra[ASPECTRA_kz] > 0) {
    Akz = new nca(nZ, nZ); 
    Akz -> write_v_time = true;

    Akz -> time_dims[0] = time_dim;
    Akz -> time_dims[1] = nkz;
    
    Akz -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Akzst", NC_FLOAT, 2, Akz -> time_dims, &Akz -> time))  ERR(retval);
    
    Akz -> time_count[1] = grids_->Nz;      
  } else {
    Akz = new nca(0); 
  }
  
  ////////////////////////////
  //                        //
  //   A (z)  adiabatic     //
  //                        //
  ////////////////////////////

  if (pars_->aspectra[ASPECTRA_z] > 0) {
    Az = new nca(nZ); 
    Az -> write_v_time = true;

    Az -> time_dims[0] = time_dim;
    Az -> time_dims[1] = nz;
    
    Az -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Azst", NC_FLOAT, 2, Az -> time_dims, &Az -> time))  ERR(retval);
    
    Az -> time_count[1] = grids_->Nz;      
  } else {
    Az = new nca(0); 
  }
  
  ////////////////////////////
  //                        //
  //   W (kx,ky) adiabatic  //
  //                        //
  ////////////////////////////

  if (pars_->aspectra[ASPECTRA_kxky] > 0) {
    Akxky = new nca(nX * nY, nXk * nYk); 
    Akxky -> write_v_time = true;

    Akxky -> time_dims[0] = time_dim;
    Akxky -> time_dims[1] = ky_dim;
    Akxky -> time_dims[2] = kx_dim;
    
    Akxky -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Akxkyst", NC_FLOAT, 3, Akxky -> time_dims, &Akxky -> time))  ERR(retval);
    
    Akxky -> time_count[1] = grids_->Naky;      
    Akxky -> time_count[2] = grids_->Nakx;
  } else {
    Akxky = new nca(0); 
  }
  
  ////////////////////////////
  //                        //
  // Lag-Herm spectrum      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_lm] > 0) {
    Wlm = new nca(nL*nM*nS); 
    Wlm -> write_v_time = true;
  
    Wlm -> time_dims[0] = time_dim;
    Wlm -> time_dims[1] = s_dim;
    Wlm -> time_dims[2] = m_dim;
    Wlm -> time_dims[3] = l_dim;
    
    Wlm -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Wlmst", NC_FLOAT, 4, Wlm -> time_dims, &Wlm -> time))  ERR(retval);
    
    Wlm -> time_count[1] = grids_->Nspecies;
    Wlm -> time_count[2] = grids_->Nm;
    Wlm -> time_count[3] = grids_->Nl;      
  } else {
    Wlm = new nca(0); 
  }
    
  ////////////////////////////
  //                        //
  // Laguerre spectrum      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_l] > 0) {
    Wl = new nca(nL*nS); 
    Wl -> write_v_time = true;
    
    Wl -> time_dims[0] = time_dim;
    Wl -> time_dims[1] = s_dim;
    Wl -> time_dims[2] = l_dim;

    Wl -> file = nc_sp;     
    if (retval = nc_def_var(nc_sp, "Wlst", NC_FLOAT, 3, Wl -> time_dims, &Wl -> time))  ERR(retval);
    
    Wl -> time_count[1] = grids_->Nspecies;
    Wl -> time_count[2] = grids_->Nl;
  } else {
    Wl = new nca(0); 
  }
  
  ////////////////////////////
  //                        //
  //  Hermite spectrum      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_m] > 0) {
    Wm = new nca(nM*nS); 
    Wm -> write_v_time = true;
  
    Wm -> time_dims[0] = time_dim;
    Wm -> time_dims[1] = s_dim;
    Wm -> time_dims[2] = m_dim;
    
    Wm -> file = nc_sp; 
    if (retval = nc_def_var(nc_sp, "Wmst", NC_FLOAT, 3, Wm -> time_dims, &Wm -> time))  ERR(retval);
    
    Wm -> time_count[1] = grids_->Nspecies;    
    Wm -> time_count[2] = grids_->Nm;          
  } else {
    Wm = new nca(0); 
  }

  bool linked = (not pars_->local_limit && not pars_->boundary_option_periodic);

  /*
  if (linked && false) {
    zkxky[0] = nz;
    zkxky[1] = kx_dim; 
    zkxky[2] = ky_dim;
    
    zkxky -> file = nc_special; 
    if (retval = nc_def_var(nc_special, "theta_x",  NC_FLOAT, 3, zkxky, &theta_x))  ERR(retval);
  }
  */
  
  ////////////////////////////////
  //  Zonal, y-component of v_E //
  //  <v_ExB>_y,z (x)           // 
  //                            //
  ////////////////////////////////

  if (pars_->write_vEy) {
    vEy = new nca(grids_->NxNyNz, grids_->Nx);
    vEy -> write_v_time = true;

    vEy -> time_dims[0] = time_dim;
    vEy -> time_dims[1] = x_dim;
    
    vEy -> file = nc_zonal;  
    if (retval = nc_def_var(nc_zonal, "vEy_xt", NC_FLOAT, 2, vEy->time_dims, &vEy->time))  ERR(retval);
    
    vEy -> time_count[1] = grids_->Nx;
    vEy -> xdata = true;
    vEy -> dx = true;
  } else {
    vEy = new nca(0);
  }

  if (pars_->write_avg_zvE) {
    avg_zvE = new nca(grids_->NxNyNz, grids_->Nx); 
    avg_zvE -> write_v_time = true;

    avg_zvE -> time_dims[0] = time_dim;
    
    avg_zvE -> file = nc_flux;  
    if (retval = nc_def_var(nc_flux, "avg_zvE_t", NC_FLOAT, 1,
			    avg_zvE->time_dims, &avg_zvE->time))  ERR(retval);
    avg_zvE -> scalar = true;
    avg_zvE -> dx = true;
  } else {
    avg_zvE = new nca(0); 
  }

  if (pars_->write_avg_zkxvEy) {
    avg_zkxvEy = new nca(grids_->NxNyNz, grids_->Nx); 
    avg_zkxvEy -> write_v_time = true;

    avg_zkxvEy -> time_dims[0] = time_dim;
    
    avg_zkxvEy -> file = nc_flux;  
    if (retval = nc_def_var(nc_flux, "avg_zkxvEy_t", NC_FLOAT, 1,
			    avg_zkxvEy->time_dims, &avg_zkxvEy->time))  ERR(retval);
    avg_zkxvEy -> scalar = true;
    avg_zkxvEy -> d2x = true;
  } else {
    avg_zkxvEy = new nca(0); 
  }

  if (pars_->write_avg_zkden) {
    avg_zkden = new nca(grids_->NxNyNz, grids_->Nx); 
    avg_zkden -> write_v_time = true;

    avg_zkden -> time_dims[0] = time_dim;
    
    avg_zkden -> file = nc_flux;  
    if (retval = nc_def_var(nc_flux, "avg_zkden_t", NC_FLOAT, 1,
			    avg_zkden->time_dims, &avg_zkden->time))  ERR(retval);
    avg_zkden -> scalar = true;
    avg_zkden -> dx = true;
  } else {
    avg_zkden = new nca(0); 
  }

  if (pars_->write_avg_zkUpar) {
    avg_zkUpar = new nca(grids_->NxNyNz, grids_->Nx); 
    avg_zkUpar -> write_v_time = true;

    avg_zkUpar -> time_dims[0] = time_dim;
    
    avg_zkUpar -> file = nc_flux;  
    if (retval = nc_def_var(nc_flux, "avg_zkUpar_t", NC_FLOAT, 1,
			    avg_zkUpar->time_dims, &avg_zkUpar->time))  ERR(retval);
    avg_zkUpar -> scalar = true;
    avg_zkUpar -> dx = true;
  } else {
    avg_zkUpar = new nca(0); 
  }

  if (pars_->write_avg_zkTpar) {
    avg_zkTpar = new nca(grids_->NxNyNz, grids_->Nx); 
    avg_zkTpar -> write_v_time = true;

    avg_zkTpar -> time_dims[0] = time_dim;
    
    avg_zkTpar -> file = nc_flux;  
    if (retval = nc_def_var(nc_flux, "avg_zkTpar_t", NC_FLOAT, 1,
			    avg_zkTpar->time_dims, &avg_zkTpar->time))  ERR(retval);
    avg_zkTpar -> scalar = true;
    avg_zkTpar -> dx = true;
    avg_zkTpar -> adj = sqrtf(2.0);
  } else {
    avg_zkTpar = new nca(0); 
  }

  if (pars_->write_avg_zkqpar) {
    avg_zkqpar = new nca(grids_->NxNyNz, grids_->Nx); 
    avg_zkqpar -> write_v_time = true;

    avg_zkqpar -> time_dims[0] = time_dim;
    
    avg_zkqpar -> file = nc_flux;  
    if (retval = nc_def_var(nc_flux, "avg_zkqpar_t", NC_FLOAT, 1,
			    avg_zkqpar->time_dims, &avg_zkqpar->time))  ERR(retval);
    avg_zkqpar -> scalar = true;
    avg_zkqpar -> dx = true;
    avg_zkqpar -> adj = sqrtf(6.0);
  } else {
    avg_zkqpar = new nca(0); 
  }

  if (pars_->write_avg_zkTperp) {
    avg_zkTperp = new nca(grids_->NxNyNz, grids_->Nx); 
    avg_zkTperp -> write_v_time = true;

    avg_zkTperp -> time_dims[0] = time_dim;
    
    avg_zkTperp -> file = nc_flux;  
    if (retval = nc_def_var(nc_flux, "avg_zkTperp_t", NC_FLOAT, 1,
			    avg_zkTperp->time_dims, &avg_zkTperp->time))  ERR(retval);
    avg_zkTperp -> scalar = true;
    avg_zkTperp -> dx = true;
  } else {
    avg_zkTperp = new nca(0); 
  }

  ////////////////////////////
  //                        //
  //  <d/dx v_ExB>_y,z (x)  // 
  //                        //
  ////////////////////////////

  if (pars_->write_kxvEy) {
    kxvEy = new nca(grids_->NxNyNz, grids_->Nx);    
    kxvEy -> write_v_time = true;

    kxvEy -> time_dims[0] = time_dim;
    kxvEy -> time_dims[1] = x_dim;
    
    kxvEy -> file = nc_zonal;  
    if (retval = nc_def_var(nc_zonal, "kxvEy_xt", NC_FLOAT, 2, kxvEy -> time_dims, &kxvEy -> time))  ERR(retval);
    
    kxvEy -> time_count[1] = grids_->Nx;
    kxvEy -> xdata = true;
    kxvEy -> d2x = true;
  } else {
    kxvEy = new nca(0);    
  }
  
  ////////////////////////////
  //                        //
  // <d/dx denh>_y,z (x)    // 
  //                        //
  ////////////////////////////

  if (pars_->write_kden) {
    kden = new nca(grids_->NxNyNz, grids_->Nx);
    kden -> write_v_time = true;

    kden -> time_dims[0] = time_dim;
    kden -> time_dims[1] = x_dim;
    
    kden -> file = nc_zonal;  
    if (retval = nc_def_var(nc_zonal, "kden_xt", NC_FLOAT, 2, kden->time_dims, &kden->time))  ERR(retval);
    
    kden -> time_count[1] = grids_->Nx; 
    kden -> xdata = true;
    kden -> dx = true;
  } else {
    kden = new nca(0);
  }

  ////////////////////////////
  //                        //
  // <d/dx uparh>_y,z (x)   // 
  //                        //
  ////////////////////////////

  if (pars_->write_kUpar) {
    kUpar = new nca(grids_->NxNyNz, grids_->Nx);
    kUpar -> write_v_time = true;

    kUpar -> time_dims[0] = time_dim;
    kUpar -> time_dims[1] = x_dim;
    
    kUpar -> file = nc_zonal;  
    if (retval = nc_def_var(nc_zonal, "kUpar_xt", NC_FLOAT, 2, kUpar->time_dims, &kUpar->time))  ERR(retval);
    
    kUpar->time_count[1] = grids_->Nx;
    kUpar->xdata = true;
    kUpar -> dx = true;
  } else {
    kUpar = new nca(0);
  }
  
  ////////////////////////////
  //                        //
  // <d/dx Tparh>_y,z (x)   // 
  //                        //
  ////////////////////////////

  if (pars_->write_kTpar) {
    kTpar = new nca(grids_->NxNyNz, grids_->Nx);
    kTpar->write_v_time = true;

    kTpar -> time_dims[0] = time_dim;
    kTpar -> time_dims[1] = x_dim;
    
    kTpar -> file = nc_zonal;  
    if (retval = nc_def_var(nc_zonal, "kTpar_xt", NC_FLOAT, 2, kTpar->time_dims, &kTpar->time))  ERR(retval);
    
    kTpar -> time_count[1] = grids_->Nx;
    kTpar -> xdata = true;
    kTpar -> dx = true;
    kTpar -> adj = sqrtf(2.0);
  } else {
    kTpar = new nca(0);
  }
  
  ////////////////////////////
  //                        //
  // <d/dx Tperph>_y,z (x)  // 
  //                        //
  ////////////////////////////

  if (pars_->write_kTperp) {
    kTperp = new nca(grids_->NxNyNz, grids_->Nx);
    kTperp -> write_v_time = true;

    kTperp -> time_dims[0] = time_dim;
    kTperp -> time_dims[1] = x_dim;
    
    kTperp -> file = nc_zonal;  
    if (retval = nc_def_var(nc_zonal, "kTperp_xt", NC_FLOAT, 2, kTperp->time_dims, &kTperp->time))  ERR(retval);
    
    kTperp -> time_count[1] = grids_->Nx;
    kTperp -> xdata = true;
    kTperp -> dx = true;
  } else {
    kTperp = new nca(0);
  }
  
  ////////////////////////////
  //                        //
  // <d/dx qparh>_y,z (x)   // 
  //                        //
  ////////////////////////////

  if (pars_->write_kqpar) {
    kqpar = new nca(grids_->NxNyNz, grids_->Nx);
    kqpar -> write_v_time = true;
    
    kqpar -> time_dims[0] = time_dim;
    kqpar -> time_dims[1] = x_dim;
    
    kqpar -> file = nc_zonal;  
    if (retval = nc_def_var(nc_zonal, "kqpar_xt", NC_FLOAT, 2, kqpar -> time_dims, &kqpar->time))  ERR(retval);
    
    kqpar -> time_count[1] = grids_->Nx;
    kqpar -> xdata = true;
    kqpar -> dx = true;
    kqpar -> adj = sqrtf(6.0);
  } else {
    kqpar = new nca(0);
  }

  ////////////////////////////
  // Non-zonal              //
  // <v_ExB> (x, y)         // 
  // y-component            //
  ////////////////////////////

  if (pars_->write_xyvEy) {
    xyvEy = new nca(grids_->NxNyNz, grids_->NxNy);
    xyvEy->write_v_time = true;
  
    xyvEy -> time_dims[0] = ztime_dim;
    xyvEy -> time_dims[1] = zy_dim;  // Transpose to accommodate ncview
    xyvEy -> time_dims[2] = zx_dim;
    
    xyvEy -> file = z_file;
    if (retval = nc_def_var(z_file, "vEy_xyt", NC_FLOAT, 3, xyvEy -> time_dims, &xyvEy->time)) ERR(retval);
    
    xyvEy -> time_count[1] = grids_->Ny;      
    xyvEy -> time_count[2] = grids_->Nx;          

    xyvEy -> xydata = true;
    xyvEy -> dx = true;
  } else {
    xyvEy = new nca(0);
  }
  
  ////////////////////////////
  //                        //
  // v_ExB (x, y)           // 
  // x-component            //
  ////////////////////////////

  if (pars_->write_xyvEx) {
    xyvEx = new nca(grids_->NxNyNz, grids_->NxNy);
    xyvEx->write_v_time = true;
  
    xyvEx -> time_dims[0] = ztime_dim;
    xyvEx -> time_dims[1] = zy_dim;  // Transpose to accommodate ncview
    xyvEx -> time_dims[2] = zx_dim;
    
    xyvEx -> file = z_file;
    if (retval = nc_def_var(z_file, "vEx_xyt", NC_FLOAT, 3, xyvEx -> time_dims, &xyvEx->time)) ERR(retval);
    
    xyvEx -> time_count[1] = grids_->Ny;      
    xyvEx -> time_count[2] = grids_->Nx;          

    xyvEx -> xydata = true;
    xyvEx -> mdy = true; // Requesting that the diagnostic write -d/dy because this is part of v_E
  } else {
    xyvEx = new nca(0);
  }
  
  ////////////////////////////
  // Non-zonal              //
  // <d/dx v_ExB,y> (x, y)  // 
  //                        //
  ////////////////////////////

  if (pars_ -> write_xykxvEy) {
    xykxvEy = new nca(grids_->NxNyNz, grids_->NxNy);
    xykxvEy -> write_v_time = true;
  
    xykxvEy -> time_dims[0] = ztime_dim;
    xykxvEy -> time_dims[1] = zy_dim;  // Transpose to accommodate ncview
    xykxvEy -> time_dims[2] = zx_dim;
    
    xykxvEy -> file = z_file;
    if (retval = nc_def_var(z_file, "kxvEy_xyt", NC_FLOAT, 3, xykxvEy -> time_dims, &xykxvEy->time)) ERR(retval);
    
    xykxvEy -> time_count[1] = grids_->Ny;      
    xykxvEy -> time_count[2] = grids_->Nx;

    xykxvEy -> xydata = true;
    xykxvEy -> d2x = true;
  } else {
    xykxvEy = new nca(0);
  }
  
  ////////////////////////////
  // Non-zonal              //
  // <den>  (x, y)         // 
  //                        //
  ////////////////////////////

  if (pars_->write_xyden) {
    xyden = new nca(grids_->NxNyNz, grids_->NxNy);
    xyden->write_v_time = true;
  
    xyden -> time_dims[0] = ztime_dim;
    xyden -> time_dims[1] = zy_dim;  // Transpose to accommodate ncview
    xyden -> time_dims[2] = zx_dim;
    
    xyden -> file = z_file;    
    if (retval = nc_def_var(z_file, "den_xyt", NC_FLOAT, 3, xyden -> time_dims, &xyden->time)) ERR(retval);
    
    xyden -> time_count[1] = grids_->Ny;      
    xyden -> time_count[2] = grids_->Nx;

    xyden -> xydata = true;
  } else {
    xyden = new nca(0);
  }
  
  ////////////////////////////
  // Non-zonal              //
  // <Upar> (x, y)         // 
  //                        //
  ////////////////////////////

  if (pars_->write_xyUpar) {
    xyUpar = new nca(grids_->NxNyNz, grids_->NxNy);
    xyUpar->write_v_time = true;
  
    xyUpar -> time_dims[0] = ztime_dim;
    xyUpar -> time_dims[1] = zy_dim;  // Transpose to accommodate ncview
    xyUpar -> time_dims[2] = zx_dim;
    
    xyUpar -> file = z_file;    
    if (retval = nc_def_var(z_file, "upar_xyt", NC_FLOAT, 3, xyUpar -> time_dims, &xyUpar->time)) ERR(retval);
    
    xyUpar -> time_count[1] = grids_->Ny;      
    xyUpar -> time_count[2] = grids_->Nx;

    xyUpar -> xydata = true;
  } else {
    xyUpar = new nca(0);
  }    
  
  ////////////////////////////
  // Non-zonal              //
  // <Tpar> (x, y)         // 
  //                        //
  ////////////////////////////

  if (pars_->write_xyTpar) {
    xyTpar = new nca(grids_->NxNyNz, grids_->NxNy);
    xyTpar->write_v_time = true;
  
    xyTpar -> time_dims[0] = ztime_dim;
    xyTpar -> time_dims[1] = zy_dim;  // Transpose to accommodate ncview
    xyTpar -> time_dims[2] = zx_dim;
    
    xyTpar -> file = z_file;
    if (retval = nc_def_var(z_file, "Tpar_xyt", NC_FLOAT, 3, xyTpar -> time_dims, &xyTpar->time)) ERR(retval);
    
    xyTpar -> time_count[1] = grids_->Ny;      
    xyTpar -> time_count[2] = grids_->Nx;

    xyTpar -> xydata = true;
    xyTpar -> adj = sqrtf(2.0);
  } else {
    xyTpar = new nca(0);
  }    
  
  ////////////////////////////
  // Non-zonal              //
  // <Tperp> (x, y)         // 
  //                        //
  ////////////////////////////

  if (pars_->write_xyTperp) {
    xyTperp = new nca(grids_->NxNyNz, grids_->NxNy);
    xyTperp -> write_v_time = true;

    xyTperp -> time_dims[0] = ztime_dim;
    xyTperp -> time_dims[1] = zy_dim;  // Transpose to accommodate ncview
    xyTperp -> time_dims[2] = zx_dim;

    xyTperp -> file = z_file;    
    if (retval = nc_def_var(z_file, "Tperp_xyt", NC_FLOAT, 3, xyTperp -> time_dims, &xyTperp->time))  ERR(retval);
    
    xyTperp -> time_count[1] = grids_->Ny;      
    xyTperp -> time_count[2] = grids_->Nx;

    xyTperp -> xydata = true;
  } else {
    xyTperp = new nca(0);
  }    

  ////////////////////////////
  // Non-zonal              //
  // <qpar> (x, y)          // 
  //                        //
  ////////////////////////////

  if (pars_->write_xyqpar) {
    xyqpar = new nca(grids_->NxNyNz, grids_->NxNy);
    xyqpar->write_v_time = true;
  
    xyqpar -> time_dims[0] = ztime_dim;
    xyqpar -> time_dims[1] = zy_dim;  // Transpose to accommodate ncview
    xyqpar -> time_dims[2] = zx_dim;
    
    xyqpar -> file = z_file;    
    if (retval = nc_def_var(z_file, "qpar_xyt", NC_FLOAT, 3, xyqpar -> time_dims, &xyqpar->time)) ERR(retval);
    
    xyqpar -> time_count[1] = grids_->Ny;      
    xyqpar -> time_count[2] = grids_->Nx;

    xyqpar -> xydata = true;
    xyqpar -> adj = sqrtf(6.0);
    
  } else {
    xyqpar = new nca(0);
  }    

  if (pars_->ks && pars_->ResWrite) {
    r_y = new nca(pars_->ResQ * grids_->NxNyNz * grids_->Nmoms);
    r_y -> write_v_time = true;

    r_y -> time_dims[0] = rtime_dim;
    r_y -> time_dims[1] = res_dim; 

    r_y -> file = r_file;
    if (retval = nc_def_var(r_file, "r", NC_DOUBLE, 2, r_y -> time_dims, &r_y -> time))  ERR(retval);

    r_y -> time_count[1] = pars_->ResQ * grids_->NxNyNz*grids_->Nmoms;

  } else {
    r_y = new nca(0);
  }
  
  ////////////////////////////
  //                        //
  //   g(y) for K-S eqn     // 
  //                        //
  ////////////////////////////

  if (pars_->ks && pars_->write_ks) {
    g_y = new nca(grids_->Ny); 
    g_y -> write_v_time = true;
    
    g_y -> time_dims[0] = time_dim;
    g_y -> time_dims[1] = y_dim;
    
    g_y -> file = nc_special;
    if (retval = nc_def_var(nc_special, "g_yt", NC_FLOAT, 2, g_y -> time_dims, &g_y -> time))  ERR(retval);

    g_y -> time_count[1] = grids_->Ny;

    int nbatch = 1;
    grad_perp = new GradPerp(grids_, nbatch, grids_->Nyc);    
  } else {
    g_y = new nca(0); 
  }    
  
  ////////////////////////////
  //                        //
  //   Free energy          //
  //                        //
  ////////////////////////////

  if (pars_->write_free_energy) {
    Wtot = new nca(0); 
    Wtot -> write_v_time = true;
  
    Wtot -> time_dims[0] = time_dim;

    Wtot -> file = nc_sp;
    if (retval = nc_def_var(nc_sp, "W", NC_FLOAT, 1, Wtot -> time_dims, &Wtot -> time)) ERR(retval);

    totW = 0.;
  } else {
    Wtot = new nca(0); 
  }    

  ////////////////////////////
  //                        //
  //    Heat fluxes         //
  //                        //
  ////////////////////////////

  if (pars_->write_fluxes ) {
    qs = new nca(nS); 
    qs -> write_v_time = true; 
  
    qs -> time_dims[0] = time_dim;
    qs -> time_dims[1] = s_dim;
    
    qs -> file = nc_flux;
    if (retval = nc_def_var(nc_flux, "qflux", NC_FLOAT, 2, qs -> time_dims, &qs -> time)) ERR(retval);

    qs -> time_count[1] = grids_->Nspecies;

    all_red = new Species_Reduce(nR, nS);  cudaDeviceSynchronize();  CUDA_DEBUG("Reductions: %s \n");
  } else {
    qs = new nca(0); 
  }

  DEBUGPRINT("ncdf:  ending definition mode for NetCDF \n");
  
  if (retval = nc_enddef(file)) ERR(retval);
  
  if (pars_->write_xymom) {
    if (retval = nc_enddef(z_file)) ERR(retval);
  }

  if (pars_->ResWrite) {
    if (retval = nc_enddef(r_file)) ERR(retval);
  }
  
  ///////////////////////////////////
  //                               //
  //        x                      //
  //                               //
  ///////////////////////////////////
  x_start[0] = 0;
  x_count[0] = grids_->Nx;

  if (retval = nc_put_vara(file, x, x_start, x_count, grids_->x_h))         ERR(retval);

  if (pars_->write_xymom) {
    if (retval = nc_put_vara(z_file, zx, x_start, x_count, grids_->x_h))   ERR(retval);
  }

  ///////////////////////////////////
  //                               //
  //        y                      //
  //                               //
  ///////////////////////////////////
  y_start[0] = 0;
  y_count[0] = grids_->Ny;

  if (retval = nc_put_vara(file, y, y_start, y_count, grids_->y_h))         ERR(retval);

  if (pars_->write_xymom) {
    if (retval = nc_put_vara(z_file, zy, y_start, y_count, grids_->y_h))     ERR(retval);
  }
  
  ///////////////////////////////////
  //                               //
  //         z                     //
  //                               //
  ///////////////////////////////////
  z_start[0] = 0;
  z_count[0] = grids_->Nz;

  //  if (retval = nc_put_vara(file, z, z_start, z_count, z_h))      ERR(retval);

  ///////////////////////////////////
  //                               //
  //        kz                     //
  //                               //
  ///////////////////////////////////
  kz_start[0] = 0;
  kz_count[0] = grids_->Nz;

  if(geo_) {
    for (int i=0; i<grids_->Nz; i++) grids_->kpar_outh[i] = geo_->gradpar*grids_->kz_outh[i];
    if (retval = nc_put_vara(file, kz, kz_start, kz_count, grids_->kpar_outh))      ERR(retval);
  }

  ///////////////////////////////////
  //                               //
  //        ky                     //
  //                               //
  ///////////////////////////////////
  ky_start[0] = 0;
  ky_count[0] = grids_->Naky;

  if (retval = nc_put_vara(file, ky, ky_start, ky_count, grids_->ky_h))         ERR(retval);
  
  ///////////////////////////////////
  //                               //
  //        kx                     //
  //                               //
  ///////////////////////////////////
  kx_start[0] = 0;
  kx_count[0] = grids_->Nakx;

  if (retval = nc_put_vara(file, kx, kx_start, kx_count, grids_->kx_outh))      ERR(retval);

  ///////////////////////////////////
  //                               //
  //  geometric information        //
  //                               //
  ///////////////////////////////////
  if(geo_) {
    geo_start[0] = 0;
    geo_count[0] = grids_->Nz;

    if (retval = nc_put_vara(file, theta,    geo_start, geo_count, geo_->z_h))         ERR(retval);

    if (linked && false) {
      
      int Nx = grids_->Nx;
      int Ny = grids_->Ny;
      int Nz = grids_->Nz;
      
      int Naky = grids_->Naky;
      
      zkxky_count[0] = Nz;
      zkxky_count[1] = 1;
      zkxky_count[2] = 1;
      
      size_t size = sizeof(float)*Nz;
      theta_extended = (float*) malloc(size);
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
          if (retval = nc_put_vara(nc_geo, theta_x, zkxky_start, zkxky_count, theta_extended)) ERR(retval);
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
          if (retval = nc_put_vara(nc_geo, theta_x, zkxky_start, zkxky_count, theta_extended)) ERR(retval);
        }
      }
      if (theta_extended) free(theta_extended);
    }

    //  if (retval = nc_put_vara(file, theta,    geo_start, geo_count, geo_->z_h))         ERR(retval);
    if (retval = nc_put_vara(nc_geo, bmag,     geo_start, geo_count, geo_->bmag_h))      ERR(retval);
    if (retval = nc_put_vara(nc_geo, bgrad,    geo_start, geo_count, geo_->bgrad_h))     ERR(retval);
    if (retval = nc_put_vara(nc_geo, gbdrift,  geo_start, geo_count, geo_->gbdrift_h))   ERR(retval);
    if (retval = nc_put_vara(nc_geo, gbdrift0, geo_start, geo_count, geo_->gbdrift0_h))  ERR(retval);
    if (retval = nc_put_vara(nc_geo, cvdrift,  geo_start, geo_count, geo_->cvdrift_h))   ERR(retval);
    if (retval = nc_put_vara(nc_geo, cvdrift0, geo_start, geo_count, geo_->cvdrift0_h))  ERR(retval);
    if (retval = nc_put_vara(nc_geo, gds2,     geo_start, geo_count, geo_->gds2_h))      ERR(retval);
    if (retval = nc_put_vara(nc_geo, gds21,    geo_start, geo_count, geo_->gds21_h))     ERR(retval);  
    if (retval = nc_put_vara(nc_geo, gds22,    geo_start, geo_count, geo_->gds22_h))     ERR(retval);
    if (retval = nc_put_vara(nc_geo, grho,     geo_start, geo_count, geo_->grho_h))      ERR(retval);
    if (retval = nc_put_vara(nc_geo, jacobian, geo_start, geo_count, geo_->jacobian_h))  ERR(retval);

    if (retval = nc_put_var (nc_geo, ivar, &geo_->gradpar)) ERR(retval);
  }
  
  idum = pars_->boundary_option_periodic ? 1 : 0;
  if (retval = nc_put_var(file, periodic,      &idum))     ERR(retval);

  idum = pars_->local_limit ? 1 : 0;
  if (retval = nc_put_var(file, local_limit,   &idum))     ERR(retval);
}

NetCDF_ids::~NetCDF_ids() {

  if (primary)      cudaFreeHost ( primary   );
  if (secondary)    cudaFreeHost ( secondary );
  if (tertiary)     cudaFreeHost ( tertiary  );

  if (amom)         cudaFree ( amom );
  if (df)           cudaFree ( df   );
  if (favg)         cudaFree ( favg );
  
  if (red)          delete red;
  if (pot)          delete pot;
  if (ph2)          delete ph2;
  if (all_red)      delete all_red;
}

void NetCDF_ids::write_zonal_nc(nca *D, bool endrun) {
  int retval;

  if (D->write && endrun) {
    if (retval=nc_put_vara(D->file, D->idx,  D->start,      D->count,      &D->zonal)) ERR(retval);
  } 
  if (D->write_v_time)    {
    if (retval=nc_put_vara(D->file, D->time, D->time_start, D->time_count, &D->zonal)) ERR(retval);
  }
  D->increment_ts(); 
}

void NetCDF_ids::write_nc(nca *D, bool endrun) {
  int retval;

  if (D->write && endrun) {if (retval=nc_put_vara(D->file, D->idx,  D->start,      D->count,      D->cpu)) ERR(retval);} 
  if (D->write_v_time)    {if (retval=nc_put_vara(D->file, D->time, D->time_start, D->time_count, D->cpu)) ERR(retval);}
  D->increment_ts(); 
}

void NetCDF_ids::write_nc(nca *D, double data, bool endrun) {
  int retval;

  if (D->write && endrun) {if (retval=nc_put_vara(D->file, D->idx,  D->start,      D->count,      &data)) ERR(retval);} 
  if (D->write_v_time)    {if (retval=nc_put_vara(D->file, D->time, D->time_start, D->time_count, &data)) ERR(retval);}
  D->increment_ts(); 
}

void NetCDF_ids::write_nc(nca *D, float data, bool endrun) {
  int retval;

  if (D->write && endrun) {if (retval=nc_put_vara(D->file, D->idx,  D->start,      D->count,      &data)) ERR(retval);} 
  if (D->write_v_time)    {if (retval=nc_put_vara(D->file, D->time, D->time_start, D->time_count, &data)) ERR(retval);}
  D->increment_ts(); 
}

/*
void NetCDF_ids::pzt(MomentsG* G, Fields* f)
{
  int threads=256;
  int blocks=(grids_->NxNycNz+threads-1)/threads;
  
  primary[0]=0.; secondary[0]=0.; tertiary[0]=0.;
  
  Tbar <<<blocks, threads>>> (t_bar, G->G(), f->phi, geo_->kperp2);
  get_pzt <<<blocks, threads>>> (&primary[0], &secondary[0], &tertiary[0], f->phi, t_bar);
}
*/

void NetCDF_ids::write_Pky(float* P2, bool endrun)
{
  if (Pky -> write_v_time || (Pky -> write && endrun)) {
    int i = grids_->Nyc*grids_->Nspecies;

    pot->Sum(P2, Pky->data, PSPECTRA_ky);               CP_TO_CPU(Pky->tmp, Pky->data, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      for (int ik = 0; ik < grids_->Naky; ik++) {
	Pky->cpu[ik + is*grids_->Naky] = Pky->tmp[ik + is*grids_->Nyc];
      }
    }
    write_nc(Pky, endrun);      
  }
}

void NetCDF_ids::write_Pkx(float* P2, bool endrun)
{
  if (Pkx -> write_v_time || (Pkx -> write && endrun)) {
    int i = grids_->Nx*grids_->Nspecies;
    int NK = grids_->Nakx/2;
    int NX = grids_->Nx;
    
    pot->Sum(P2, Pkx->data, PSPECTRA_kx);               CP_TO_CPU(Pkx->tmp, Pkx->data, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      int it  = 0;
      int itp = it + NK;
      Pkx->cpu[itp + is*grids_->Nakx] = Pkx->tmp[it  + is*grids_->Nx];
      
      for (int it = 1; it < NK+1; it++) {
	int itp = NK + it;
	int itn = NK - it;
	int itm = NX - it;
	Pkx->cpu[itp + is*grids_->Nakx] = Pkx->tmp[it  + is*grids_->Nx];
	Pkx->cpu[itn + is*grids_->Nakx] = Pkx->tmp[itm + is*grids_->Nx];	
      }
    }  
    write_nc(Pkx, endrun);     
  }
}

void NetCDF_ids::write_Pz(float* P2, bool endrun)
{
  if (Pz -> write_v_time || (Pz -> write && endrun)) {
    int i = grids_->Nz*grids_->Nspecies;
    
    pot->Sum(P2, Pz->data, PSPECTRA_z);         CP_TO_CPU(Pz->cpu, Pz->data, sizeof(float)*i);
    write_nc(Pz, endrun);        
  }
}

void NetCDF_ids::write_Pkz(float* P2, bool endrun)
{
  if (Pkz -> write_v_time || (Pkz -> write && endrun)) {
    int i = grids_->Nz*grids_->Nspecies;     int Nz = grids_->Nz;
    
    pot->Sum(P2, Pkz->data, PSPECTRA_kz);       CP_TO_CPU(Pkz->tmp, Pkz->data, sizeof(float)*i);

    for (int is = 0; is < grids_->Nspecies; is++) {
      if (Nz>1) {
	for (int i = 0; i < Nz; i++) Pkz->cpu[i + is*Nz] = Pkz->tmp[ (i + Nz/2 + 1) % Nz + is*Nz ];
      } else {
	for (int i = 0; i < Nz; i++) Pkz->cpu[i + is*Nz] = Pkz->tmp[ i + is*Nz ];
      }
    }
    
    write_nc(Pkz, endrun);      
  }
}

void NetCDF_ids::write_Pkxky(float* P2, bool endrun)
{
  if (Pkxky -> write_v_time || (Pkxky -> write && endrun)) {

    int i = grids_->Nyc*grids_->Nx*grids_->Nspecies;

    int NK = grids_->Nakx/2;
    int NX = grids_->Nx; 
    
    pot->Sum(P2, Pkxky->data, PSPECTRA_kxky);
    CP_TO_CPU(Pkxky->tmp, Pkxky->data, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      int it = 0;
      int itp = it + NK;
      for (int ik = 0; ik < grids_->Naky; ik++) {
	int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc *grids_->Nx;
	Pkxky->cpu[Qp] = Pkxky->tmp[Rp];
      }	
      for (int it = 1; it < NK+1; it++) {
	int itp = NK + it;
	int itn = NK - it;
	int itm = NX - it;
	
	for (int ik = 0; ik < grids_->Naky; ik++) {

	  int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	  int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc * NX;

	  int Qn = itn + ik *grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	  int Rm = ik  + itm*grids_->Nyc  + is*grids_->Nyc * NX;

	  Pkxky->cpu[Qp] = Pkxky->tmp[Rp];
	  Pkxky->cpu[Qn] = Pkxky->tmp[Rm];
	}
      }
    }
    write_nc(Pkxky, endrun);     
  }
}

void NetCDF_ids::write_Wz(float *G2, bool endrun)
{
  if (Wz -> write_v_time || (Wz -> write && endrun)) {
    int i = grids_->Nz*grids_->Nspecies;
    
    red->Sum(G2, Wz->data, WSPECTRA_z);          CP_TO_CPU(Wz->cpu, Wz->data, sizeof(float)*i);
    write_nc(Wz, endrun);        
  }
}

void NetCDF_ids::write_Wkz(float *G2, bool endrun)
{
  if (Wkz -> write_v_time || (Wkz -> write && endrun)) {
    int i = grids_->Nz*grids_->Nspecies;     int Nz = grids_->Nz;
    
    red->Sum(G2, Wkz->data, WSPECTRA_kz);        CP_TO_CPU(Wkz->tmp, Wkz->data, sizeof(float)*i);

    for (int is = 0; is < grids_->Nspecies; is++) {
      if (Nz>1) {
	for (int i = 0; i < Nz; i++) Wkz->cpu[i+is*Nz] = Wkz->tmp[ (i + Nz/2 + 1) % Nz + is*Nz ];
      } else {
	for (int i = 0; i < Nz; i++) Wkz->cpu[i+is*Nz] = Wkz->tmp[ i + is*Nz ];
      }
    }
    
    write_nc (Wkz, endrun);      
  }
}

void NetCDF_ids::write_Ws(float* G2, bool endrun)
{
  if (Ws -> write_v_time) {
    red->Sum(G2, Ws->data, WSPECTRA_species);    CP_TO_CPU(Ws->cpu, Ws->data, sizeof(float)*grids_->Nspecies);
    write_nc(Ws, endrun);        

    if (Wtot -> write_v_time) {
      for (int is=0; is < grids_->Nspecies; is++) totW += Ws->cpu[is];
    }
  }
}

void NetCDF_ids::write_Wky(float* G2, bool endrun)
{
  if (Wky -> write_v_time || (Wky -> write && endrun)) {
    int i = grids_->Nyc*grids_->Nspecies;
    
    red->Sum(G2, Wky->data, WSPECTRA_ky);                CP_TO_CPU(Wky->tmp, Wky->data, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      for (int ik = 0; ik < grids_->Naky; ik++) {
	Wky->cpu[ik + is*grids_->Naky] = Wky->tmp[ik + is*grids_->Nyc];
      }
    }
    write_nc(Wky, endrun);      
  }
}

void NetCDF_ids::write_Wkx(float* G2, bool endrun)
{  
  if (Wkx -> write_v_time || (Wkx -> write && endrun)) {
    int i = grids_->Nx*grids_->Nspecies; 
    int NX = grids_->Nx;
    int NK = grids_->Nakx/2;
        
    red->Sum(G2, Wkx->data, WSPECTRA_kx);                CP_TO_CPU(Wkx->tmp, Wkx->data, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      int it = 0;
      int itp = it + NK;
      Wkx->cpu[itp + is*grids_->Nakx] = Wkx->tmp[it  + is*grids_->Nx];
      
      for (int it = 1; it < NK+1; it++) {
	int itp = NK + it;
	int itn = NK - it;
	int itm = NX - it;

	Wkx->cpu[itp + is*grids_->Nakx] = Wkx->tmp[it  + is*grids_->Nx];
	Wkx->cpu[itn + is*grids_->Nakx] = Wkx->tmp[itm + is*grids_->Nx];
      }
    }
    write_nc(Wkx, endrun);      
  }
}

void NetCDF_ids::write_Wkxky(float* G2, bool endrun)
{
  if (Wkxky -> write_v_time || (Wkxky -> write && endrun)) {
    int i = grids_->Nyc*grids_->Nx*grids_->Nspecies;  // int NK = (grids_->Nx-1)/3+1;

    int NK = grids_->Nakx/2;
    int NX = grids_->Nx; 
    
    red->Sum(G2, Wkxky->data, WSPECTRA_kxky);            CP_TO_CPU(Wkxky->tmp, Wkxky->data, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      int it = 0;
      int itp = it + NK;
      for (int ik = 0; ik < grids_->Naky; ik++) {     
	int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc *grids_->Nx;
	Wkxky->cpu[Qp] = Wkxky->tmp[Rp];
      }
      
      for (int it = 1; it < NK+1; it++) {
	int itp = NK + it; 
	int itn = NK - it; 
	int itm = NX - it; 
	for (int ik = 0; ik < grids_->Naky; ik++) {     

	  int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	  int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc * NX;

	  int Qn = itn + ik *grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	  int Rm = ik  + itm*grids_->Nyc  + is*grids_->Nyc * NX;

	  Wkxky->cpu[Qp] = Wkxky->tmp[Rp];
	  Wkxky->cpu[Qn] = Wkxky->tmp[Rm];
	}
      }
    }
    write_nc(Wkxky, endrun);  
  }
}

void NetCDF_ids::write_Wm(float* G2, bool endrun)
{
  if (Wm -> write_v_time || (Wm -> write && endrun)) {  
    int i = grids_->Nm*grids_->Nspecies;

    red-> Sum(G2, Wm -> data, WSPECTRA_m);         CP_TO_CPU(Wm->cpu, Wm->data, sizeof(float)*i);
    write_nc(Wm, endrun);       
  }
}

void NetCDF_ids::write_Wlm(float* G2, bool endrun)
{
  if (Wlm -> write_v_time || (Wlm -> write && endrun)) {  
    int i = grids_->Nmoms*grids_->Nspecies;
    
    red->Sum(G2, Wlm->data, WSPECTRA_lm);        CP_TO_CPU(Wlm->cpu, Wlm->data, sizeof(float)*i);
    write_nc(Wlm, endrun);      
  }
}

void NetCDF_ids::write_Wl(float* G2, bool endrun)
{
  if (Wl -> write_v_time || (Wl -> write && endrun)) {  
    int i = grids_->Nl*grids_->Nspecies;
    
    red->Sum(G2, Wl->data, WSPECTRA_l);          CP_TO_CPU(Wl->cpu, Wl->data, sizeof(float)*i);
    write_nc(Wl, endrun);        
  }
}

void NetCDF_ids::write_Ps(float* P2, bool endrun)
{
  if (Ps -> write_v_time) {
    
    pot->Sum(P2, Ps->data, PSPECTRA_species);     CP_TO_CPU(Ps->cpu, Ps->data, sizeof(float)*grids_->Nspecies);
    write_nc(Ps, endrun);          

    if (Wtot -> write_v_time) {
      totW = 0.;
      for (int is=0; is < grids_->Nspecies; is++) totW += Ps->cpu[is];
    }
  }
    
}

void NetCDF_ids::write_Aky(float* P2, bool endrun)
{
  if (Aky -> write_v_time || (Aky -> write && endrun)) {
    int i = grids_->Naky;
    
    ph2->Sum(P2, Aky->data, ASPECTRA_ky);       CP_TO_CPU(Aky->cpu, Aky->data, sizeof(float)*i);
    write_nc(Aky, endrun);     
  }
}

void NetCDF_ids::write_Az(float* P2, bool endrun)
{
  if (Az -> write_v_time || (Az -> write && endrun)) {
    int i = grids_->Nz;
    
    ph2->Sum(P2, Az->data, ASPECTRA_z);         CP_TO_CPU(Az->cpu, Az->data, sizeof(float)*i);
    write_nc(Az, endrun);       
  }
}

void NetCDF_ids::write_Akz(float* P2, bool endrun)
{
  if (Akz -> write_v_time || (Akz -> write && endrun)) {
    int Nz = grids_->Nz;
    
    ph2->Sum(P2, Akz->data, ASPECTRA_kz);       CP_TO_CPU(Akz->tmp, Akz->data, sizeof(float)*Nz);

    if (Nz>1) {
      for (int i = 0; i < Nz; i++) Akz->cpu[i] = Akz->tmp[ (i + Nz/2 + 1) % Nz ];
    } else {
      for (int i = 0; i < Nz; i++) Akz->cpu[i] = Akz->tmp[ i ];
    }    
    write_nc(Akz, endrun);      
  }
}

void NetCDF_ids::write_Akx(float* P2, bool endrun)
{
  if (Akx -> write_v_time || (Akx -> write && endrun)) {
    int NX = grids_->Nx;
    int NK = grids_->Nakx/2;
    
    ph2->Sum(P2, Akx->data, ASPECTRA_kx);               CP_TO_CPU(Akx->tmp, Akx->data, sizeof(float)*NX);
    
    int it = 0;
    int itp = it + NK;
    Akx->cpu[itp] = Akx->tmp[it ];;
    
    for (int it = 1; it < NK+1; it++) {
      int itp = NK + it;
      int itn = NK - it;
      int itm = NX - it;
      
      Akx->cpu[itp] = Akx->tmp[it ];;
      Akx->cpu[itn] = Akx->tmp[itm];;
    }
    write_nc(Akx, endrun);      
  }
}

void NetCDF_ids::write_Akxky(float* P2, bool endrun)
{
  if (Akxky -> write_v_time || (Akxky -> write && endrun)) {
    int i = grids_->Nyc*grids_->Nx; int NK = grids_->Nakx/2;  int NX = grids_->Nx;
    
    ph2->Sum(P2, Akxky->data, ASPECTRA_kxky);    CP_TO_CPU(Akxky->tmp, Akxky->data, sizeof(float)*i);
    
    int it = 0;
    int itp = it + NK;
    for (int ik = 0; ik < grids_->Naky; ik++) {
      int Qp = itp + ik*grids_->Nakx ;
      int Rp = ik  + it*grids_->Nyc  ;
      Akxky->cpu[Qp] = Akxky->tmp[Rp];
    }
    
    for (int it = 1; it < NK+1; it++) {
      int itp = NK + it;
      int itn = NK - it;
      int itm = NX - it;

      for (int ik = 0; ik < grids_->Naky; ik++) {

	int Qp = itp + ik*grids_->Nakx ;
	int Rp = ik  + it*grids_->Nyc  ;
	
	int Qn = itn + ik *grids_->Nakx ;
	int Rm = ik  + itm*grids_->Nyc  ;

	Akxky->cpu[Qp] = Akxky->tmp[Rp];
	Akxky->cpu[Qn] = Akxky->tmp[Rm];
      }
    }
    write_nc(Akxky, endrun);  
  }
}

void NetCDF_ids::write_As(float *P2, bool endrun)
{
  if (As -> write_v_time) {  
    ph2->Sum(P2, As->data, ASPECTRA_species);    CP_TO_CPU (As->cpu, As->data, sizeof(float));
    write_nc(As, endrun);         

    if (Wtot -> write_v_time) totW += *As->cpu;
  }
}

void NetCDF_ids::write_Q (float* Q, bool endrun)
{
  if (qs -> write_v_time) {
    all_red->Sum(Q, qs->data);                   CP_TO_CPU (qs->cpu, qs->data, sizeof(float)*grids_->Nspecies);
    write_nc(qs, endrun);       

    for (int is=0; is<grids_->Nspecies; is++) printf ("%e \t ",qs->cpu[is]);
    printf("\n");
  }
}

void NetCDF_ids::write_omg(cuComplex *W, bool endrun)
{
  CP_TO_CPU (omg->z_tmp, W, sizeof(cuComplex)*grids_->NxNyc);

  reduce2k(omg->cpu, omg->z_tmp);
  write_nc(omg, endrun);
}

void NetCDF_ids::write_Wtot()
{
  if (Wtot -> write_v_time) {  write_nc(Wtot, totW);        totW = 0.;}
}

void NetCDF_ids::close_nc_file() {
  int retval;
  if (retval = nc_close(  file)) ERR(retval);
  if (pars_->write_xymom) {
    if (retval = nc_close(pars_->nczid)) ERR(retval);
  }
}

void NetCDF_ids::write_moment(nca *D, cuComplex *f, float* vol_fac) {
  
  //
  // If D->dy = true, take one derivative in y
  // If D->dx = true, take one derivative in x
  // If D->d2x = true, take two derivatives in x
  // Multiply by D->adj
  // If D->xydata = true, output is function of (x, y) with zonal component subtracted
  // If D->xdata = true, output is function of x only
  // If D->scalar = true, output is sqrt (sum_kx <<f**2(kx)>>)
  
  if (!D->write_v_time) return;

  cuComplex zz = make_cuComplex(0., 0.);  setval loop_R (amom, zz, D->N_);
  
  // Perform any desired derivatives
  if (D->d2x) {    d2x Gmom (amom, f, grids_->kx);   } else
  if (D->dx)  {    ddx Gmom (amom, f, grids_->kx);   } else
  if (D->dy)  {    ddy Gmom (amom, f, grids_->ky);   } else            
  if (D->mdy) {   mddy Gmom (amom, f, grids_->ky);   } else            {
    CP_ON_GPU (amom, f, sizeof(cuComplex)*grids_->NxNycNz);
  }

  // Hermite -> physical moments
  if (D->adj > 1.0) {
    scale_singlemom_kernel loop_R (amom, amom, D->adj); // loop_R has more elements than required but it is safe
  }
  
  if (D->xydata) {
    fieldlineaverage GFLA (favg, df, amom, vol_fac); // D->tmp = <<f>>(kx), df = f - <<f>>
    grad_phi -> C2R(df, D->data);
    xytranspose loop_xy (D->data, D->tmp_d); // For now, take the first plane in the z-direction by default
    CP_TO_CPU(D->cpu, D->tmp_d, sizeof(float)*D->Nwrite_);
    write_nc(D);
    return;
  }
   
  grad_phi -> C2R(amom, D->data);
  yzavg loop_x (D->data, D->tmp_d, vol_fac);     
  CP_TO_CPU (D->cpu, D->tmp_d, sizeof(float)*D->Nwrite_);

  if (D->xdata) {
    write_nc(D);
    return;
  }
  
  if (D->scalar) {
    D->zonal = 0.;
    for (int idx = 0; idx<grids_->Nx; idx++) D->zonal += D->cpu[idx] * D->cpu[idx];
    D->zonal = sqrtf(D->zonal/((float) grids_->Nx));
    write_zonal_nc(D);
    return;
  }    
}

void NetCDF_ids::write_ks_data(nca *D, cuComplex *G) {
  if (!D->write_v_time) return;

  grad_perp->C2R(G, D->data);
  CP_TO_CPU (D->cpu, D->data, sizeof(float)*D->N_);
  write_nc(D);
}

void NetCDF_ids::write_ks_data(nca *D, float *G) {
  if (!D->write_v_time) return;

  CP_TO_CPU (D->cpu, G, sizeof(float)*D->N_);
  write_nc(D);
}

// condense a (ky,kx) object for netcdf output, taking into account the mask
// and changing the type from cuComplex to float
void NetCDF_ids::reduce2k(float *fk, cuComplex* f) {
  
  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;

  int NK = grids_->Nakx/2;
  
  int it = 0;
  int itp = it + NK;
  for (int ik=0; ik<Naky; ik++) {
    int Qp = itp + ik*Nakx;
    int Rp = ik  + it*Nyc;
    fk[2*Qp  ] = f[Rp].x;
    fk[2*Qp+1] = f[Rp].y;
  }

  for (int it = 1; it < NK+1; it++) {
    int itp = NK + it;
    int itn = NK - it;
    int itm = Nx - it;
    for (int ik=0; ik<Naky; ik++) {
      int Qp = itp + ik*Nakx;
      int Rp = ik  + it*Nyc;

      int Qn = itn + ik*Nakx;
      int Rm = ik  + itm*Nyc;
      fk[2*Qp  ] = f[Rp].x;
      fk[2*Qp+1] = f[Rp].y;

      fk[2*Qn  ] = f[Rm].x;
      fk[2*Qn+1] = f[Rm].y;
    }
  }  
}



