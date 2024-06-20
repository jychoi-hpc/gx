#include "geometry.h"
#include "vmec_variables.h"
#include "geometric_coefficients.h"
#define GGEO <<< dimGrid, dimBlock >>>

#include "geometry_modules/vmec/include/solver.h"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// wrapper for initializing a particular Geometry class
Geometry* init_geo(Parameters* pars, Grids* grids)
{
  Geometry* geo;

  int igeo = pars->igeo;
  std::string geo_option = pars->geo_option;
  if(grids->iproc==0) DEBUGPRINT("Initializing geometry...\n");
  if(geo_option=="s-alpha" || geo_option=="slab" || geo_option=="const-curv" || igeo==0) {
    if(geo_option=="slab") pars->slab = true;
    if(geo_option=="const-curv") pars->const_curv = true;

    geo = new S_alpha_geo(pars, grids);
    if(grids->iproc==0) CUDA_DEBUG("Initializing geometry s_alpha: %s \n");
    if(igeo==0) {
      if(grids->iproc==0) printf(ANSI_COLOR_RED);
      if(grids->iproc==0) printf("Warning: igeo is being deprecated. Use geo_option=\"s-alpha\" instead of igeo=0.\n"); 
      if(grids->iproc==0) printf(ANSI_COLOR_RESET);
    }
  }
  else if(geo_option=="miller") {
    // call python geometry module to write an eik.out geo file
    // GX_PATH is defined at compile time via a -D flag
    pars->geofilename = std::string(pars->run_name) + ".eik.out";
    if(grids->iproc == 0) {
      char command[300];
      sprintf(command, "python %s/geometry_modules/miller/gx_geo.py %s.in %s > %s.gx_geo.log", GX_PATH, pars->run_name, pars->geofilename.c_str(), pars->run_name);
      printf("Using Miller geometry. Generating geometry file %s with\n> %s\n", pars->geofilename.c_str(), command);
      system(command);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // now read the eik file that was generated
    geo = new Eik_geo(pars, grids);
    if(grids->iproc==0) CUDA_DEBUG("Initializing miller geometry: %s \n");
  } 
  else if(geo_option=="vmec_c") {
    bool usenc;
    if(grids->iproc == 0) {
      char nml_file[512];
      strcpy (nml_file, pars->run_name);
      strcat (nml_file, ".in");
      VMEC_variables *vmec = new VMEC_variables(nml_file);
      Geometric_coefficients *vmec_geo = new Geometric_coefficients(nml_file, vmec);
      usenc = vmec_geo->usenc;
      if(usenc) {
        pars->geofilename = vmec_geo->outnc_name;
      }
      else {
        pars->geofilename = vmec_geo->outfile_name;
      }
      delete vmec_geo;
      delete vmec;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int size = pars->geofilename.size();
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(grids->iproc != 0) pars->geofilename.resize(size);
    MPI_Bcast((void*) pars->geofilename.c_str(), size, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*) &usenc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // now read the eik file that was generated
    if (usenc) {
      geo = new geo_nc(pars, grids);
    }
    else {
      geo = new Eik_geo(pars, grids);
    }
  }
  else if(geo_option=="vmec" || geo_option == "pyvmec") {
    // call python geometry module to write an eik.out geo file
    // GX_PATH is defined at compile time via a -D flag
    pars->geofilename = std::string(pars->run_name) + ".eik.nc";
    if(grids->iproc == 0) {
      char command[300];
      sprintf(command, "python %s/geometry_modules/pyvmec/gx_geo_vmec.py %s.in %s", GX_PATH, pars->run_name, pars->geofilename.c_str(), pars->run_name);
      printf("Using vmec geometry. Generating geometry file %s with\n> %s\n", pars->geofilename.c_str(), command);
      system(command);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // now read the eik nc file that was generated
    geo = new geo_nc(pars, grids);
  } 
  else if(geo_option=="desc") {
    // call python geometry module to write an eik.out geo file
    // GX_PATH is defined at compile time via a -D flag
    pars->geofilename = std::string(pars->run_name) + ".eik.out";
    if(grids->iproc == 0) {
      char command[300];
      sprintf(command, "python %s/geometry_modules/desc/gx_desc_geo.py %s.in %s > %s.gx_geo.log", GX_PATH, pars->run_name, pars->geofilename.c_str(), pars->run_name);
      printf("Using DESC geometry. Generating geometry file %s with\n> %s\n", pars->geofilename.c_str(), command);
      system(command);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // now read the eik file that was generated
    geo = new Eik_geo(pars, grids);
    if(grids->iproc==0) CUDA_DEBUG("Initializing miller geometry: %s \n");
  } 
#ifdef GS2_PATH
  else if(geo_option=="gs2_geo") {
    if(grids->iproc == 0) {
      // write an eik.in file
      write_eiktest_in(pars, grids);
      char command[300];
      sprintf(command, "unset SLURM_NODELIST; %s/bin/eiktest %s.eik.in > eiktest.log", GS2_PATH, pars->run_name);
      printf("Generating geometry file %s.eik.out.nc with\n> %s\n", pars->run_name, command);
      int err = system(command);
      if (err) {
        printf("ERROR in system command\n", err);
      }
      
    }
    MPI_Barrier(MPI_COMM_WORLD);
    pars->geofilename = std::string(pars->run_name) + ".eik.out.nc"; // need this on all procs

    // now read the netcdf file that was generated
    geo = new geo_nc(pars, grids);
  }
#else
  else if(geo_option=="gs2_geo") {
    printf("Error: gx was not compiled with gs2 geometry support.\n");
    exit(1);
  }
#endif
  else if(geo_option=="eik" || igeo==1) {
    // read already existing eik.out geo file (don't run any geometry module) 
    geo = new Eik_geo(pars, grids);
    if(grids->iproc==0) CUDA_DEBUG("Initializing geometry from eik.out file: %s \n", pars->geofilename.c_str());
    if(igeo==1) {
      if(grids->iproc==0) printf(ANSI_COLOR_RED);
      if(grids->iproc==0) printf("Warning: igeo is being deprecated. Use geo_option=\"eik\" instead of igeo=1.\n"); 
      if(grids->iproc==0) printf(ANSI_COLOR_RESET);
    }
  }
  else if(geo_option=="nc" || igeo==2) {
    geo = new geo_nc(pars, grids);
    if(grids->iproc==0) CUDA_DEBUG("Initializing geometry from NetCDF file: %s \n");
    if(igeo==2) {
      if(grids->iproc==0) printf(ANSI_COLOR_RED);
      if(grids->iproc==0) printf("Warning: igeo is being deprecated. Use geo_option=\"nc\" instead of igeo=2.\n"); 
      if(grids->iproc==0) printf(ANSI_COLOR_RESET);
    }
  } 
  else {
    if(grids->iproc==0) printf("Error: geo_option = \"%s\" is invalid.\n", geo_option.c_str());
    if(grids->iproc==0) printf("Options are: geo_option = {\"s-alpha\", \"miller\", \"vmec\", \"eik\", \"nc\", \"slab\", \"const-curv\"}\n");
    exit(1);
  }
  return geo;
}

void write_eiktest_in(Parameters *pars, Grids *grids) {
  FILE *fptr;

  char fname[300];
  sprintf(fname, "%s.eik.in", pars->run_name);
  fptr = fopen(fname, "w");
  fprintf(fptr, "&stuff\n");
  fprintf(fptr, " ntheta = %d\n", pars->nz_in/(2*pars->nperiod-1));
  fprintf(fptr, " nperiod = %d\n", pars->nperiod);
  fprintf(fptr, " geoType = %d\n", pars->geoType);
  fprintf(fptr, " rmaj = %.9e\n", pars->rmaj);
  fprintf(fptr, " akappri = %.9e\n", pars->akappri);
  fprintf(fptr, " akappa = %.9e\n", pars->akappa);
  fprintf(fptr, " shift = %.9e\n", pars->shift);
  fprintf(fptr, " equal_arc = .true. ! this is required for GX\n");
  fprintf(fptr, " rhoc = %.9e\n", pars->rhoc);
  fprintf(fptr, " itor = 1\n");
  fprintf(fptr, " qinp = %.9e\n", pars->qsf);
  fprintf(fptr, " iflux = %d\n", pars->iflux);
  fprintf(fptr, " delrho = %f\n", pars->delrho);
  fprintf(fptr, " tri = %.9e\n", pars->tri);
  fprintf(fptr, " bishop = %d\n", pars->bishop);
  fprintf(fptr, " irho = %d\n", pars->irho);
  fprintf(fptr, " isym = %d\n", pars->isym);
  fprintf(fptr, " tripri = %.9e\n", pars->tripri);
  fprintf(fptr, " R_geo = %.9e\n", pars->r_geo);
  fprintf(fptr, " eqfile = '%s'\n", pars->eqfile.c_str());
  if(pars->efit_eq)
    fprintf(fptr, " efit_eq = .true.\n");
  else
    fprintf(fptr, " efit_eq = .false.\n");
  if(pars->dfit_eq)
    fprintf(fptr, " dfit_eq = .true.\n");
  else
    fprintf(fptr, " dfit_eq = .false.\n");
  if(pars->gen_eq)
    fprintf(fptr, " gen_eq = .true.\n");
  else
    fprintf(fptr, " gen_eq = .false.\n");
  if(pars->ppl_eq)
    fprintf(fptr, " ppl_eq = .true.\n");
  else
    fprintf(fptr, " ppl_eq = .false.\n");
  if(pars->local_eq)
    fprintf(fptr, " local_eq = .true.\n");
  else
    fprintf(fptr, " local_eq = .false.\n");
  if(pars->idfit_eq)
    fprintf(fptr, " idfit_eq = .true.\n");
  else
    fprintf(fptr, " idfit_eq = .false.\n");
  if(pars->chs_eq)
    fprintf(fptr, " chs_eq = .true.\n");
  else
    fprintf(fptr, " chs_eq = .false.\n");
  if(pars->transp_eq)
    fprintf(fptr, " transp_eq = .true.\n");
  else
    fprintf(fptr, " transp_eq = .false.\n");
  if(pars->gs2d_eq)
    fprintf(fptr, " gs2d_eq = .true.\n");
  else
    fprintf(fptr, " gs2d_eq = .false.\n");
  fprintf(fptr, " s_hat_input = %.9e\n", pars->s_hat_input);
  fprintf(fptr, " p_prime_input = %.9e\n", pars->p_prime_input);
  fprintf(fptr, " beta_prime_input = %.9e\n", pars->beta_prime_input);
  fprintf(fptr, " invLp_input = %.9e\n", pars->invLp_input);
  fprintf(fptr, " alpha_input = %.9e\n", pars->alpha_input);
  fprintf(fptr, "/\n");
  fclose(fptr);
}

Geometry::Geometry() {

  operator_arrays_allocated_=false;

  z_h          = nullptr;  gbdrift_h  = nullptr;  grho_h     = nullptr;  cvdrift_h  = nullptr;
  bmag_h       = nullptr;  bmagInv_h  = nullptr;  bgrad_h    = nullptr;  gds2_h     = nullptr;
  gds21_h      = nullptr;  gds22_h    = nullptr;  cvdrift0_h = nullptr;  gbdrift0_h = nullptr;
  jacobian_h   = nullptr;

  z            = nullptr;  gbdrift    = nullptr;  grho       = nullptr;  cvdrift    = nullptr;
  bmag         = nullptr;  bmagInv    = nullptr;  bgrad      = nullptr;  gds2       = nullptr;
  gds21        = nullptr;  gds22      = nullptr;  cvdrift0   = nullptr;  gbdrift0   = nullptr;
  jacobian     = nullptr;

  gradpar_arr  = nullptr;  Rplot      = nullptr;  Zplot      = nullptr;  aplot      = nullptr;
  Xplot        = nullptr;  Yplot      = nullptr;  Rprime     = nullptr;  Zprime     = nullptr;
  aprime       = nullptr;  deltaFL    = nullptr; 
  
  bmag_complex = nullptr;  bgrad_temp = nullptr; 
    
  // operator arrays
  kperp2       = nullptr;  omegad     = nullptr;  cv_d       = nullptr;   gb_d      = nullptr;
  kperp2_h     = nullptr;
  m0           = nullptr;  deltaKx    = nullptr;  ftwist     = nullptr;

}

Geometry::~Geometry() {
  if (z)         cudaFree(z);
  if (bmag)      cudaFree(bmag);
  if (bmagInv)   cudaFree(bmagInv);
  if (bgrad)     cudaFree(bgrad);
  if (gds2)      cudaFree(gds2);	
  if (gds21)     cudaFree(gds21);	
  if (gds22)     cudaFree(gds22);	
  if (gbdrift)   cudaFree(gbdrift);	
  if (gbdrift0)  cudaFree(gbdrift0);	
  if (cvdrift)   cudaFree(cvdrift);	
  if (cvdrift0)  cudaFree(cvdrift0);	
  if (grho)      cudaFree(grho);	
  if (jacobian)  cudaFree(jacobian);	

  if (z_h)         free(z_h);
  if (bmag_h)      free(bmag_h);
  if (bmagInv_h)   free(bmagInv_h);
  if (bgrad_h)     free(bgrad_h);
  if (gds2_h)      free(gds2_h);	
  if (gds21_h)     free(gds21_h);	
  if (gds22_h)     free(gds22_h);	
  if (gbdrift_h)   free(gbdrift_h);	
  if (gbdrift0_h)  free(gbdrift0_h);	
  if (cvdrift_h)   free(cvdrift_h);	
  if (cvdrift0_h)  free(cvdrift0_h);	
  if (grho_h)      free(grho_h);	
  if (jacobian_h)  free(jacobian_h);	

  if(operator_arrays_allocated_) {
    if (kperp2) cudaFree(kperp2);
    if (omegad) cudaFree(omegad);
    if (cv_d)   cudaFree(cv_d);
    if (gb_d)   cudaFree(gb_d);
    if (m0)     cudaFree(m0);
    if (deltaKx) cudaFree(deltaKx);
    if (ftwist) cudaFree(ftwist);
  }
}

S_alpha_geo::S_alpha_geo(Parameters *pars, Grids *grids) 
{
  int Nz = grids->Nz;
  float theta;
  operator_arrays_allocated_=false;
  size_t size = sizeof(float)*Nz;
  z_h = (float*) malloc (size);
  bmag_h = (float*) malloc (size);
  bmagInv_h = (float*) malloc (size);
  bgrad_h = (float*) malloc (size);
  gds2_h = (float*) malloc (size);
  gds21_h = (float*) malloc (size);
  gds22_h = (float*) malloc (size);
  gbdrift_h = (float*) malloc (size);
  gbdrift0_h = (float*) malloc (size);
  cvdrift_h = (float*) malloc (size);
  cvdrift0_h = (float*) malloc (size);
  grho_h = (float*) malloc (size);
  jacobian_h = (float*) malloc (size);

  // kperp2_h = (float*) malloc(sizeof(float)*grids->NxNycNz);
  
  cudaMalloc ((void**) &z, size);
  cudaMalloc ((void**) &bmag, size);
  cudaMalloc ((void**) &bmagInv, size);
  cudaMalloc ((void**) &bgrad, size);
  cudaMalloc ((void**) &gds2, size);
  cudaMalloc ((void**) &gds21, size);
  cudaMalloc ((void**) &gds22, size);
  cudaMalloc ((void**) &gbdrift, size);
  cudaMalloc ((void**) &gbdrift0, size);
  cudaMalloc ((void**) &cvdrift, size);
  cudaMalloc ((void**) &cvdrift0, size);
  cudaMalloc ((void**) &grho, size);
  cudaMalloc ((void**) &jacobian, size);
  
  qsf = pars->qsf;
  float beta_e = pars->beta;
  rmaj = pars->rmaj;
  specie* species = pars->species_h;
  
  gradpar = (float) abs(1./(qsf*rmaj));
  zero_shat_ = pars->zero_shat;
  shat = pars->shat;
  drhodpsi = pars->drhodpsi = 1.; 
  kxfac = pars->kxfac = 1.;
  
  if(pars->shift < 0.) {
    pars->shift = 0.;
    for(int s=0; s<pars->nspec_in; s++) { 
      pars->shift += qsf*qsf*rmaj*beta_e*
	(species[s].temp/species[pars->nspec_in-1].temp)*
	(species[s].tprim + species[s].fprim);
    }
  }
  shift = pars->shift;
 
  if(grids->iproc==0) DEBUGPRINT("\n\n Using s-alpha geometry: \n\n");
  for(int k=0; k<Nz; k++) {
    z_h[k] = 2.*M_PI *pars->Zp *(k-Nz/2)/Nz;
    if(grids->iproc==0) DEBUGPRINT("theta[%d] = %f \n",k,z_h[k]);
    if(pars->local_limit) {z_h[k] = 0.;} // outboard-midplane
    theta = z_h[k];
    
    bmag_h[k] = 1. / (1. + pars->eps * cos(theta));
    bgrad_h[k] = gradpar * pars->eps * sin(theta) * bmag_h[k]; 

    gds2_h[k] = 1. + pow((shat * theta - shift * sin(theta)), 2);
    gds21_h[k] = -shat * (shat * theta - shift * sin(theta));
    gds22_h[k] = pow(shat,2);

    gbdrift_h[k] = 1. / rmaj * (cos(theta) + (shat * theta - shift * sin(theta)) * sin(theta));
    cvdrift_h[k] = gbdrift_h[k];

    gbdrift0_h[k] = - shat * sin(theta) / rmaj;
    cvdrift0_h[k] = gbdrift0_h[k];

    grho_h[k] = 1;

    if(pars->const_curv) {
      cvdrift_h[k] = 1./rmaj;
      gbdrift_h[k] = 1./rmaj;
      cvdrift0_h[k] = 0.;
      gbdrift0_h[k] = 0.;
    }
    
    if(pars->slab) {
      cvdrift_h[k] = 0.;
      gbdrift_h[k] = 0.;       
      cvdrift0_h[k] = 0.;
      gbdrift0_h[k] = 0.;
      bgrad_h[k] = 0.;
      bmag_h[k] = 1.;
      gradpar = 1.;
      if (pars->z0 > 0.) gradpar = 1./pars->z0;
      printf("z0: %f   ",pars->z0);
      if (pars->zero_shat) {
	gds21_h[k] = 0.0;
	gds22_h[k] = 1.0;
	shat = pars->shat = 0.0;	
      }
    }
    if(pars->local_limit) { z_h[k] = 2 * M_PI * pars->Zp * (k-Nz/2) / Nz; gradpar = 1.; }

    // calculate these derived coefficients after slab overrides
    bmagInv_h[k] = 1./bmag_h[k];
    jacobian_h[k] = 1. / abs(drhodpsi * gradpar * bmag_h[k]);
  }  

  CP_TO_GPU (z,        z_h,        size);
  CP_TO_GPU (gbdrift,  gbdrift_h,  size);
  CP_TO_GPU (grho,     grho_h,     size);
  CP_TO_GPU (cvdrift,  cvdrift_h,  size);
  CP_TO_GPU (bmag,     bmag_h,     size);
  CP_TO_GPU (bmagInv,  bmagInv_h,  size);
  CP_TO_GPU (bgrad,    bgrad_h,    size);
  CP_TO_GPU (gds2,     gds2_h,     size);
  CP_TO_GPU (gds21,    gds21_h,    size);
  CP_TO_GPU (gds22,    gds22_h,    size);
  CP_TO_GPU (cvdrift0, cvdrift0_h, size);
  CP_TO_GPU (gbdrift0, gbdrift0_h, size);
  CP_TO_GPU (jacobian, jacobian_h, size);

  cudaDeviceSynchronize();
  
  // initialize the drift arrays and kperp2
  initializeOperatorArrays(pars, grids);
}

Gs2_geo::Gs2_geo() {

}

geo_nc::geo_nc(Parameters *pars, Grids *grids)
{
  if(grids->iproc==0) printf("READING NC GEO\n");
  operator_arrays_allocated_=false;
  size_t size = sizeof(float)*grids->Nz;
  size_t dsize = sizeof(double)*(grids->Nz+1);

  char stra[NC_MAX_NAME+1];
  char strb[513];
  strcpy(strb, pars->geofilename.c_str());

    // open the netcdf file
  int retval;
  int ncgeo;
  if (retval = nc_open(strb, NC_NOWRITE, &ncgeo)) { printf("file: %s \n",strb); ERR(retval);}

  // get the array dimensions
  int id_z;
  size_t N; 
  if (retval = nc_inq_dimid(ncgeo, "z",  &id_z))
    if (retval = nc_inq_dimid(ncgeo, "nt",  &id_z))       ERR(retval);
  if (retval = nc_inq_dim  (ncgeo, id_z, stra, &N))    ERR(retval);

  // do basic sanity check
  if (grids->Nz != (int) N-1) {
    if(grids->iproc==0) printf("Number of points along the field line in geometry file %d does not match input %d \n", N-1, grids->Nz);
    exit (1);
  }

  // allocate space for variables on the CPU
  double* dtmp = (double*) malloc(dsize);
  double* nc_z_h = (double*) malloc (dsize);
  double* nc_bmag_h = (double*) malloc (dsize);
  double* nc_bmagInv_h = (double*) malloc (dsize);
  double* nc_gds2_h = (double*) malloc (dsize);
  double* nc_gds21_h = (double*) malloc (dsize);
  double* nc_gds22_h = (double*) malloc (dsize);
  double* nc_gbdrift_h = (double*) malloc (dsize);
  double* nc_gbdrift0_h = (double*) malloc (dsize);
  double* nc_cvdrift_h = (double*) malloc (dsize);
  double* nc_cvdrift0_h = (double*) malloc (dsize);
  double* nc_grho_h = (double*) malloc (dsize);
  double* nc_gradpar_h = (double*) malloc (dsize);
  double* nc_jacobian_h = (double*) malloc (dsize);

  z_h = (float*) malloc (size);
  bmag_h = (float*) malloc (size);
  bmagInv_h = (float*) malloc (size);
  gds2_h = (float*) malloc (size);
  gds21_h = (float*) malloc (size);
  gds22_h = (float*) malloc (size);
  gbdrift_h = (float*) malloc (size);
  gbdrift0_h = (float*) malloc (size);
  cvdrift_h = (float*) malloc (size);
  cvdrift0_h = (float*) malloc (size);
  grho_h = (float*) malloc (size);
  jacobian_h = (float*) malloc (size);

  // read the data with nc_get_var
  int id;
  if (retval = nc_inq_varid(ncgeo, "theta", &id))        ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) nc_z_h[n] = dtmp[n];
  
  if (retval = nc_inq_varid(ncgeo, "bmag", &id))         ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) nc_bmag_h[n] = dtmp[n];
  for (int n=0; n<N; n++) nc_bmagInv_h[n] = 1./nc_bmag_h[n];

  if (retval = nc_inq_varid(ncgeo, "gradpar", &id))      ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) nc_gradpar_h[n] = dtmp[n];
  if(nc_gradpar_h[0] != nc_gradpar_h[N/2]) {
    if(grids->iproc==0) printf("Error: GX requires an equal-arc theta coordinate, so that gradpar = const.\nFor gs2 geometry module, use equal_arc = true. Exiting...\n");
    fflush(stdout);
    abort();
  } else {
    gradpar = nc_gradpar_h[0];
  }

  if (retval = nc_inq_varid(ncgeo, "grho", &id))         ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) nc_grho_h[n] = dtmp[n];
  
  if (retval = nc_inq_varid(ncgeo, "gds2", &id))         ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) nc_gds2_h[n] = dtmp[n];
  
  if (retval = nc_inq_varid(ncgeo, "gds21", &id))        ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) nc_gds21_h[n] = dtmp[n];
  
  if (retval = nc_inq_varid(ncgeo, "gds22", &id))        ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) nc_gds22_h[n] = dtmp[n];
  
  if (retval = nc_inq_varid(ncgeo, "gbdrift", &id))      ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) nc_gbdrift_h[n] = dtmp[n] / 2.0;
  
  if (retval = nc_inq_varid(ncgeo, "gbdrift0", &id))     ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) nc_gbdrift0_h[n] = dtmp[n] / 2.0;
  
  if (retval = nc_inq_varid(ncgeo, "cvdrift", &id))      ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) nc_cvdrift_h[n] = dtmp[n] / 2.0;
  
  if (retval = nc_inq_varid(ncgeo, "cvdrift0", &id))     ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) nc_cvdrift0_h[n] = dtmp[n] / 2.0;
  
  free(dtmp);

  // interpolate to equally-spaced theta grid
  for(int k=0; k<grids->Nz; k++) {
    z_h[k] = 2.*M_PI *pars->Zp *(k-grids->Nz/2)/grids->Nz;
  }
  int ntgrid = N/2;

  interp_to_new_grid(nc_bmag_h, bmag_h, nc_z_h, z_h, grids->Nz+1, grids->Nz);
  interp_to_new_grid(nc_bmagInv_h, bmagInv_h, nc_z_h, z_h, grids->Nz+1, grids->Nz);
  interp_to_new_grid(nc_gds2_h, gds2_h, nc_z_h, z_h, grids->Nz+1, grids->Nz);
  interp_to_new_grid(nc_gds21_h, gds21_h, nc_z_h, z_h, grids->Nz+1, grids->Nz);
  interp_to_new_grid(nc_gds22_h, gds22_h, nc_z_h, z_h, grids->Nz+1, grids->Nz);
  interp_to_new_grid(nc_gbdrift_h, gbdrift_h, nc_z_h, z_h, grids->Nz+1, grids->Nz);
  interp_to_new_grid(nc_gbdrift0_h, gbdrift0_h, nc_z_h, z_h, grids->Nz+1, grids->Nz);
  interp_to_new_grid(nc_cvdrift_h, cvdrift_h, nc_z_h, z_h, grids->Nz+1, grids->Nz);
  interp_to_new_grid(nc_cvdrift0_h, cvdrift0_h, nc_z_h, z_h, grids->Nz+1, grids->Nz);
  interp_to_new_grid(nc_grho_h, grho_h, nc_z_h, z_h, grids->Nz+1, grids->Nz);
  interp_to_new_grid(nc_jacobian_h, jacobian_h, nc_z_h, z_h, grids->Nz+1, grids->Nz);

  double stmp; 
  
  if (retval = nc_inq_varid(ncgeo, "drhodpsi", &id))     ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
  drhodpsi = pars->drhodpsi = (float) stmp;
  
  for (int n=0; n<N; n++) jacobian_h[n] = 1./abs(drhodpsi*gradpar*bmag_h[n]);
      
  if (retval = nc_inq_varid(ncgeo, "kxfac", &id))        ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
  kxfac = pars->kxfac = (float) stmp;

  if (retval = nc_inq_varid(ncgeo, "shat", &id))         ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
  shat = pars->shat = (float) stmp;
  //  printf("geometry: shat = %f \n",shat);
  
  if (retval = nc_inq_varid(ncgeo, "Rmaj", &id))
    if (retval = nc_inq_varid(ncgeo, "rmaj", &id))           ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
  rmaj = pars->rmaj = (float) stmp;

  if (retval = nc_inq_varid(ncgeo, "q", &id))            ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
  qsf = pars->qsf = (float) stmp;

  // these are parameters useful for T3D that are only included in newer eik.out.nc files
  if (!nc_inq_varid(ncgeo, "B_T", &id)) {
    if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
    pars->B_ref = (float) stmp;
  }
  if (!nc_inq_varid(ncgeo, "aminor", &id)) {
    if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
    pars->a_ref = (float) stmp;
  }
  if (!nc_inq_varid(ncgeo, "grhoavg", &id)) {
    if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
    pars->grhoavg = (float) stmp;
  }
  if (!nc_inq_varid(ncgeo, "surfarea", &id)) {
    if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
    pars->surfarea = (float) stmp;
  }

  // these are gx-specific parameters, which will not be included for .nc files generated from gs2
  if (retval = nc_inq_varid(ncgeo, "scale", &id)) {
    stmp = 1.0;
  } else {
    if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
  }
  theta_scale = (float) stmp;

  if (retval = nc_inq_varid(ncgeo, "nfp", &id)) {
    stmp = 1;
  } else {
    if (retval = nc_get_var  (ncgeo, id, &nfp))           ERR(retval);
  }

  if (retval = nc_inq_varid(ncgeo, "alpha", &id)) {
    stmp = 0.0;
  } else {
    if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
  }
  alpha = (float) stmp;

  if (retval = nc_inq_varid(ncgeo, "zeta_center", &id)) {
    stmp = 0.0;
  } else {
    if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
  }
  zeta_center = (float) stmp;

  // close the netcdf file with nc_close
  if (retval = nc_close(ncgeo)) ERR(retval);

  // allocate space for variables on the GPU
  cudaMalloc ((void**) &z, size);
  cudaMalloc ((void**) &bmag, size);
  cudaMalloc ((void**) &bmagInv, size);
  cudaMalloc ((void**) &gds2, size);
  cudaMalloc ((void**) &gds21, size);
  cudaMalloc ((void**) &gds22, size);
  cudaMalloc ((void**) &gbdrift, size);
  cudaMalloc ((void**) &gbdrift0, size);
  cudaMalloc ((void**) &cvdrift, size);
  cudaMalloc ((void**) &cvdrift0, size);
  cudaMalloc ((void**) &grho, size);
  cudaMalloc ((void**) &jacobian, size);
  
  // move the data to the GPU
  CP_TO_GPU (z,        z_h,        size);
  CP_TO_GPU (gbdrift,  gbdrift_h,  size);
  CP_TO_GPU (grho,     grho_h,     size);
  CP_TO_GPU (cvdrift,  cvdrift_h,  size);
  CP_TO_GPU (bmag,     bmag_h,     size);
  CP_TO_GPU (bmagInv,  bmagInv_h,  size);
  CP_TO_GPU (gds2,     gds2_h,     size);
  CP_TO_GPU (gds21,    gds21_h,    size);
  CP_TO_GPU (gds22,    gds22_h,    size);
  CP_TO_GPU (cvdrift0, cvdrift0_h, size);
  CP_TO_GPU (gbdrift0, gbdrift0_h, size);
  CP_TO_GPU (jacobian, jacobian_h, size);

  // synchronize memory
  cudaDeviceSynchronize();

  // initialize omegad and kperp2
  initializeOperatorArrays(pars, grids);
  
  // calculate bgrad
  calculate_bgrad(grids);
  if(grids->iproc==0) CUDA_DEBUG("calc bgrad: %s \n");
}

// MFM - 07/09/17
Eik_geo::Eik_geo(Parameters *pars, Grids *grids)
{

  if(grids->iproc==0) printf("READING FILE GEO: %s\n", pars->geofilename.c_str());
  operator_arrays_allocated_=false;

  size_t eiksize = sizeof(double)*(grids->Nz+1); 
  double* eik_z_h = (double*) malloc (eiksize);
  double* eik_bmag_h = (double*) malloc (eiksize);
  double* eik_bmagInv_h = (double*) malloc (eiksize);
  double* eik_gradpar_h = (double*) malloc (eiksize);
  double* eik_gds2_h = (double*) malloc (eiksize);
  double* eik_gds21_h = (double*) malloc (eiksize);
  double* eik_gds22_h = (double*) malloc (eiksize);
  double* eik_gbdrift_h = (double*) malloc (eiksize);
  double* eik_gbdrift0_h = (double*) malloc (eiksize);
  double* eik_cvdrift_h = (double*) malloc (eiksize);
  double* eik_cvdrift0_h = (double*) malloc (eiksize);
  double* eik_grho_h = (double*) malloc (eiksize);
  double* eik_jacobian_h = (double*) malloc (eiksize);

  size_t size = sizeof(float)*(grids->Nz); 
  z_h = (float*) malloc (size);
  bmag_h = (float*) malloc (size);
  bmagInv_h = (float*) malloc (size);
  gds2_h = (float*) malloc (size);
  gds21_h = (float*) malloc (size);
  gds22_h = (float*) malloc (size);
  gbdrift_h = (float*) malloc (size);
  gbdrift0_h = (float*) malloc (size);
  cvdrift_h = (float*) malloc (size);
  cvdrift0_h = (float*) malloc (size);
  grho_h = (float*) malloc (size);
  jacobian_h = (float*) malloc (size);

  cudaMalloc ((void**) &z, size);
  cudaMalloc ((void**) &bmag, size);
  cudaMalloc ((void**) &bmagInv, size);
  cudaMalloc ((void**) &gds2, size);
  cudaMalloc ((void**) &gds21, size);
  cudaMalloc ((void**) &gds22, size);
  cudaMalloc ((void**) &gbdrift, size);
  cudaMalloc ((void**) &gbdrift0, size);
  cudaMalloc ((void**) &cvdrift, size);
  cudaMalloc ((void**) &cvdrift0, size);
  cudaMalloc ((void**) &grho, size);
  cudaMalloc ((void**) &jacobian, size);
  
  FILE * geoFile = fopen(pars->geofilename.c_str(), "r");
  
  if (geoFile == NULL) {
    if(grids->iproc==0) printf("Cannot open file %s \n", pars->geofilename.c_str());
    exit(0);
  } else if(grids->iproc==0) DEBUGPRINT("Using igeo = 1. Opened geo file %s \n", pars->geofilename.c_str());

  int nlines=0;
  fpos_t lineStartPos;
  int ch;

  int ntgrid;
  int oldNz, oldnperiod;
  
  //  rewind(geoFile);
  nlines=0;
  using namespace std;
  string datline;
  ifstream myfile (pars->geofilename.c_str());
  oldNz = grids->Nz;
  int newNz = oldNz;

  if (myfile.is_open())
    {
      getline (myfile, datline);  // text
      getline (myfile, datline);  
      stringstream ss(datline);      string element;       
      ss >> element; ntgrid         = stoi(element);    
      ss >> element; nperiod = pars->nperiod  = stoi(element);
      ss >> element; newNz          = stoi(element);   
      ss >> element; drhodpsi = pars->drhodpsi = stof(element);
      ss >> element; rmaj = pars->rmaj     = stof(element);
      ss >> element; shat = pars->shat     = stof(element);
      ss >> element; kxfac = pars->kxfac    = stof(element);       
      ss >> element; qsf = pars->qsf      = stof(element);       
      if (!ss.eof()) { // newer eik.out files may have additional data
        ss >> element; pars->B_ref    = stof(element);
        ss >> element; pars->a_ref    = stof(element);
        ss >> element; pars->grhoavg  = stof(element);
        ss >> element; pars->surfarea = stof(element);
      }

      oldnperiod = pars->nperiod;

      newNz = (2*pars->nperiod-1)*newNz;
      
      if(grids->iproc==0) DEBUGPRINT("\n\nIN READ_GEO_INPUT:\nntgrid = %d, nperiod = %d, Nz = %d, rmaj = %f, shat = %f\n\n\n",
		 ntgrid, pars->nperiod, newNz, pars->rmaj, pars->shat);

      if(oldNz != newNz) {
        if(grids->iproc==0) printf("old Nz = %d \t new Nz = %d \n",oldNz,newNz);
        if(grids->iproc==0) printf("You must set ntheta in the namelist equal to ntheta in the geofile. Exiting...\n");
        fflush(stdout);
        abort();
      }
      int Nz = newNz;
      if(oldnperiod != pars->nperiod) {
        if(grids->iproc==0) printf("You must set nperiod in the namelist equal to nperiod in the geofile. Exiting...\n");
        fflush(stdout);
        abort();
      }
      
      getline (myfile, datline);  // text
      for (int idz=0; idz < newNz+1; idz++) {
	getline (myfile, datline); stringstream ss(datline);
        ss >> element; eik_gbdrift_h[idz] = stod(element); eik_gbdrift_h[idz] *= 0.5;
        ss >> element; eik_gradpar_h[idz] = stod(element);
        ss >> element; eik_grho_h[idz]    = stod(element);
        ss >> element; eik_z_h[idz]       = stod(element);
      }
      if(eik_gradpar_h[0] != eik_gradpar_h[Nz/2]) {
        if(grids->iproc==0) printf("Error: GX requires an equal-arc theta coordinate, so that gradpar = const.\nFor gs2 geometry module, use equal_arc = true. Exiting...\n");
        fflush(stdout);
        abort();
      } else {
        gradpar = eik_gradpar_h[0];
      }
     
      if(grids->iproc==0) DEBUGPRINT("gbdrift[0]: %.7e    gbdrift[end]: %.7e\n",2.*gbdrift_h[0],2.*gbdrift_h[Nz-1]);
      if(grids->iproc==0) DEBUGPRINT("z[0]: %.7e    z[end]: %.7e\n",z_h[0],z_h[Nz-1]);
      
      getline (myfile, datline);  // text
      for (int idz=0; idz < newNz+1; idz++) {
        getline (myfile, datline); stringstream ss(datline);
        ss >> element; eik_cvdrift_h[idz] = stod(element);
        eik_cvdrift_h[idz] *= 0.5;
        ss >> element; eik_gds2_h[idz]    = stod(element);
        ss >> element; eik_bmag_h[idz]    = stod(element);
        eik_bmagInv_h[idz]  = 1./eik_bmag_h[idz];
        eik_jacobian_h[idz] = 1./abs(drhodpsi*gradpar*eik_bmag_h[idz]);
      }

      if(grids->iproc==0) DEBUGPRINT("cvdrift[0]: %.7e    cvdrift[end]: %.7e\n",2.*cvdrift_h[0],2.*cvdrift_h[Nz-1]);
      if(grids->iproc==0) DEBUGPRINT("bmag[0]: %.7e    bmag[end]: %.7e\n",bmag_h[0],bmag_h[Nz-1]);
      if(grids->iproc==0) DEBUGPRINT("gds2[0]: %.7e    gds2[end]: %.7e\n",gds2_h[0],gds2_h[Nz-1]);

      getline(myfile, datline); // text
      for (int idz=0; idz < newNz+1; idz++) {
        getline (myfile, datline); stringstream ss(datline);
        ss >> element; eik_gds21_h[idz] = stod(element); 
        ss >> element; eik_gds22_h[idz] = stod(element);
      }

      if(grids->iproc==0) DEBUGPRINT("gds21[0]: %.7e    gds21[end]: %.7e\n",gds21_h[0],gds21_h[Nz-1]);
      if(grids->iproc==0) DEBUGPRINT("gds22[0]: %.7e    gds22[end]: %.7e\n",gds22_h[0],gds22_h[Nz-1]);

            getline(myfile, datline); // text
      for (int idz=0; idz < newNz+1; idz++) {
        getline (myfile, datline); stringstream ss(datline);
        ss >> element; eik_cvdrift0_h[idz] = stod(element); eik_cvdrift0_h[idz] *= 0.5;
        ss >> element; eik_gbdrift0_h[idz] = stod(element); eik_gbdrift0_h[idz] *= 0.5;
      }

      if(grids->iproc==0) DEBUGPRINT("gds21[0]: %.7e    gds21[end]: %.7e\n",gds21_h[0],gds21_h[Nz-1]);
      if(grids->iproc==0) DEBUGPRINT("gds22[0]: %.7e    gds22[end]: %.7e\n",gds22_h[0],gds22_h[Nz-1]);
      
      myfile.close();      
    }
  else if(grids->iproc==0)  cout << "Failed to open";    

  // interpolate to equally-spaced theta grid
  for(int k=0; k<newNz; k++) {
    z_h[k] = 2.*M_PI *pars->Zp *(k-newNz/2)/newNz;
  }
  interp_to_new_grid(eik_bmag_h, bmag_h, eik_z_h, z_h, newNz+1, newNz);
  interp_to_new_grid(eik_bmagInv_h, bmagInv_h, eik_z_h, z_h, newNz+1, newNz);
  interp_to_new_grid(eik_gds2_h, gds2_h, eik_z_h, z_h, newNz+1, newNz);
  interp_to_new_grid(eik_gds21_h, gds21_h, eik_z_h, z_h, newNz+1, newNz);
  interp_to_new_grid(eik_gds22_h, gds22_h, eik_z_h, z_h, newNz+1, newNz);
  interp_to_new_grid(eik_gbdrift_h, gbdrift_h, eik_z_h, z_h, newNz+1, newNz);
  interp_to_new_grid(eik_gbdrift0_h, gbdrift0_h, eik_z_h, z_h, newNz+1, newNz);
  interp_to_new_grid(eik_cvdrift_h, cvdrift_h, eik_z_h, z_h, newNz+1, newNz);
  interp_to_new_grid(eik_cvdrift0_h, cvdrift0_h, eik_z_h, z_h, newNz+1, newNz);
  interp_to_new_grid(eik_grho_h, grho_h, eik_z_h, z_h, newNz+1, newNz);
  interp_to_new_grid(eik_jacobian_h, jacobian_h, eik_z_h, z_h, newNz+1, newNz);
  
  //copy host variables to device variables
  CP_TO_GPU (z,        z_h,        size);
  CP_TO_GPU (gbdrift,  gbdrift_h,  size);
  CP_TO_GPU (grho,     grho_h,     size);
  CP_TO_GPU (cvdrift,  cvdrift_h,  size);
  CP_TO_GPU (bmag,     bmag_h,     size);
  CP_TO_GPU (bmagInv,  bmagInv_h,  size);
  CP_TO_GPU (gds2,     gds2_h,     size);
  CP_TO_GPU (gds21,    gds21_h,    size);
  CP_TO_GPU (gds22,    gds22_h,    size);
  CP_TO_GPU (cvdrift0, cvdrift0_h, size);
  CP_TO_GPU (gbdrift0, gbdrift0_h, size);
  CP_TO_GPU (jacobian, jacobian_h, size);

  cudaDeviceSynchronize();

  // initialize omegad and kperp2
  initializeOperatorArrays(pars, grids);

  // calculate bgrad
  calculate_bgrad(grids);
  if(grids->iproc==0) CUDA_DEBUG("calc bgrad: %s \n");
}

void Geometry::initializeOperatorArrays(Parameters* pars, Grids* grids) {
  // set this flag so we know to deallocate
  operator_arrays_allocated_ = true;

  cudaMalloc ((void**) &kperp2, sizeof(float)*grids->NxNycNz);
  cudaMalloc ((void**) &omegad, sizeof(float)*grids->NxNycNz);
  cudaMalloc ((void**) &cv_d,   sizeof(float)*grids->NxNycNz);
  cudaMalloc ((void**) &gb_d,   sizeof(float)*grids->NxNycNz);
  if (pars->nonTwist) {
    cudaMalloc ((void**) &ftwist, sizeof(float)*grids->Nz);
    cudaMalloc ((void**) &m0, sizeof(int)*grids->NycNz); 
    cudaMalloc ((void**) &deltaKx, sizeof(float)*grids->NycNz);
  }
  checkCuda  (cudaGetLastError());

  cudaMemset (kperp2, 0., sizeof(float)*grids->NxNycNz);
  cudaMemset (omegad, 0., sizeof(float)*grids->NxNycNz);
  cudaMemset (cv_d,   0., sizeof(float)*grids->NxNycNz);
  cudaMemset (gb_d,   0., sizeof(float)*grids->NxNycNz);
  if (pars->nonTwist) {
    cudaMemset (ftwist, 0., sizeof(float)*grids->Nz);
    cudaMemset (m0, 0., sizeof(int)*grids->NycNz);
    cudaMemset (deltaKx, 0., sizeof(float)*grids->NycNz);
  }
  
  dim3 dimBlock (32, 4, 4);
  dim3 dimGrid  (1+(grids->Nyc-1)/dimBlock.x, 1+(grids->Nx-1)/dimBlock.y, 1+(grids->Nz-1)/dimBlock.z);

  // set jtwist and x0, now that we know the final value of shat from geometry
  pars->set_jtwist_x0(&shat, gds21_h, gds22_h);
  // initialize k and coordinate arrays
  grids->init_ks_and_coords();

  // initialize operator arrays
  if (pars->nonTwist) {
    dim3 dimBlock_ntft (32,16);
    dim3 dimGrid_ntft (1+(grids->Nyc-1)/dimBlock.x, 1+(grids->Nz-1)/dimBlock.y);

    printf("Using non-twisting flux tube \n"); 

    // see (87), (44), and (45) in Ball 2020, respectively
    init_ftwist <<< (1 + (grids->Nz-1)/dimBlock.z), 32 >>> (ftwist, gds21, gds22, shat);
    init_m0 <<< dimGrid_ntft, dimBlock_ntft >>> (m0, pars->x0, grids->ky, ftwist, shat, pars->kxfac);
    CP_TO_GPU (grids->m0_h, m0, sizeof(int)*grids->NycNz);
    init_deltaKx <<<dimGrid_ntft, dimBlock_ntft >>> (deltaKx, m0, pars->x0, grids->ky, ftwist);


    init_kperp2_ntft GGEO (kperp2, grids->kx, grids->ky, gds2, gds21, gds22, ftwist, bmagInv, shat, deltaKx);
    init_omegad_ntft GGEO (omegad, cv_d, gb_d, grids->kx, grids->ky, cvdrift, gbdrift, cvdrift0, gbdrift0, shat, m0, pars->x0);

    if (!pars->linear) {
      CP_TO_GPU (grids->x, grids->x_h, sizeof(float)*grids->Nx);
      init_iKx GGEO (grids->iKx, grids->kx, deltaKx);
      init_phasefac_ntft GGEO (grids->phasefac_ntft, grids->x, deltaKx, true);
      init_phasefac_ntft GGEO (grids->phasefacminus_ntft, grids->x, deltaKx, false);
    }
  }
  else { 
    init_kperp2 GGEO (kperp2, grids->kx, grids->ky, gds2, gds21, gds22, bmagInv, shat);
    init_omegad GGEO (omegad, cv_d, gb_d, grids->kx, grids->ky, cvdrift, gbdrift, cvdrift0, gbdrift0, shat);
  }

  // initialize volume integral weight quantities needed for some diagnostics
  float volDenom = 0.;  
  vol_fac_h = (float*) malloc (sizeof(float) * grids->Nz);
  cudaMalloc (&vol_fac, sizeof(float) * grids->Nz);
  for (int i=0; i < grids->Nz; i++) volDenom   += jacobian_h[i]; 
  for (int i=0; i < grids->Nz; i++) vol_fac_h[i]  = jacobian_h[i] / volDenom;
  CP_TO_GPU(vol_fac, vol_fac_h, sizeof(float)*grids->Nz);

  // volume integrals for fluxes contain a factor of 1/grho == 1/|\nabla x|
  float fluxDenom = 0.;  
  flux_fac_h = (float*) malloc (sizeof(float) * grids->Nz);
  cudaMalloc(&flux_fac, sizeof(float)*grids->Nz);
  for (int i=0; i<grids->Nz; i++) fluxDenom   += jacobian_h[i]*grho_h[i];
  for (int i=0; i<grids->Nz; i++) flux_fac_h[i]  = jacobian_h[i] / fluxDenom;
  CP_TO_GPU(flux_fac, flux_fac_h, sizeof(float)*grids->Nz);

  // compute max values of gbdrift, cvdrift, gbdrift0, cvdrift0
  bmag_max = 0.;
  gbdrift_max = 0.;
  gbdrift0_max = 0.;
  cvdrift_max = 0.;
  cvdrift0_max = 0.;
  for(int i=0; i<grids->Nz; i++) {
    gbdrift_max = max(gbdrift_max, abs(gbdrift_h[i]));
    gbdrift0_max = max(gbdrift0_max, abs(gbdrift0_h[i]));
    cvdrift_max = max(cvdrift_max, abs(cvdrift_h[i]));
    cvdrift0_max = max(cvdrift0_max, abs(cvdrift0_h[i]));
    bmag_max = max(bmag_max, abs(bmag_h[i]));
  }

  if (pars->nonTwist) {
    grids->m0_max = 0;
    float m0_omega0 = 0; // need to maximize this quantity to find max frequency for the NTFT
    for (int idz = 0; idz < grids->Nz; idz++) { //only need to loop through Nz since m0 scales with ky, max ky will have max m0
      if (grids->m0_h[grids->Nyc-1 + grids->Nyc*idz] * (grids->vpar_max * grids->vpar_max*abs(cvdrift0_h[idz]) + grids->muB_max * abs(gbdrift0_h[idz]))) {
        m0_omega0 = grids->m0_h[grids->Nyc-1 + grids->Nyc*idz] * (grids->vpar_max * grids->vpar_max*abs(cvdrift0_h[idz]) + grids->muB_max * abs(gbdrift0_h[idz]));
	grids->m0_max = abs(grids->m0_h[grids->Nyc-1 + grids->Nyc*idz]);
	gbdrift0_max = abs(gbdrift0_h[idz]);
	cvdrift0_max = abs(cvdrift0_h[idz]);
      }
    }
  }

  /*
  kperp2_h = (float*) malloc(sizeof(float)*grids->NxNycNz);
  CP_TO_GPU (kperp2_h,    kperp2, sizeof(float)*grids->NxNycNz);

  for (int iz=0; iz < grids->Nz; iz++) {
    for (int ikx=0; ikx < grids->Nx; ikx++) {
      for (int iky=0; iky< grids->Nyc; iky++) {
	printf("kperp2(%d,%d,%d) = %e \n", iky, ikx, iz, kperp2_h[iky + grids->Nyc*ikx + grids->Nyc*grids->Nx*iz]);
      }
      printf("\n");
    }
    printf("\n");
  }
  */
}

// MFM - 07/25/17
void Geometry::calculate_bgrad(Grids* grids)
{
  operator_arrays_allocated_=false;

  size_t size = sizeof(float)*grids->Nz;
  bgrad_h = (float*) malloc (size);

  cudaMalloc ((void**) &bgrad, size);
  cudaMalloc ((void**) &bgrad_temp, size);

  CP_ON_GPU (bgrad_temp, bmag, size);
  
  GradParallel1D* grad_par = new GradParallel1D(grids);

  //bgrad = d/dz ln(B(z)) = 1/B dB/dz
  grad_par->dz1D(bgrad_temp); // FFT and k-space derivative

  calc_bgrad <<< 1 + (grids->Nz-1)/512, 512 >>> (bgrad, bgrad_temp, bmag, gradpar);

  CP_TO_CPU (bgrad_h, bgrad, size);
  if (bgrad_temp) cudaFree(bgrad_temp);

  delete grad_par;

//  for(int i=0; i<grids->Nz; i++) {
//    printf("bgrad_h[%d]: %.4e\n",i,bgrad_h[i]);
//  }
  cudaDeviceSynchronize();
}

