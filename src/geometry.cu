#include "geometry.h"
#define GGEO <<< dimGrid, dimBlock >>>

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

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
  
  float qsf = pars->qsf;
  float beta_e = pars->beta;
  float rmaj = pars->rmaj;
  specie* species = pars->species_h;
  
  gradpar = (float) abs(1./(qsf*rmaj));
  zero_shat_ = pars->zero_shat;
  shat = pars->shat;
  pars->drhodpsi = 1.; 
  pars->kxfac = 1.;
  
  if(pars->shift < 0.) {
    pars->shift = 0.;
    for(int s=0; s<pars->nspec_in; s++) { 
      pars->shift += qsf*qsf*rmaj*beta_e*
	(species[s].temp/species[pars->nspec_in-1].temp)*
	(species[s].tprim + species[s].fprim);
    }
  }
  float shift = pars->shift;
 
  DEBUGPRINT("\n\n Using s-alpha geometry: \n\n");
  for(int k=0; k<Nz; k++) {
    z_h[k] = 2.*M_PI *pars->Zp *(k-Nz/2)/Nz;
    DEBUGPRINT("theta[%d] = %f \n",k,z_h[k]);
    if(pars->local_limit) {z_h[k] = 0.;} // outboard-midplane
    theta = z_h[k];
    
    bmag_h[k] = 1. / (1. + pars->eps * cos(theta));
    bgrad_h[k] = gradpar * pars->eps * sin(theta) * bmag_h[k]; 

    gds2_h[k] = 1. + pow((shat * theta - shift * sin(theta)), 2);
    gds21_h[k] = -shat * (shat * theta - shift * sin(theta));
    gds22_h[k] = pow(shat,2);

    gbdrift_h[k] = 1. / (2.*rmaj) *
               (cos(theta) + (shat * theta - shift * sin(theta)) * sin(theta));
    cvdrift_h[k] = gbdrift_h[k];

    gbdrift0_h[k] = - shat * sin(theta) / (2.*rmaj);
    cvdrift0_h[k] = gbdrift0_h[k];

    grho_h[k] = 1;

    if(pars->const_curv) {
      cvdrift_h[k] = 1./(2.*rmaj);
      gbdrift_h[k] = 1./(2.*rmaj);
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
    }
    if(pars->local_limit) { z_h[k] = 2 * M_PI * pars->Zp * (k-Nz/2) / Nz; }

    // calculate these derived coefficients after slab overrides
    bmagInv_h[k] = 1./bmag_h[k];
    jacobian_h[k] = 1. / abs(pars->drhodpsi * gradpar * bmag_h[k]);
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
  initializeOperatorArrays(grids);
}

Gs2_geo::Gs2_geo() {

}

geo_nc::geo_nc(Parameters *pars, Grids *grids)
{
  operator_arrays_allocated_=false;
  size_t size = sizeof(float)*grids->Nz;

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
  if (retval = nc_inq_dimid(ncgeo, "z",  &id_z))       ERR(retval);
  if (retval = nc_inq_dim  (ncgeo, id_z, stra, &N))    ERR(retval);

  // do basic sanity check
  if (grids->Nz != (int) N) {
    printf("Number of points along the field line in geometry file %d does not match input %d \n", N, grids->Nz);
    exit (1);
  }

  // allocate space for variables on the CPU
  double * dtmp;
  dtmp = (double*) malloc(sizeof(double)*N);
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
  for (int n=0; n<N; n++) z_h[n] = (float) dtmp[n];
  
  if (retval = nc_inq_varid(ncgeo, "bmag", &id))         ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) bmag_h[n] = (float) dtmp[n];
  for (int n=0; n<N; n++) bmagInv_h[n] = 1./bmag_h[n];

  if (retval = nc_inq_varid(ncgeo, "gradpar", &id))      ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  gradpar = (float) dtmp[0];

  if (retval = nc_inq_varid(ncgeo, "grho", &id))         ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) grho_h[n] = (float) dtmp[n];
  
  if (retval = nc_inq_varid(ncgeo, "gds2", &id))         ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) gds2_h[n] = (float) dtmp[n];
  
  if (retval = nc_inq_varid(ncgeo, "gds21", &id))        ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) gds21_h[n] = (float) dtmp[n];
  
  if (retval = nc_inq_varid(ncgeo, "gds22", &id))        ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) gds22_h[n] = (float) dtmp[n];
  
  if (retval = nc_inq_varid(ncgeo, "gbdrift", &id))      ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) gbdrift_h[n] = (float) dtmp[n];
  
  if (retval = nc_inq_varid(ncgeo, "gbdrift0", &id))     ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) gbdrift0_h[n] = (float) dtmp[n];
  
  if (retval = nc_inq_varid(ncgeo, "cvdrift", &id))      ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) cvdrift_h[n] = (float) dtmp[n];
  
  if (retval = nc_inq_varid(ncgeo, "cvdrift0", &id))     ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, dtmp))            ERR(retval);
  for (int n=0; n<N; n++) cvdrift0_h[n] = (float) dtmp[n];
  
  free(dtmp);

  double stmp; 
  
  if (retval = nc_inq_varid(ncgeo, "drhodpsi", &id))     ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
  pars->drhodpsi = (float) stmp;
  drhodpsi = pars->drhodpsi;
  
  for (int n=0; n<N; n++) jacobian_h[n] = 1./abs(drhodpsi*gradpar*bmag_h[n]);
      
  if (retval = nc_inq_varid(ncgeo, "kxfac", &id))        ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
  pars->kxfac = (float) stmp;

  if (retval = nc_inq_varid(ncgeo, "shat", &id))         ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
  pars->shat = (float) stmp;
  shat = (float) stmp;
  
  if (retval = nc_inq_varid(ncgeo, "Rmaj", &id))         ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
  pars->rmaj = (float) stmp;

  if (retval = nc_inq_varid(ncgeo, "q", &id))        ERR(retval);
  if (retval = nc_get_var  (ncgeo, id, &stmp))           ERR(retval);
  pars->qsf = (float) stmp;

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
  initializeOperatorArrays(grids);
  
  // calculate bgrad
  calculate_bgrad(grids);
  CUDA_DEBUG("calc bgrad: %s \n");
}

// MFM - 07/09/17
File_geo::File_geo(Parameters *pars, Grids *grids)
{

  operator_arrays_allocated_=false;
  size_t size = sizeof(float)*grids->Nz; 
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
    printf("Cannot open file %s \n", pars->geofilename.c_str());
    exit(0);
  } else DEBUGPRINT("Using igeo = 1. Opened geo file %s \n", pars->geofilename.c_str());

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
      ss >> element; pars->nperiod  = stoi(element);
      ss >> element; newNz          = stoi(element);   
      ss >> element; pars->drhodpsi = stof(element);
      ss >> element; pars->rmaj     = stof(element);
      ss >> element; pars->shat     = stof(element);
      ss >> element; pars->kxfac    = stof(element);       
      ss >> element; pars->qsf      = stof(element);       

      shat       = pars->shat;
      drhodpsi   = pars->drhodpsi;
      oldnperiod = pars->nperiod;
      
      DEBUGPRINT("\n\nIN READ_GEO_INPUT:\nntgrid = %d, nperiod = %d, Nz = %d, rmaj = %f, shat = %f\n\n\n",
		 ntgrid, pars->nperiod, newNz, pars->rmaj, pars->shat);

      if(oldNz != newNz) {
        printf("old Nz = %d \t new Nz = %d \n",oldNz,newNz);
        printf("You must set ntheta in the namelist equal to ntheta in the geofile. Exiting...\n");
        fflush(stdout);
        abort();
      }
      int Nz = newNz;
      if(oldnperiod != pars->nperiod) {
        printf("You must set nperiod in the namelist equal to nperiod in the geofile. Exiting...\n");
        fflush(stdout);
        abort();
      }
      
      getline (myfile, datline);  // text
      for (int idz=0; idz < newNz; idz++) {
	getline (myfile, datline); stringstream ss(datline);
        ss >> element; gbdrift_h[idz] = stof(element); gbdrift_h[idz] *= 0.25;
        ss >> element; gradpar        = stof(element);
        ss >> element; grho_h[idz]    = stof(element);
        ss >> element; z_h[idz]       = stof(element);
      }
      getline(myfile, datline); // periodic points (not always periodic, but extra)
     
      DEBUGPRINT("gbdrift[0]: %.7e    gbdrift[end]: %.7e\n",4.*gbdrift_h[0],4.*gbdrift_h[Nz-1]);
      DEBUGPRINT("z[0]: %.7e    z[end]: %.7e\n",z_h[0],z_h[Nz-1]);
      
      getline (myfile, datline);  // text
      for (int idz=0; idz < newNz; idz++) {
        getline (myfile, datline); stringstream ss(datline);
        ss >> element; cvdrift_h[idz] = stof(element);
        cvdrift_h[idz] *= 0.25;
        ss >> element; gds2_h[idz]    = stof(element);
        ss >> element; bmag_h[idz]    = stof(element);
        bmagInv_h[idz]  = 1./bmag_h[idz];
        jacobian_h[idz] = 1./abs(drhodpsi*gradpar*bmag_h[idz]);
      }
      getline(myfile, datline); // periodic points (not always periodic, but extra)

      DEBUGPRINT("cvdrift[0]: %.7e    cvdrift[end]: %.7e\n",4.*cvdrift_h[0],4.*cvdrift_h[Nz-1]);
      DEBUGPRINT("bmag[0]: %.7e    bmag[end]: %.7e\n",bmag_h[0],bmag_h[Nz-1]);
      DEBUGPRINT("gds2[0]: %.7e    gds2[end]: %.7e\n",gds2_h[0],gds2_h[Nz-1]);

      getline(myfile, datline); // text
      for (int idz=0; idz < newNz; idz++) {
        getline (myfile, datline); stringstream ss(datline);
        ss >> element; gds21_h[idz] = stof(element); 
        ss >> element; gds22_h[idz] = stof(element);
      }
      getline(myfile, datline); // periodic points (not always periodic, but extra)

      DEBUGPRINT("gds21[0]: %.7e    gds21[end]: %.7e\n",gds21_h[0],gds21_h[Nz-1]);
      DEBUGPRINT("gds22[0]: %.7e    gds22[end]: %.7e\n",gds22_h[0],gds22_h[Nz-1]);

            getline(myfile, datline); // text
      for (int idz=0; idz < newNz; idz++) {
        getline (myfile, datline); stringstream ss(datline);
        ss >> element; cvdrift0_h[idz] = stof(element); cvdrift0_h[idz] *= 0.25;
        ss >> element; gbdrift0_h[idz] = stof(element); gbdrift0_h[idz] *= 0.25;
      }
      getline(myfile, datline); // periodic points (not always periodic, but extra)

      DEBUGPRINT("gds21[0]: %.7e    gds21[end]: %.7e\n",gds21_h[0],gds21_h[Nz-1]);
      DEBUGPRINT("gds22[0]: %.7e    gds22[end]: %.7e\n",gds22_h[0],gds22_h[Nz-1]);
      
      myfile.close();      
    }
  else cout << "Failed to open";    
  
  //copy host variables to device variables
  CP_TO_GPU (z,        z_h,        size);
  CP_TO_GPU (gbdrift,  gbdrift_h,  size);
  CP_TO_GPU (grho,     grho_h,     size);
  CP_TO_GPU (cvdrift,  cvdrift_h,  size);
  CP_TO_GPU (bmag,     bmag_h,     size);
  CP_TO_GPU (bmagInv,  bmagInv_h,  size);
  //  CP_TO_GPU (bgrad,    bgrad_h,    size);
  CP_TO_GPU (gds2,     gds2_h,     size);
  CP_TO_GPU (gds21,    gds21_h,    size);
  CP_TO_GPU (gds22,    gds22_h,    size);
  CP_TO_GPU (cvdrift0, cvdrift0_h, size);
  CP_TO_GPU (gbdrift0, gbdrift0_h, size);
  CP_TO_GPU (jacobian, jacobian_h, size);

  cudaDeviceSynchronize();

  // initialize omegad and kperp2
  initializeOperatorArrays(grids);

  // calculate bgrad
  calculate_bgrad(grids);
  CUDA_DEBUG("calc bgrad: %s \n");
}

void Geometry::initializeOperatorArrays(Grids* grids) {
  // set this flag so we know to deallocate
  operator_arrays_allocated_ = true;

  cudaMalloc ((void**) &kperp2, sizeof(float)*grids->NxNycNz);
  cudaMalloc ((void**) &omegad, sizeof(float)*grids->NxNycNz);
  cudaMalloc ((void**) &cv_d,   sizeof(float)*grids->NxNycNz);
  cudaMalloc ((void**) &gb_d,   sizeof(float)*grids->NxNycNz);
  checkCuda  (cudaGetLastError());

  cudaMemset (kperp2, 0., sizeof(float)*grids->NxNycNz);
  cudaMemset (omegad, 0., sizeof(float)*grids->NxNycNz);
  cudaMemset (cv_d,   0., sizeof(float)*grids->NxNycNz);
  cudaMemset (gb_d,   0., sizeof(float)*grids->NxNycNz);
  
  dim3 dimBlock (32, 4, 4);
  dim3 dimGrid  (1+(grids->Nyc-1)/dimBlock.x, 1+(grids->Nx-1)/dimBlock.y, 1+(grids->Nz-1)/dimBlock.z);

  init_kperp2 GGEO (kperp2, grids->kx, grids->ky, gds2, gds21, gds22, bmagInv, shat);
  init_omegad GGEO (omegad, cv_d, gb_d, grids->kx, grids->ky, cvdrift, gbdrift, cvdrift0, gbdrift0, shat);

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

