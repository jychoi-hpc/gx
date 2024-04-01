#pragma once
#include "parameters.h"
#include "device_funcs.h"
#include "get_error.h"
#include "nccl.h"

class Grids {

 public:
  Grids(Parameters* pars);
  ~Grids();
  void init_ks_and_coords();

  int Nx;
  int Ny;
  int Nz;
  int Nspecies; 
  int Nspecies_glob;
  int Nm;
  int Nm_glob;
  int Nl;
  int Nj;
  int Nyc;
  int Naky;
  int Nakx;
  int NxNyc;
  int NxNy;
  int NxNycNz;
  int NxNyNz;
  int NxNz;
  int NycNz;
  int Nmoms;
  size_t size_G;
  int * kzm;  
  float * ky;       float * kx;    float * kz;    float * kxs;
  float * ky_h;     float * kx_h;  float * kz_h;  float * kzp;
  float * kx_outh;  float * kz_outh;
  float * kpar_outh;
  float *y_h, *x_h, *z_h;
  int * m0_h;
  cuComplex * phasefac_ntft;
  cuComplex * phasefacminus_ntft;
  cuComplex * iKx;
  float * x;
  cuComplex * phasefac_exb;
  cuComplex * phasefacminus_exb;
  float * kxstar;
  int * kxbar_ikx_new, * kxbar_ikx_old;

  
  float * theta0_h ;
  float * th0; 
  float Zp;
  float kx_max, ky_max, kz_max, vpar_max, muB_max;
  int m0_max;
  float kperp_min;

  ncclComm_t ncclComm, ncclComm_s, ncclComm_m, ncclComm_m0;
  ncclUniqueId ncclId, ncclId_m0;
  std::vector<ncclUniqueId> ncclId_s, ncclId_m;
  cudaStream_t ncclStream;

  int iproc, nprocs;
  int iproc_m, nprocs_m;
  int iproc_s, nprocs_s;
  int is_lo, is_up;
  int m_lo, m_up;
  int m_ghost;

  int proc(int iproc_m_in, int iproc_s_in) { return iproc_m_in + nprocs_m*iproc_s_in; };
  int procLeft() {return proc(iproc_m-1, iproc_s);}
  int procRight() {return proc(iproc_m+1, iproc_s);}
  int procLeft2() {return proc(iproc_m-2, iproc_s);}
  int procRight2() {return proc(iproc_m+2, iproc_s);}

  dim3 dB, dG;

  /* Flow shear arrays*/
  //  float * kx_shift ;
  //  int * jump ;
  
 private:
  Parameters * pars_ ; 
};

