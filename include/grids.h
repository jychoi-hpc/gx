#pragma once
#include "parameters.h"
#include "device_funcs.h"
#include "get_error.h"

class Grids {

 public:
  Grids(Parameters* pars);
  ~Grids();

  int Nx;
  int Ny;
  int Nz;
  int Nspecies; 
  int Nm;
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
  
  float * theta0_h ;
  float * th0; 

  /* Flow shear arrays*/
  //  float * kx_shift ;
  //  int * jump ;
  
 private:
  Parameters * pars_ ; 
};

