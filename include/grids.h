#pragma once

//#include "cufft.h"
#include "parameters.h"

class Grids {

 public:
  Grids(Parameters* pars);
  ~Grids();

  const int Nx;
  const int Ny;
  const int Nz;
  const int Nspecies; 
  const int Nm;
  const int Nl;
  const int Nyc;
  //  const int Nakz;
  const int Naky;
  const int Nakx;
  const int NxNyc;
  const int NxNy;
  const int NxNycNz;
  const int NxNyNz;
  const int NxNz;
  const int NycNz;
  const int Nmoms;

  float * ky;
  float * kx;
  float * kz;

  float * ky_h;
  float * kx_h;
  float * kz_h;

  float * theta0_h;

  /* A grid the size of kx, true if in the dealiased zone*/
  bool * kx_mask;
  
  /* Flow shear arrays*/
  float * kx_shift;
  int * jump;
  
  /* Stuff for z covering */
  int nClasses;
  /* Note: the arrays below are allocated in initialize_z_covering, not
   * in the main allocation routines. Also note that the ** 
   * pointers to pointers point to arrays of pointers on the
   * host. The pointers themselves then may point to arrays 
   * on either the device or host */
  int * nLinks;
  int * nChains;
  int ** kxCover;
  int ** kyCover;
  //cuComplex ** g_covering;
  float ** kz_covering;
  
  int ** kxCover_d;
  int ** kyCover_d;
  //cuComplex ** g_covering_d;
  float ** kz_covering_d;
  //int * nLinks_d;
  //int * nChains_d;
  
  float * covering_scaler;
  
 private:
  const Parameters* pars_; //local Parameters object for convenience

};

