#pragma once

#include "parameters.h"
#include "device_funcs.h"
#include "reductions.h"
#include "grids.h"
#include "get_error.h"
#include "cusolverDn.h"

class Reservoir {
 public:
  Reservoir(Parameters *pars, Grids *grids, int m);
  ~Reservoir();
  void conclude_training(void);
  void add_data(float* G);
  void predict (double* G);
  bool predicting(void);

  void fake_data(float* G);
  
private:
  
  cusolverDnHandle_t handle = NULL;
  int invSize;
  
  void update_reservoir(double* G);
  
  dim3 blocks_m, threads_m;
  dim3 blocks_n, threads_n;
  dim3 blocks_mn, threads_mn;
  dim3 blocks_n2, threads_n2;
  dim3 blocks_nk, threads_nk;
  dim3 blocks_MN, threads_MN;
  dim3 blocks_NN, threads_NN;
  dim3 blocks_QM, threads_QM;
  
  bool addNoise_;
  unsigned int N_, M_, K_;
  unsigned int nW;
  unsigned int ResQ_, nT_, iT_;
  float sigNoise_;
  double beta_, ResRadius_, sigma_;

  float *fake_G;
  double *invWork, *B, *V, *W, *R, *R2, *x, *P, *A_in, *W_in, *dG;
  int *A_col, *info;

  // local private copies
  Parameters * pars_  ;
  Grids      * grids_ ;
  Reduction<double> * red;
  DenseM * dense;
};
