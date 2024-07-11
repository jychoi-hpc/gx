#pragma once
#include "netcdf.h"
#include "parameters.h"
#include "grids.h"
#include "device_funcs.h"
#include "get_error.h"
#include "geometry.h"

#include <cusolverSp.h>
#include "cusolver_utils.h"
#include <cusolverDn.h>
/*
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include "helper_cuda.h"
#include "helper_cusolver.h"
#include "cusolver_utils.h"
#include "cusolver_utils.h"
*/
class Cusolve {
 public:
  Cusolve(Parameters* pars, Grids* grids, Geometry* geo, double p, double r, double u,bool sdirk, double dt_in, double vte);
  ~Cusolve();

  void factorize();
  void print_matrix(const int m, const int n, const cuComplex *A, const int lda);
  void invert(cuComplex* rhs, cuComplex* res, int iz); 
  cuComplex** A_bounce;

 private:
  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;
  
  dim3 dG;
  dim3 dB;

  double p_;
  double r_;
  double u_;
  bool sdirk_;
  double dt_;
  double vte_;
  int64_t** d_Ipiv;
  int LM;
  int* d_info;
  int pivot_on;

  cusolverDnHandle_t cusolverH;
  cudaStream_t stream;
  cusolverDnParams_t params;

  cublasHandle_t cublasH = NULL;
  cudaStream_t stream_cublas = NULL;

  const cuComplex alpha = make_cuComplex(1.0,0.0);
  const cuComplex beta = make_cuComplex(-1.0,0.0);

  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  
  cuComplex* test;
  cuComplex* test2;

};

