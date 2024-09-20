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
class Cublas_test{
 public:
  Cublas_test(Parameters* pars, Grids* grids, Geometry* geo, double p, double r, double u,bool sdirk, double dt_in, double vte);
  ~Cublas_test();

  void invert_stream(cuComplex* G, int stage);
  void invert_sherman_morrison(cuComplex* u);

  cuComplex** A_bounce;
  cuComplex     ** bounce_rhs;
  cuComplex     ** res;
  cuComplex** d_A_bounce;
  cuComplex** d_bounce_rhs;


 private:
  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;
  
  dim3 dG;
  dim3 dB;
  dim3 dG_b;
  dim3 dB_b;
  dim3 dG_bd;
  dim3 dB_bd;
  dim3 dG_lu;
  dim3 dB_lu;
  dim3 dG_lu_sm;
  dim3 dB_lu_sm;

  double p_;
  double r_;
  double u_;
  bool sdirk_;
  double dt_;
  double vte_;
  int* d_Ipiv;
//  int64_t ** d_Ipiv;
  int* infoArray;
  int LM;
  int* infoArray_h;
  int pivot_on;
  int num_coeff;

  cublasHandle_t cublasH = NULL;
  cudaStream_t stream_cublas = NULL;

  const cuComplex alpha = make_cuComplex(1.0,0.0);
  const cuComplex beta = make_cuComplex(-1.0,0.0);

  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;





};

