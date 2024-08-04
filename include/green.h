#pragma once
#include "netcdf.h"
#include "parameters.h"
#include "grids.h"
#include "device_funcs.h"
#include "get_error.h"
#include "geometry.h"
#include "grad_parallel.h"
#include <cusolverSp.h>
#include "cusolver_utils.h"
#include <cusolverDn.h>
#include "solver.h"
/*
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include "helper_cuda.h"
#include "helper_cusolver.h"
#include "cusolver_utils.h"
#include "cusolver_utils.h"
*/
class Green {
 public:
  Green(Parameters* pars, Grids* grids, Geometry* geo, Solver* solver, double p, double r, double u,bool sdirk, double dt_in, double vte);
  ~Green();
  void invert(cuComplex* phi_i);
  cuComplex** A_phi;
  cuComplex** d_A_phi;

  cuComplex** A_phi_copy;

  cuComplex     ** phi_rhs;
  cuComplex     ** d_phi_rhs;
  cuComplex     ** res;
  cuComplex     ** prod;


 private:
  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;
  Solver* solver_;
  MomentsG** G;
  cuComplex* phi_r;
  GradParallel * grad_par   ;

  dim3 dG;
  dim3 dB;
  dim3 dG_b;
  dim3 dB_b;
  dim3 dG_p;
  dim3 dB_p;
  dim3 dG_s;
  dim3 dB_s;


  double p_;
  double r_;
  double u_;
  bool sdirk_;
  double dt_;
  double vte_;
  int* d_Ipiv;
  int* d_info;
  int pivot_on;
  int num_coeff;
  int* infoArray;
  int* infoArray_h;


  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;

  const cuComplex alpha = make_cuComplex(1.0,0.0);
  const cuComplex beta = make_cuComplex(-1.0,0.0);
  const cuComplex zeta = make_cuComplex(0.0,0.0);

  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  
  cuComplex* test;
  cuComplex* test2;


};

