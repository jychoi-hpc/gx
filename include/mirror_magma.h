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

//includes specific to magma test routines
#include "testings.h"

class Mirror_magma{
 public:
  Mirror_magma(Parameters* pars, Grids* grids, Geometry* geo, double p, double r, double u,bool sdirk, double dt_in, double vte);
  ~Mirror_magma();

  void invert_stream(cuComplex* G, int stage);
  void invert_apar(cuComplex* u);
  void fill_B(cuComplex* B, int N, int nrhs, int ldb, int batchCount);
  void fill_A(cuComplex* A, cuComplex* h_diags, int* h_offsets, int N, int lda, int batchCount, int KL, int KU, int M, int num_diags);
  void fill_diags(cuComplex* diags, int N, int M, int L, int batchCount, int Nband, double coeff);
  void save_matrix(cuComplex* A, cuComplex* A_full, int* h_offsets, int KL, int KU, int num_diags, int N, int lda, int batchCount, int ind);
  void save_rhs(cuComplex* h_B, cuComplex* h_X_test, int ldb, int ind, int id, int nrhs);
  void fill_dB_array(cuComplex** dB_array,int N, int nrhs, int ldb, int batchCount);
  void invert(cuComplex* G, bool copy);

  cuComplex* h_A;
  cuComplex* A_bounce;
  cuComplex** A_inv_bounce;

  cuComplex** bounce_rhs;
  cuComplex** bounce_rhs_sol;
  cuComplex     ** res;
  cuComplex** d_A_bounce;
  cuComplex** d_Ainv_bounce;

  cuComplex** d_bounce_rhs;
  cuComplex** d_bounce_rhs_sol;
  cuComplex** bounce_rhs_apar;
  cuComplex** d_bounce_rhs_apar;

  int N, KL, KU, nrhs, ldda, lddb, batchCount;
  int** dipiv_array;
  int* dinfo_array;
  cuComplex** dA_array;
  cuComplex** dB_array;
  cuComplex** dB_array_apar;

  magma_queue_t my_queue;

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
  int info;

  cuComplex* test;

  cublasHandle_t cublasH = NULL;
  cudaStream_t stream_cublas = NULL;

  const cuComplex alpha = make_cuComplex(1.0,0.0);
  const cuComplex beta = make_cuComplex(0.0,0.0);

  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;





};

