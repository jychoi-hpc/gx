#include "hermite_transform.h"

HermiteTransform::HermiteTransform(Grids* grids) :
  grids_(grids), M(grids->Nm), 
  toGrid(nullptr), toSpectral(nullptr), halfHermiteCoeff(nullptr), roots(nullptr)
{
  double * toGrid_h     = nullptr;  double * toSpectral_h = nullptr;  double * roots_h      = nullptr;
  double * halfHermiteCoeff_h = nullptr;
  toGrid_h     = (double*) malloc(sizeof(double)*M*M);
  toSpectral_h = (double*) malloc(sizeof(double)*M*M);
  halfHermiteCoeff_h = (double*) malloc(sizeof(double)*M*M);
  roots_h      = (double*) malloc(sizeof(double)*M);

  cudaMalloc ((void**) &toGrid,     sizeof(double)*M*M);
  cudaMalloc ((void**) &toSpectral, sizeof(double)*M*M);
  cudaMalloc ((void**) &halfHermiteCoeff, sizeof(double)*M*M);
  cudaMalloc ((void**) &roots,      sizeof(double)*M);

  initTransforms(toGrid_h, toSpectral_h, halfHermiteCoeff_h, roots_h);
  printf("\nhalfHermiteCoeff:\n");
  for(int i=0; i<M*M; i++) printf("%f  ", halfHermiteCoeff_h[i]);

  CP_TO_GPU (toGrid,     toGrid_h,     sizeof(double)*M*M);
  CP_TO_GPU (toSpectral, toSpectral_h, sizeof(double)*M*M);
  CP_TO_GPU (halfHermiteCoeff,      halfHermiteCoeff_h,      sizeof(double)*M*M);
  CP_TO_GPU (roots,      roots_h,      sizeof(double)*M);

  if (toGrid_h)     free (toGrid_h);
  if (toSpectral_h) free (toSpectral_h);
  if (halfHermiteCoeff_h) free (halfHermiteCoeff_h);
  if (roots_h)      free (roots_h);
}

HermiteTransform::~HermiteTransform()
{
  if (toGrid)     cudaFree(toGrid);
  if (toSpectral) cudaFree(toSpectral);
  if (halfHermiteCoeff) cudaFree(halfHermiteCoeff);
  if (roots)      cudaFree(roots);
}

void HermiteTransform::initTransforms(double* toGrid_h, double* toSpectral_h, double* halfHermiteCoeff_h, double* roots_h)
{
  int i, j;
  gsl_matrix *Jacobi = gsl_matrix_alloc(M,M);
  gsl_matrix_set_zero (Jacobi);
  
  for (i = 0; i < M-1; i++) {
    gsl_matrix_set(Jacobi, i, i+1, sqrt(i+1.));
    gsl_matrix_set(Jacobi, i+1, i, sqrt(i+1.));
  }
    
  gsl_vector *eval = gsl_vector_alloc (M);
  gsl_matrix *evec = gsl_matrix_alloc (M, M);
  gsl_eigen_symmv_workspace *wrk = gsl_eigen_symmv_alloc (M);
  gsl_eigen_symmv (Jacobi, eval, evec, wrk);                 
  gsl_eigen_symmv_free (wrk);
  gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_VAL_ASC);

  gsl_matrix *toGrid = gsl_matrix_alloc(M,M);
  gsl_matrix_set_zero (toGrid);

  gsl_matrix *toGridPosHalf = gsl_matrix_alloc(M,M);
  gsl_matrix_set_zero (toGridPosHalf);

  gsl_matrix *toSpectral = gsl_matrix_alloc(M, M);
  gsl_matrix_set_zero (toSpectral);

  for (i = 0; i < M; i++) {
    double x_i = gsl_vector_get(eval, i);
    roots_h[i] = (float) x_i;
    // use analytic formula for weights instead of eigenvectors 
    // because eigenvectors inaccurate for large M
    double wgt_i = pow(gsl_sf_hermite_func(M-1, x_i/sqrtf(2.0))*pow(M_PI,.25),-2)/M;
    for (j = 0; j < M; j++) {
      double poly_ij = gsl_sf_hermite_func(j, x_i/sqrtf(2.0))*pow(M_PI,0.25);
      gsl_matrix_set(toGrid, i, j, poly_ij);
      gsl_matrix_set(toSpectral, j, i, poly_ij*wgt_i);
      if(x_i>0.) {
        gsl_matrix_set(toGridPosHalf, i, j, poly_ij);
      } else {
        gsl_matrix_set(toGridPosHalf, i, j, 0.);
      }
    }
  }

  gsl_matrix *halfHermiteCoeff = gsl_matrix_alloc(M, M);
  gsl_matrix_set_zero(halfHermiteCoeff);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, toSpectral, toGridPosHalf, 0.0, halfHermiteCoeff);

  memcpy(toGrid_h, toGrid->data, sizeof(double)*M*M);
  memcpy(toSpectral_h, toSpectral->data, sizeof(double)*M*M);
  memcpy(halfHermiteCoeff_h, halfHermiteCoeff->data, sizeof(double)*M*M);

  gsl_matrix_free(toGrid);
  gsl_matrix_free(toGridPosHalf);
  gsl_matrix_free(toSpectral);
  gsl_matrix_free(halfHermiteCoeff);
  gsl_matrix_free(Jacobi);
  gsl_vector_free(eval);
  gsl_matrix_free(evec);
}

void HermiteTransform::initTransforms2(double* toGrid_h, double* toSpectral_h, double* halfHermiteCoeff_h, double* roots_h)
{
  int i, j;
  gsl_matrix *toGrid = gsl_matrix_alloc(M,M);
  gsl_matrix_set_zero (toGrid);

  gsl_matrix *toGridPosHalf = gsl_matrix_alloc(M,M);
  gsl_matrix_set_zero (toGridPosHalf);

  gsl_vector *roots = gsl_vector_alloc(M);

  // get all roots of Mth hermite probabilist polynomial
  // note: gsl_sf_hermite_prob_zero only valid for 1<=i<=M/2 (positive roots only)
  if(M%2 == 0) {
    for (i = 1; i <= M/2; i++) {
      double x_i = gsl_sf_hermite_prob_zero(M, i);
      gsl_vector_set(roots, i+M/2-1, x_i);
      gsl_vector_set(roots, M/2-i, -x_i);
    }
  } else {
    gsl_vector_set(roots, M/2, 0.);
    for (i = 1; i <= M/2; i++) {
      double x_i = gsl_sf_hermite_prob_zero(M, i);
      gsl_vector_set(roots, i+M/2, x_i);
      gsl_vector_set(roots, M/2-i, -x_i);
    }
  }

  for (i = 0; i < M; i++) {
    double x_i = gsl_vector_get(roots, i);
    for (j = 0; j < M; j++) {
      gsl_matrix_set(toGrid, j, i, gsl_sf_hermite_prob(j, x_i)/sqrtf(gsl_sf_fact(j))*gsl_sf_exp(-x_i*x_i/2));
      if(x_i>0.) {
        gsl_matrix_set(toGridPosHalf, j, i, gsl_sf_hermite_prob(j, x_i)/sqrtf(gsl_sf_fact(j))*gsl_sf_exp(-x_i*x_i/2));
      } else {
        gsl_matrix_set(toGridPosHalf, j, i, 0.);        
      }
    }
  }

  gsl_matrix *toSpectral = gsl_matrix_alloc(M, M);
  gsl_matrix_memcpy(toSpectral, toGrid);
  gsl_permutation *p = gsl_permutation_alloc(M);
  int s;
  //  gsl_linalg_LU_decomp(toSpectral, p, &s);
  //  gsl_linalg_LU_invx(toSpectral, p);

  gsl_matrix *halfHermiteCoeff = gsl_matrix_alloc(M, M);
  gsl_matrix_set_zero(halfHermiteCoeff);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, toGridPosHalf, toSpectral, 0.0, halfHermiteCoeff);

  memcpy(toGrid_h, toGrid->data, sizeof(double)*M*M);
  memcpy(toSpectral_h, toSpectral->data, sizeof(double)*M*M);
  memcpy(halfHermiteCoeff_h, halfHermiteCoeff->data, sizeof(double)*M*M);
  memcpy(roots_h, roots->data, sizeof(double)*M);

  gsl_matrix_free(toGrid);
  gsl_matrix_free(toGridPosHalf);
  gsl_matrix_free(toSpectral);
  gsl_matrix_free(halfHermiteCoeff);
  gsl_vector_free(roots);
  gsl_permutation_free(p);
}
