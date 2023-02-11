#include "hermite_transform.h"

HermiteTransform::HermiteTransform(Grids* grids, float vmax) :
  grids_(grids), M(grids->Nm_glob), vmax(vmax), 
  toGrid(nullptr), toSpectral(nullptr), roots(nullptr)
{
  double * toGrid_h     = nullptr;  double * toSpectral_h = nullptr;  double * roots_h      = nullptr;
  toGrid_h     = (double*) malloc(sizeof(double)*M*M);
  toSpectral_h = (double*) malloc(sizeof(double)*M*M);
  roots_h      = (double*) malloc(sizeof(double)*M);

  //cudaMalloc ((void**) &toGrid,     sizeof(double)*M*M);
  //cudaMalloc ((void**) &toSpectral, sizeof(double)*M*M);
  //cudaMalloc ((void**) &roots,      sizeof(double)*M);

  initTransforms(toGrid_h, toSpectral_h, roots_h);

  //CP_TO_GPU (toGrid,     toGrid_h,     sizeof(double)*M*M);
  //CP_TO_GPU (toSpectral, toSpectral_h, sizeof(double)*M*M);
  //CP_TO_GPU (roots,      roots_h,      sizeof(double)*M);

  if (toGrid_h)     free (toGrid_h);
  if (toSpectral_h) free (toSpectral_h);
  if (roots_h)      free (roots_h);
}

HermiteTransform::~HermiteTransform()
{
  if (toGrid)     cudaFree(toGrid);
  if (toSpectral) cudaFree(toSpectral);
  if (roots)      cudaFree(roots);
}

void HermiteTransform::initTransforms(double* toGrid_h, double* toSpectral_h, double* roots_h)
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

  gsl_matrix *toSpectral = gsl_matrix_alloc(M, M);
  gsl_matrix_set_zero (toSpectral);

  // compute scaling factor, scale_fac, to enforce vmax
  float x_M = gsl_vector_get(eval, M-1);
  if (vmax>0) scale_fac = vmax/x_M;
  else {scale_fac = 1.0; vmax = x_M;}

  for (i = 0; i < M; i++) {
    double x_i = gsl_vector_get(eval, i)/scale_fac;
    roots_h[i] = (float) x_i;
    // use analytic formula for weights instead of eigenvectors 
    // because eigenvectors inaccurate for large M
    double wgt_i = pow(gsl_sf_hermite_func(M-1, scale_fac*x_i/sqrtf(2.0))*pow(M_PI,.25),-2)/M;
    for (j = 0; j < M; j++) {
      double poly_ij = gsl_sf_hermite_func(j, scale_fac*x_i/sqrtf(2.0))*pow(M_PI,0.25);
      gsl_matrix_set(toGrid, i, j, poly_ij);
      gsl_matrix_set(toSpectral, j, i, poly_ij*wgt_i);
    }
  }

  gsl_matrix_free(toGrid);
  gsl_matrix_free(toSpectral);
  gsl_matrix_free(Jacobi);
  gsl_vector_free(eval);
  gsl_matrix_free(evec);
}
