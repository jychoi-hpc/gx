#include "laguerre_transform.h"

LaguerreTransform::LaguerreTransform(Grids* grids, int batch_size) :
  grids_(grids), L(grids->Nl), J(grids->Nj), batch_size_(batch_size),
  toGrid(nullptr), toSpectral(nullptr), roots(nullptr)
{
  float * toGrid_h     = nullptr;  float * toSpectral_h = nullptr;  float * roots_h      = nullptr;
  toGrid_h     = (float*) malloc(sizeof(float)*L*J);
  toSpectral_h = (float*) malloc(sizeof(float)*L*J);
  roots_h      = (float*) malloc(sizeof(float)*J);

  cudaMalloc ((void**) &toGrid,     sizeof(float)*L*J);
  cudaMalloc ((void**) &toSpectral, sizeof(float)*L*J);
  cudaMalloc ((void**) &roots,      sizeof(float)*J);

  initTransforms(toGrid_h, toSpectral_h, roots_h);

  CP_TO_GPU (toGrid,     toGrid_h,     sizeof(float)*L*J);
  CP_TO_GPU (toSpectral, toSpectral_h, sizeof(float)*L*J);
  CP_TO_GPU (roots,      roots_h,      sizeof(float)*J);

  cublasCreate (&handle);
  if (toGrid_h)     free (toGrid_h);
  if (toSpectral_h) free (toSpectral_h);
  if (roots_h)      free (roots_h);
}

LaguerreTransform::~LaguerreTransform()
{
  if (toGrid)     cudaFree(toGrid);
  if (toSpectral) cudaFree(toSpectral);
  if (roots)      cudaFree(roots);
}

void LaguerreTransform::initTransforms(float* toGrid_h, float* toSpectral_h, float* roots_h)
{
  int i, j;
  gsl_matrix *Jacobi = gsl_matrix_alloc(J,J);
  gsl_matrix_set_zero (Jacobi);
  
  for (i = 0; i < J-1; i++) {
    gsl_matrix_set(Jacobi, i, i, 2.*i+1.);
    gsl_matrix_set(Jacobi, i, i+1, i+1.);
    gsl_matrix_set(Jacobi, i+1, i, i+1.);
  }
  gsl_matrix_set(Jacobi, J-1, J-1, 2*J-1);
    
  gsl_vector *eval = gsl_vector_alloc (J);
  gsl_matrix *evec = gsl_matrix_alloc (J, J);
  gsl_eigen_symmv_workspace *wrk = gsl_eigen_symmv_alloc (J);
  gsl_eigen_symmv (Jacobi, eval, evec, wrk);                 
  gsl_eigen_symmv_free (wrk);
  gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);

  // eval: eigenvalues of Jacobi matrix; roots of the (J)th Laguerre
  // evec: Laguerre cardinal polynomials; use these to get the weights 

  gsl_matrix *poly = gsl_matrix_alloc (J, J);               // pointer to defined type gsl_matrix
  gsl_matrix_set_zero (poly);
  for (i=0; i<J; i++) {      
    for (j=0; j<i+1; j++) {
      // i-th polynomial, j-th coefficient
      double tmp = gsl_sf_choose (i, j);      
      //      printf("tmp = %d\n",tmp);
      tmp *= gsl_pow_int(-1.0, j) / gsl_sf_fact(j) * gsl_pow_int(-1.0, i);
      //      printf("i, j, tmp = %d, %d, %e \n", i, j, tmp);
      gsl_matrix_set(poly, i, j, tmp);
    }
  }
  int ell;
  double x_i, wgt, Lmat;
  gsl_vector_view polyvec;

  for (j=0; j<J; j++) {
    x_i = gsl_vector_get (eval, j); 
    roots_h[j] = (float) x_i; // Used in argument of J0
    //    printf("roots_h[%d] =  %f \n",j,roots_h[j]);
    wgt = pow (gsl_matrix_get (evec, 0, j), 2); // square first element of j_th eigenvector
    //    printf("wgt = %e \n",wgt);
    // evaluate the ell-th polynomial at x(j) = x_j and multiply by weight(j) as needed
    for (ell=0; ell<L; ell++) {
      polyvec = gsl_matrix_row (poly, ell);
      Lmat = gsl_poly_eval(polyvec.vector.data, ell+1, x_i);
      //printf("j, l, L(j,l) = %d, %d, %e \n",j,ell,(float) Lmat);
      toGrid_h[ell + L*j] = (float) Lmat;
      Lmat = Lmat * wgt;
      toSpectral_h[j + J*ell] = (float) Lmat;
    }	
  }
  gsl_matrix_free(Jacobi);
  gsl_matrix_free(poly);
  gsl_vector_free(eval);
  gsl_matrix_free(evec);
}

void LaguerreTransform::transformToGrid(float* G_in, float* g_res)
{
  int m = grids_->NxNyNz;
  int n = J;
  int k = L;
  float alpha = 1.;
  int lda = grids_->NxNyNz;
  int strideA = grids_->NxNyNz*L;
  int ldb = L;
  int strideB = 0;
  float beta = 0.;
  int ldc = grids_->NxNyNz;
  int strideC = grids_->NxNyNz*J;
  //  printf("J = %d \n",J);
  int status; 
  status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
     m, n, k, &alpha,
     G_in, lda, strideA,
     toGrid, ldb, strideB,
     &beta, g_res, ldc, strideC,
     batch_size_); 
}

void LaguerreTransform::transformToSpectral(float* g_in, float* G_res)
{
  int m = grids_->NxNyNz;
  int n = L;
  int k = J;
  float alpha = 1.;
  int lda = grids_->NxNyNz;
  int strideA = grids_->NxNyNz*J;
  int ldb = J;
  int strideB = 0;
  float beta = 0.;
  int ldc = grids_->NxNyNz;
  int strideC = grids_->NxNyNz*L;

  // column-major order, according to the manuals. WTF?
  // So I should transpose everything? 

  // COLUMN MAJOR
  // G(l) = T(l, j) g(j)   (for every k)
  // G(k, l) = g(k, j) T(j, l)  <<<<<< Note that T is transposed...
  // G has leading dimension K and a new block starts K*L away 
  // g has leading dimension K and a new block starts K*J away 
  // T has leading dimension J and a stride of zero 

  // T has to be T[j + J*l] for this to hold together. 
  // ... and it does; the Python numpy flatten trick does that. 

  // C = A B ==> lda=K, strideA = K*J;  ldc=K, strideC=K*L; ldb=J and strideB=0
  // A has K rows so ::::::::: m = K   ////////////// yes
  // B has L columns so :::::: n = L-1   ////////////// yes
  // A has J columns so :::::: k = J-1   ////////////// yes

  // lda=K  ///////////////////////// yes
  // strideA=K*J    /////////////// yes
  // ldb=J          //////////////// yes
  // strideB=0 ////////////////////// yes
  // ldc=K  ///////////////////////// yes
  // strideC=K*L    //////////////  yes

  int status;
  status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
     m, n, k, &alpha,
     g_in, lda, strideA,
     toSpectral, ldb, strideB,
     &beta, G_res, ldc, strideC,
     batch_size_); 
}



