#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include "parameters.h"
#include "device_funcs.h"

void smith_par_getAs (int n, int q, cuComplex *x_answer);
void get_power_series (cuDoubleComplex *power_series, int q);
void fill_r_matrix (cuDoubleComplex *power_series, cuDoubleComplex **rMatrix, int q);
void get_normalized_hermite_coefficients (cuDoubleComplex **matrix, int n, double scaling, char type);
int linearSolverLU (cusolverDnHandle_t handle, int n, const cuDoubleComplex *Acopy, int lda, const cuDoubleComplex *b, cuDoubleComplex *x); 

/*
int main() {	
  
  int n,q;

	//Request number of A's to find
	printf("q (# of A terms): ");
	scanf("%d", &q);
	
	//Request a degree for the normalized hermite polynomial
	printf("n (degree of Hermite moment being closed): ");
	scanf("%d", &n);
  printf("\n");
  
  cuComplex *x_answer = (cuComplex*) malloc(q*sizeof(cuComplex));
  
  smith_par_getAs(n,q, x_answer);
  
  for (int i = 0; i < q; i++) {
    printf("A_n-%d: %f + %fi\n", i+1, cuCrealf(x_answer[i]),cuCimagf(x_answer[i]));
  }
  
  free(x_answer);
  
  return 0;
  } */

/* Pade Approximant to find A coefficients
 * Calculation can be found on pg. 37-38 of Smith's thesis
 * n is the degree of the Hermite moment being closed
 * q is the number of coefficients used for the nth moment closure
 * x_answer stores the final result */
void smith_par_getAs(int n, int q, cuComplex *x_answer) {
  int i, j, k;
  
  // Create matrices for r, P coefficients, Q coefficients, P, Q, and the final LHS matrix
  cuDoubleComplex **rMatrix       = (cuDoubleComplex **) malloc(q *     sizeof(cuDoubleComplex *));
  cuDoubleComplex **PCoefficients = (cuDoubleComplex **) malloc((n+1) * sizeof(cuDoubleComplex *));
  cuDoubleComplex **QCoefficients = (cuDoubleComplex **) malloc((n+1) * sizeof(cuDoubleComplex *));
  cuDoubleComplex **PMatrix       = (cuDoubleComplex **) malloc(q *     sizeof(cuDoubleComplex *));
  cuDoubleComplex **QMatrix       = (cuDoubleComplex **) malloc(q *     sizeof(cuDoubleComplex *));
  cuDoubleComplex **lhsMatrix     = (cuDoubleComplex **) malloc(q *     sizeof(cuDoubleComplex *));
  
  for (i = 0; i < q; i++) {
    rMatrix[i] =       (cuDoubleComplex *) calloc(q,   sizeof(cuDoubleComplex));
    PCoefficients[i] = (cuDoubleComplex *) calloc(i+1, sizeof(cuDoubleComplex));
    QCoefficients[i] = (cuDoubleComplex *) calloc(i+1, sizeof(cuDoubleComplex));
    PMatrix[i] =       (cuDoubleComplex *) calloc(q,   sizeof(cuDoubleComplex));
    QMatrix[i] =       (cuDoubleComplex *) calloc(q,   sizeof(cuDoubleComplex));
    lhsMatrix[i] =     (cuDoubleComplex *) calloc(q,   sizeof(cuDoubleComplex)); 
  }
  
  for (i = q; i <= n; i++) {
    PCoefficients[i] = (cuDoubleComplex *) calloc(i+1, sizeof(cuDoubleComplex));
    QCoefficients[i] = (cuDoubleComplex *) calloc(i+1, sizeof(cuDoubleComplex)); 
  }

  // Create Pn and Qn vectors
  cuDoubleComplex *Pn = (cuDoubleComplex *) calloc(q, sizeof(cuDoubleComplex));
  cuDoubleComplex *Qn = (cuDoubleComplex *) calloc(q, sizeof(cuDoubleComplex));

  // Create lhsVector and rhsVector (lhsVector is the final LHS matrix as a single vector;
  // rhsVector is the b vector in Ax = b)
  cuDoubleComplex *lhsVector = (cuDoubleComplex *) calloc(q*q, sizeof(cuDoubleComplex));
  cuDoubleComplex *rhsVector = (cuDoubleComplex *) calloc(q,   sizeof(cuDoubleComplex));
  
//Create power_series array
  cuDoubleComplex *power_series = (cuDoubleComplex *) calloc(q, sizeof(cuDoubleComplex));

  cuDoubleComplex I = make_cuDoubleComplex(0., 1.);
  
  // Fill power_series array and rMatrix
  get_power_series(power_series, q);
  fill_r_matrix(power_series, rMatrix, q);

  // Get P and Q coefficients
  get_normalized_hermite_coefficients(PCoefficients, n, 1/sqrt(2), 'P');
  get_normalized_hermite_coefficients(QCoefficients, n, 1/sqrt(2), 'Q');
  
  //Fill P and Q matrices and vectors
  
  int col = 0; // column of P and Q matrix filled
  int d; // degree of polynomial filling each column
  
  for(d = n; d >= n - q; d--){	
    if (d != n) {
      for (i = 0; i < q; i++) {
	
	// Only filling indices where the degree of the term is less than
	// or equal to the degree of the corresponding polynomial
	if (i <= d) {
	  PMatrix[i][col] = PCoefficients[d][i];
	  QMatrix[i][col] = QCoefficients[d][i];
	}
      }
      col++;
    }
    else {
      for (i = 0; i < q; i++) {
	Pn[i] = PCoefficients[n][i];
	Qn[i] = QCoefficients[n][i];
      }
    }
  }
  
  // Freeing PCoefficients and QCoefficients
  for (i = 0; i <= n; i++) {
    free(PCoefficients[i]);
    free(QCoefficients[i]);
  }
  
  free(PCoefficients);
  free(QCoefficients);
  
  // Get LHS matrix
  cuDoubleComplex sum;
  for(i = 0; i < q; i++){
    for(j = 0; j < q; j++){
      sum.x = 0;
      sum.y = 0;
      for(k = 0; k < q; k++){
	sum = cuCadd(sum, cuCmul(rMatrix[i][k], PMatrix[k][j]));
      }
      lhsMatrix[i][j] = cuCsub(sum, cuCmul(I,QMatrix[i][j]));
    }		
  }
  
  //Transpose of lhsMatrix and putting into one vector for cuSolver/cuBlas call below
  for(i = 0; i < q; i++){
    for(j = 0; j < q; j++){
      lhsVector[j + q*i] = lhsMatrix[j][i];
    }
  }
  
  // Get RHS vector	
  for(j = 0; j < q; j++){
    sum.x = 0;
    sum.y = 0;
    for(k = 0; k < q; k++){
      sum = cuCadd(sum, cuCmul(rMatrix[j][k],Pn[k]));
    }
    
    rhsVector[j] = cuCsub(sum, cuCmul(I,Qn[j]));
  }
  
  // Creating CUDA array copy of lhsVector
  cuDoubleComplex * lhsVector_d = nullptr;
  cudaMalloc(&lhsVector_d, q*q*sizeof(cuDoubleComplex));
  //  cudaMemcpy(lhsVector_d, (cuDoubleComplex*) lhsVector, q*q*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice); 
  CP_TO_GPU (lhsVector_d, (cuDoubleComplex*) lhsVector, q*q*sizeof(cuDoubleComplex));
  
  // Creating CUDA array copy of rhsVector
  cuDoubleComplex *rhsVector_d = nullptr;
  cudaMalloc(&rhsVector_d, q*sizeof(cuDoubleComplex));
  //  cudaMemcpy(rhsVector_d, (cuDoubleComplex*) rhsVector, q*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  CP_TO_GPU (rhsVector_d, (cuDoubleComplex*) rhsVector, q*sizeof(cuDoubleComplex));
  
  // Setting up CUDA handlers and stream
  cusolverDnHandle_t handle = NULL;
  cublasHandle_t cublasHandle = NULL;
  cudaStream_t stream = NULL;
  
  cusolverDnCreate(&handle);
  cublasCreate(&cublasHandle);
  cudaStreamCreate(&stream);
  cusolverDnSetStream(handle, stream);
  cublasSetStream(cublasHandle, stream);
  
  // Calling cuSolver and cuBLAS libraries to solve Ax = b 
  linearSolverLU(handle, q, lhsVector_d, q, rhsVector_d, rhsVector_d);
  
  // Converting cuDoubleComplex answer to cuComplex and storing it in x_answer
  cuComplex * rhsVector_d_float = nullptr;
  cudaMalloc(&rhsVector_d_float, q*q*sizeof(cuComplex));
  
  castDoubleToFloat<<<1,1>>>(rhsVector_d, rhsVector_d_float, q);
  
  CP_TO_CPU (x_answer, rhsVector_d_float, sizeof(cuComplex)*q);
  
  // Print only if debugging
  if (0==1) {
    
    // Print r values
    printf("r coefficients\n");
    for (i = 0; i < q; i++) printf("r_%d: %f + i%f\n", i, power_series[i].x,power_series[i].y);
    
    // Print r matrix
    printf("\nMatrix r\n");
    for (i = 0; i < q; i++) {
      for (j = 0; j < q; j++) {
	printf("%2.2f + %2.2fi ",rMatrix[i][j].x,rMatrix[i][j].y);
      }
      printf("\n");
    }
    
    printf("\n");
    
    // Print P matrix
    printf("Matrix P\n");
    for (i = 0; i < q; i++) {
      for (j = 0; j < q; j++) {
	printf("%2.4f  ",PMatrix[i][j].x);
      }
      printf("\n");
    }
    
    printf("\n");
    
    // Print Q Matrix
    printf("Matrix Q\n");
    for (i = 0; i < q; i++) {
      for (j = 0; j < q; j++) {
	printf("%2.4f  ",QMatrix[i][j].x);
      }
      printf("\n");
    }
    
    printf("\n");
    
    // Print final LHS matrix and RHS vector for Ax = b
    printf("Before\n");
    
    printf("LHS Matrix\n");
    for (i = 0; i < q; i++) {
      for (j = 0; j < q; j++) {
	printf("%2.2f + %2.2fi ",lhsMatrix[i][j].x,lhsMatrix[i][j].y);
      }
      printf("\n");
    }
    
    printf("\nRHS Vector\n");
    for (i = 0; i < q; i++) printf("%2.4f + %2.4fi\n",rhsVector[i].x,rhsVector[i].y);
    
    printf("\nAfter\n");
    for (int i = 0; i < q; i++) printf("A_n-%d: %f + %fi\n", i+1, cuCrealf(x_answer[i]),cuCimagf(x_answer[i]));
  }
  
  // Freeing dynamically allocated arrays and matrices
  
  for (i = 0; i < q; i++) {
    free(rMatrix[i]);
    free(PMatrix[i]);
    free(QMatrix[i]);
    free(lhsMatrix[i]);
  }
  
  free(power_series);
  free(rMatrix);
  free(PMatrix);
  free(QMatrix);
  free(lhsMatrix);
  free(Pn);
  free(Qn);
  free(lhsVector);
  
  if (lhsVector_d)       cudaFree(lhsVector_d);
  if (rhsVector_d)       cudaFree(rhsVector_d);
  if (rhsVector_d_float) cudaFree(rhsVector_d_float);
}

/* This function finds the coefficients of the Taylor series expansion of
 * R00, which is (-1/sqrt(2))*Z(w/sqrt(2)). */
//void get_power_series(double complex *power_series, int q) {
void get_power_series(cuDoubleComplex *power_series, int q) {
  int j;
  cuDoubleComplex twoI = make_cuDoubleComplex(0., 2.);
  cuDoubleComplex fac  = make_cuDoubleComplex(1., 0.);
  cuDoubleComplex fac2 = make_cuDoubleComplex(sqrt(2.), 0.);
  cuDoubleComplex root2= make_cuDoubleComplex(sqrt(2.), 0.);
  cuDoubleComplex res;
  
  for (j = 0; j < q; j++) {
    // Below definition comes from Smith's thesis on pg. 13
    res = cuCdiv(fac, fac2);
    // Alternate definition of above formula, works better for large number of coefficients
    power_series[j].x = res.x * sqrt(M_PI)*pow(2, -j)/tgamma(j/2.0 +1);
    power_series[j].y = res.y * sqrt(M_PI)*pow(2, -j)/tgamma(j/2.0 +1);
    fac = cuCmul(fac, twoI);
    fac2 = cuCmul(fac2, root2);
  }
}

/* This function fills the rMatrix with the power series coefficients found in get_power_series */
void fill_r_matrix(cuDoubleComplex *power_series, cuDoubleComplex **rMatrix, int q) {
  int i,j;
  
  for (j = 0; j < q; j++) {
    for (i = j; i < q; i++) {
      rMatrix[i][j].x = power_series[i-j].x;
      rMatrix[i][j].y = power_series[i-j].y;
    }
  }
}

/* Fills matrix with coefficients of the first n Hermite polynomial with specified scaling
 * and type (P for regular normalized Hermite, Q for conjugate polynomials)
 * Generates physicists' Hermite polynomials from 0 to n of form: 1/sqrt(j!2^j)H_j(scaling*x) */
void get_normalized_hermite_coefficients(cuDoubleComplex **matrix, int n, double scaling, char type) {

  int i,j;
  
  if (type == 'P') {
    matrix[0][0].x = 1;    matrix[0][0].y = 0;
    matrix[1][0].x = 0;    matrix[1][0].y = 0;
    matrix[1][1].x = sqrt(2)*scaling;    matrix[1][1].y = 0;
} else if (type == 'Q') {
    matrix[0][0].x = 0;    matrix[0][0].y = 0;
    matrix[1][0].x = sqrt(2)*scaling;    matrix[1][0].y = 0;
    matrix[1][1].x = 0;    matrix[1][1].y = 0;
  }
  
  /* Calculation using standard recurrence relation of normalized Hermite polynomials:
   * h_i(x) = sqrt(2/i)*x*h_(i-1)(x) - sqrt((i-1)/i)*h_(i-2)(x) */
  for (i = 2; i <= n; i++) {
    for (j = 0; j <= i; j++){
      double di = (double) i;
      if (j == 0) {
	matrix[i][0].x = -sqrt((di -1)/di)*matrix[i-2][0].x;
	matrix[i][0].y = -sqrt((di -1)/di)*matrix[i-2][0].y;
      } else {
	if (i-2 >= j)   matrix[i][j].x += -sqrt((di -1)/di)     * matrix[i-2][j].x;
	if (i-2 >= j)   matrix[i][j].y += -sqrt((di -1)/di)     * matrix[i-2][j].y;
	if (i-1 >= j-1) matrix[i][j].x +=  sqrt(2/di) * scaling * matrix[i-1][j-1].x;
	if (i-1 >= j-1) matrix[i][j].y +=  sqrt(2/di) * scaling * matrix[i-1][j-1].y;
      }
    }
  }
}


/* Solve Ax = b by LU decomposition with partial pivoting */
int linearSolverLU(cusolverDnHandle_t handle, int n, const cuDoubleComplex *Acopy,
		   int lda, const cuDoubleComplex *b, cuDoubleComplex *x) {
  int bufferSize = 0;
  int *info = nullptr;
  cuDoubleComplex *buffer = nullptr;
  cuDoubleComplex *A = nullptr;
  int *ipiv = nullptr; // pivoting sequence
  int h_info = 0;
  
  cusolverDnZgetrf_bufferSize(handle, n, n, (cuDoubleComplex*)Acopy, lda, &bufferSize);
  
  cudaMalloc(&info,   sizeof(int));
  cudaMalloc(&buffer, sizeof(cuDoubleComplex)*bufferSize);
  cudaMalloc(&A,      sizeof(cuDoubleComplex)*lda*n);
  cudaMalloc(&ipiv,   sizeof(int)*n);
    
  // Prepare a copy of A because getrf will overwrite A with L
  CP_TO_GPU(A, Acopy, sizeof(cuDoubleComplex)*lda*n);
  cudaMemset(info, 0, sizeof(int));
  
  cusolverDnZgetrf(handle, n, n, A, lda, buffer, ipiv, info);
  CP_TO_CPU(&h_info, info, sizeof(int));
  
  if ( 0 != h_info ){
    fprintf(stderr, "Error: LU factorization failed\n");
  }
  
  CP_ON_GPU(x, b, sizeof(cuDoubleComplex)*n);
  cusolverDnZgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info);
  cudaDeviceSynchronize();
  
  if (info  ) { cudaFree(info  ); }
  if (buffer) { cudaFree(buffer); }
  if (A     ) { cudaFree(A); }
  if (ipiv  ) { cudaFree(ipiv);}
  
  return 0;
}

