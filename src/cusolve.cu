#include "cusolve.h"

Cusolve::Cusolve(Parameters *pars, Grids *grids, Geometry *geo, double p, double r, double u, bool sdirk, double dt_in, double vte):
	p_(p), r_(r), u_(u), sdirk_(sdirk), grids_(grids), pars_(pars), geo_(geo), A_bounce(nullptr), dt_(dt_in), vte_(vte)
{
  
  A_bounce = nullptr;	

  size_t nzlm = sizeof(int) * grids_->Nz * grids_->Nz * pars_->nm_in * pars_->nl_in;
  LM = pars_->nm_in * pars_-> nl_in;
  size_t LM2 = sizeof(cuComplex) *LM*LM;
  int num_coeff = 0;
  if (sdirk_){
    num_coeff = 1;
  }
  else{
    if (p_ != 0){
      num_coeff = 2;
    }
    else{
      num_coeff = 3;
    }
  }
     
  A_bounce = (cuComplex**) malloc(sizeof(cuComplex*)*num_coeff*grids_->Nz);
  cuComplex* LU = (cuComplex*) malloc(sizeof(cuComplex)*LM*LM); 
  d_Ipiv = (int64_t**)malloc(sizeof(int64_t*)*grids_->Nz);
 
//  double* dcoeff = nullptr;
 // checkCuda(cudaMalloc((void**) &dcoeff, num_coeff));
  double* dcoeff = (double*) malloc(sizeof(double)*num_coeff);
  if (num_coeff == 1){
    dcoeff[0] = r_;
  }
  else if (num_coeff == 2){
    dcoeff[0] = r_;
    dcoeff[1] = u_;
  }
  else{
    dcoeff[0] = p_;
    dcoeff[1] = r_;
    dcoeff[2] = u_;

  }


  for (int i = 0; i < num_coeff; i++){
    for (int j = 0; j < grids_->Nz; j++){
      checkCuda(cudaMalloc((void**) &A_bounce[j + i*num_coeff], LM2));
      checkCuda(cudaMalloc((void**) &d_Ipiv[j + i*num_coeff], sizeof(int64_t)*LM)); 

    }
  }

  DEBUGPRINT("Allocated an A_bounce array of size %.2f MB\n", nzlm/1024./1024.);

  int nn1, nt1, nb1, nn2, nt2, nb2, nn3, nt3, nb3;

  nn1 = LM;		    nt1 = min(nn1, 32 );   nb1 = 1 + (nn1-1)/nt1;
  nn2 = 1;         	    nt2 = min(nn2,  4 );   nb2 = 1 + (nn2-1)/nt2;
  nn3 = 1;         	    nt3 = min(nn3,  4 );   nb3 = 1 + (nn3-1)/nt3;
  
  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3);
  printf("num_coeff is %d\n", num_coeff);
  printf("coeff is %f\n", dcoeff[0]);
  for (int i = 0; i < num_coeff; i++){
    for (int j = 0; j < grids_->Nz; j++){
       initialize_A_bounce<<<dG, dB>>>(A_bounce[j + i*num_coeff], LM, pars_->nm_in, pars_->nl_in, geo_->bgrad, dcoeff[i], dt_,vte,j);

    }
  }
 
/*  checkCuda(cudaMemcpy(LU, A_bounce[11], LM2, cudaMemcpyDeviceToHost));
  
  printf("bgrad is %f\n", geo_->bgrad_h[11]);

  for (int i = 0; i < LM; i++) {
        for (int j = 0; j < LM; j++) {
            std::printf("%0.6f ", LU[j * LM + i].x);
        }
        std::printf("\n");
  }
*/

  d_info = nullptr;     /* error info */

  size_t workspaceInBytesOnDevice = 0; /* size of workspace */
  void *d_work = nullptr;              /* device workspace for getrf */
  size_t workspaceInBytesOnHost = 0;   /* size of workspace */
  void *h_work = nullptr;              /* host workspace for getrf */

  pivot_on = 1;
  const int algo = 0;
  
  cusolverH = NULL;
  stream = NULL;

  if (pivot_on) {
      std::printf("pivot is on : compute P*A = L*U \n");
  } else {
      std::printf("pivot is off: compute A = L*U (not numerically stable)\n");
  }
    
  /* step 1: create cusolver handle, bind a stream */
  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

  checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
  
  using data_type = cuComplex;

    /* Create advanced params */
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));
  if (algo == 0) {
      std::printf("Using New Algo\n");
      CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0));
  } else {
      std::printf("Using Legacy Algo\n");
      CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_1));
  }


 // print_matrix(LM, LM, LU, LM);

  int ind;
  for (int i = 0; i < num_coeff; i++){
    for(int iz = 0; iz < grids_->Nz; iz++){
	ind = iz + i*num_coeff; 
   	CUSOLVER_CHECK(
	    cusolverDnXgetrf_bufferSize(cusolverH, params, LM, LM, CUDA_C_32F, A_bounce[ind],
					LM, CUDA_C_32F, &workspaceInBytesOnDevice,
					&workspaceInBytesOnHost));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));

	if (0 < workspaceInBytesOnHost) {
	    h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
	    if (h_work == nullptr) {
		throw std::runtime_error("Error: h_work not allocated.");
	    }
	}
	
	if (pivot_on) {
	    CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, LM, LM, CUDA_C_32F,
					    A_bounce[ind], LM, d_Ipiv[ind], CUDA_C_32F, d_work,
					    workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info));
	} else {
	    CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, LM, LM, CUDA_C_32F,
					    A_bounce[ind], LM, nullptr, CUDA_C_32F,
					    d_work, workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info));
	}


	checkCuda(cudaStreamSynchronize(stream));
   
    }
  }
  
/*  for (int iz = 0; iz < grids_->Nz; iz++){
    checkCuda(cudaMemcpy(LU, A_bounce[iz], LM2, cudaMemcpyDeviceToHost));
    for (int i = 0; i < LM; i++) {
          for (int j = 0; j < LM; j++) {
  //            std::printf("%0.2f + %0.2fj ", LU[j * LM + i].x, LU[j * LM + i].y);
  	      if (j == i && (LU[j*LM+i].x != 1.0 || LU[j*LM + i].y != 0.0)){
  	        printf("Wrong at iz = %d, i = %d, j = %d\n", iz, i, j);
  	      }
  	      if (j != i && (LU[j*LM+i].x != 0.0 || LU[j*LM + i].y != 0.0)){
  	        printf("Wrong at iz = %d, i = %d, j = %d\n", iz, i, j);
  	      }
  
          }
      }
    printf("Finished iz = %d\n", iz); 
  }
*/
  CUBLAS_CHECK(cublasCreate(&cublasH));

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_cublas, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream_cublas));

 test = (cuComplex*) malloc(sizeof(cuComplex)*LM*grids_->Nx*grids_->Nyc);
 test2 = (cuComplex*) malloc(sizeof(cuComplex)*LM*grids_->Nx*grids_->Nyc);

}

void Cusolve::invert(cuComplex* rhs, cuComplex* res, int iz){

//  checkCuda(cudaMemcpy(test, rhs, sizeof(cuComplex)*LM*grids_->Nx*grids_->Nyc, cudaMemcpyDeviceToHost));
//printf("test.x = %f, test.y = %f\n", test[0].x, test[0].y);
  if (pivot_on) {
      CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, LM, grids_->Nx*grids_->Nyc,
  				CUDA_C_32F, A_bounce[iz], LM, d_Ipiv[iz],
  				CUDA_C_32F, rhs, LM, d_info));
  } else {
      CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, LM, grids_->Nx*grids_->Nyc,
  				CUDA_C_32F, A_bounce[iz], LM, nullptr,
  				CUDA_C_32F, rhs, LM, d_info));
  }
/*  CUBLAS_CHECK(
    cublasCgemm(cublasH, transa, transb, LM,grids_->Nx*grids_->Nyc,LM, &alpha, A_bounce[iz], LM, rhs, LM, &beta, res, LM));*/

/*  checkCuda(cudaMemcpy(test2, rhs, sizeof(cuComplex)*LM*grids_->Nx*grids_->Nyc, cudaMemcpyDeviceToHost));
  for (int i = 0; i < LM*grids_->Nx*grids_->Nyc; i++){
    if (test[i].x != test2[i].x || test[i].y != test2[i].y){
//      printf("Error of at i = %d\n", i);
//      printf("");
    }
  }*/
//  checkCuda(cudaMemcpy(test, rhs[0], sizeof(cuComplex), cudaMemcpyDeviceToHost));
//  printf("test.x = %f, test.y = %f\n", test.x, test.y);


}

void print_matrix(const int m, const int n, const cuComplex *A, const int lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}


void Cusolve::factorize(){
  
}

/*
void Cusolve::factor_bounce(const int p, const int q, const int r, const bool sdirk){
  

}*/
