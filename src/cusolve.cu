#include "cusolve.h"

Cusolve::Cusolve(Parameters *pars, Grids *grids, Geometry *geo, double p, double r, double u, bool sdirk, double dt_in, double vte):
	p_(p), r_(r), u_(u), sdirk_(sdirk), grids_(grids), pars_(pars), geo_(geo), A_bounce(nullptr), dt_(dt_in), vte_(vte)
{
  
  A_bounce = nullptr;	

  size_t nzlm = sizeof(int) * grids_->Nz * grids_->Nz * pars_->nm_in * pars_->nl_in;
  LM = pars_->nm_in * pars_-> nl_in;
  size_t LM2 = sizeof(cuComplex) *LM*LM;
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
  h_Ipiv = (int64_t**)malloc(sizeof(int64_t*)*grids_->Nz);

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
  
  int nn4 = grids_->Nyc;             int nt4 = min(nn4, 16);   int nb4 = 1 + (nn4-1)/nt4;
  int nn5 = grids_->Nx;              int nt5 = min(nn5,  4);   int nb5 = 1 + (nn5-1)/nt5;
  int nn6 = pars_->nm_in * pars_->nl_in; int nt6 = min(nn6, 4); int nb6 = 1 + (nn6-1)/nt6;

  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3); 

  dG_b = dim3(nt4, nt5, nt2);
  dB_b = dim3(nb4, nb5, nb2);

  for (int i = 0; i < num_coeff; i++){
    for (int j = 0; j < grids_->Nz; j++){
       initialize_A_bounce<<<dG, dB>>>(A_bounce[j + i*num_coeff], LM, pars_->nm_in, pars_->nl_in, geo_->bgrad, dcoeff[i], dt_,vte,j);

    }
  }
 


  d_info = nullptr;     /* error info */

  size_t workspaceInBytesOnDevice = 0; /* size of workspace */
  void *d_work = nullptr;              /* device workspace for getrf */
  size_t workspaceInBytesOnHost = 0;   /* size of workspace */
  void *h_work = nullptr;              /* host workspace for getrf */

  pivot_on = 1;
  const int algo = 0;
  if (pivot_on) {
      std::printf("pivot is on : compute P*A = L*U \n");
  } else {
      std::printf("pivot is off: compute A = L*U (not numerically stable)\n");
  }

  
 
  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream)); 
  /* Create advanced params */
  CUSOLVER_CHECK(cusolverDnCreateParams(&params));
  if (algo == 0) {
    std::printf("Using New Algo\n");
    CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0));
  } else {
    std::printf("Using Legacy Algo\n");
    CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_1));
  }


  using data_type = cuComplex;
 // print_matrix(LM, LM, LU, LM);
  printf("BEFORE FACTORIZATION\n");

  int ind;
  for (int i = 0; i < num_coeff; i++){
    for(int iz = 0; iz < grids_->Nz; iz++){
	ind = iz + i*num_coeff;
	CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream)); 
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
	    printf("PIVOTTTTTING\n");
	    CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, LM, LM, CUDA_C_32F,
					    A_bounce[ind], LM, d_Ipiv[ind], CUDA_C_32F, d_work,
					    workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info));
	} else {
	    printf("NOT PIVOTTTTTING\n");
	    CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, LM, LM, CUDA_C_32F,
					    A_bounce[ind], LM, nullptr, CUDA_C_32F,
					    d_work, workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info));
	}   
    }
  }
  checkCuda(cudaStreamSynchronize(stream));

  printf("AFTER FACTORIZATION\n");


/*  for(int iz = 0; iz < grids_->Nz; iz++){
    h_Ipiv[iz] = (int64_t*) malloc(sizeof(int64_t)*LM);
    checkCuda(cudaMemcpy(h_Ipiv[iz], d_Ipiv[iz], sizeof(int64_t)*LM, cudaMemcpyDeviceToHost));
    for(int idlm = 0; idlm < pars_->nm_in*pars_->nl_in; idlm++){
      if(h_Ipiv[iz][idlm] != idlm+1){
        printf("PIVOTING AT IZ = %d, IDLM = %d, value is %d\n", iz, idlm, h_Ipiv[iz][idlm]);
      }
    }
  }*/

  checkCuda(cudaMemcpy(LU, A_bounce[0], LM2, cudaMemcpyDeviceToHost));
  
  printf("bgrad is %f\n", geo_->bgrad_h[0]);

  for (int i = 0; i < LM; i++) {
        for (int j = 0; j < LM; j++) {
            std::printf("%2.3e ", LU[j * LM + i].x);
        }
        std::printf("\n\n");
  }


  bounce_rhs = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nz);
  size_t brhs_size = sizeof(cuComplex)*pars_->nm_in*pars_->nl_in*grids_->Nyc*grids_->Nx;
  for (int j = 0; j < grids_->Nz; j++){
    checkCuda(cudaMalloc((void**) &bounce_rhs[j], brhs_size));
  }

  printf("FINISHED INIT\n");
}

Cusolve::~Cusolve(){
  for (int i = 0; i < num_coeff; i++){
    for(int iz = 0; iz < grids_->Nz; iz++){
       if(A_bounce[iz + i*num_coeff] != nullptr) cudaFree(A_bounce[iz + i*num_coeff]);
    }
  }
  for(int iz = 0; iz < grids_->Nz; iz++){
    if (bounce_rhs[iz] != nullptr) cudaFree(bounce_rhs[iz]);
//    if (cusolverH[iz] != nullptr) cusolverDnDestroy(cusolverH[iz]); 
//    if (stream[iz] != nullptr) cudaStreamDestroy(stream[iz]);

  }
  if(bounce_rhs != nullptr) free(bounce_rhs);
  if(A_bounce != nullptr) free(A_bounce);
  if (cusolverH != nullptr) free(cusolverH);
}

void Cusolve::invert_stream(cuComplex* G, int stage){

  for(int iz = 0; iz < grids_->Nz; iz++){
//    copy_brhs_from_g<<<dG_b, dB_b>>>(bounce_rhs[iz], G, iz);
/*    if (pivot_on) {
        CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, LM, grids_->Nx*grids_->Nyc,
    				CUDA_C_32F, A_bounce[iz], LM, d_Ipiv[iz],
  				CUDA_C_32F, bounce_rhs[iz], LM, d_info));
    } else {
	printf("NOT PIVOTTTTTING SOLVE\n");
        CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, LM, grids_->Nx*grids_->Nyc,
  				CUDA_C_32F, A_bounce[iz], LM, nullptr,
  				CUDA_C_32F, bounce_rhs[iz], LM, d_info));
    }*/
    lu_backsub_bounce<<<dG_b, dB_b>>>(A_bounce[iz], G, iz);
//    copy_g_from_brhs<<<dG_b, dB_b>>>(G,bounce_rhs[iz], iz);


  }
  checkCudaErrors(cudaGetLastError());

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
