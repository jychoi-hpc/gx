#include "cublas_test.h"

Cublas_test::Cublas_test(Parameters *pars, Grids *grids, Geometry *geo, double p, double r, double u, bool sdirk, double dt_in, double vte):
	p_(p), r_(r), u_(u), sdirk_(sdirk), grids_(grids), pars_(pars), geo_(geo), A_bounce(nullptr), dt_(dt_in), vte_(vte)
{
  
  A_bounce = nullptr;	
  d_A_bounce = nullptr;
  d_bounce_rhs = nullptr;
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
//  d_Ipiv = (int64_t**)malloc(sizeof(int64_t*)*grids_->Nz*LM);
  checkCuda(cudaMalloc((void**) &d_Ipiv, sizeof(int)*grids_->Nz*LM));
  checkCuda(cudaMalloc((void**) &infoArray, sizeof(int)*grids_->Nz)); 
  infoArray_h = (int*) malloc(sizeof(int)*grids_->Nz);

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
//      checkCuda(cudaMalloc((void**) &d_Ipiv[j + i*num_coeff], sizeof(int64_t)*LM)); 

    }
  }

  checkCuda(cudaMalloc((void**) &d_A_bounce, sizeof(cuComplex*)*grids_->Nz));

  DEBUGPRINT("Allocated an A_bounce array of size %.2f MB\n", nzlm/1024./1024.);

  int nn1, nt1, nb1, nn2, nt2, nb2, nn3, nt3, nb3;

  nn1 = LM;		    nt1 = min(nn1, 32 );   nb1 = 1 + (nn1-1)/nt1;
  nn2 = 1;         	    nt2 = min(nn2,  4 );   nb2 = 1 + (nn2-1)/nt2;
  nn3 = 1;         	    nt3 = min(nn3,  4 );   nb3 = 1 + (nn3-1)/nt3;
  
  int nn4 = grids_->Nyc;             int nt4 = min(nn4, 16);   int nb4 = 1 + (nn4-1)/nt4;
  int nn5 = grids_->Nx;              int nt5 = min(nn5,  4);   int nb5 = 1 + (nn5-1)/nt5;
  int nn6 = pars_->nm_in * pars_->nl_in; int nt6 = min(nn6, 4); int nb6 = 1 + (nn6-1)/nt6;
  int nn7 = grids_->Nx*grids_->Nz;   int nt7 = min(nn7,  16);   int nb7 = 1 + (nn7-1)/nt7;
  int nn8 = grids_->Nz;   int nt8 = min(nn8,  4);   int nb8 = 1 + (nn8-1)/nt8;


  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3); 

  dG_b = dim3(nt4, nt5, nt6);
  dB_b = dim3(nb4, nb5, nb6);

  dG_bd = dim3(nt4, nt7, nt6);
  dB_bd = dim3(nb4, nb7, nb6);

  dG_lu = dim3(nt4, nt5, nt8);
  dB_lu = dim3(nb4, nb5, nb8);

  for (int i = 0; i < num_coeff; i++){
    for (int j = 0; j < grids_->Nz; j++){
       initialize_A_bounce<<<dG, dB>>>(A_bounce[j + i*num_coeff], LM, pars_->nm_in, pars_->nl_in, geo_->bgrad, dcoeff[i], dt_,vte,j);

    }
  }
  checkCuda(cudaMemcpy(d_A_bounce, A_bounce, sizeof(cuComplex*)*grids_->Nz, cudaMemcpyHostToDevice));

  checkCuda(cudaMemcpy(LU, A_bounce[11], LM2, cudaMemcpyDeviceToHost));
  printf("coeff is %f\n", dcoeff[0]); 
  printf("bgrad is %f\n", geo_->bgrad_h[11]);

  for (int i = 0; i < LM; i++) {
        for (int j = 0; j < LM; j++) {
            std::printf("%0.6f ", LU[j * LM + i].x);
        }
        std::printf("\n");
  }


  checkCudaErrors(cudaGetLastError());
  

  CUBLAS_CHECK(cublasCreate(&cublasH));

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_cublas, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream_cublas));
  
  CUBLAS_CHECK(cublasCgetrfBatched(cublasH,
                                   LM,
                                   d_A_bounce,
                                   LM,
                                   d_Ipiv,
                                   infoArray,
                                   grids_->Nz));
  checkCuda(cudaMemcpy(A_bounce, d_A_bounce,sizeof(cuComplex*)*grids_->Nz, cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(LU, A_bounce[11], LM2, cudaMemcpyDeviceToHost));
  printf("coeff is %f\n", dcoeff[0]); 
  printf("bgrad is %f\n", geo_->bgrad_h[11]);

  for (int i = 0; i < LM; i++) {
        for (int j = 0; j < LM; j++) {
            std::printf("%0.6f ", LU[j * LM + i].x);
        }
        std::printf("\n");
  }


/*  checkCuda(cudaMemcpy(infoArray_h, infoArray, sizeof(int)*grids_->Nz, cudaMemcpyDeviceToHost));

  for (int iz = 0; iz < grids_->Nz; iz++){
    if (infoArray_h[iz] != 0){
      printf("Error at iz = %d\n", iz);
    }
  }
  printf("FINISHED CHECKING FOR ERRORS\n");*/


  checkCuda(cudaMalloc((void**) &d_bounce_rhs, sizeof(cuComplex*)*grids_->Nz));
  bounce_rhs = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nz);
  res = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nz);
  size_t brhs_size = sizeof(cuComplex)*pars_->nm_in*pars_->nl_in*grids_->Nyc*grids_->Nx;
  for (int j = 0; j < grids_->Nz; j++){
    checkCuda(cudaMalloc((void**) &bounce_rhs[j], brhs_size));
//    checkCuda(cudaMalloc((void**) &d_bounce_rhs[j], brhs_size));
    checkCuda(cudaMalloc((void**) &res[j], brhs_size));
  }
  checkCuda(cudaMemcpy(d_bounce_rhs, bounce_rhs, sizeof(cuComplex*)*grids_->Nz, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaGetLastError());
}

Cublas_test::~Cublas_test(){
  for (int i = 0; i < num_coeff; i++){
    for(int iz = 0; iz < grids_->Nz; iz++){
       if(A_bounce[iz + i*num_coeff] != nullptr) cudaFree(A_bounce[iz + i*num_coeff]);
    }
  }
  for(int iz = 0; iz < grids_->Nz; iz++){
    if (bounce_rhs[iz] != nullptr) cudaFree(bounce_rhs[iz]);
    if (res[iz] != nullptr) cudaFree(res[iz]);
    if (d_A_bounce[iz] != nullptr) cudaFree(d_A_bounce[iz]);
    if (d_bounce_rhs[iz] != nullptr) cudaFree(d_bounce_rhs[iz]);

  }
  if(bounce_rhs != nullptr) free(bounce_rhs);
  if(res != nullptr) free(res);
  if(A_bounce != nullptr) free(A_bounce);
  if (d_A_bounce != nullptr) cudaFree(d_A_bounce);
  if (d_bounce_rhs != nullptr) cudaFree(d_bounce_rhs);
  if(d_Ipiv != nullptr) cudaFree(d_Ipiv);
  if(infoArray != nullptr) cudaFree(infoArray);
  if(infoArray_h != nullptr) free(infoArray_h);
  if(cublasH != nullptr) cublasDestroy(cublasH);
  if(stream_cublas != nullptr) cudaStreamDestroy(stream_cublas);
}

void Cublas_test::invert_stream(cuComplex* G, int stage){
/*  for (int iz = 0; iz < grids_->Nz; iz++){
    copy_brhs_from_g<<<dG_b, dB_b>>>(bounce_rhs[iz],G, iz);
    checkCudaErrors(cudaGetLastError());
  }*/
//  copy_brhs_from_g_d<<<dG_bd, dB_bd>>>(d_bounce_rhs,G);
  lu_backsub_bounce_d<<<dG_lu, dB_lu>>>(d_A_bounce, G);
/*  checkCuda(cudaMemcpy(bounce_rhs, d_bounce_rhs, sizeof(cuComplex*)*grids_->Nz, cudaMemcpyDeviceToHost));

  CUBLAS_CHECK(cublasCgetrsBatched(cublasH,
                                   CUBLAS_OP_N,
                                   LM,
                                   grids_->NxNyc,
                                   d_A_bounce,
                                   LM,
                                   d_Ipiv,
                                   bounce_rhs,
                                   LM,
                                   infoArray_h,
                                   grids_->Nz));

  checkCuda(cudaMemcpy(d_bounce_rhs, bounce_rhs, sizeof(cuComplex*)*grids_->Nz, cudaMemcpyHostToDevice));*/
//  checkCuda(cudaMemcpy(A_bounce,  d_A_bounce, sizeof(cuComplex*)*grids_->Nz, cudaMemcpyDeviceToHost));


/*  for (int iz = 0; iz < grids_->Nz; iz++){
    if (infoArray_h[iz] != 0){
      printf("Error at iz = %d\n", iz);
    }
  CUBLAS_CHECK(
    cublasCgemm(cublasH, transa, transb, LM,grids_->Nx*grids_->Nyc,LM, &alpha, A_bounce[iz], LM, bounce_rhs[iz], LM, &beta, res[iz], LM));
    check_residual<<<dG_b, dB_b>>>(res[iz]);

    copy_g_from_brhs<<<dG_b, dB_b>>>(G,bounce_rhs[iz], iz);
  }*/

//  copy_g_from_brhs_d<<<dG_bd, dB_bd>>>(G,d_bounce_rhs);

  checkCudaErrors(cudaGetLastError());

}


