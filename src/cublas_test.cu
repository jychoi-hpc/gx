#include "cublas_test.h"

Cublas_test::Cublas_test(Parameters *pars, Grids *grids, Geometry *geo, double p, double r, double u, bool sdirk, double dt_in, double vte):
	p_(p), r_(r), u_(u), sdirk_(sdirk), grids_(grids), pars_(pars), geo_(geo), A_bounce(nullptr), dt_(dt_in), vte_(vte)
{
  
  A_bounce = nullptr;	
  d_A_bounce = nullptr;
  d_bounce_rhs = nullptr;
  bounce_rhs = nullptr;
  size_t nzlm = sizeof(int) * grids_->Nz * grids_->Nz * pars_->nm_in * pars_->nl_in;
  LM = pars_->nm_in * pars_-> nl_in;
  size_t LM2 = sizeof(cuComplex) *LM*LM;
  test = (cuComplex*) malloc(sizeof(cuComplex)*grids_->Nyc*grids_->Nx*grids_->Nm*grids_->Nl);
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
//  d_Ipiv = (int64_t**)malloc(sizeof(int64_t*)*grids_->Nz);
  checkCuda(cudaMalloc((void**) &d_Ipiv, sizeof(int)*grids_->Nz*LM));
  checkCuda(cudaMalloc((void**) &infoArray, sizeof(int)*grids_->Nz)); 
  infoArray_h = (int*) malloc(sizeof(int)*grids_->Nz);

  info = 0;
//  checkCuda(cudaMalloc((void**) &infoArray, sizeof(int)*grids_->Nz)); 
 

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

  bounce_rhs = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nz);
  size_t brhs_size = sizeof(cuComplex)*pars_->nm_in*pars_->nl_in*grids_->Nyc*grids_->Nx;
  checkCuda(cudaMalloc((void**) &d_bounce_rhs, sizeof(cuComplex*)*grids_->Nz));

  for(int i = 0; i < grids_->Nz; i++){
    checkCuda(cudaMalloc((void**) &bounce_rhs[i], brhs_size));
    checkCuda(cudaMemset(bounce_rhs[i],0.,brhs_size));

  }
  checkCuda(cudaMemcpy(d_bounce_rhs, bounce_rhs, sizeof(cuComplex*)*grids_->Nz, cudaMemcpyHostToDevice));

  for (int i = 0; i < num_coeff; i++){
    for (int j = 0; j < grids_->Nz; j++){
      checkCuda(cudaMalloc((void**) &A_bounce[j + i*num_coeff], LM2));
/*      checkCuda(cudaMemset(A_bounce[j + i*num_coeff], 0., LM2));
      checkCuda(cudaStreamSynchronize(0));

//      checkCuda(cudaMalloc((void**) &d_Ipiv[j + i*num_coeff], sizeof(int64_t)*LM));
      checkCuda(cudaMalloc((void**) &bounce_rhs[j], brhs_size));
      checkCuda(cudaMemset(bounce_rhs[j],0.,brhs_size));
      checkCuda(cudaStreamSynchronize(0));*/

    }
  }

  checkCuda(cudaMalloc((void**) &d_A_bounce, sizeof(cuComplex*)*grids_->Nz));

  DEBUGPRINT("Allocated an A_bounce array of size %.2f MB\n", nzlm/1024./1024.);

  int nn1, nt1, nb1, nn2, nt2, nb2, nn3, nt3, nb3;

  nn1 = LM;		    nt1 = min(nn1, 512 );   nb1 = 1 + (nn1-1)/nt1;
  nn2 = 1;         	    nt2 = min(nn2,  1 );   nb2 = 1 + (nn2-1)/nt2;
  nn3 = 1;         	    nt3 = min(nn3,  1 );   nb3 = 1 + (nn3-1)/nt3;
  
  int nn4 = grids_->Nyc;             int nt4 = min(nn4, 16);   int nb4 = 1 + (nn4-1)/nt4;
  int nn5 = grids_->Nx;              int nt5 = min(nn5,  4);   int nb5 = 1 + (nn5-1)/nt5;
  int nn8 = grids_->Nz;   	     int nt8 = min(nn8,  8);   int nb8 = 1 + (nn8-1)/nt8;
  int nn6 = pars_->nm_in * pars_->nl_in; int nt6 = min(nn6, 4); int nb6 = 1 + (nn6-1)/nt6;
  int nn7 = grids_->Nx*grids_->Nz;   int nt7 = min(nn7,  8);   int nb7 = 1 + (nn7-1)/nt7;


  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3); 

//  dG_b = dim3(nt4, nt5, nt6);
//  dB_b = dim3(nb4, nb5, nb6);

  dB_b = dim3(nt4, nt5, nt6);
  dG_b = dim3(nb4, nb5, nb6);



  dB_bd = dim3(nt4, nt7, nt6);
  dG_bd = dim3(nb4, nb7, nb6);


  dB_lu = dim3(nt4, nt5, nt8);
  dG_lu = dim3(nb4, nb5, nb8);

  dB_lu_sm = dim3(nt8, 1, 1);
  dG_lu_sm = dim3(nb8, 1, 1);

  float* bgrad_h_test = (float*) malloc(sizeof(float)*grids_->Nz);
  checkCuda(cudaMemcpy(bgrad_h_test, geo_->bgrad, sizeof(float)*grids_->Nz, cudaMemcpyDeviceToHost));
  printf("bgrad is %f\n", bgrad_h_test[10]);



  for (int i = 0; i < num_coeff; i++){
    for (int j = 0; j < grids_->Nz; j++){
       initialize_A_bounce<<<dG, dB>>>(A_bounce[j + i*num_coeff], LM, pars_->nm_in, pars_->nl_in, geo_->bgrad, dcoeff[i], dt_,vte,j);
       transpose_A<<<dG, dB>>>(A_bounce[j + i*num_coeff], LM);
//       initialize_A_bounce_loop(A_bounce[j + i*num_coeff], LM, pars_->nm_in, pars_->nl_in, geo_->bgrad, dcoeff[i], dt_,vte,j);

    }
  }
  checkCuda(cudaMemcpy(d_A_bounce, A_bounce, sizeof(cuComplex*)*grids_->Nz, cudaMemcpyHostToDevice));

/*  checkCuda(cudaMemcpy(LU, A_bounce[11], LM2, cudaMemcpyDeviceToHost));
  bgrad_h_test = (float*) malloc(sizeof(float)*grids_->Nz);
  checkCuda(cudaMemcpy(bgrad_h_test, geo_->bgrad, sizeof(float)*grids_->Nz, cudaMemcpyDeviceToHost));

  printf("coeff is %f\n", dcoeff[0]); 
  printf("bgrad is %f\n", bgrad_h_test[11]);
  printf("dt_ is %f\n", dt_);
  printf("vte is %f\n", vte);

  for (int i = 0; i < LM; i++) {
        for (int j = 0; j < LM; j++) {
            std::printf("%2.3e ", LU[j * LM + i].x);
        }
        std::printf("\n\n");
  }*/


  checkCudaErrors(cudaGetLastError());


  CUBLAS_CHECK(cublasCreate(&cublasH));

//  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_cublas, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_cublas, cudaStreamDefault));

  CUBLAS_CHECK(cublasSetStream(cublasH, stream_cublas));
  
  CUBLAS_CHECK(cublasCgetrfBatched(cublasH,
                                   LM,
                                   d_A_bounce,
                                   LM,
                                   d_Ipiv,
//				   NULL,
                                   infoArray,
                                   grids_->Nz));
  checkCuda(cudaStreamSynchronize(stream_cublas));

  checkCuda(cudaMemcpy(A_bounce, d_A_bounce,sizeof(cuComplex*)*grids_->Nz, cudaMemcpyDeviceToHost));
//  checkCuda(cudaMemcpy(LU, A_bounce[11], LM2, cudaMemcpyDeviceToHost));



//  checkCuda(cudaStreamSynchronize(stream_cublas));
/*  printf("AFTER FACTORIZATION\n");
  printf("coeff is %f\n", dcoeff[0]); 
  printf("bgrad is %f\n", geo_->bgrad_h[11]);
  printf("dt_ is %f\n", dt_);
  printf("vte is %f\n", vte);

  for (int i = 0; i < LM; i++) {
        for (int j = 0; j < LM; j++) {
            std::printf("%2.3e ", LU[j * LM + i].x);
        }
        std::printf("\n\n");
  }*/





/*  checkCuda(cudaMalloc((void**) &d_bounce_rhs, sizeof(cuComplex*)*grids_->Nz));
  bounce_rhs = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nz);
  res = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nz);
  size_t brhs_size = sizeof(cuComplex)*pars_->nm_in*pars_->nl_in*grids_->Nyc*grids_->Nx;
  for (int j = 0; j < grids_->Nz; j++){
    checkCuda(cudaMalloc((void**) &bounce_rhs[j], brhs_size));
//    checkCuda(cudaMalloc((void**) &d_bounce_rhs[j], brhs_size));
    checkCuda(cudaMalloc((void**) &res[j], brhs_size));
  }
//  checkCuda(cudaMemcpy(d_bounce_rhs, bounce_rhs, sizeof(cuComplex*)*grids_->Nz, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaGetLastError());*/
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

  copy_brhs_from_g_d<<<dG_bd, dB_bd>>>(d_bounce_rhs, G, false);

  CUBLAS_CHECK(cublasCgetrsBatched(cublasH,
                                   CUBLAS_OP_N,
                                   LM,
                                   grids_->Nyc*grids_->Nx,
                                   d_A_bounce,
                                   LM,
                                   d_Ipiv,
//				   NULL,
                                   d_bounce_rhs,
                                   LM,
//                                   infoArray_h,
				   &info,
                                   grids_->Nz));

  copy_g_from_brhs_d<<<dG_bd, dB_bd>>>(G, d_bounce_rhs);
  checkCuda(cudaStreamSynchronize(stream_cublas));


//  lu_backsub_bounce_d<<<dG_lu, dB_lu>>>(d_A_bounce, G);


  checkCudaErrors(cudaGetLastError());

}

void Cublas_test::invert_sherman_morrison(cuComplex* u)
{
  lu_backsub_bounce_d_sm<<<dG_lu_sm, dB_lu_sm>>>(d_A_bounce, u);

}


