#include "green.h"

Green::Green(Parameters *pars, Grids *grids, Geometry *geo, Solver *solver, double p, double r, double u, bool sdirk, double dt_in, double vte):
	p_(p), r_(r), u_(u), sdirk_(sdirk), grids_(grids), pars_(pars), geo_(geo), solver_(solver), A_phi(nullptr), dt_(dt_in), vte_(vte),
	phi_rhs(nullptr), d_Ipiv(nullptr), infoArray(nullptr), infoArray_h(nullptr), cublasH(nullptr), stream(nullptr)
{
  
  size_t nz2 = sizeof(cuComplex)*grids_->Nz*grids_->Nz;
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
  
  A_phi = (cuComplex**) malloc(sizeof(cuComplex*)*num_coeff*grids_->NxNyc);
  d_A_phi = nullptr;
  d_phi_rhs = nullptr;
  A_phi_copy = (cuComplex**) malloc(sizeof(cuComplex*)*num_coeff*grids_->NxNyc);

  phi_rhs = (cuComplex**) malloc(sizeof(cuComplex*)*num_coeff*grids_->NxNyc);
  res = (cuComplex**) malloc(sizeof(cuComplex*)*num_coeff*grids_->NxNyc);
  prod = (cuComplex**) malloc(sizeof(cuComplex*)*num_coeff*grids_->NxNyc);

  cuComplex* LU = (cuComplex*) malloc(nz2); 
  checkCuda(cudaMalloc((void**) &d_Ipiv, sizeof(int)*grids_->Nz*grids_->NxNyc));
  checkCuda(cudaMalloc((void**) &infoArray, sizeof(int)*grids_->NxNyc));
  infoArray_h = (int*) malloc(sizeof(int)*grids_->NxNyc);
  G = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G[is] = new MomentsG (pars_, grids_, is_glob);
    G[is]->set_zero();
  }
  checkCuda(cudaMalloc((void**) &phi_r, sizeof(cuComplex)*grids_->NxNycNz));
  checkCuda(cudaMalloc((void**) &d_A_phi, sizeof(cuComplex*)*grids_->NxNyc));

//  checkCuda(cudaMalloc((void**) &phi_r, sizeof(cuComplex)*grids_->NxNycNz));


  if (pars_->local_limit) {
    grad_par = new GradParallelLocal(grids_);
  }
//  else if (pars_->boundary_option_periodic) {
//   grad_par = new GradParallelPeriodic(grids_);
//  }
  else {
    grad_par = new GradParallelLinked(pars_, grids_);
  }


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
  checkCuda(cudaMalloc((void**) &d_phi_rhs, sizeof(cuComplex*)*grids_->NxNyc*num_coeff));

  for (int i = 0; i < num_coeff; i++){
    for (int j = 0; j < grids_->NxNyc; j++){
      checkCuda(cudaMalloc((void**) &A_phi[j + i*num_coeff], nz2));
      checkCuda(cudaMalloc((void**) &A_phi_copy[j + i*num_coeff], nz2));
      checkCuda(cudaMalloc((void**) &phi_rhs[j + i*num_coeff], sizeof(cuComplex)*grids_->Nz));
      checkCuda(cudaMalloc((void**) &res[j + i*num_coeff], sizeof(cuComplex)*grids_->Nz));
      checkCuda(cudaMalloc((void**) &prod[j + i*num_coeff], sizeof(cuComplex)*grids_->Nz));

    } 
  }

  checkCuda(cudaMemcpy(d_phi_rhs, phi_rhs, sizeof(cuComplex*)*grids_->NxNyc*num_coeff, cudaMemcpyHostToDevice));

  int nn1, nt1, nb1, nn2, nt2, nb2, nn3, nt3, nb3;

  nn1 = grids_->Nz;	    nt1 = min(nn1, 32 );   nb1 = 1 + (nn1-1)/nt1;
  nn2 = 1;         	    nt2 = min(nn2,  4 );   nb2 = 1 + (nn2-1)/nt2;
  nn3 = 1;         	    nt3 = min(nn3,  4 );   nb3 = 1 + (nn3-1)/nt3;
  
  int nn4 = grids_->Nyc;             int nt4 = min(nn4, 16);   int nb4 = 1 + (nn4-1)/nt4;
  int nn5 = grids_->Nx;              int nt5 = min(nn5,  4);   int nb5 = 1 + (nn5-1)/nt5;
  int nn7 = 1;            int nt7 = min(nn7, 4);    int nb7 = 1 + (nn7-1)/nt7;
  int nn8 = grids_->Nz; int nt8 = min(nn8, 4);    int nb8 = 1 + (nn8-1)/nt8;



  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3); 

  dG_p = dim3(nt4, nt5, nt7);
  dB_p = dim3(nb4, nb5, nb7);

  dG_s = dim3(nt4, nt5, nt8);
  dB_s = dim3(nb4, nb5, nb8);


  for (int i = 0; i < num_coeff; i++){
    for (int is = 0; is < grids_->Nspecies; is++){
      for (int iz = 0; iz < grids_->Nz; iz++){ 
/*        set_delta_phi<<<dG_p,dB_p>>>(phi_r,iz,1.0f,geo_->kperp2,*(G[is]->species));
        checkCudaErrors(cudaGetLastError());
        grad_par->zft_nmoms(phi_r,phi_r, grids_->Nl);*/
	
	for(int il = 0; il < grids_->Nl; il++){
          set_delta_phi<<<dG_p,dB_p>>>(phi_r,iz,1.0f,geo_->kperp2,*(G[is]->species), il);
          checkCudaErrors(cudaGetLastError());
          grad_par->zft(phi_r,phi_r);
          compute_homogenous_sol_loop<<<dG_s, dB_s>>>(G[is]->G(),phi_r,grids_->kz,*(G[is]->species),dcoeff[i]*dt_,geo_->gradpar, il);
          checkCuda(cudaMemset(phi_r,0., sizeof(cuComplex)*grids_->NxNycNz));
          checkCudaErrors(cudaGetLastError());

	}

	grad_par->zft_inverse(G[is]);
	for (int ik = 0; ik < grids_->NxNyc; ik++){
          compute_response_matrix<<<dG, dB>>>(A_phi[ik + i*num_coeff],G[is]->G(),*(G[is]->species),geo_->kperp2,solver_->getQneutDenom(),ik,iz);
	}
        checkCudaErrors(cudaGetLastError());
      }
    }
  }

/*  for (int i = 0; i < num_coeff; i++){
    for (int is = 0; is < grids_->Nspecies; is++){
      for (int iz = 0; iz < grids_->Nz; iz++){ 
        set_delta_phi<<<dG_p,dB_p>>>(phi_r,iz,1.0f,geo_->kperp2,*(G[is]->species));
        checkCudaErrors(cudaGetLastError());
        grad_par->zft(phi_r,phi_r);
	
	for(int il = 0; il < grids_->Nl; il++){
          compute_homogenous_sol_loop<<<dG_s, dB_s>>>(G[is]->G(),phi_r,grids_->kz,*(G[is]->species),dcoeff[i]*dt_,geo_->gradpar, il);
	}
        checkCuda(cudaMemset(phi_r,0., sizeof(cuComplex)*grids_->NxNycNz));
        checkCudaErrors(cudaGetLastError());

	grad_par->zft_inverse(G[is]);
	for (int ik = 0; ik < grids_->NxNyc; ik++){
          compute_response_matrix<<<dG, dB>>>(A_phi[ik + i*num_coeff],G[is]->G(),*(G[is]->species),geo_->kperp2,solver_->getQneutDenom(),ik,iz);
	}
        checkCudaErrors(cudaGetLastError());
      }
    }
  }*/

  
  for(int i = 0; i < num_coeff; i++){
    for(int ik = 0; ik < grids_->NxNyc; ik++){
      add_id_response_matrix<<<dG, dB>>>(A_phi[ik+i*num_coeff]);
      checkCuda(cudaMemcpy(A_phi_copy[ik], A_phi[ik], nz2, cudaMemcpyDeviceToDevice));

    }
  }

  
  checkCuda(cudaMemcpy(d_A_phi, A_phi, sizeof(cuComplex*)*grids_->NxNyc, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaGetLastError());
  

  CUBLAS_CHECK(cublasCreate(&cublasH));

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));
  
  CUBLAS_CHECK(cublasCgetrfBatched(cublasH,
                                   grids_->Nz,
                                   d_A_phi,
                                   grids_->Nz,
                                   d_Ipiv,
                                   infoArray,
                                   grids_->NxNyc*num_coeff));
  checkCudaErrors(cudaGetLastError());

  checkCuda(cudaMemcpy(A_phi, d_A_phi, sizeof(cuComplex*)*grids_->NxNyc, cudaMemcpyDeviceToHost));
/*  checkCuda(cudaMemcpy(LU, A_phi[2], nz2, cudaMemcpyDeviceToHost));
  for (int i = 0; i < grids_->Nz; i++) {
        for (int j = 0; j < grids_->Nz; j++) {
            std::printf("%0.6f + %0.6fj ", LU[j * grids_->Nz + i].x, LU[j * grids_->Nz + i].y);
        }
        std::printf("\n");
  }*/



}

Green::~Green(){
  for (int i = 0; i < num_coeff; i++){
    for(int ik = 0; ik < grids_->NxNyc; ik++){
       if(A_phi[ik + num_coeff*i]) cudaFree(A_phi[ik + num_coeff*i]);
    }
  }
  for(int ik = 0; ik < grids_->NxNyc; ik++){
    if (phi_rhs[ik]) cudaFree(phi_rhs[ik]);

  }
  if(phi_rhs) free(phi_rhs);
  if(A_phi) free(A_phi);
  if(infoArray != nullptr) cudaFree(infoArray);
  if(infoArray_h != nullptr) free(infoArray_h);
  if(cublasH != nullptr) cublasDestroy(cublasH);
  if(stream != nullptr) cudaStreamDestroy(stream);

  if(G) delete G;
}

void Green::invert(cuComplex* phi_i)
{ 
  for (int ik = 0; ik < grids_->NxNyc; ik++){
   copy_prhs_from_p<<<dG, dB>>>(phi_rhs[ik],phi_i, ik);
//   print_phi<<<dG, dB>>>(phi_i, ik);
//   copy_prhs_from_p<<<dG, dB>>>(res[ik],phi_i, ik);

  }

  for (int ik = 0; ik < grids_->NxNyc; ik++){
    lu_backsub<<<dG, dB>>>(A_phi[ik], phi_rhs[ik], phi_i, ik);  
  }

///  copy_prhs_from_p_d<<<dG_s, dB_s>>>(d_phi_rhs, *phi_i);
/*  CUBLAS_CHECK(cublasCgetrsBatched(cublasH,
                                 CUBLAS_OP_N,
                                 grids_->Nz,
                                 1,
                                 d_A_phi,
                                 grids_->Nz,
                                 d_Ipiv,
                                 phi_rhs,
                                 grids_->Nz,
                                 infoArray_h,
                                 grids_->NxNyc));*/
//  copy_p_from_prhs_d<<<dG_s, dB_s>>>(*phi_i,d_phi_rhs);

  for (int ik = 0; ik < grids_->NxNyc; ik++){ 
/*    compute_residual<<<dG, dB>>>(A_phi_copy[ik], phi_rhs[ik], phi_i, res[ik], ik, false);
    compute_residual<<<dG, dB>>>(A_phi_copy[ik], phi_rhs[ik], phi_i, prod[ik], ik, true);*/

/*    CUBLAS_CHECK(
      cublasCgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, grids_->Nz,1,grids_->Nz, &alpha, A_phi[ik], grids_->Nz, phi_rhs[ik], grids_->Nz, &beta, res[ik], grids_->Nz));
    CUBLAS_CHECK(
      cublasCgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, grids_->Nz,1,grids_->Nz, &alpha, A_phi[ik], grids_->Nz, phi_rhs[ik], grids_->Nz, &zeta, prod[ik], grids_->Nz));*/

/*    check_residual_phi<<<dG, dB>>>(res[ik],phi_i,prod[ik], ik);
    checkCuda(cudaMemset(res[ik],0., sizeof(cuComplex)*grids_->Nz));
    checkCuda(cudaMemset(prod[ik],0., sizeof(cuComplex)*grids_->Nz));*/

    copy_p_from_prhs<<<dG, dB>>>(phi_i,phi_rhs[ik], ik);
  }
/*  for (int ik = 0; ik < grids_->NxNyc; ik++){
    printf("infoarray is %d\n", infoArray_h[ik]);
 
  }*/

}

