#include "green.h"

Green::Green(Parameters *pars, Grids *grids, Geometry *geo, double p, double r, double u, bool sdirk, double dt_in, double vte):
	p_(p), r_(r), u_(u), sdirk_(sdirk), grids_(grids), pars_(pars), geo_(geo), A_phi(nullptr), dt_(dt_in), vte_(vte),
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
  phi_rhs = (cuComplex**) malloc(sizeof(cuComplex*)*num_coeff*grids_->NxNyc);
  cuComplex* LU = (cuComplex*) malloc(nz2); 
  checkCuda(cudaMalloc((void**) &d_Ipiv, sizeof(int)*grids_->Nz*grids_->NxNyc));
  checkCuda(cudaMalloc((void**) &infoArray, sizeof(int)*grids_->NxNyc));
  infoArray_h = (int*) malloc(sizeof(int)*grids_->NxNyc);
  G = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G[is] = new MomentsG (pars_, grids_, is_glob);
  }
  checkCuda(cudaMalloc((void**) &phi_r, sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl));

  if (pars_->local_limit) {
    grad_par = new GradParallelLocal(grids_);
  }
//  else if (pars_->boundary_option_periodic) {
//   grad_par = new GradParallelPeriodic(grids_);
//  }
  else {
    printf("USING GRADPARALLELLINKED!!!\n");
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

  for (int i = 0; i < num_coeff; i++){
    for (int j = 0; j < grids_->NxNyc; j++){
      checkCuda(cudaMalloc((void**) &A_phi[j + i*num_coeff], nz2));
      checkCuda(cudaMalloc((void**) &phi_rhs[j + i*num_coeff], sizeof(cuComplex)*grids_->Nz));
      checkCuda(cudaMalloc((void**) &d_Ipiv[j + i*num_coeff], sizeof(int64_t)*grids_->Nz));
    } 
  }
  int nn1, nt1, nb1, nn2, nt2, nb2, nn3, nt3, nb3;

  nn1 = grids_->Nz;	    nt1 = min(nn1, 32 );   nb1 = 1 + (nn1-1)/nt1;
  nn2 = 1;         	    nt2 = min(nn2,  4 );   nb2 = 1 + (nn2-1)/nt2;
  nn3 = 1;         	    nt3 = min(nn3,  4 );   nb3 = 1 + (nn3-1)/nt3;
  
  int nn4 = grids_->Nyc;             int nt4 = min(nn4, 16);   int nb4 = 1 + (nn4-1)/nt4;
  int nn5 = grids_->Nx;              int nt5 = min(nn5,  4);   int nb5 = 1 + (nn5-1)/nt5;
  int nn6 = pars_->nm_in * pars_->nl_in; int nt6 = min(nn6, 4); int nb6 = 1 + (nn6-1)/nt6;
  int nn7 = pars_->nl_in;            int nt7 = min(nn7, 4);    int nb7 = 1 + (nn7-1)/nt7;
  int nn8 = pars_->nl_in*grids_->Nz; int nt8 = min(nn8, 4);    int nb8 = 1 + (nn8-1)/nt8;



  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3); 

  dG_p = dim3(nt4, nt5, nt7);
  dB_p = dim3(nb4, nb5, nb7);

  dG_s = dim3(nt4, nt5, nt8);
  dB_s = dim3(nb4, nb5, nb8);


  for (int i = 0; i < num_coeff; i++){
    for (int is = 0; is < grids_->Nspecies; is++){
      for (int iz = 0; iz < grids_->Nz; iz++){ 
        set_delta_phi<<<dG_p,dB_p>>>(phi_r,iz,1.0f,geo_->kperp2,*(G[is]->species));
        grad_par->zft(phi_r,phi_r);
        compute_homogenous_sol<<<dG_s, dB_s>>>(G[is]->G(),phi_r,grids_->kz,*(G[is]->species),dcoeff[i]*dt_,geo_->gradpar);
        cudaMemset(phi_r,0., sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl);
	grad_par->zft_inverse(G[is]);
	for (int ik = 0; ik < grids_->NxNyc; ik++){
          compute_response_matrix<<<dG, dB>>>(A_phi[ik + i*num_coeff],G[is]->G(),*(G[is]->species),geo_->kperp2,ik,iz);
	}
      }
    }
  }
  checkCudaErrors(cudaGetLastError());
  

  CUBLAS_CHECK(cublasCreate(&cublasH));

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));
  
  CUBLAS_CHECK(cublasCgetrfBatched(cublasH,
                                   LM,
                                   A_phi,
                                   nz2,
                                   d_Ipiv,
                                   infoArray,
                                   grids_->NxNyc*num_coeff));
 


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
  }
  CUBLAS_CHECK(cublasCgetrsBatched(cublasH,
                                 CUBLAS_OP_N,
                                 grids_->Nz,
                                 1,
                                 A_phi,
                                 grids_->Nz,
                                 d_Ipiv,
                                 phi_rhs,
                                 grids_->Nz,
                                 infoArray_h,
                                 grids_->NxNyc));

  for (int ik = 0; ik < grids_->NxNyc; ik++){
   copy_prhs_from_p<<<dG, dB>>>(phi_i,phi_rhs[ik], ik);
  }

}

