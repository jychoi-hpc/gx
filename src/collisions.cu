#include "collisions.h"

LorentzCollisionOperator::LorentzCollisionOperator(Parameters* pars, Grids* grids) :
  grids_(grids), Nlm(grids->Nmoms), Nk(grids_->NxNycNz)
{
  cublasCreate (&handle);
  cudaMalloc ((void**) &collMat, sizeof(cuComplex)*Nlm*Nlm);
  cuComplex *collMat_h = (cuComplex*) malloc(sizeof(cuComplex)*Nlm*Nlm);

  initCollisionMatrix(collMat_h);
  CP_TO_GPU (collMat, collMat_h, sizeof(cuComplex)*Nlm*Nlm);

  tmpG = new MomentsG (pars, grids);

  if (collMat_h) free(collMat_h);

  int nn1 = grids_->NxNycNz;   int nt1 = pars->i_share     ;  int nb1 = 1 + (nn1-1)/nt1;
  int nn2 = 1;                 int nt2 = min(grids_->Nl, 4 ); int nb2 = 1 + (nn2-1)/nt2;
  int nn3 = 1;                 int nt3 = min(grids_->Nm, 4 ); int nb3 = 1 + (nn3-1)/nt3;

  dimBlock = dim3(nt1, nt2, nt3);
  dimGrid  = dim3(nb1, nb2, nb3);
  
  if(grids_->m_ghost == 0)
    sharedSize = nt1 * (grids_->Nl+2) * (grids_->Nm+4) * sizeof(cuComplex);
  else 
    sharedSize = nt1 * (grids_->Nl+2) * (grids_->Nm+2*grids_->m_ghost) * sizeof(cuComplex);

  int dev;
  cudaDeviceProp prop;
  checkCuda( cudaGetDevice(&dev) );
  checkCuda( cudaGetDeviceProperties(&prop, dev) );
  maxSharedSize = prop.sharedMemPerBlockOptin;
}

LorentzCollisionOperator::~LorentzCollisionOperator()
{
  if (collMat) cudaFree(collMat);
  if (tmpG) delete tmpG;
  cublasDestroy(handle);
}

void LorentzCollisionOperator::initCollisionMatrix(cuComplex* mat)
{
  for(int i=0; i<Nlm; i++) {
    for(int j=0; j<Nlm; j++) {
      int il = i%grids_->Nl; 
      int im = i/grids_->Nl;
      int jl = j%grids_->Nl;
      int jm = j/grids_->Nl;
      mat[i+Nlm*j].x = nuD_matrix[il + nuD_matrix_NL*im][jl + nuD_matrix_NL*jm];
      mat[i+Nlm*j].y = 0.;
      //printf("(il,im) = (%d,%d), (jl,jm) = (%d,%d), %f\n", il, im, jl, jm, mat[i+Nlm*j].x);
    }
  }
}
  
void LorentzCollisionOperator::applyCollisionMatrix(cuComplex* G_in, cuComplex* G_res, bool accumulate)
{
  int m = Nk;
  int n = Nlm;
  int k = Nlm;
  
  cuComplex alpha; alpha.x = 1.0; alpha.y = 0.0;
  cuComplex beta; 
  if(accumulate) {
    beta.x = 1.0; beta.y = 0.0;
  } else {
    beta.x = 0.0; beta.y = 0.0;
  }
  int lda = m;
  int ldb = n;
  int ldc = m;

  int status = cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
     m, n, k, &alpha,
     G_in, lda,
     collMat, ldb,
     &beta, G_res, ldc);
}

void LorentzCollisionOperator::rhs(MomentsG* G, Fields* f, Geometry* geo, MomentsG* GRhs, bool accumulate)
{
  cudaFuncSetAttribute(lorentz_rhs, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedSize);
  lorentz_rhs<<<dimGrid, dimBlock, sharedSize>>>
      	(G->G(), f->phi, f->apar, f-> bpar, geo->kperp2, 
	*(G->species), tmpG->G()); // no accumulate!

  applyCollisionMatrix(tmpG->G(), GRhs->G(), accumulate);
}
