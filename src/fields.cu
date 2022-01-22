#include "fields.h"
#include "get_error.h"

Fields::Fields(Parameters* pars, Grids* grids) :
  size_(sizeof(cuComplex)*grids->NxNycNz), N(grids->NxNycNz), pars_(pars), grids_(grids),
  phi(nullptr), phi_h(nullptr), apar(nullptr), apar_h(nullptr)
{
  checkCuda(cudaMalloc((void**) &phi, size_));

  int nn = grids->NxNycNz; int nt = min(nn, 512); int nb = 1 + (nn-1)/nt;  cuComplex zero = make_cuComplex(0.,0.);
  setval <<< nb, nt >>> (phi, zero, nn);

  //  cudaMemset(phi, 0., size_);

  phi_h = (cuComplex*) malloc(size_);
  printf("Allocated a field array of size %.2f MB\n", size_/1024./1024.);

  if (pars_->beta > 0. || pars_->krehm) {
    checkCuda(cudaMalloc((void**) &apar, size_));
    printf("Allocated a field array of size %.2f MB\n", size_/1024./1024.);

    cudaMemset(apar, 0., size_); setval <<< nb, nt >>> (apar, zero, nn);

    apar_h = (cuComplex*) malloc(size_);
  }

}

Fields::~Fields() {
  if (phi)     cudaFree(phi);
  if (phi_h)   free(phi_h);
  if (apar)    cudaFree(apar);
  if (apar_h)  free(apar_h);
}

void Fields::print_phi(void)
{
  CP_TO_CPU(phi_h, phi, size_);
  printf("\n");
  for (int j=0; j<N; j++) printf("phi(%d) = (%e, %e) \n",j, phi_h[j].x, phi_h[j].y);
  printf("\n");
}

void Fields::print_apar(void)
{
  CP_TO_CPU(apar_h, apar, size_);
  printf("\n");
  for (int j=0; j<N; j++) printf("apar(%d) = (%e, %e) \n",j, apar_h[j].x, apar_h[j].y);
  printf("\n");
}

void Fields::rescale(float * phi_max) {
  int nn1 = grids_->NxNyc; int nt1 = min(nn1, 32); int nb1 = 1 + (nn1-1)/nt1;
  int nn2 = grids_->Nz;    int nt2 = min(nn2, 32); int nb2 = 1 + (nn2-1)/nt2;
  dim3 dB, dG;
  dB = dim3(nt1, nt2, 1);
  dG = dim3(nb1, nb2, 1);
  rescale_kernel <<< dG, dB >>> (phi, phi_max, 1);
}
