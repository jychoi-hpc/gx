#include "grad_parallel.h"
#include <stdlib.h>
#include "get_error.h"
#define GCHAINS <<< dG[c], dB[c] >>>

// Included for prime factorization, needed for checking if primes > 127 are present
#include "util.h"

// For std::max_element
#include <algorithm>

// For std::cerr
#include <iostream>

GradParallelLinked::GradParallelLinked(Parameters* pars, Grids* grids)
 : pars_(pars), grids_(grids)
{
  int jtwist = pars_->jtwist;
  nLinks       = nullptr;  nChains      = nullptr;
  ikxLinked_h  = nullptr;  ikyLinked_h  = nullptr;
  ikxLinked    = nullptr;  ikyLinked    = nullptr;
  kzLinked     = nullptr;  G_linked     = nullptr;
  dG           = nullptr;  dB           = nullptr;
 
  zft_plan_forward           = nullptr;
  zft_plan_inverse           = nullptr;

  zft_plan_forward_singlemom = nullptr;
  //  zft_plan_inverse_singlemom = nullptr;       

  dz_plan_forward            = nullptr;
  dz_plan_inverse            = nullptr;

  dz2_plan_forward            = nullptr;

  dz_plan_forward_singlemom  = nullptr;
  dz_plan_inverse_singlemom  = nullptr;       

  hyperz_plan_forward            = nullptr;
  hyperz_plan_inverse            = nullptr;

  abs_dz_plan_forward = nullptr;
  dz2_plan_forward_singlemom  = nullptr;
  abs_dz_plan_forward_singlemom = nullptr;

  int naky = grids_->Naky;
  int nakx = grids_->Nakx;

  int idxRight[naky*nakx];
  int idxLeft[naky*nakx];

  int linksR[naky*nakx];
  int linksL[naky*nakx];

  int n_k[naky*nakx];

  nClasses = get_nClasses(idxRight, idxLeft, linksR, linksL, n_k, naky, nakx, jtwist);

  nLinks   = (int*) malloc(sizeof(int)*nClasses);
  nChains  = (int*) malloc(sizeof(int)*nClasses);

  get_nLinks_nChains(nLinks, nChains, n_k, nClasses, naky, nakx);

  for( int c=0; c < nClasses; c++ ) {
    std::vector<int> factors = factorize<int>( nLinks[ c ] );
    int max_prime_factor = *( std::max_element( factors.begin(), factors.end() ) );
    if( max_prime_factor >= 127 ) {
      if( pars_->iproc == 0 ) {
        std::cerr << std::endl;
        std::cerr << "ERROR: In constructing the extended flux tubes, nLinks[" << c << "] = " << nLinks[c] << " which has prime factors larger than 127." << std::endl;
        std::cerr << "ERROR: cuFFT callbacks do not support fourier transforms with large prime factors. Aborting." << std::endl << std::endl;
        std::cerr << "We suggest adjusting nx to a nearby value." << std::endl << std::endl;
      }
      exit(-2);
    }
  }

  ikxLinked_h = (int**) malloc(sizeof(int*)*nClasses);
  ikyLinked_h = (int**) malloc(sizeof(int*)*nClasses);

  for(int c=0; c<nClasses; c++) {
    ikxLinked_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
    ikyLinked_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
  }

  kFill(nClasses, nChains, nLinks, ikyLinked_h, ikxLinked_h, linksL, linksR, idxRight, naky, nakx);

  dG = (dim3*) malloc(sizeof(dim3)*nClasses);
  dB = (dim3*) malloc(sizeof(dim3)*nClasses);

  zft_plan_forward = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);
  zft_plan_inverse = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);
  zft_plan_forward_singlemom = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);
  // zft_plan_inverse_singlemom = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);

  dz_plan_forward = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);
  dz_plan_inverse = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);
  dz_plan_forward_singlemom = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);
  dz_plan_inverse_singlemom = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);

  hyperz_plan_forward = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);
  hyperz_plan_inverse = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);

  abs_dz_plan_forward = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);
  dz2_plan_forward = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);
  dz2_plan_forward_singlemom = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);
  
  abs_dz_plan_forward_singlemom = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);

  // these are arrays of pointers to device memory
  ikxLinked = (int**) malloc(sizeof(int*)*nClasses);
  ikyLinked = (int**) malloc(sizeof(int*)*nClasses);
  G_linked = (cuComplex**) malloc(sizeof(cuComplex*)*nClasses);
  kzLinked = (float**) malloc(sizeof(float*)*nClasses);

  //  printf("nClasses = %d\n", nClasses);
  for(int c=0; c<nClasses; c++) {
    //    printf("\tClass %d: nChains = %d, nLinks = %d\n", c, nChains[c], nLinks[c]);

    // allocate and copy into device memory
    int nLC = nLinks[c]*nChains[c];
    cudaMalloc ((void**) &ikxLinked[c],      sizeof(int)*nLC);
    cudaMalloc ((void**) &ikyLinked[c],      sizeof(int)*nLC);

    CP_TO_GPU(ikxLinked[c], ikxLinked_h[c], sizeof(int)*nLC);
    CP_TO_GPU(ikyLinked[c], ikyLinked_h[c], sizeof(int)*nLC);

    size_t sLClmz = sizeof(cuComplex)*nLC*grids_->Nl*grids_->Nm*grids_->Nz;

    checkCuda(cudaMalloc((void**) &G_linked[c], sLClmz));
    cudaMemset(G_linked[c], 0., sLClmz);

    cudaMalloc((void**) &kzLinked[c], sizeof(float)*grids_->Nz*nLinks[c]);
    cudaMemset(kzLinked[c], 0.,       sizeof(float)*grids_->Nz*nLinks[c]);

    // set up transforms
    cufftCreate(    &zft_plan_forward[c]);
    cufftCreate(    &zft_plan_inverse[c]);
    cufftCreate(    &zft_plan_forward_singlemom[c]);
    //    cufftCreate(    &zft_plan_inverse_singlemom[c]);

    cufftCreate(    &dz_plan_forward[c]);
    cufftCreate(    &dz_plan_inverse[c]);
    cufftCreate(    &dz_plan_forward_singlemom[c]);
    cufftCreate(    &dz_plan_inverse_singlemom[c]);

    cufftCreate(&hyperz_plan_forward[c]);
    cufftCreate(&hyperz_plan_inverse[c]);

    cufftCreate(&abs_dz_plan_forward[c]);
    cufftCreate(    &dz2_plan_forward[c]);
    cufftCreate(    &dz2_plan_forward_singlemom[c]);

    cufftCreate(&abs_dz_plan_forward_singlemom[c]);

    int size = nLinks[c]*grids_->Nz;
    size_t workSize;
    int nClm = nChains[c]*grids_->Nl*grids_->Nm; 
    checkCuda(cufftMakePlanMany(zft_plan_forward[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize));
    checkCuda(cufftMakePlanMany(zft_plan_inverse[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize));

    checkCuda(cufftMakePlanMany(zft_plan_forward_singlemom[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nChains[c], &workSize));
    //    cufftMakePlanMany(zft_plan_inverse_singlemom[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nChains[c], &workSize);

    checkCuda(cufftMakePlanMany(dz_plan_forward[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize));
    checkCuda(cufftMakePlanMany(dz_plan_inverse[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize));

    checkCuda(cufftMakePlanMany(dz_plan_forward_singlemom[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nChains[c], &workSize));
    checkCuda(cufftMakePlanMany(dz_plan_inverse_singlemom[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nChains[c], &workSize));

    checkCuda(cufftMakePlanMany(hyperz_plan_forward[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize));
    checkCuda(cufftMakePlanMany(hyperz_plan_inverse[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize));

    checkCuda(cufftMakePlanMany(abs_dz_plan_forward[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize));
    checkCuda(cufftMakePlanMany(dz2_plan_forward[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize));
    checkCuda(cufftMakePlanMany(dz2_plan_forward_singlemom[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nChains[c], &workSize));

    checkCuda(cufftMakePlanMany(abs_dz_plan_forward_singlemom[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 
                      nChains[c], &workSize));

    // initialize kzLinked
    init_kzLinked <<<1,1>>> (kzLinked[c], nLinks[c], false);

    int nn1, nn2, nn3, nt1, nt2, nt3, nb1, nb2, nb3;

    nn1 = grids_->Nz;                   nt1 = min( nn1, 32 );    nb1 = 1 + (nn1-1)/nt1;
    nn2 = nLinks[c]*nChains[c];         nt2 = min( nn2,  4 );    nb2 = 1 + (nn2-1)/nt2; 
    nn3 = grids_->Nmoms;                nt3 = min( nn3,  4 );    nb3 = 1 + (nn3-1)/nt3;
    
    dB[c] = dim3(nt1, nt2, nt3);
    dG[c] = dim3(nb1, nb2, nb3);
    
    //    dB[c] = dim3(32,4,4);
    //    dG[c] = dim3(1 + (grids_->Nz-1)/dB[c].x,
    //		 1 + (nLinks[c]*nChains[c]-1)/dB[c].y,
    //		 1 + (grids_->Nmoms-1)/dB[c].z);
  }

  set_callbacks();
  
  if(pars_->debug && pars_->iproc == 0)  this->linkPrint();
}

GradParallelLinked::~GradParallelLinked()
{
  if (nLinks)  free(nLinks);
  if (nChains) free(nChains);
  if (dB)      free(dB);
  if (dG)      free(dG);

  for(int c=0; c<nClasses; c++) {

    cufftDestroy(    zft_plan_forward[c]          );
    cufftDestroy(    zft_plan_inverse[c]          );
    cufftDestroy(    zft_plan_forward_singlemom[c]);
    //    cufftDestroy(    zft_plan_inverse_singlemom[c]);

    cufftDestroy(    dz_plan_forward[c]           );
    cufftDestroy(    dz_plan_inverse[c]           );
    cufftDestroy(    dz_plan_forward_singlemom[c] );
    cufftDestroy(    dz_plan_inverse_singlemom[c] );

    cufftDestroy(hyperz_plan_forward[c]           );
    cufftDestroy(hyperz_plan_inverse[c]           );

    cufftDestroy(abs_dz_plan_forward[c]);
    cufftDestroy(   dz2_plan_forward[c]           );
    cufftDestroy(   dz2_plan_forward_singlemom[c] );
    cufftDestroy(abs_dz_plan_forward_singlemom[c]);

    if (ikxLinked_h[c])       free(ikxLinked_h[c]);
    if (ikyLinked_h[c])       free(ikyLinked_h[c]);
    if (ikxLinked[c])         cudaFree(ikxLinked[c]);
    if (ikyLinked[c])         cudaFree(ikyLinked[c]);
    if (kzLinked[c])          cudaFree(kzLinked[c]);
    if (G_linked[c])          cudaFree(G_linked[c]);
  }
  if (zft_plan_forward)              free(    zft_plan_forward);
  if (zft_plan_inverse)              free(    zft_plan_inverse);
  if (zft_plan_forward_singlemom)    free(    zft_plan_forward_singlemom);
  //  if (zft_plan_inverse_singlemom)    free(    zft_plan_inverse_singlemom);

  if (dz_plan_forward)               free(    dz_plan_forward);
  if (dz_plan_inverse)               free(    dz_plan_inverse);
  if (dz_plan_forward_singlemom)     free(    dz_plan_forward_singlemom);
  if (dz_plan_inverse_singlemom)     free(    dz_plan_inverse_singlemom);
  if (hyperz_plan_forward)           free(hyperz_plan_forward);
  if (hyperz_plan_inverse)           free(hyperz_plan_inverse);
  if (abs_dz_plan_forward) free(abs_dz_plan_forward);
  if (dz2_plan_forward)              free(   dz2_plan_forward);
  if (dz2_plan_forward_singlemom)    free(   dz2_plan_forward_singlemom);
  if (abs_dz_plan_forward_singlemom) free(abs_dz_plan_forward_singlemom);

  if (ikxLinked_h)         free(ikxLinked_h);
  if (ikyLinked_h)         free(ikyLinked_h);

  if (ikxLinked)           free(ikxLinked);
  if (ikyLinked)           free(ikyLinked);
  if (G_linked)            free(G_linked);
  if (kzLinked)            free(kzLinked);
}

void GradParallelLinked::dealias(MomentsG* G)
{
  // not yet implemented
}

void GradParallelLinked::dealias(cuComplex* f)
{
  // not yet implemented
}

void GradParallelLinked::zft(MomentsG* G) 
{
  for(int c=0; c<nClasses; c++) {
    /*
    int nlin, nch;

    int *ifac;
    cudaMalloc((void**) &ifac, sizeof(int)*nlin*nch);
    CP_TO_CPU(&ifac, ikxLinked[c], sizeof(int)*nlin*nch);
    for (int j=0; j<nlin*nch; j++) printf("ikxLinked[%d] = %d \n", j, ifac[j]);
    cudaFree(ifac);
    */				       
    linkedCopy GCHAINS (G->G(), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);

    checkCuda(cufftExecC2C (zft_plan_forward[c], G_linked[c], G_linked[c], CUFFT_FORWARD));

    linkedCopyBack GCHAINS (G_linked[c], G->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
  }
}

void GradParallelLinked::zft_inverse(MomentsG* G) 
{
  for(int c=0; c<nClasses; c++) {
    linkedCopy GCHAINS (G->G(), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);

    cufftExecC2C (zft_plan_inverse[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    linkedCopyBack GCHAINS (G_linked[c], G->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
  }
}

// for a single moment m 
void GradParallelLinked::zft(cuComplex* m, cuComplex* res)
{
  int nMoms=1;

  for(int c=0; c<nClasses; c++) {  // these only use the G(0,0) part of G_linked
    linkedCopy GCHAINS (m, G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);

    cufftExecC2C(zft_plan_forward_singlemom[c], G_linked[c], G_linked[c], CUFFT_FORWARD);

    linkedCopyBack GCHAINS (G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
  }
}
/*
// for a single moment m 
void GradParallelLinked::zft_inverse(cuComplex* m, cuComplex* res)
{
  int nMoms=1;

  for(int c=0; c<nClasses; c++) {  // these only use the G(0,0) part of G_linked
    linkedCopy GCHAINS (m, G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);

    cufftExecC2C(zft_plan_inverse_singlemom[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    linkedCopyBack GCHAINS (G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
  }
}
*/

void GradParallelLinked::applyBCs(MomentsG* G, MomentsG* GRhs, Fields* f, float* kperp2, double dt)
{
  for(int c=0; c<nClasses; c++) {
    // each "class" has a different number of links in the chains, and a different number of chains.
    dampEnds_linked GCHAINS (G->G(), f->phi, f->apar, f->bpar, kperp2, *(G->species), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms, GRhs->G(), pars_->damp_ends_widthfrac, (float) pars_->damp_ends_amp/dt);
  }
}

void GradParallelLinked::dz(MomentsG* G, MomentsG* res, bool accumulate) 
{
  for(int c=0; c<nClasses; c++) {
    // each "class" has a different number of links in the chains, and a different number of chains.
    linkedCopy GCHAINS (G->G(), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);

    cufftExecC2C (dz_plan_forward[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C (dz_plan_inverse[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    if(accumulate) {
      linkedAccumulateBack GCHAINS (G_linked[c], res->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms, 1.0);
    } else {
      linkedCopyBack GCHAINS (G_linked[c], res->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
    }
  }
}

void GradParallelLinked::dz2(MomentsG* G) 
{
  for(int c=0; c<nClasses; c++) {
    // each "class" has a different number of links in the chains, and a different number of chains.
    linkedCopy GCHAINS (G->G(), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);

    cufftExecC2C (dz2_plan_forward[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C (dz_plan_inverse[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    linkedCopyBack GCHAINS (G_linked[c], G->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
  }
}

// for a single moment m 
void GradParallelLinked::dz(cuComplex* m, cuComplex* res, bool accumulate)
{
  int nMoms=1;

  for(int c=0; c<nClasses; c++) {
    // these only use the G(0,0) part of G_linked
    linkedCopy GCHAINS (m, G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);

    cufftExecC2C(dz_plan_forward_singlemom[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C(dz_plan_inverse_singlemom[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    if(accumulate) {
      linkedAccumulateBack GCHAINS (G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms, 1.0);
    } else {
      linkedCopyBack GCHAINS (G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
    }
  }
}

void GradParallelLinked::hyperz(MomentsG* G, MomentsG* res, float nu, bool accumulate) 
{
  for(int c=0; c<nClasses; c++) {
    // each "class" has a different number of links in the chains, and a different number of chains.
    linkedCopy GCHAINS (G->G(), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);

    cufftExecC2C (hyperz_plan_forward[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C (hyperz_plan_inverse[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    if(accumulate) {
      linkedAccumulateBack GCHAINS (G_linked[c], res->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms, nu);
    } else {
      linkedCopyBack GCHAINS (G_linked[c], res->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
    }
  }
}

void GradParallelLinked::abs_dz(MomentsG* G, MomentsG* res, bool accumulate) 
{
  for(int c=0; c<nClasses; c++) {
    // each "class" has a different number of links in the chains, and a different number of chains.
    linkedCopy GCHAINS (G->G(), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);

    cufftExecC2C (abs_dz_plan_forward[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C (dz_plan_inverse[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    if(accumulate) {
      linkedAccumulateBack GCHAINS (G_linked[c], res->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms, 1.0);
    } else {
      linkedCopyBack GCHAINS (G_linked[c], res->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
    }
  }
}

// for a single moment m 
void GradParallelLinked::dz2(cuComplex* m, cuComplex* res)
{
  int nMoms=1;

  for(int c=0; c<nClasses; c++) {
    // these only use the G(0,0) part of G_linked
    linkedCopy GCHAINS (m, G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);

    cufftExecC2C(dz2_plan_forward_singlemom[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C(dz_plan_inverse_singlemom[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    linkedCopyBack GCHAINS (G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
  }
}

// for a single moment m
void GradParallelLinked::abs_dz(cuComplex* m, cuComplex* res, bool accumulate)
{
  int nMoms=1;

  for(int c=0; c<nClasses; c++) {
    // these only use the G(0,0) part of G_linked
    linkedCopy GCHAINS (m, G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);

    cufftExecC2C(abs_dz_plan_forward_singlemom[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C(    dz_plan_inverse_singlemom[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    linkedCopyBack GCHAINS (G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
  }
}

int compare (const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}

int GradParallelLinked::get_nClasses(int *idxRight, int *idxLeft, int *linksR, int *linksL,
				     int *n_k, int naky, int nakx, int jshift0)
{  
  int idx0, idxL, idxR;

  //  printf("naky, nakx, jshift0 = %d \t %d \t %d \n",naky, nakx, jshift0);
  
  for(int idx=0; idx<nakx; idx++) {
    for(int idy=0; idy<naky; idy++) {
      
      //map indices to kx indices
      if(idx < (nakx+1)/2 ) {
        idx0 = idx;     
      } else {
        idx0 = idx - nakx;
      }       
              
      if(idy == 0) {                 
        idxL = idx0;
	idxR = idx0;
      } else {
        // signs here are correct according to Mike Beer's thesis
        idxL = idx0 + idy*jshift0;
	idxR = idx0 - idy*jshift0;
      }
      
      //remap to usual indices
      if(idxL >= 0 && idxL < (nakx+1)/2) {
        idxLeft[idy + naky*idx] = idxL;
      } else if( idxL+nakx >= (nakx+1)/2 && idxL+nakx < nakx ) {
        idxLeft[idy + naky*idx] = idxL + nakx;                   //nshift
      } else {
        idxLeft[idy + naky*idx] = -1;
      }
      
      if(idxR >= 0 && idxR < (nakx+1)/2) {
        idxRight[idy + naky*idx] = idxR;
      } else if( idxR+nakx >= (nakx+1)/2 && idxR+nakx <nakx ) {
        idxRight[idy + naky*idx] = idxR + nakx;
      } else {
        idxRight[idy + naky*idx] = -1;
      }
    }
  }

  /*
  for(int idx=0; idx<nakx; idx++) {
    for(int idy=0; idy<naky; idy++) {
      printf("idxLeft[%d,%d]= %d  ", idy, idx, idxLeft[idy + naky*idx]);
    }
    printf("\n");
  }
  for(int idx=0; idx<nakx; idx++) {
    for(int idy=0; idy<naky; idy++) {
      printf("idxRight[%d,%d]= %d  ", idy, idx, idxRight[idy + naky*idx]);
    }
    printf("\n");
  }
  */
  
  for(int idx=0; idx<nakx; idx++) {
    for(int idy=0; idy<naky; idy++) {
      
      
      //count the links for each region
      //linksL = number of links to the left of current position
      
      linksL[idy + naky*idx] = 0;     
      
      int idx_star = idx;
      
      while(idx_star != idxLeft[idy + naky*idx_star] && idxLeft[idy + naky*(idx_star)] >= 0) {
        //increment left links counter, and move to next link to left
	//until idx of link to left is negative or same as current idx
	linksL[idy + naky*idx]++;
	idx_star = idxLeft[idy + naky*(idx_star)];
      }	  
     
      //linksR = number of links to the right
      linksR[idy + naky*idx] = 0;     
      idx_star = idx;
      while(idx_star != idxRight[idy + naky*idx_star] && idxRight[idy + naky*idx_star] >= 0) {
        linksR[idy + naky*idx]++;
        idx_star = idxRight[idy + naky*idx_star];
      }	       
    }
  }

  /*
  for(int idx=0; idx<nakx; idx++) {
    for(int idy=0; idy<naky; idy++) {
      printf("linksL[%d,%d]= %d  ", idy, idx, linksL[idy + naky*idx]);
    }
    printf("\n");
  }
  for(int idx=0; idx<nakx; idx++) {
    for(int idy=0; idy<naky; idy++) {
      printf("linksR[%d,%d]= %d  ", idy, idx, linksR[idy + naky*idx]);
    }
    printf("\n");
  } 
  */
  
  //now we set up class array
  //nClasses = # of classes  
  
  //first count number of links for each (kx,ky)
  int k = 0;
  
  for(int idx=0; idx<nakx; idx++) {
    for(int idy=0; idy<naky; idy++) {
      n_k[k] = 1 + linksL[idy + naky*idx] + linksR[idy + naky*idx];
      k++;
    }
  }

  /*
  for(int idx=0; idx<nakx; idx++) {
    for(int idy=0; idy<naky; idy++) {
      printf("nLinks[%d,%d]= %d  ", idy, idx, n_k[idy+naky*idx]);
    }
    printf("\n");
  }
  */
  
  //count how many unique values of n_k there are, which is the number of classes
  
  //sort...
  qsort(n_k, naky*nakx, sizeof(int), compare);   
  
  //then count
  int nClasses = 1;
  for(int k=0; k<naky*nakx-1; k++) {
    if(n_k[k] != n_k[k+1])
      nClasses= nClasses + 1;
  }
  return nClasses;  
}

void GradParallelLinked::get_nLinks_nChains(int *nLinks, int *nChains, int *n_k, int nClasses, int naky, int nakx)
{
  for(int c=0; c<nClasses; c++) {
    nChains[c] = 1;
    nLinks[c] = 0;
  }
  
  //fill the nChains and nLinks arrays
  int c = 0;
  for(int k=1; k<naky*nakx; k++) {
    if(n_k[k] == n_k[k-1])
      nChains[c]++;
    else {
      nLinks[c] = n_k[k-1];
      nChains[c] = nChains[c]/nLinks[c];
      c++;
    }
  }
  c = nClasses-1;
  nLinks[c] = n_k[naky*nakx-1];
  nChains[c] = nChains[c]/nLinks[c];
  
}  


void kt2ki(int idy, int idx, int *c, int *p, int* linksL, int* linksR, int nClasses, int* nLinks, int naky)
{
  //get nLinks in the current chain
  int np_k = 1 + linksL[idy + naky*idx] + linksR[idy + naky*idx];
  
  //find which class corresponds to this nLinks
  for(int i=0; i<nClasses; i++) {
    if(nLinks[i] == np_k) {
      *c= i;
      break;
    }
  }
  
  *p = linksL[idy + naky*idx];
} 


void fill(int *ky, int *kx, int idy, int idx, int *idxRight,
	  int c, int p, int n, int naky, int nakx, int nshift, int nLinks) {
  int idx0;
  if(idx < (nakx+1)/2)
    idx0=idx;
  else
    idx0=idx+nshift;
  
  ky[p+nLinks*n] = idy;              
  kx[p+nLinks*n] = idx0;
  int idxR=idx;
  
  for(p=1; p<nLinks; p++) {
    idxR = idxRight[idy + naky*idxR];     
    
    ky[p + nLinks*n] = idy;
    if(idxR < (nakx+1)/2) {      
      kx[p + nLinks*n] = idxR;
    } else {
      kx[p + nLinks*n] = idxR+nshift;
    }
  }  
}   
  
void GradParallelLinked::kFill(int nClasses, int *nChains, int *nLinks,
			       int **ky, int **kx, int *linksL, int *linksR, int *idxRight, int naky, int nakx) 
{
  int nshift = grids_->Nx-nakx;
  //fill the kx and ky arrays
  for(int ic=0; ic<nClasses; ic++) {
    int n = 0;
    int  p, c;
    for(int idy=0; idy<naky; idy++) {
      for(int idx=0; idx<nakx; idx++) {
        kt2ki(idy, idx, &c, &p, linksL, linksR, nClasses, nLinks, naky);
     	if(c==ic) {	  
	  if(p==0) {	 
	    fill(ky[c], kx[c], idy, idx, idxRight, c, p, n, naky, nakx, nshift, nLinks[c]);
	    
	    n++;
	  }
	}
      }
    }
  }
}      	

void GradParallelLinked::linkPrint() {
  printf("Printing links...\n");

  for(int c=0; c<nClasses; c++) {
    for(int n=0; n<nChains[c]; n++) {
      for(int p=0; p<nLinks[c]; p++) {
	if(ikxLinked_h[c][p+nLinks[c]*n]<(grids_->Nx-1)/3+1) {
          printf("(%d,%d) ", ikyLinked_h[c][p+nLinks[c]*n], ikxLinked_h[c][p+nLinks[c]*n]);
        }
        else {
          printf("(%d,%d) ",ikyLinked_h[c][p+nLinks[c]*n], ikxLinked_h[c][p+nLinks[c]*n]-grids_->Nx);	
        }
        if(ikxLinked_h[c][p+nLinks[c]*n]>(grids_->Nx-1)/3 && ikxLinked_h[c][p+nLinks[c]*n]<2*(grids_->Nx/3)+1) {
          printf("->DEALIASING ERROR");
        }	
        /* *counter= *counter+1; */
      }
      printf("\n");
    }
    printf("\n\n");
  }
}

void GradParallelLinked::set_callbacks()
{
  cudaDeviceSynchronize();
  cufftCallbackStoreC  zfts_Linked_callbackPtr_h;
  cufftCallbackStoreC   i_kzLinked_callbackPtr_h;
  cufftCallbackStoreC  mkz2_Linked_callbackPtr_h;
  cufftCallbackStoreC abs_kzLinked_callbackPtr_h;
  cufftCallbackStoreC   hyperkzLinked_callbackPtr_h;
  checkCuda(cudaMemcpyFromSymbol(&zfts_Linked_callbackPtr_h, 
                     zfts_Linked_callbackPtr, 
                     sizeof(zfts_Linked_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&i_kzLinked_callbackPtr_h, 
                     i_kzLinked_callbackPtr, 
                     sizeof(i_kzLinked_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&hyperkzLinked_callbackPtr_h, 
                     hyperkzLinked_callbackPtr, 
                     sizeof(hyperkzLinked_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&mkz2_Linked_callbackPtr_h, 
                     mkz2_Linked_callbackPtr, 
                     sizeof(i_kzLinked_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&abs_kzLinked_callbackPtr_h, 
                     abs_kzLinked_callbackPtr, 
                     sizeof(abs_kzLinked_callbackPtr_h)));

  int *hyperdata_h, *hyperdata_d;
  hyperdata_h = (int*) malloc(sizeof(int)*2);
  cudaMalloc((void**) &hyperdata_d, sizeof(int)*2);
  for(int c=0; c<nClasses; c++) {
    // set up callback functions
    checkCuda(cufftXtSetCallback(    zft_plan_forward[c],
		       (void**)   &zfts_Linked_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]));

    checkCuda(cufftXtSetCallback(    dz_plan_forward[c],
		       (void**)   &i_kzLinked_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]));

    checkCuda(cufftXtSetCallback(    dz_plan_forward_singlemom[c],
		       (void**)   &i_kzLinked_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]));

    hyperdata_h[0] = nLinks[c];
    hyperdata_h[1] = pars_->p_hyper_z;
    CP_TO_GPU(hyperdata_d, hyperdata_h, sizeof(int)*2);
    checkCuda(cufftXtSetCallback(    hyperz_plan_forward[c],
		       (void**)   &hyperkzLinked_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&hyperdata_d));

    checkCuda(cufftXtSetCallback(abs_dz_plan_forward[c],
		       (void**) &abs_kzLinked_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]));
    checkCuda(cufftXtSetCallback(    dz2_plan_forward[c],
		       (void**)   &mkz2_Linked_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]));

    checkCuda(cufftXtSetCallback(    dz2_plan_forward_singlemom[c],
		       (void**)   &mkz2_Linked_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]));

    checkCuda(cufftXtSetCallback(abs_dz_plan_forward_singlemom[c],
		       (void**) &abs_kzLinked_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]));

  }
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
}

void GradParallelLinked::clear_callbacks()
{
  for(int c=0; c<nClasses; c++) {
    // set up callback functions
    cudaDeviceSynchronize();
    cufftXtClearCallback(    zft_plan_inverse[c],           CUFFT_CB_ST_COMPLEX);
    //    cufftXtClearCallback(    zft_plan_inverse_singlemom[c], CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(    dz_plan_forward[c],            CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(    dz_plan_forward_singlemom[c],  CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(hyperz_plan_forward[c],            CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(abs_dz_plan_forward[c],  CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(abs_dz_plan_forward_singlemom[c],  CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(    dz2_plan_forward[c],            CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(    dz2_plan_forward_singlemom[c],  CUFFT_CB_ST_COMPLEX);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
}
          

