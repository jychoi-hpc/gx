#include "grad_parallel.h"
#include <stdlib.h>
#include "get_error.h"
#define GCHAINS <<< dG[c], dB[c] >>>

GradParallelNTFT::GradParallelNTFT(Parameters* pars, Grids* grids)
 : pars_(pars), grids_(grids)
{
  int jtwist = pars_->jtwist;
  nLinks       = nullptr;  nChains      = nullptr;
  ikxLinked_h  = nullptr;  ikyLinked_h  = nullptr;
  ikxLinked    = nullptr;  ikyLinked    = nullptr;
  kzLinked     = nullptr;  G_linked     = nullptr;
  dG           = nullptr;  dB           = nullptr;
  nExtra       = nullptr;
 
  zft_plan_forward           = nullptr;
  zft_plan_inverse           = nullptr;

  zft_plan_forward_singlemom = nullptr;
  //  zft_plan_inverse_singlemom = nullptr;       

  dz_plan_forward            = nullptr;
  dz_plan_inverse            = nullptr;  
  
  dz2_plan_forward            = nullptr;
  dz2_plan_forward_singlemom  = nullptr;

  dz_plan_forward_singlemom  = nullptr;
  dz_plan_inverse_singlemom  = nullptr;       
  
  hyperz_plan_forward            = nullptr;
  hyperz_plan_inverse            = nullptr;
  

  abs_dz_plan_forward = nullptr;
  abs_dz_plan_forward_singlemom = nullptr;

  int naky = grids_->Naky;
  int nakx = grids_->Nakx;
  int nz = grids_->Nz;
  int nx = grids_->Nx;
  int nLinks_max = 3000; // above this value, we cut off nLinks to be multiple of nz or else we are fft-ing chains with large prime factors
  int nLinks_min = 0;   // not used anymore // JMH
  
  int mode_nums[naky*nakx*nz] = {0}; //array that lists what mode a point is part of
    
  mode = get_mode_nums_ntft(mode_nums, nz, naky, nakx, jtwist, grids_->m0_h, grids_->Nyc, grids_->ky_h);
  int mode_size[mode] = {0}; // this will be sorted, used for nLinks/nChains
  int mode_size_ref[mode] = {0}; //this won't be sorted, used for filling kx/ky grids
  nExtra = (int*) malloc(sizeof(int)*mode); // this is used when chains are extremely long (>3000 grid points), cut off nExtra grid points to make nLinks integer multiple of nz for fft efficiency
  nClasses = get_nClasses_ntft(mode_size, mode_size_ref, mode_nums, nExtra, naky, nakx, nz, mode, nLinks_max, nLinks_min);
    
  nChains = (int*) malloc(sizeof(int)*nClasses);
  nLinks = (int*) malloc(sizeof(int)*nClasses); //this is number of grid points, not 2pi segments

  get_nChains_nLinks_ntft(mode_size, nLinks, nChains, nClasses, nakx, naky, nLinks_min, mode);
  
  // ikxLinked for NTFT stores both ikx and idz via combined index
  ikxLinked_h = (int**) malloc(sizeof(int*)*nClasses); 
  ikyLinked_h = (int**) malloc(sizeof(int*)*nClasses);

  for(int c=0; c<nClasses; c++) {
    ikxLinked_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
    ikyLinked_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
  }

  kFill_ntft(nClasses, nChains, nLinks, nExtra, ikyLinked_h, ikxLinked_h, naky, nakx, jtwist, nz, mode, mode_size_ref, mode_nums, nx, grids_->m0_h, grids_->Nyc);

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
  
  dz2_plan_forward = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);
  dz2_plan_forward_singlemom = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);
  
  abs_dz_plan_forward = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);
  abs_dz_plan_forward_singlemom = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);

  // these are arrays of pointers to device memory
  ikxLinked = (int**) malloc(sizeof(int*)*nClasses);
  ikyLinked = (int**) malloc(sizeof(int*)*nClasses);
  G_linked = (cuComplex**) malloc(sizeof(cuComplex*)*nClasses);
  kzLinked = (float**) malloc(sizeof(float*)*nClasses);

  //  printf("nClasses = %d\n", nClasses);
  for(int c=0; c<nClasses; c++) {
        //printf("\tClass %d: nChains = %d, nLinks = %d\n", c, nChains[c], nLinks[c]);

    // The NTFT treats nLinks as the number of grid points per chain because it is not always a
    // a multiple of Nz, so this section is the same as the linked version without multiplying by Nz
    // JMH  

    // allocate and copy into device memory
    int nLC = nLinks[c]*nChains[c];
    cudaMalloc ((void**) &ikxLinked[c],      sizeof(int)*nLC);
    cudaMalloc ((void**) &ikyLinked[c],      sizeof(int)*nLC);

    CP_TO_GPU(ikxLinked[c], ikxLinked_h[c], sizeof(int)*nLC);
    CP_TO_GPU(ikyLinked[c], ikyLinked_h[c], sizeof(int)*nLC);

    size_t sLClmz = sizeof(cuComplex)*nLC*grids_->Nl*grids_->Nm;

    checkCuda(cudaMalloc((void**) &G_linked[c], sLClmz));
    cudaMemset(G_linked[c], 0., sLClmz);

    cudaMalloc((void**) &kzLinked[c], sizeof(float)*nLinks[c]);
    cudaMemset(kzLinked[c], 0.,       sizeof(float)*nLinks[c]);


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
    
    cufftCreate(    &dz2_plan_forward[c]);
    cufftCreate(    &dz2_plan_forward_singlemom[c]);
    
    cufftCreate(&abs_dz_plan_forward[c]);
    cufftCreate(&abs_dz_plan_forward_singlemom[c]);

    int size = nLinks[c];
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
    
    checkCuda(cufftMakePlanMany(dz2_plan_forward[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize));
    checkCuda(cufftMakePlanMany(dz2_plan_forward_singlemom[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nChains[c], &workSize));
    
    checkCuda(cufftMakePlanMany(abs_dz_plan_forward[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize));
    checkCuda(cufftMakePlanMany(abs_dz_plan_forward_singlemom[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 
                      nChains[c], &workSize));

    // initialize kzLinked
    init_kzLinkedNTFT <<<1,1>>> (kzLinked[c], nLinks[c], false);

    int nn1, nn2, nn3, nt1, nt2, nt3, nb1, nb2, nb3;

    nn1 = nLinks[c];                    nt1 = min( nn1, 32 );    nb1 = 1 + (nn1-1)/nt1;
    nn2 = nChains[c];                   nt2 = min( nn2,  4 );    nb2 = 1 + (nn2-1)/nt2; 
    nn3 = grids_->Nmoms;                nt3 = min( nn3,  4 );    nb3 = 1 + (nn3-1)/nt3;
    
    dB[c] = dim3(nt1, nt2, nt3);
    dG[c] = dim3(nb1, nb2, nb3);
    
    //    dB[c] = dim3(32,4,4);
    //    dG[c] = dim3(1 + (grids_->Nz-1)/dB[c].x,
    //		 1 + (nLinks[c]*nChains[c]-1)/dB[c].y,
    //		 1 + (grids_->Nmoms-1)/dB[c].z);
  }
  set_callbacks();
  
  if(pars_->debug) this->linkPrint();
}

GradParallelNTFT::~GradParallelNTFT()
{
  if (nLinks)  free(nLinks);
  if (nChains) free(nChains);
  if (nExtra)  free(nExtra);
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
    
    cufftDestroy(   dz2_plan_forward[c]           );
    cufftDestroy(   dz2_plan_forward_singlemom[c] );
    
    cufftDestroy(abs_dz_plan_forward[c]);
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
  
  if (dz2_plan_forward)              free(   dz2_plan_forward);
  if (dz2_plan_forward_singlemom)    free(   dz2_plan_forward_singlemom);
  
  if (abs_dz_plan_forward) free(abs_dz_plan_forward);
  if (abs_dz_plan_forward_singlemom) free(abs_dz_plan_forward_singlemom);

  if (ikxLinked_h)         free(ikxLinked_h);
  if (ikyLinked_h)         free(ikyLinked_h);

  if (ikxLinked)           free(ikxLinked);
  if (ikyLinked)           free(ikyLinked);
  if (G_linked)            free(G_linked);
  if (kzLinked)            free(kzLinked);
}

void GradParallelNTFT::dealias(MomentsG* G)
{
  // not yet implemented
}

void GradParallelNTFT::dealias(cuComplex* f)
{
  // not yet implemented
}

void GradParallelNTFT::zft(MomentsG* G) 
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
    linkedCopyNTFT GCHAINS (G->G(), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);

    checkCuda(cufftExecC2C (zft_plan_forward[c], G_linked[c], G_linked[c], CUFFT_FORWARD));

    linkedCopyBackNTFT GCHAINS (G_linked[c], G->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
  }
}

void GradParallelNTFT::zft_inverse(MomentsG* G) 
{
  for(int c=0; c<nClasses; c++) {
    linkedCopyNTFT GCHAINS (G->G(), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);

    cufftExecC2C (zft_plan_inverse[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    linkedCopyBackNTFT GCHAINS (G_linked[c], G->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
  }
}

// for a single moment m 
void GradParallelNTFT::zft(cuComplex* m, cuComplex* res)
{
  int nMoms=1;

  for(int c=0; c<nClasses; c++) {  // these only use the G(0,0) part of G_linked
    linkedCopyNTFT GCHAINS (m, G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);

    cufftExecC2C(zft_plan_forward_singlemom[c], G_linked[c], G_linked[c], CUFFT_FORWARD);

    linkedCopyBackNTFT GCHAINS (G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
  }
}
/*
// for a single moment m 
void GradParallelNTFT::zft_inverse(cuComplex* m, cuComplex* res)
{
  int nMoms=1;

  for(int c=0; c<nClasses; c++) {  // these only use the G(0,0) part of G_linked
    linkedCopyNTFT GCHAINS (m, G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);

    cufftExecC2C(zft_plan_inverse_singlemom[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    linkedCopyBackNTFT GCHAINS (G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
  }
}
*/

void GradParallelNTFT::applyBCs(MomentsG* G, MomentsG* GRhs, Fields* f, float* kperp2, double dt)
{
  for(int c=0; c<nClasses; c++) {
    // each "class" has a different number of links in the chains, and a different number of chains.
    dampEnds_linkedNTFT GCHAINS (G->G(), f->phi, f->apar, f->bpar, kperp2, *(G->species), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms, GRhs->G(), pars_->damp_ends_widthfrac, (float) pars_->damp_ends_amp/dt);
  }
}

void GradParallelNTFT::dz(MomentsG* G, MomentsG* res, bool accumulate) 
{
  for(int c=0; c<nClasses; c++) {
    // each "class" has a different number of links in the chains, and a different number of chains.
    linkedCopyNTFT GCHAINS (G->G(), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);

    cufftExecC2C (dz_plan_forward[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C (dz_plan_inverse[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    if(accumulate) {
      linkedAccumulateBackNTFT GCHAINS (G_linked[c], res->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms, 1.0);
    } else {
      linkedCopyBackNTFT GCHAINS (G_linked[c], res->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
    }
  }
}

void GradParallelNTFT::dz2(MomentsG* G) 
{
  for(int c=0; c<nClasses; c++) {
    // each "class" has a different number of links in the chains, and a different number of chains.
    linkedCopyNTFT GCHAINS (G->G(), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);

    cufftExecC2C (dz2_plan_forward[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C (dz_plan_inverse[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    linkedCopyBackNTFT GCHAINS (G_linked[c], G->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
  }
}

// for a single moment m 
void GradParallelNTFT::dz(cuComplex* m, cuComplex* res, bool accumulate)
{
  int nMoms=1;

  for(int c=0; c<nClasses; c++) {
    // these only use the G(0,0) part of G_linked
    linkedCopyNTFT GCHAINS (m, G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);

    cufftExecC2C(dz_plan_forward_singlemom[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C(dz_plan_inverse_singlemom[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    if(accumulate) {
      linkedAccumulateBackNTFT GCHAINS (G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms, 1.0);
    } else {
      linkedCopyBackNTFT GCHAINS (G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
    }
  }
}

void GradParallelNTFT::hyperz(MomentsG* G, MomentsG* res, float nu, bool accumulate) 
{
  for(int c=0; c<nClasses; c++) {
    // each "class" has a different number of links in the chains, and a different number of chains.
    linkedCopyNTFT GCHAINS (G->G(), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);

    cufftExecC2C (hyperz_plan_forward[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C (hyperz_plan_inverse[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    if(accumulate) {
      linkedAccumulateBackNTFT GCHAINS (G_linked[c], res->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms, nu);
    } else {
      linkedCopyBackNTFT GCHAINS (G_linked[c], res->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
    }
  }
}

void GradParallelNTFT::abs_dz(MomentsG* G, MomentsG* res, bool accumulate) 
{
  for(int c=0; c<nClasses; c++) {
    // each "class" has a different number of links in the chains, and a different number of chains.
    linkedCopyNTFT GCHAINS (G->G(), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);

    cufftExecC2C (abs_dz_plan_forward[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C (dz_plan_inverse[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    if(accumulate) {
      linkedAccumulateBackNTFT GCHAINS (G_linked[c], res->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms, 1.0);
    } else {
      linkedCopyBackNTFT GCHAINS (G_linked[c], res->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
    }
  }
}

void GradParallelNTFT::dz2(cuComplex* m, cuComplex* res)
{
  int nMoms=1;
  for(int c=0; c<nClasses; c++) {
    // these only use the G(0,0) part of G_linked
    linkedCopyNTFT GCHAINS (m, G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);

    cufftExecC2C(dz2_plan_forward_singlemom[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C(dz_plan_inverse_singlemom[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    linkedCopyBackNTFT GCHAINS (G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
  }
}

// for a single moment m
void GradParallelNTFT::abs_dz(cuComplex* m, cuComplex* res, bool accumulate)
{
  int nMoms=1;

  for(int c=0; c<nClasses; c++) {
    // these only use the G(0,0) part of G_linked
    linkedCopyNTFT GCHAINS (m, G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);

    cufftExecC2C(abs_dz_plan_forward_singlemom[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C(    dz_plan_inverse_singlemom[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    linkedCopyBackNTFT GCHAINS (G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
  }
}

int compare_ntft (const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}

int GradParallelNTFT::get_mode_nums_ntft(int *mode_nums, int nz, int naky, int nakx, int jtwist, int *m0, int nyc, float *ky) // JMH
{
  // this function assigns every grid point in the 3D array mode_nums to a corresponding NTFT ballooning mode
  // also identifies total number of ballooning modes "mode"
  // for jtwist > 0, we start search for new modes in bottom left, but piece together modes 
  // from right to left
  int idz_prime, idx_constant, idx_prime, idz_start; 
  int mode = 0;
  
  for(int idy=0; idy<naky; idy++) { 
    if (ky[idy] < 1e-10) { // special case for zonal mode
      for(int idx=0; idx<nakx; idx++) {
	 mode++;
	 for (int idz=0; idz<nz; idz++) {
	    mode_nums[idy + naky * (idx + nakx * idz)] = mode;
	 }
      }
    } else {
      for(int idx=0; idx<nakx; idx++) {
        for(int idz=0; idz<nz; idz++) {
          if (mode_nums[idy + naky * (idx + nakx * idz)] == 0) { //if you find a grid point not assigned to a mode
  	    
  	    // once new mode is found, need to find farthest -z point to start assembling
            idz_start = idz;
  	    idx_constant = idx + m0[idy + nyc * idz];
  	    while (idx_constant - m0[idy + nyc * ((idz_start-1+nz)%nz)] - floor((idz_start-1)/(1.0*nz))*jtwist*idy >= 0 && idx_constant - m0[idy + nyc * ((idz_start-1+nz) %nz)] - floor((idz_start-1)/(1.0*nz))*jtwist*idy < nakx) {
  	      if (idz_start == 0) {
  	        idx_constant = idx_constant + jtwist * idy;
  	        idz_start = nz - 1;
  	      } else {
  	        idz_start--;
  	      }
  	    }
  	     
  	    mode++; // increment the mode number once you find start of mode
    	    idz_prime = idz_start;
  	    while(idx_constant - m0[idy + nyc * idz_prime] < nakx && idx_constant - m0[idy + nyc * idz_prime] >= 0) {
  	      idx_prime = idx_constant - m0[idy+ nyc * idz_prime];
  	      mode_nums[idy + naky * (idx_prime + nakx * idz_prime)] = mode;
  	      if (idz_prime == nz - 1) { // if at end of row, shift upwards and restart from right
  		idx_constant = idx_constant - jtwist * idy;
  		idz_prime = 0;
              } else { 
  		idz_prime++;
  	      }
  	    }
  	  }
  	}
      }
    } 
  }
  return mode;
}
int GradParallelNTFT::get_nClasses_ntft(int *mode_size, int *mode_size_ref, int *mode_nums, int *nExtra, int naky, int nakx, int nz, int mode, int nLinks_max, int nLinks_min)
{ // JMH
 
  // this uses the data from the get mode nums function to identify the number of classes
  // loop through grid and count how many grid points in each ballooning mode
  for(int idy=0; idy<naky; idy++) {
    for(int idx=nakx-1; idx>=0; idx--) {
       for(int idz=0; idz<nz; idz++) {
	 // add one to the mode length corresponding to that grid point, this is analagous to n_k
	 mode_size[mode_nums[idy + naky * (idx + nakx * idz)]-1]++;
       	 mode_size_ref[mode_nums[idy + naky * (idx + nakx * idz)]-1]++; //should be identical arrays
	 //printf("%3d ", mode_nums[idy + naky * (idx + nakx * idz)]); //uncomment these three print statements to see a visual of the NTFT kx/z grid
       }
       //printf("\n");
    }
    //printf(" \n\n\n\n");
  }
  
  // Because NTFT doesn't require multiple of nz grid points per chain, nLinks at low ky can get messy in high resolution scans (large prime factors). 
  // This section makes long chains (more than nLinks_max grid points) a multiple of nz by cutting off the highest kperp (damped) grid points when we do kFill below
  for (int k=0; k<mode; k++) {
    if (mode_size[k] > nLinks_max) {
      nExtra[k] = mode_size[k] % nz;
      mode_size[k] = mode_size[k] - nExtra[k]; // modes with different sizes that are cut down to the same length are in the same class
      mode_size_ref[k] = mode_size_ref[k] - nExtra[k];
    } else {
      nExtra[k] = 0;
    }
  }
  qsort(mode_size, mode, sizeof(int), compare_ntft); //sort mode_size into increasing order

  // count how many different classes
  int nClasses = 1;
  for(int k=0; k<mode-1; k++){
    if(mode_size[k] != mode_size[k+1] && mode_size[k] > nLinks_min) { // we don't fft extremely short links (<nLinks_min) in the NTFT, creates unphysical timestep restriction
      nClasses++;
    }
  }
  return nClasses;
}
void GradParallelNTFT::get_nChains_nLinks_ntft(int *mode_size, int *nLinks, int *nChains, int nClasses, int nakx, int naky, int nLinks_min, int mode) // JMH
{
  // this function fills nLinks and nChains arrays for each class (where a class represents a ballooning mode of different size)
  // nLinks[c] = number of GRID POINTS (not 2pi segments) in a ballooning mode of class c
  // nChains[c] = number of ballooning modes with length nLinks[c] in class c
  for(int c=0; c<nClasses; c++) {
    nChains[c] = 1;
    nLinks[c] = 0;
  }
  int c=0;
  for(int k=1; k<mode; k++) {
    if(mode_size[k-1] > nLinks_min) {
      if(mode_size[k] == mode_size[k-1]) {
       nChains[c]++;
      } else {
        nLinks[c] = mode_size[k-1];
        c++;
      }
    }
  }
  nLinks[nClasses-1] = mode_size[mode-1];
}

void GradParallelNTFT::kFill_ntft(int nClasses, int *nChains, int *nLinks, int *nExtra, int **ikyNTFT, int **ikxdzNTFT, int naky, int nakx, int jtwist, int nz, int mode, int *mode_size_ref, int *mode_nums, int nx, int* m0, int nyc) // JMH
{
 
  // this function fills the ky and kx index arrays corresponding to each class c
  // fill order in kx depends on sign of jtwist, but will always fill from -z to +z
 
  int nshift = nx - nakx;
  int n, p, idx0;
  int idz_prime, idx_constant, idx_prime, idz_start; 
  int extraLinkCounter;
  for(int ic=0; ic<nClasses; ic++) {
    n = -1; //keeps track of number of chains with same nLinks for indexing purposes
    for(int i=0; i<mode; i++) { 
      //check if the # of grid points at that class index is = to the # of grid points of the mode
      if (nLinks[ic] == mode_size_ref[i]) {
	n++; //chain number index
	p=0; //grid point number in chain index
        for(int idy=0; idy<naky; idy++) {
          if (idy == 0) { // special case for zonal mode
            for(int idx=0; idx<nakx; idx++) {
              if (mode_nums[idy + naky * (idx + nakx * 0)] == i + 1) {
		idx0 = calc_idx0(idx, nshift, nakx);
	        for (int idz=0; idz<nz; idz++) {
	          ikxdzNTFT[ic][p + nLinks[ic] * n] = idx0 + nx * idz;
	          ikyNTFT[ic][p+ nLinks[ic] * n] = idy;
	          p++;
	        }
              }
            }
	  } else {
            for(int idx=0; idx<nakx; idx++) {
              for(int idz=0; idz<nz; idz++) {
                if (mode_nums[idy + naky * (idx + nakx * idz)] == i + 1) { //if you find a grid point assigned to the mode number you're looking for
      
		  extraLinkCounter = 0; // used to cut down long chains by nExtra[ic] grid points	
                  // find the start of the mode
                  idz_start = idz;
                  idx_constant = idx + m0[idy + nyc * idz];
                  while (idx_constant - m0[idy + nyc * ((idz_start-1+nz)%nz)] - floor((idz_start-1)/(1.0*nz))*jtwist*idy >= 0 && idx_constant - m0[idy + nyc * ((idz_start-1+nz) %nz)] - floor((idz_start-1)/(1.0*nz))*jtwist*idy < nakx) {
                    if (idz_start == 0) {
  	              idx_constant = idx_constant + jtwist * idy;
                      idz_start = nz - 1;
                    } else {
  	              idz_start--;
                    }
                  }
        
                  //assemble ikxdz and iky grids from -z to +z
  	          idz_prime = idz_start;
              	  while(idx_constant - m0[idy + nyc * idz_prime] < nakx && idx_constant - m0[idy + nyc * idz_prime] >= 0 && p < nLinks[ic]) {
		    if (extraLinkCounter < nExtra[i]/2) { //only fill the k grids if we're not cutting off these points, nExtra[i]/2 on each side
		      extraLinkCounter++;
		    } else {
  	              idx_prime = idx_constant - m0[idy+ nyc * idz_prime];
		      idx0 = calc_idx0(idx_prime, nshift, nakx);
                      ikxdzNTFT[ic][p + nLinks[ic] * n] = idx0 + nx * idz_prime; // overwriting by nExtra/2 points
                      ikyNTFT[ic][p + nLinks[ic] * n] = idy;
                      p++;
		    }
                    if (idz_prime == nz - 1) { // if at end of row, shift upwards and restart from right
  	              idx_constant = idx_constant - jtwist * idy;
  	      	      idz_prime = 0;
  	      	     } else { 
  	              idz_prime++;
  	      	    }
                  }
                  idx = nakx - 1; //terminate for loops after you find a mode once
                  idz = nz - 1; 
                }
              }
            }
	  }
        }
      }  
    }
  }
}

int GradParallelNTFT::calc_idx0(int idx, int nshift, int nakx) {
// this function converts the idx in NTFT terms [0->nakx-1] in sequential indexing from neg to pos
// to the nonsequential idx0 used in the rest of the code [0->nx-1]
// note: nakx is always odd // JMH

  int idx0;

  // this converts from NTFT neg to pos idx grid to nonsequential idx grid used in conventional,
  // both in range [0->nakx-1]
  if (idx < (nakx/2)) {
    idx0 = idx + (nakx/2) + 1;
  } else {
    idx0 = idx - (nakx/2);
  }

  // this is copied from fill function, converts nonsequential [0->nakx-1] to nonsequential [0->nx-1]
  if (idx0 < (nakx+1)/2) {
    idx0 = idx0;
  } else {
    idx0 = idx0 + nshift;
  }

  return idx0;
}

void GradParallelNTFT::linkPrint() {
  printf("Printing links...\n");

  for(int c=0; c<nClasses; c++) {
    for(int n=0; n<nChains[c]; n++) {
      for(int p=0; p<nLinks[c]; p++) {
	if(((-ikxLinked_h[c][p+nLinks[c]*n]-1) % grids_->Nx)<(grids_->Nx-1)/3+1) {
          printf("(%d,%d) ", ikyLinked_h[c][p+nLinks[c]*n], ((-ikxLinked_h[c][p+nLinks[c]*n]-1) % grids_->Nx));
        }
        else {
          printf("(%d,%d) ",ikyLinked_h[c][p+nLinks[c]*n], ((-ikxLinked_h[c][p+nLinks[c]*n]-1) % grids_->Nx) - grids_->Nx);	
        }
        if(((-ikxLinked_h[c][p+nLinks[c]*n]-1) % grids_->Nx)>(grids_->Nx-1)/3 && ((-ikxLinked_h[c][p+nLinks[c]*n]-1) % grids_->Nx)<2*(grids_->Nx/3)+1) {
          printf("->DEALIASING ERROR");
        }	
        /* *counter= *counter+1; */
      }
      printf("\n");
    }
    printf("\n\n");
  }
}

void GradParallelNTFT::set_callbacks()
{
  cudaDeviceSynchronize();
  cufftCallbackStoreC   zfts_LinkedNTFT_callbackPtr_h;
  cufftCallbackStoreC    i_kzLinkedNTFT_callbackPtr_h;
  cufftCallbackStoreC  abs_kzLinkedNTFT_callbackPtr_h;
  cufftCallbackStoreC   mkz2_LinkedNTFT_callbackPtr_h;
  cufftCallbackStoreC hyperkzLinkedNTFT_callbackPtr_h;


  checkCuda(cudaMemcpyFromSymbol(&zfts_LinkedNTFT_callbackPtr_h, 
                     zfts_LinkedNTFT_callbackPtr, 
                     sizeof(zfts_LinkedNTFT_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&i_kzLinkedNTFT_callbackPtr_h, 
                     i_kzLinkedNTFT_callbackPtr, 
                     sizeof(i_kzLinkedNTFT_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&hyperkzLinkedNTFT_callbackPtr_h, 
                     hyperkzLinkedNTFT_callbackPtr, 
                     sizeof(hyperkzLinkedNTFT_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&mkz2_LinkedNTFT_callbackPtr_h, 
                     mkz2_LinkedNTFT_callbackPtr, 
                     sizeof(i_kzLinkedNTFT_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&abs_kzLinkedNTFT_callbackPtr_h, 
                     abs_kzLinkedNTFT_callbackPtr, 
                     sizeof(abs_kzLinkedNTFT_callbackPtr_h)));

  int *hyperdata_h, *hyperdata_d;
  hyperdata_h = (int*) malloc(sizeof(int)*2);
  cudaMalloc((void**) &hyperdata_d, sizeof(int)*2);

  for(int c=0; c<nClasses; c++) {
    // set up callback functions
    checkCuda(cufftXtSetCallback(    zft_plan_forward[c],
		       (void**)   &zfts_LinkedNTFT_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]));

    checkCuda(cufftXtSetCallback(    dz_plan_forward[c],
		       (void**)   &i_kzLinkedNTFT_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]));

    checkCuda(cufftXtSetCallback(    dz_plan_forward_singlemom[c],
		       (void**)   &i_kzLinkedNTFT_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]));
    hyperdata_h[0] = nLinks[c];
    hyperdata_h[1] = pars_->p_hyper_z;
    CP_TO_GPU(hyperdata_d, hyperdata_h, sizeof(int)*2);

    checkCuda(cufftXtSetCallback(    hyperz_plan_forward[c],
		       (void**)   &hyperkzLinkedNTFT_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&hyperdata_d));

    checkCuda(cufftXtSetCallback(abs_dz_plan_forward[c],
		       (void**) &abs_kzLinkedNTFT_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]));

    checkCuda(cufftXtSetCallback(    dz2_plan_forward[c],
		       (void**)   &mkz2_LinkedNTFT_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]));

    checkCuda(cufftXtSetCallback(    dz2_plan_forward_singlemom[c],
		       (void**)   &mkz2_LinkedNTFT_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]));

    checkCuda(cufftXtSetCallback(abs_dz_plan_forward_singlemom[c],
		       (void**) &abs_kzLinkedNTFT_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]));

  }
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
}

void GradParallelNTFT::clear_callbacks()
{
  for(int c=0; c<nClasses; c++) {
    // set up callback functions
    cudaDeviceSynchronize();
    cufftXtClearCallback(    zft_plan_inverse[c],           CUFFT_CB_ST_COMPLEX);
    //    cufftXtClearCallback(    zft_plan_inverse_singlemom[c], CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(    dz_plan_forward[c],            CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(hyperz_plan_forward[c],            CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(abs_dz_plan_forward[c],  CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(    dz_plan_forward_singlemom[c],  CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(abs_dz_plan_forward_singlemom[c],  CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(    dz2_plan_forward[c],            CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(    dz2_plan_forward_singlemom[c],  CUFFT_CB_ST_COMPLEX);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
}
          

