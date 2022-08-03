#include "grad_parallel.h"
#include <stdlib.h>
#include "get_error.h"
#define GCHAINS <<< dG[c], dB[c] >>>

GradParallelLinked::GradParallelLinked(Grids* grids, int jtwist, bool nonTwist, int *m0_h) // JMH
 : grids_(grids), nonTwist(nonTwist)
{
  nLinks       = nullptr;  nChains      = nullptr;
  ikxLinked_h  = nullptr;  ikyLinked_h  = nullptr;
  ikxLinked    = nullptr;  ikyLinked    = nullptr;
  kzLinked     = nullptr;  G_linked     = nullptr;
  dG           = nullptr;  dB           = nullptr;
  //mode_nums    = nullptr;  mode_size    = nullptr; //JMH
  //mode_size_ref= nullptr; // JMH
 
  zft_plan_forward           = nullptr;
  zft_plan_inverse           = nullptr;

  zft_plan_forward_singlemom = nullptr;
  //  zft_plan_inverse_singlemom = nullptr;       

  dz_plan_forward            = nullptr;
  dz_plan_inverse            = nullptr;

  dz_plan_forward_singlemom  = nullptr;
  dz_plan_inverse_singlemom  = nullptr;       

  abs_dz_plan_forward_singlemom = nullptr;

  int naky = grids_->Naky;
  int nakx = grids_->Nakx;
  int nz = grids_->Nz; // JMH
  int nx = grids_->Nx; // JMH
  
  if (nonTwist) { // JMH
     //initialize grids
    //mode_nums = (int*) malloc(sizeof(int)*naky*nakx*nz);

    int mode_nums[naky*nakx*nz] = {0}; //array that lists what mode a point is part of
    //int mode; //counter for number of modes
    
    mode = get_mode_nums_ntft(mode_nums, nz, naky, nakx, jtwist, m0_h, grids_->Nyc, grids_->ky_h);
    //mode_size = (int*) calloc(mode, sizeof(int));
    //mode_size_ref = (int*) calloc(mode, sizeof(int));
    int mode_size[mode] = {0}; // this will be sorted, used for nLinks/nChains
    int mode_size_ref[mode] = {0}; //this won't be sorted, used for filling kx/ky grids
    nClasses = get_nClasses_ntft(mode_size, mode_size_ref, mode_nums, naky, nakx, nz, mode);
    
    nChains = (int*) malloc(sizeof(int)*nClasses);
    nLinks = (int*) malloc(sizeof(int)*nClasses); //this is number of grid points, not 2pi segments

    get_nChains_nLinks_ntft(mode_size, nLinks, nChains, nClasses, nakx, naky, mode);

    // note that ikxLinked for NTFT will be negative and stores both ikx and idz via combined index
    ikxLinked_h = (int**) malloc(sizeof(int*)*nClasses); 
    ikyLinked_h = (int**) malloc(sizeof(int*)*nClasses);


    for(int c=0; c<nClasses; c++) {
      ikxLinked_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
      ikyLinked_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
    }

    kFill_ntft(nClasses, nChains, nLinks, ikyLinked_h, ikxLinked_h, naky, nakx, jtwist, nz, mode, mode_size_ref, mode_nums, nx);
    
  }
  else { // conventional flux tube, nothing changed // JMH
    int idxRight[naky*nakx];
    int idxLeft[naky*nakx];

    int linksR[naky*nakx];
    int linksL[naky*nakx];

    int n_k[naky*nakx];
  
    nClasses = get_nClasses(idxRight, idxLeft, linksR, linksL, n_k, naky, nakx, jtwist);
    
    nLinks   = (int*) malloc(sizeof(int)*nClasses);
    nChains  = (int*) malloc(sizeof(int)*nClasses);
    get_nLinks_nChains(nLinks, nChains, n_k, nClasses, naky, nakx);
    
    ikxLinked_h = (int**) malloc(sizeof(int*)*nClasses);
    ikyLinked_h = (int**) malloc(sizeof(int*)*nClasses);

    for(int c=0; c<nClasses; c++) {
      ikxLinked_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
      ikyLinked_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
    }

    kFill(nClasses, nChains, nLinks, ikyLinked_h, ikxLinked_h, linksL, linksR, idxRight, naky, nakx);
  }
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

  abs_dz_plan_forward_singlemom = (cufftHandle*) malloc(sizeof(cufftHandle*)*nClasses);

  // these are arrays of pointers to device memory
  ikxLinked = (int**) malloc(sizeof(int*)*nClasses);
  ikyLinked = (int**) malloc(sizeof(int*)*nClasses);
  G_linked = (cuComplex**) malloc(sizeof(cuComplex*)*nClasses);
  kzLinked = (float**) malloc(sizeof(float*)*nClasses);
  
  // for NTFT, nLinks[c] is number of grid points per chain, so we don't need to multiply by Nz
  // changed all grids->Nz to nz below to not have to add larger if statement // JMH
  if (nonTwist) {
    nz = 1;
  }

  //  printf("nClasses = %d\n", nClasses);
  for(int c=0; c<nClasses; c++) {
    // allocate and copy into device memory

    int nLC = nLinks[c]*nChains[c];
    cudaMalloc ((void**) &ikxLinked[c],      sizeof(int)*nLC);
    cudaMalloc ((void**) &ikyLinked[c],      sizeof(int)*nLC);
    printf("nLinks[%d] = %d, nChains = %d \n", c, nLinks[c], nChains[c]); // JMH

    CP_TO_GPU(ikxLinked[c], ikxLinked_h[c], sizeof(int)*nLC);
    CP_TO_GPU(ikyLinked[c], ikyLinked_h[c], sizeof(int)*nLC);
    
    for(int i=0; i<nLinks[c]*nChains[c]; i++) { // JMH
      printf("ikxLinked[%d][%d] = %d \n", c, i, ikxLinked_h[c][i]); 
    }

    size_t sLClmz = sizeof(cuComplex)*nLC*grids_->Nl*grids_->Nm*nz;

    checkCuda(cudaMalloc((void**) &G_linked[c], sLClmz));
    cudaMemset(G_linked[c], 0., sLClmz);

    cudaMalloc((void**) &kzLinked[c], sizeof(float)*nz*nLinks[c]);
    cudaMemset(kzLinked[c], 0.,       sizeof(float)*nz*nLinks[c]);

    // set up transforms
    cufftCreate(    &zft_plan_forward[c]);
    cufftCreate(    &zft_plan_inverse[c]);
    cufftCreate(    &zft_plan_forward_singlemom[c]);
    //    cufftCreate(    &zft_plan_inverse_singlemom[c]);

    cufftCreate(    &dz_plan_forward[c]);
    cufftCreate(    &dz_plan_inverse[c]);
    cufftCreate(    &dz_plan_forward_singlemom[c]);
    cufftCreate(    &dz_plan_inverse_singlemom[c]);

    cufftCreate(&abs_dz_plan_forward_singlemom[c]);

    int size = nLinks[c]*nz;
    size_t workSize;
    int nClm = nChains[c]*grids_->Nl*grids_->Nm; 
    cufftMakePlanMany(zft_plan_forward[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize);
    cufftMakePlanMany(zft_plan_inverse[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize);

    cufftMakePlanMany(zft_plan_forward_singlemom[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nChains[c], &workSize);
    //    cufftMakePlanMany(zft_plan_inverse_singlemom[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nChains[c], &workSize);

    cufftMakePlanMany(dz_plan_forward[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize);
    cufftMakePlanMany(dz_plan_inverse[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize);

    cufftMakePlanMany(dz_plan_forward_singlemom[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nChains[c], &workSize);
    cufftMakePlanMany(dz_plan_inverse_singlemom[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nChains[c], &workSize);

    cufftMakePlanMany(abs_dz_plan_forward_singlemom[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 
                      nChains[c], &workSize);

    // initialize kzLinked
    init_kzLinked <<<1,1>>> (kzLinked[c], nLinks[c], false, nonTwist); // added nonTwist // JMH
     
    int nn1, nn2, nn3, nt1, nt2, nt3, nb1, nb2, nb3;

    nn1 = (nonTwist) ? nLinks[c] : nz;                    nn1 = nz; nt1 = min( nn1, 32 );    nb1 = 1 + (nn1-1)/nt1; //JMH
    nn2 = (nonTwist) ? nChains[c] : nLinks[c]*nChains[c]; nn2 = nLinks[c]*nChains[c]; nt2 = min( nn2,  4 );    nb2 = 1 + (nn2-1)/nt2; //JMH
    nn3 = grids_->Nmoms;                		                                      nt3 = min( nn3,  4 );    nb3 = 1 + (nn3-1)/nt3;
   
    printf("nn1 = %d, nn2 = %d, nn3 = %d \n", nn1, nn2, nn3); // JMH
    dB[c] = dim3(nt1, nt2, nt3);
    dG[c] = dim3(nb1, nb2, nb3);
    //    dB[c] = dim3(32,4,4);
    //    dG[c] = dim3(1 + (grids_->Nz-1)/dB[c].x,
    //		 1 + (nLinks[c]*nChains[c]-1)/dB[c].y,
    //		 1 + (grids_->Nmoms-1)/dB[c].z);
  }

  set_callbacks();
  
  hermite = new HermiteTransform(grids_);
  //  this->linkPrint();
}

GradParallelLinked::~GradParallelLinked()
{
  if (nLinks)        free(nLinks);
  if (nChains)       free(nChains);
  if (dB)            free(dB);
  if (dG)            free(dG);
  //if (mode_nums)     free(mode_nums); //JMH
  //if (mode_size)     free(mode_size); // JMH
  //if (mode_size_ref) free(mode_size_ref); //JMH


  for(int c=0; c<nClasses; c++) {

    cufftDestroy(    zft_plan_forward[c]          );
    cufftDestroy(    zft_plan_inverse[c]          );
    cufftDestroy(    zft_plan_forward_singlemom[c]);
    //    cufftDestroy(    zft_plan_inverse_singlemom[c]);

    cufftDestroy(    dz_plan_forward[c]           );
    cufftDestroy(    dz_plan_inverse[c]           );
    cufftDestroy(    dz_plan_forward_singlemom[c] );
    cufftDestroy(    dz_plan_inverse_singlemom[c] );

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
  for (int is=0; is < grids_->Nspecies; is++) {
    for(int c=0; c<nClasses; c++) {
      /*
      int nlin, nch;

      int *ifac;
      cudaMalloc((void**) &ifac, sizeof(int)*nlin*nch);
      CP_TO_CPU(&ifac, ikxLinked[c], sizeof(int)*nlin*nch);
      for (int j=0; j<nlin*nch; j++) printf("ikxLinked[%d] = %d \n", j, ifac[j]);
      cudaFree(ifac);
      */				       
      linkedCopy GCHAINS (G->G(0,0,is), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
      cufftExecC2C (zft_plan_forward[c], G_linked[c], G_linked[c], CUFFT_FORWARD);

      linkedCopyBack GCHAINS (G_linked[c], G->G(0,0,is), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
    }
  }
}

void GradParallelLinked::zft_inverse(MomentsG* G) 
{
  for (int is=0; is < grids_->Nspecies; is++) {
    for(int c=0; c<nClasses; c++) {
      linkedCopy GCHAINS (G->G(0,0,is), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);

      cufftExecC2C (zft_plan_inverse[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

      linkedCopyBack GCHAINS (G_linked[c], G->G(0,0,is), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
    }
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

void GradParallelLinked::applyBCs(MomentsG* G, MomentsG* GRhs, Fields* f, float* kperp2)
{
  for (int is=0; is < grids_->Nspecies; is++) {
    for(int c=0; c<nClasses; c++) {
      // each "class" has a different number of links in the chains, and a different number of chains.
      dampEnds_linked GCHAINS (G->G(0,0,is), f->phi, f->apar, kperp2, G->zt(is), G->vt(is), G->r2(is), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms, GRhs->G(0,0,is));
    }
  }
}

void GradParallelLinked::dz(MomentsG* G) 
{
  for (int is=0; is < grids_->Nspecies; is++) {
    for(int c=0; c<nClasses; c++) {
      // each "class" has a different number of links in the chains, and a different number of chains.
     linkedCopy GCHAINS (G->G(0,0,is), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);

     cufftExecC2C (dz_plan_forward[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
      
      //cudaDeviceSynchronize();
      //checkCudaErrors(cudaGetLastError());
      
     cufftExecC2C (dz_plan_inverse[c], G_linked[c], G_linked[c], CUFFT_INVERSE);
      
     linkedCopyBack GCHAINS (G_linked[c], G->G(0,0,is), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
    }
  }
}

// for a single moment m 
void GradParallelLinked::dz(cuComplex* m, cuComplex* res)
{
  int nMoms=1;

  for(int c=0; c<nClasses; c++) {
    // these only use the G(0,0) part of G_linked
    linkedCopy GCHAINS (m, G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);

    cufftExecC2C(dz_plan_forward_singlemom[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C(dz_plan_inverse_singlemom[c], G_linked[c], G_linked[c], CUFFT_INVERSE);

    linkedCopyBack GCHAINS (G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
  }
}

// for a single moment m
void GradParallelLinked::abs_dz(cuComplex* m, cuComplex* res)
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
  for(int c=0; c<nClasses; c++) {
    // set up callback functions
    cudaDeviceSynchronize();
    if (nonTwist) {
      cufftXtSetCallback(    dz_plan_forward[c],
		       (void**)   &i_kzLinkedNTFT_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]);

      cufftXtSetCallback(    dz_plan_forward_singlemom[c],
		       (void**)   &i_kzLinkedNTFT_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]);
    }
    else {
      cufftXtSetCallback(    dz_plan_forward[c],
		       (void**)   &i_kzLinked_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]);

      cufftXtSetCallback(    dz_plan_forward_singlemom[c],
		       (void**)   &i_kzLinked_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]);
    }

    cufftXtSetCallback(    zft_plan_forward[c],
		       (void**)   &zfts_Linked_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]);

    cufftXtSetCallback(abs_dz_plan_forward_singlemom[c],
		       (void**) &abs_kzLinked_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]);

    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
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
    cufftXtClearCallback(abs_dz_plan_forward_singlemom[c],  CUFFT_CB_ST_COMPLEX);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
}

int GradParallelLinked::get_mode_nums_ntft(int *mode_nums, int nz, int naky, int nakx, int jtwist, int *m0, int nyc, float *ky) // JMH
{
  // this function assigns every grid point in the 3D array mode_nums to a corresponding NTFT ballooning mode
  // also identifies total number of ballooning modes "mode"
  // for jtwist > 0, we start search for new modes in bottom left, but piece together modes 
  // from right to left

  int idz_prime, idx_constant, idx_prime, idz_temp; 
  int mode = 0;
  
  // fill order depends on jtwist sign, starts either bottom left or bottom right
 // printf("naky = %d, nakx = %d, nyc = %d, nz = %d \n", naky, nakx, nyc, nz);
  for(int idy=0; idy<naky; idy++) { // add in an if statement to distinguish whether 
    if (ky[idy] < 1e-10) { // special case for zonal mode
      for(int idx=0; idx<nakx; idx++) {
	 mode++;
	 for (int idz=0; idz<nz; idz++) {
	    mode_nums[idy + naky * (idx + nakx * idz)] = mode;
	 }
      }
    } else {
    for(int idx=0; idx<nakx; idx++) {
      if (jtwist<0) { //positive sloping lines, start in bottom right
        for(int idz=nz-1; idz>=0; idz--) {
          if (mode_nums[idy + naky * (idx + nakx * idz)] == 0) { //if you find a grid point not assigned to a mode
	    
            // once new mode is found, need to start assembling at farthest left point 
            idz_temp = idz;
	    while (m0[idy + nyc * idz_temp] == m0[idy + nyc * ((idz_temp - 1) % nz)] && idz_temp >= 0 ) {
	      idz_temp--;
	    }
	    
	    mode++; // increment the mode number
  	    idz_prime = idz;
	    idx_constant = idx + m0[idy + nyc * idz_prime]; //i_constant -> m+m0 constant -> Kx constant
	    while(idx_constant - m0[idy + nyc * idz_prime] < nakx) { // while kx < kx_allowed
		idx_prime = idx_constant - m0[idy+ nyc * idz_prime];
	        mode_nums[idy + naky * (idx_prime + nakx * idz_prime)] = mode; //is there a fastest way to index the data?
	        if (idz_prime == nz - 1) { // if at end of row
		  idx_constant = idx_constant + jtwist * idy; // shift upwards by ky*jtwist
		  idz_prime = 0; // restart at left hand side and continue
		} else { 
		  idz_prime++;
		}
	    }
	  }
	}
      }
      else if (jtwist>0) { // start in bottom left
        for(int idz=0; idz<nz; idz++) {
          if (mode_nums[idy + naky * (idx + nakx * idz)] == 0) { //if you find a grid point not assigned to a mode
	    
	    // once new mode is found, need to start assembling at farthest right point 
            idz_temp = idz;
	    while (m0[idy + nyc * idz_temp] == m0[idy + nyc * ((idz_temp + 1) % nz)] && idz_temp < nz ) {
	      idz_temp++;
	    }
	    if (idz_temp == nz) {
	      idz_temp = nz - 1;
	    } 
	    mode++; // increment the mode number once you find start of mode
  	    idz_prime = idz_temp; 
	    idx_constant = idx + m0[idy + nyc * idz_prime]; //i_constant -> m+m0 constant -> Kx constant
	    while(idx_constant - m0[idy + nyc * idz_prime] < nakx) {
		idx_prime = idx_constant - m0[idy+ nyc * idz_prime];
	        mode_nums[idy + naky * (idx_prime + nakx * idz_prime)] = mode;
	        if (idz_prime == 0) { // if at end of row, shift upwards and restart from right
		  idx_constant = idx_constant + jtwist * idy;
		  idz_prime = nz-1;
		} else { 
		  idz_prime--;
		}
	    }
	  }
	}
      }
    }
    }
  }
  printf("number of modes = %d \n", mode);
  return mode;
}

int GradParallelLinked::get_nClasses_ntft(int *mode_size, int *mode_size_ref, int *mode_nums, int naky, int nakx, int nz, int mode)
{ // JMH
 	
  // this uses the data from the get mode nums function to identify the number of classes
  // loop through grid and count how many grid points in each ballooning mode
  for(int idy=0; idy<naky; idy++) {
    for(int idx=nakx-1; idx>=0; idx--) {
       for(int idz=0; idz<nz; idz++) {
	 // add one to the mode length corresponding to that grid point, this is analagous to n_k
	 mode_size[mode_nums[idy + naky * (idx + nakx * idz)]-1]++;
       	 mode_size_ref[mode_nums[idy + naky * (idx + nakx * idz)]-1]++; //should be identical arrays
	 printf("%d ", mode_nums[idy + naky * (idx + nakx * idz)]); 
       }
       printf("\n");
    }
    printf(" \n\n\n\n");
  }

  qsort(mode_size, mode, sizeof(int), compare); //sort mode_size into increasing order

  // count how many different classes
  int nClasses = 1;
  for(int k=0; k<mode-1; k++){
    if(mode_size[k] != mode_size[k+1]) {
      nClasses++;
    }
  }
//  printf("nClasses = %d \n", nClasses);
  return nClasses;
}

void GradParallelLinked::get_nChains_nLinks_ntft(int *mode_size, int *nLinks, int *nChains, int nClasses, int nakx, int naky, int mode) // JMH
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
    if(mode_size[k] == mode_size[k-1]) {
      nChains[c]++;
    } else {
      //note that here, nLinks[c] represents # of grid points, not segments
      nLinks[c] = mode_size[k-1]; 
//      printf("nLinks(%d) = %d, nChains(%d) = %d \n", c, nLinks[c], c, nChains[c]);
      c++;
    }
  }
  nLinks[nClasses-1] = mode_size[mode-1];

//  printf("nLinks(%d) = %d, nChains(%d) = %d \n", nClasses-1, nLinks[nClasses-1], nClasses-1, nChains[nClasses-1]);
}

void GradParallelLinked::kFill_ntft(int nClasses, int *nChains, int *nLinks, int **ikyNTFT, int **neg_ikxdzNTFT, int naky, int nakx, int jtwist, int nz, int mode, int *mode_size_ref, int *mode_nums, int nx) // JMH
{
 
  // this function fills the ky and kx index arrays corresponding to each class c
  // again, fill order depends on sign of jtwist, but will always fill from -z to +z
 
  int nshift = nx-nakx;
  int n,p, idx0, idy, idx, idz;

  for(int ic=0; ic<nClasses; ic++) {
    n = -1; //keeps track of number of chains with same nLinks for indexing purposes
    for(int i=0; i<mode; i++) { 
      //check if the # of grid points at that class index is = to the # of grid points of the mode
      if (nLinks[ic] == mode_size_ref[i]) {
	n++; //chain number index
	p=0; //grid point number index
	printf("nLinks[%d] = %d; mode_num = %d \n", ic, nLinks[ic], i+1);
        for(idy=0; idy<naky; idy++) {
	  if (jtwist<0) { //positive sloping lines, start in bottom left
	    for(idx=0; idx<nakx; idx++) {
	      if (idx >= (nakx - 1)/2) { // transform idx to ikx (nonsequential)
	        idx0 = idx - nshift;
	      } else {
	        idx0 = idx + nakx;
	      }
              for(idz=0; idz<nz; idz++) {
	        if (mode_nums[idy + naky * (idx + nakx * idz)] == i+1) {
		  neg_ikxdzNTFT[ic][p + nLinks[ic] * n] = -(1 + idx0 + nx * idz); // this stores both ikx and idz, negative so it can be distinguished from conventional, 1 is added to make sure it is < 0 and not 0 (might not be needed?)  
	          ikyNTFT[ic][p + nLinks[ic] * n] = idy;
		  //printf("ikxNTFT[%d][%d] = %d; ikyNTFT[%d][%d] = %d idx0 = %d, idz = %d \n", ic, p + nLinks[ic] * n, neg_ikxdzNTFT[ic][p+nLinks[ic] * n], ic, p + nLinks[ic] * n, idy, idx0, idz);
		  p++;
		}
	      }
	    }
	  }
	  else { //if jtwist > 0, negative sloping lines, start in top left
	    for(idx=nakx-1; idx>=0; idx--) {
	      if (idx >= (nakx - 1)/2) { // transform idx to ikx (nonsequential)
	        idx0 = idx - nshift;
	      }  else {
	        idx0 = idx + nakx;
	      }
              for(idz=0; idz<nz; idz++) {
	        if (mode_nums[idy + naky * (idx + nakx * idz)] == i+1) {
		  neg_ikxdzNTFT[ic][p + nLinks[ic] * n] = -(1 + idx0 + nx * idz); 
	          ikyNTFT[ic][p+ nLinks[ic] * n] = idy;
		 // printf("ikxNTFT[%d][%d] = %d; ikyNTFT[%d][%d] = %d idx0 = %d, idz = %d \n", ic, p + nLinks[ic] * n, neg_ikxdzNTFT[ic][p+nLinks[ic] * n], ic, p + nLinks[ic] * n, idy, idx0, idz);
		  p++;
		}
	      }
	    }
	  }
	}
      }
    }
  }
}



