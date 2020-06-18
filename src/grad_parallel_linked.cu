#include "grad_parallel.h"
#include "cuda_constants.h"
#include "device_funcs.h"
#include "cufftXt.h"
#include "cufft.h"
#include <stdlib.h>
#include "get_error.h"

GradParallelLinked::GradParallelLinked(Grids* grids, int jtwist)
 : grids_(grids)
{
  //  int naky = grids_->Nyc;
  //  int nakx = grids_->Nakx;

  int naky = grids_->Naky;
  int nakx = grids_->Nakx;

  int idxRight[naky*nakx];
  int idxLeft[naky*nakx];

  int linksR[naky*nakx];
  int linksL[naky*nakx];

  int n_k[naky*nakx];

  nClasses = get_nClasses(idxRight, idxLeft, linksR, linksL, n_k, naky, nakx, jtwist);

  nLinks = (int*) malloc(sizeof(int)*nClasses);
  nChains = (int*) malloc(sizeof(int)*nClasses);
  get_nLinks_nChains(nLinks, nChains, n_k, nClasses, naky, nakx);

  ikxLinked_h = (int**) malloc(sizeof(int*)*nClasses);
  ikyLinked_h = (int**) malloc(sizeof(int*)*nClasses);
  for(int c=0; c<nClasses; c++) {
    ikxLinked_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
    ikyLinked_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
  }

  kFill(nClasses, nChains, nLinks, ikyLinked_h, ikxLinked_h, linksL, linksR, idxRight, naky, nakx);

  dG = (dim3*) malloc(sizeof(dim3)*nClasses);
  dB = (dim3*) malloc(sizeof(dim3)*nClasses);

  cudaMallocHost((void**) &dz_plan_forward,               sizeof(cufftHandle*)*nClasses);
  cudaMallocHost((void**) &dz_plan_inverse,               sizeof(cufftHandle*)*nClasses);
  cudaMallocHost((void**) &dz_plan_forward_singlemom,     sizeof(cufftHandle*)*nClasses);
  cudaMallocHost((void**) &abs_dz_plan_forward_singlemom, sizeof(cufftHandle*)*nClasses);
  cudaMallocHost((void**) &dz_plan_inverse_singlemom,     sizeof(cufftHandle*)*nClasses);

  // these are arrays of pointers to device memory
  cudaMallocHost((void**) &ikxLinked, sizeof(int*)      *nClasses);
  cudaMallocHost((void**) &ikyLinked, sizeof(int*)      *nClasses);
  cudaMallocHost((void**) &G_linked,  sizeof(cuComplex*)*nClasses);
  cudaMallocHost((void**) &kzLinked,  sizeof(float*)    *nClasses);

  //  printf("nClasses = %d\n", nClasses);
  for(int c=0; c<nClasses; c++) {
    //    printf("\tClass %d: nChains = %d, nLinks = %d\n", c, nChains[c], nLinks[c]);
    // allocate and copy into device memory
    int nLC = nLinks[c]*nChains[c];
    cudaMalloc((void**) &ikxLinked[c],      sizeof(int)*nLC);
    cudaMalloc((void**) &ikyLinked[c],      sizeof(int)*nLC);

    CP_TO_GPU(ikxLinked[c], ikxLinked_h[c], sizeof(int)*nLC);
    CP_TO_GPU(ikyLinked[c], ikyLinked_h[c], sizeof(int)*nLC);

    size_t sLClmz = sizeof(cuComplex)*nLC*grids_->Nl*grids_->Nm*grids_->Nz;
    checkCuda(cudaMalloc((void**) &G_linked[c], sLClmz));
    cudaMemset(G_linked[c], 0., sLClmz);

    cudaMalloc((void**) &kzLinked[c], sizeof(float)*grids_->Nz*nLinks[c]);
    cudaMemset(kzLinked[c], 0., sizeof(float)*grids_->Nz*nLinks[c]);

    // set up transforms
    cufftCreate(    &dz_plan_forward[c]);
    cufftCreate(    &dz_plan_inverse[c]);
    cufftCreate(    &dz_plan_forward_singlemom[c]);
    cufftCreate(&abs_dz_plan_forward_singlemom[c]);
    cufftCreate(    &dz_plan_inverse_singlemom[c]);
    int size = nLinks[c]*grids_->Nz;
    size_t workSize;
    int nClm = nChains[c]*grids_->Nl*grids_->Nm;
    cufftMakePlanMany(dz_plan_forward[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize);
    cufftMakePlanMany(dz_plan_inverse[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nClm, &workSize);

    cufftMakePlanMany(dz_plan_forward_singlemom[c],     1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 
                      nChains[c], &workSize);
    cufftMakePlanMany(abs_dz_plan_forward_singlemom[c], 1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 
                      nChains[c], &workSize);
    cufftMakePlanMany(dz_plan_inverse_singlemom[c],     1, &size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 
                      nChains[c], &workSize);

    // initialize kzLinked
    init_kzLinked<<<1,1>>>(kzLinked[c], nLinks[c]);
    
    dB[c] = dim3(32,4,4);
    dG[c] = dim3(grids_->Nz/dB[c].x+1, nLinks[c]*nChains[c]/dB[c].y+1, grids_->Nmoms/dB[c].z+1);
  }

  set_callbacks();
  //  this->linkPrint();
}

GradParallelLinked::~GradParallelLinked()
{
  free(nLinks);
  free(nChains);
  free(dB);
  free(dG);
  for(int c=0; c<nClasses; c++) {
    cufftDestroy(    dz_plan_forward[c]);
    cufftDestroy(    dz_plan_inverse[c]);
    cufftDestroy(    dz_plan_forward_singlemom[c]);
    cufftDestroy(abs_dz_plan_forward_singlemom[c]);
    cufftDestroy(    dz_plan_inverse_singlemom[c]);
    free(ikxLinked_h[c]);
    free(ikyLinked_h[c]);
    cudaFree(ikxLinked[c]);
    cudaFree(ikyLinked[c]);
    cudaFree(kzLinked[c]);
    cudaFree(G_linked[c]);
  }
  cudaFreeHost(    dz_plan_forward);
  cudaFreeHost(    dz_plan_inverse);
  cudaFreeHost(    dz_plan_forward_singlemom);
  cudaFreeHost(abs_dz_plan_forward_singlemom);
  cudaFreeHost(    dz_plan_inverse_singlemom);
  free(ikxLinked_h);
  free(ikyLinked_h);
  cudaFreeHost(ikxLinked);
  cudaFreeHost(ikyLinked);
  cudaFreeHost(G_linked);
  cudaFreeHost(kzLinked);
}


void GradParallelLinked::dz(MomentsG* G) 
{
  G->reality(); // why is this required?
  for(int c=0; c<nClasses; c++) {
    // each "class" has a different number of links in the chains, and a different number of chains.
    linkedCopy <<<dG[c],dB[c]>>>
      (G->G(), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
    cufftExecC2C (dz_plan_forward[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C (dz_plan_inverse[c], G_linked[c], G_linked[c], CUFFT_INVERSE);
    linkedCopyBack <<<dG[c],dB[c]>>>
      (G_linked[c], G->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
  }
  G->reality(); // why is this here? 
}

// for a single moment m
void GradParallelLinked::dz(cuComplex* m, cuComplex* res)
{
  int nMoms=1;
  reality_singlemom_kernel<<<dim3(32,32,1),dim3(grids_->Nx/32+1, grids_->Nz/32+1,1)>>>(m);
  for(int c=0; c<nClasses; c++) {
    // these only use the G(0,0) part of G_linked
    linkedCopy<<<dG[c],dB[c]>>>(m, G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
    cufftExecC2C(dz_plan_forward_singlemom[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C(dz_plan_inverse_singlemom[c], G_linked[c], G_linked[c], CUFFT_INVERSE);
    linkedCopyBack<<<dG[c],dB[c]>>>(G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
  }
  reality_singlemom_kernel<<<dim3(32,32,1),dim3(grids_->Nx/32+1, grids_->Nz/32+1,1)>>>(res);
}

// for a single moment m
void GradParallelLinked::abs_dz(cuComplex* m, cuComplex* res)
{
  int nMoms=1;
  reality_singlemom_kernel<<<dim3(32,32,1),dim3(grids_->Nx/32+1, grids_->Nz/32+1,1)>>>(m);
  for(int c=0; c<nClasses; c++) {
    // these only use the G(0,0) part of G_linked
    linkedCopy<<<dG[c],dB[c]>>>(m, G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
    cufftExecC2C(abs_dz_plan_forward_singlemom[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C(dz_plan_inverse_singlemom[c], G_linked[c], G_linked[c], CUFFT_INVERSE);
    linkedCopyBack<<<dG[c],dB[c]>>>(G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
  }
  reality_singlemom_kernel<<<dim3(32,32,1),dim3(grids_->Nx/32+1, grids_->Nz/32+1,1)>>>(res);
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
  } */
  
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

// for testing
void GradParallelLinked::identity(MomentsG* G)
{
  G->reality();
  for(int c=0; c<nClasses; c++) {
    // each "class" has a different number of links in the chains, and a different number of chains.
    linkedCopy<<<dG[c],dB[c]>>>(G->G(), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
    linkedCopyBack<<<dG[c],dB[c]>>>(G_linked[c], G->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
  }
  G->reality();
}

void GradParallelLinked::set_callbacks()
{
  for(int c=0; c<nClasses; c++) {
    // set up callback functions
    cudaDeviceSynchronize();
    cufftXtSetCallback(    dz_plan_forward[c],
		       (void**)   &i_kzLinked_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]);
    cufftXtSetCallback(    dz_plan_forward_singlemom[c],
		       (void**)   &i_kzLinked_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]);
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
    cufftXtClearCallback(    dz_plan_forward[c],           CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(    dz_plan_forward_singlemom[c], CUFFT_CB_ST_COMPLEX);
    cufftXtClearCallback(abs_dz_plan_forward_singlemom[c], CUFFT_CB_ST_COMPLEX);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
}
          

