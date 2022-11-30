#include "grad_perp.h"
#include "get_error.h"

GradPerp::GradPerp(Grids* grids, int batch_size, int mem_size)
  : grids_(grids), batch_size_(batch_size), mem_size_(mem_size), tmp(nullptr)
{
  // 2D
  cufftCreate(&gradperp_plan_R2C);
  cufftCreate(&gradperp_plan_C2R);
  cufftCreate(&gradperp_plan_dxC2R);
  cufftCreate(&gradperp_plan_dyC2R);

  // 1D
  cufftCreate(&gradperp_plan_R2Cx);
  cufftCreate(&gradperp_plan_C2Rx);
  cufftCreate(&gradperp_plan_dxC2Rx);
  cufftCreate(&gradperp_plan_dyC2Rx);

  // 1D
  cufftCreate(&gradperp_plan_R2Cy);
  cufftCreate(&gradperp_plan_C2Ry);
  cufftCreate(&gradperp_plan_dxC2Ry);
  cufftCreate(&gradperp_plan_dyC2Ry);

  // Use MakePlanMany to enable callbacks
  // Order of Nx, Ny is correct here

  cudaMalloc (&tmp, sizeof(cuComplex)*mem_size_); // modify?

  int maxthreads = 1024; 
  int nthreads = min(maxthreads, mem_size_);
  int nblocks = 1 + (mem_size_-1)/nthreads;

  dB = dim3(nthreads, 1, 1);
  dG = dim3(nblocks,  1, 1);
  
  int NLPSfftdims[2] = {grids->Nx, grids->Ny};

  // 1D
  int NLPSfftdimx = grids->Nx;
  int NLPSfftdimy = grids->Nyc; // Nyc or Ny?
 
  // 1D
  // kx FFT
  int istridex = grids->Nx;           // distance between two successive input elements in innermost dimension (the FFT dimension?)
                                      // = distance between (kx=1,ky) and (kx=2,ky) = Nx
  int idistx = grids->Nx;             // distance between the first element of two consecutive signals in a batch of the input data
                                      // = distance between (kx=1,ky=1) and (kx=2,ky=1) = Nx
  int ostridex = grids->Nx;           // same, for output arrays.
  int odistx = grids->Nx;

  // 1D
  // ky FFT
  int istridey = 1;                   // distance between two successive input elements in innermost dimension (the FFT dimension?)
                                      // = distance between (kx,ky=1) and (kx,ky=2) = 1
  int idisty = grids->Nx;             // distance between the first element of two consecutive signals in a batch of the input data
                                      // = distance between (kx=1,ky=1) and (kx=2,ky=1) = Nx
  int ostridey = 1;                   // same, for output arrays.
  int odisty = grids->Nx;

  // Arguments for cufftMakePlanMany
  //cufftMakePlanMany(gradperp_plan_C2R, rank (FFT dimension), *n (size of FFT), #inembed (NULL), istride, idist, *onembed (NULL), ostride, odist, cufftType, batch_size_, size_t *workSize);

  size_t workSize;
  
  // 2D
  cufftMakePlanMany(gradperp_plan_C2R,    2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_R2C,    2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_dxC2R,  2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_dyC2R,  2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);

  // 1D
  cufftMakePlanMany(gradperp_plan_C2Rx,    1, NLPSfftdimx, NULL, istridex, idistx, NULL, ostridex, odistx, CUFFT_C2R, batch_size_, &workSize); // is input data size still Nx,Ny, not Nx?
  cufftMakePlanMany(gradperp_plan_C2Ry,    1, NLPSfftdimy, NULL, istridey, idisty, NULL, ostridey, odisty, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_R2Cx,    1, NLPSfftdimx, NULL, istridex, idistx, NULL, ostridex, odistx, CUFFT_R2C, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_R2Cy,    1, NLPSfftdimy, NULL, istridey, idisty, NULL, ostridey, odisty, CUFFT_R2C, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_dxC2Rx,  1, NLPSfftdimx, NULL, istridex, idistx, NULL, ostridex, odistx, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_dxC2Ry,  1, NLPSfftdimy, NULL, istridey, idisty, NULL, ostridey, odisty, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_dyC2Rx,  1, NLPSfftdimx, NULL, istridex, idistx, NULL, ostridex, odistx, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_dyC2Ry,  1, NLPSfftdimy, NULL, istridey, idisty, NULL, ostridey, odisty, CUFFT_C2R, batch_size_, &workSize);

  cudaDeviceSynchronize();

  // 2D
  cufftXtSetCallback(gradperp_plan_dxC2R, (void**) &i_kx_callbackPtr, 
                     CUFFT_CB_LD_COMPLEX, 
                    (void**)&grids_->kx);

  cufftXtSetCallback(gradperp_plan_dyC2R, (void**) &i_ky_callbackPtr, 
                     CUFFT_CB_LD_COMPLEX, 
                     (void**)&grids_->ky);

  cufftXtSetCallback(gradperp_plan_R2C,   (void**) &mask_and_scale_callbackPtr, 
                     CUFFT_CB_ST_COMPLEX, 
                     NULL);


  // 1D
  cufftXtSetCallback(gradperp_plan_dxC2Rx, (void**) &i_kx_callbackPtr, // (cufftHandle plan, void **callbackRoutine, cufftXtCallbackType type, void **callerInfo)
                     CUFFT_CB_LD_COMPLEX,
                     (void**)&grids_->kx);

  cufftXtSetCallback(gradperp_plan_dxC2Ry, (void**) &i_kx_callbackPtr,
                     CUFFT_CB_LD_COMPLEX,
                     (void**)&grids_->kx);

  cufftXtSetCallback(gradperp_plan_dyC2Rx, (void**) &i_ky_callbackPtr,
                     CUFFT_CB_LD_COMPLEX,
                     (void**)&grids_->ky);

  cufftXtSetCallback(gradperp_plan_dyC2Ry, (void**) &i_ky_callbackPtr,
                     CUFFT_CB_LD_COMPLEX,
                     (void**)&grids_->ky);

  cufftXtSetCallback(gradperp_plan_R2Cx,   (void**) &mask_and_scale_callbackPtr,
                     CUFFT_CB_ST_COMPLEX,
                     NULL);

  cufftXtSetCallback(gradperp_plan_R2Cy,   (void**) &mask_and_scale_callbackPtr,
                     CUFFT_CB_ST_COMPLEX,
                     NULL);

  cudaDeviceSynchronize();
}

GradPerp::~GradPerp()
{
  if (tmp)      cudaFree (tmp);
  // 2D
  cufftDestroy ( gradperp_plan_R2C    );
  cufftDestroy ( gradperp_plan_C2R    );
  cufftDestroy ( gradperp_plan_dxC2R  );
  cufftDestroy ( gradperp_plan_dyC2R  );
  // 1D
  cufftDestroy ( gradperp_plan_R2Cx    );
  cufftDestroy ( gradperp_plan_C2Rx    );
  cufftDestroy ( gradperp_plan_dxC2Rx  );
  cufftDestroy ( gradperp_plan_dyC2Rx  );
  cufftDestroy ( gradperp_plan_R2Cy    );
  cufftDestroy ( gradperp_plan_C2Ry    );
  cufftDestroy ( gradperp_plan_dxC2Ry  );
  cufftDestroy ( gradperp_plan_dyC2Ry  );
}

// Out-of-place 2D transforms in cufft now overwrite the input data. 

// 2D
void GradPerp::dxC2R(cuComplex* G, float* dxG)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);;
  cufftExecC2R(gradperp_plan_dxC2R, tmp, dxG);
}

// 1D
void GradPerp::dxC2Rx(cuComplex* G, float* dxG)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);;
  cufftExecC2R(gradperp_plan_dxC2Rx, tmp, dxG);
}

// 1D
void GradPerp::dxC2Ry(cuComplex* G, float* dxG)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);;
  cufftExecC2R(gradperp_plan_dxC2Ry, tmp, dxG);
}

// 2D
void GradPerp::qvar (cuComplex* G, int N)
{
  cuComplex* G_h;
  int Nk = grids_->Nyc*grids_->Nx;
  G_h = (cuComplex*) malloc (sizeof(cuComplex)*N);
  for (int i=0; i<N; i++) {G_h[i].x = 0.; G_h[i].y = 0.;}
  CP_TO_CPU (G_h, G, N*sizeof(cuComplex));

  printf("\n");
  for (int i=0; i<N; i++) printf("grad_perp: var(%d,%d) = (%e, %e) \n", i%Nk, i/Nk, G_h[i].x, G_h[i].y);
  printf("\n");

  free (G_h);
}

// 2D
void GradPerp::qvar (float* G, int N)
{
  float* G_h;
  int Nx = grids_->Ny*grids_->Nx;
  G_h = (float*) malloc (sizeof(float)*N);
  for (int i=0; i<N; i++) G_h[i] = 0.;
  CP_TO_CPU (G_h, G, N*sizeof(float));

  printf("\n");
  for (int i=0; i<N; i++) printf("grad_perp: var(%d,%d) = %e \n", i%Nx, i/Nx, G_h[i]);
  printf("\n");

  free (G_h);
}

// 2D
void GradPerp::dyC2R(cuComplex* G, float* dyG)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);
  cufftExecC2R(gradperp_plan_dyC2R, tmp, dyG);
}

// 1D
void GradPerp::dyC2Rx(cuComplex* G, float* dyG)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);
  cufftExecC2R(gradperp_plan_dyC2Rx, tmp, dyG);
}

// 1D
void GradPerp::dyC2Ry(cuComplex* G, float* dyG)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);
  cufftExecC2R(gradperp_plan_dyC2Ry, tmp, dyG);
}

void GradPerp::C2R(cuComplex* G, float* Gy)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);
  cufftExecC2R(gradperp_plan_C2R, tmp, Gy);
}

// 1D
void GradPerp::C2Rx(cuComplex* G, float* Gy)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);
  cufftExecC2R(gradperp_plan_C2Rx, tmp, Gy);
}

// 1D
void GradPerp::C2Ry(cuComplex* G, float* Gy)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);
  cufftExecC2R(gradperp_plan_C2Ry, tmp, Gy);
}

// An R2C that accumulates -- will be very useful
// 2D
void GradPerp::R2C(float* G, cuComplex* res, bool accumulate)
{
  if (accumulate) {
    cufftExecR2C(gradperp_plan_R2C, G, tmp);
    add_section <<< dG, dB >>> (res, tmp, mem_size_);
  } else {
    cufftExecR2C(gradperp_plan_R2C, G, res);
  }
}

// 1D
void GradPerp::R2Cx(float* G, cuComplex* res, bool accumulate)
{
  if (accumulate) {
    cufftExecR2C(gradperp_plan_R2Cx, G, tmp);
    add_section <<< dG, dB >>> (res, tmp, mem_size_);
  } else {
    cufftExecR2C(gradperp_plan_R2Cx, G, res);
  }
}

// 1D
void GradPerp::R2Cy(float* G, cuComplex* res, bool accumulate)
{
  if (accumulate) {
    cufftExecR2C(gradperp_plan_R2Cy, G, tmp);
    add_section <<< dG, dB >>> (res, tmp, mem_size_);
  } else {
    cufftExecR2C(gradperp_plan_R2Cy, G, res);
  }
}

