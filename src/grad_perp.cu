#include "grad_perp.h"
#include "get_error.h"

GradPerp::GradPerp(Grids* grids, int batch_size, int mem_size, float* phasefac, float* minusphasefac) // phasefac is a function of ky and time.
  : grids_(grids), batch_size_(batch_size), mem_size_(mem_size), tmp(nullptr)

{
  // 2D
  cufftCreate(&gradperp_plan_R2C);
  cufftCreate(&gradperp_plan_C2R);
  cufftCreate(&gradperp_plan_dxC2R);
  cufftCreate(&gradperp_plan_dyC2R);

  // 1D y,ky transforms
  cufftCreate(&gradperp_plan_R2Cy); // phi(x,y) ---> phi(x,ky). F_ky.
  cufftCreate(&gradperp_plan_C2Ry); // phi(x,ky) ---> phi(x,y) with phase factor in callback. F_ky^-1.
  cufftCreate(&gradperp_plan_C2Ry_minus); // phi(x,ky) ---> phi(x,y) with minus phase factor in callback. F_ky^-1.

  // Use MakePlanMany to enable callbacks
  // Order of Nx, Ny is correct here

  cudaMalloc (&tmp, sizeof(cuComplex)*mem_size_);

  int maxthreads = 1024; 
  int nthreads = min(maxthreads, mem_size_);
  int nblocks = 1 + (mem_size_-1)/nthreads;

  dB = dim3(nthreads, 1, 1);
  dG = dim3(nblocks,  1, 1);
  
  int NLPSfftdims[2] = {grids->Nx, grids->Ny};
  int NLPSfftdimky = grids->Ny;

  size_t workSize;
  
  // 2D, unstrided batch.
  cufftMakePlanMany(gradperp_plan_C2R,    2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_R2C,    2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_dxC2R,  2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_dyC2R,  2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);

  // 1D
  cufftMakePlanMany(gradperp_plan_C2Ry,    1, &NLPSfftdimky, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_*grids->Nx, &workSize);
  // this plan is for the minus phase factor.
  cufftMakePlanMany(gradperp_plan_C2Ry_minus,    1, &NLPSfftdimky, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_*grids->Nx, &workSize);
  cufftMakePlanMany(gradperp_plan_R2Cy,    1, &NLPSfftdimky, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, batch_size_*grids->Nx, &workSize);
  // Marker for adding cufftXtMakePlanMany for 1D FFT in future when BF16 is supported.

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
  // Use for a(x,y) --> a(x,ky). // Different masking pointer.
  cufftXtSetCallback(gradperp_plan_R2Cy,   (void**) &scale_ky_callbackPtr,
                     CUFFT_CB_ST_COMPLEX,
                     NULL);

  // Use for a(x,ky) --> a(x,y), where multiplication by phasefac*a(x,ky) in callback.
  cufftXtSetCallback(gradperp_plan_C2Ry, (void**) &phasefac_callbackPtr,
                     CUFFT_CB_LD_COMPLEX,
                     (void**)&phasefac);

  // For future use, put in minusphasefac with the same callback pointer, rather than phasefac_minus_callbackPtr as below.
  // Use for a(x,ky) --> a(x,y), where multiplication by -phasefac*a(x,ky) in callback.
  cufftXtSetCallback(gradperp_plan_C2Ry_minus, (void**) &phasefac_callbackPtr,
                     CUFFT_CB_LD_COMPLEX,
                     (void**)&minusphasefac);

  // Use for a(x,ky) --> a(x,y), where multiplication by -phasefac*a(x,ky) in callback.
  //cufftXtSetCallback(gradperp_plan_C2Ry_minus, (void**) &phasefac_minus_callbackPtr,
  //                  CUFFT_CB_LD_COMPLEX,
  //                   (void**)&phasefac);

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
  cufftDestroy ( gradperp_plan_R2Cy    );
  cufftDestroy ( gradperp_plan_C2Ry    );
}

// This method multiplies by phase factor in FFT.
// Steps:
// 1) gradperp_plan_R2Cy: G(x,y) [ky FFT] ---> G(x,ky) 
// 2) gradperp_plan_C2Ry: [CALLBACK MULTIPLY] ---> G_phase(x,ky) = G(x,ky)*phase_factor(ky) [y FFT] ---> G_phase(x,y)
// Note that the phase_mult is called with a positive sign for each term in the nonlinearity, then after multiplication is done in real space, we call phase_mult again with a negative sign.
// This is useful because it avoids FFTs in x, which are unfavorable b/c bf16 not supported with striding arrays. We want bf16 once implemented.
// 1D
void GradPerp::phase_mult(float* G, bool positive_phase)
{
  // Step 1:  G(x,y) [ky FFT] ---> G(x,ky)
  cufftExecR2C(gradperp_plan_R2Cy, G, tmp);
  // Step 2:  G(x,ky) = G(x,ky)*phase_factor(ky) [y FFT] ---> G(x,y), with positive or negative phase.
  if (positive_phase) {
    cufftExecC2R(gradperp_plan_C2Ry, tmp, G);
  } else { 
    cufftExecC2R(gradperp_plan_C2Ry_minus, tmp, G);
  }
}

// Out-of-place 2D transforms in cufft now overwrite the input data. 

void GradPerp::dxC2R(cuComplex* G, float* dxG)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);;
  cufftExecC2R(gradperp_plan_dxC2R, tmp, dxG);
}

void GradPerp::qvar (cuComplex* G, int N)
{
  cuComplex* G_h;
  int Nk = grids_->Nyc*grids_->Nx;
  G_h = (cuComplex*) malloc (sizeof(cuComplex)*N);
  for (int i=0; i<N; i++) {G_h[i].x = 0.; G_h[i].y = 0.;}
  CP_TO_CPU (G_h, G, N*sizeof(cuComplex));

  printf("\n");
  for (int i=0; i<N; i++) printf("grad_perp: var(%d,%d,%d) = (%e, %e)  \n", i%grids_->Nyc, i/grids_->Nyc%grids_->Nx, i/Nk, G_h[i].x, G_h[i].y);
  printf("\n");

  free (G_h);
}

void GradPerp::qvar (float* G, int N)
{
  float* G_h;
  int Nx = grids_->Ny*grids_->Nx;
  G_h = (float*) malloc (sizeof(float)*N);
  for (int i=0; i<N; i++) G_h[i] = 0.;
  CP_TO_CPU (G_h, G, N*sizeof(float));

  printf("\n");
  for (int i=0; i<N; i++) printf("grad_perp: var(%d,%d,%d) = %e \n", i%grids_->Ny, i/grids_->Ny%grids_->Nx, i/Nx, G_h[i]);
  printf("\n");

  free (G_h);
}

void GradPerp::dyC2R(cuComplex* G, float* dyG)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);
  cufftExecC2R(gradperp_plan_dyC2R, tmp, dyG);
}

void GradPerp::C2R(cuComplex* G, float* Gy)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);
  cufftExecC2R(gradperp_plan_C2R, tmp, Gy);
}

// An R2C that accumulates -- will be very useful
void GradPerp::R2C(float* G, cuComplex* res, bool accumulate)
{
  if (accumulate) {
    cufftExecR2C(gradperp_plan_R2C, G, tmp);
    add_section <<< dG, dB >>> (res, tmp, mem_size_);
  } else {
    cufftExecR2C(gradperp_plan_R2C, G, res);
  }
}
