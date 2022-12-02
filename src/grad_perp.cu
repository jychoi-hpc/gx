#include "grad_perp.h"
#include "get_error.h"

GradPerp::GradPerp(Grids* grids, int batch_size, int mem_size)
  : grids_(grids), batch_size_(batch_size), mem_size_(mem_size), tmp1(nullptr), tmp2(nullptr)
{
  // 2D
  cufftCreate(&gradperp_plan_R2C);
  cufftCreate(&gradperp_plan_C2R);
  cufftCreate(&gradperp_plan_dxC2R);
  cufftCreate(&gradperp_plan_dyC2R);

  // 1D y+ky transforms
  cufftCreate(&gradperp_plan_R2Cy); // used for phi(x,y) ---> phi(x,ky). F_ky.
  cufftCreate(&gradperp_plan_C2Ry); // used for phi(x,ky) ---> phi(x,y) with phase factor in callback. F_ky^-1.

  // Use MakePlanMany to enable callbacks
  // Order of Nx, Ny is correct here

  cudaMalloc (&tmp1, sizeof(cuComplex)*mem_size_); // modify?
  cudaMalloc (&tmp2, sizeof(cuComplex)*mem_size_); // modify?

  int maxthreads = 1024; 
  int nthreads = min(maxthreads, mem_size_);
  int nblocks = 1 + (mem_size_-1)/nthreads;

  dB = dim3(nthreads, 1, 1);
  dG = dim3(nblocks,  1, 1);
  
  // 2D
  int NLPSfftdims[2] = {grids->Nx, grids->Ny};
  // 1D
  int NLPSfftdimy = grids->Nyc; // size of y
  int NLPSfftdimky = grids->Ny; // size of ky
  // ky FFT
  int istridey = 1;                   // distance between two successive input elements in innermost dimension (the FFT dimension?)
                                      // = distance between (kx,ky=1) and (kx,ky=2) = 1
  int idistky = grids->Nyc;             // distance between the first element of two consecutive signals in a batch of the input data
                                      // = distance between (kx=1,ky=1) and (kx=2,ky=1) = Nyc

  int idisty = grids->Ny;             // distance between the first element of two consecutive signals in a batch of the input data
                                      // = distance between (kx=1,y=1) and (kx=2,y=1) = Ny
  int ostridey = 1;                   // same, for output arrays.
  int odistky = grids->Nyc;
  int odisty = grids->Ny;

  // Arguments for cufftMakePlanMany
  //cufftMakePlanMany(plan_name, rank (FFT dimension), *n (size of FFT), #inembed (NULL), istride, idist, *onembed (NULL), ostride, odist, cufftType, batch_size_, size_t *workSize);

  size_t workSize;
  
  // 2D
  cufftMakePlanMany(gradperp_plan_C2R,    2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_R2C,    2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_dxC2R,  2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_dyC2R,  2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);

  // 1D
  cufftMakePlanMany(gradperp_plan_C2Ry,    1, NLPSfftdimy, NULL, istridey, idisty, NULL, ostridey, odisty, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_R2Cy,    1, NLPSfftdimy, NULL, istridey, idisty, NULL, ostridey, odisty, CUFFT_R2C, batch_size_, &workSize);

  // Marker for adding cufftXtMakePlanMany in future when BF16 is supported.

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
  // Use for a(x,ky) --> a(x,y), where multiplication by phasefac*a(x,ky) in callback.
  cufftXtSetCallback(gradperp_plan_C2Ry, (void**) &phasefac_callbackPtr, // to set up phasefac_callbackPtr
                     CUFFT_CB_LD_COMPLEX,
                     (void**)&phasefac); // how to set up this phase factor?

  // Use for a(x,y) --> a(x,ky).
  cufftXtSetCallback(gradperp_plan_R2Cy,   (void**) &mask_and_scale_callbackPtr,
                     CUFFT_CB_ST_COMPLEX,
                     NULL);

  // We don't need d/dy for the ky, since ky multiplication still only occurs with the 2D transforms.

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

// phase_mult allows for multiplying by phase factor.
// Steps:
// 1) gradperp_plan_R2Cy: G(x,y) [ky FFT] ---> G(x,ky) 
// 2) gradperp_plan_C2Ry: [CALLBACK MULTIPLY] ---> G_phase(x,ky) = G(x,ky)*phase_factor(ky) [y FFT] ---> G_phase(x,y)
// This is useful because it avoids FFTs in x, which are unfavorable b/c bf16 not supported with striding arrays. We want bf16 once implemented.
// 1D
void GradPerp::phase_mult(float* G)
{
  cuComplex G_ky; // intermediate quantity G(x,ky)

  // Step 1:  G(x,y) [ky FFT] ---> G(x,ky)
  CP_ON_GPU (tmp2, G, sizeof(cuComplex)*mem_size_);;
  // G is input data, G_ky is output.
  cufftExecR2C(gradperp_plan_R2Cy, tmp2, G_ky);

  // Step 2:  G(x,ky) = G(x,ky)*phase_factor(ky) [y FFT] ---> G(x,y), now with phase.
  CP_ON_GPU (tmp2, G_ky, sizeof(cuComplex)*mem_size_);;
  // G_ky is input data, G is output.
  cufftExecC2R(gradperp_plan_C2Ry, tmp2, G);
}

// Out-of-place 2D transforms in cufft now overwrite the input data. 

// 2D
void GradPerp::dxC2R(cuComplex* G, float* dxG)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);;
  cufftExecC2R(gradperp_plan_dxC2R, tmp, dxG);
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

// 2D
void GradPerp::C2R(cuComplex* G, float* Gy)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);
  cufftExecC2R(gradperp_plan_C2R, tmp, Gy);
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
void GradPerp::R2Cy(float* G, cuComplex* res, bool accumulate)
{
  if (accumulate) {
    cufftExecR2C(gradperp_plan_R2Cy, G, tmp);
    add_section <<< dG, dB >>> (res, tmp, mem_size_);
  } else {
    cufftExecR2C(gradperp_plan_R2Cy, G, res);
  }
}

