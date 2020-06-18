#include "cufftXt.h"
#include "cufft.h"
#include "device_funcs.h"
#include "grad_perp.h"
#include "cuda_constants.h"

GradPerp::GradPerp(Grids* grids, int batch_size)
 : grids_(grids), batch_size_(batch_size)
{
  cufftCreate(&gradperp_plan_R2C);
  cufftCreate(&gradperp_plan_dxC2R);
  cufftCreate(&gradperp_plan_dyC2R);

  //  cufftCreate(&gradperp_plan_C2R);
  
  int NLPSfftdims[2] = {grids->Nx, grids->Ny};
  size_t workSize;
  cufftMakePlanMany(gradperp_plan_R2C,   2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_dxC2R, 2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_dyC2R, 2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);

  cudaDeviceSynchronize();
  cufftXtSetCallback(gradperp_plan_dxC2R, (void**) &i_kx_callbackPtr, 
                     CUFFT_CB_LD_COMPLEX, 
                     (void**)&grids_->kx);

  cufftXtSetCallback(gradperp_plan_dyC2R, (void**) &i_ky_callbackPtr, 
                     CUFFT_CB_LD_COMPLEX, 
                     (void**)&grids_->ky);

  cufftXtSetCallback(gradperp_plan_R2C,   (void**) &mask_and_scale_callbackPtr, 
                     CUFFT_CB_ST_COMPLEX, 
                     NULL);
  cudaDeviceSynchronize();
}

GradPerp::~GradPerp()
{
  cufftDestroy(gradperp_plan_R2C);
  cufftDestroy(gradperp_plan_dxC2R);
  cufftDestroy(gradperp_plan_dyC2R);
}

void GradPerp::dxC2R(cuComplex* G, float* dxG)
{ cufftExecC2R(gradperp_plan_dxC2R, G, dxG); }

void GradPerp::dyC2R(cuComplex* G, float* dyG)
{ cufftExecC2R(gradperp_plan_dyC2R, G, dyG); }

void GradPerp::R2C(float* G, cuComplex* res)
{ cufftExecR2C(gradperp_plan_R2C, G, res);   }

