#pragma once
#include "gpu_defs.h"
#include "grids.h"
#include "parameters.h"
#include "device_funcs.h"

class GradPerp {
 public:
  GradPerp(Parameters* pars, Grids* grids, int batch, int mem, cudaStream_t stream=0);
  ~GradPerp();

  void phase_mult (float* G, bool nonTwist, bool ExBshear, bool positive_phase=true);
  void dxC2R (cuComplex* G, float* dxG);
  void dyC2R (cuComplex* G, float* g);
  void C2R   (cuComplex* G, float* Gy);
  void R2C   (float* G, cuComplex* res, bool accumulate=true);
  void qvar  (cuComplex* G, int N);
  void qvar  (float* G, int N);
  
 private:
  const int batch_size_;
  const int mem_size_;
  dim3 dGk, dBk, dGx_single_ntft, dBx_single_ntft, dGx_ntft, dBx_ntft, dGphi_ntft, dBphi_ntft, dGk_ntft, dBk_ntft;
  dim3 dG, dB;
  Parameters * pars_ ;
  Grids     * grids_ ;
  cuComplex * tmp    ;
  cudaStream_t stream_;
  cufftHandle gradperp_plan_R2C;
  cufftHandle gradperp_plan_C2R;
  cufftHandle gradperp_plan_dxC2R;
  cufftHandle gradperp_plan_dyC2R;
  cufftHandle gradperp_plan_R2Cy;
  cufftHandle gradperp_plan_C2Ry;
  cufftHandle gradperp_plan_C2Ryminus;
};
