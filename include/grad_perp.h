#pragma once
#include "grids.h"
#include "cufftXt.h"
#include "cufft.h"
#include "device_funcs.h"

class GradPerp {
 public:
  GradPerp(Grids* grids, int batch, int mem);
  ~GradPerp();

  void dxC2R (cuComplex* G, float* dxG);
  void dyC2R (cuComplex* G, float* g);
  void C2R   (cuComplex* G, float* Gy);
  void R2C   (float* G, cuComplex* res, bool accumulate=true);
  void qvar  (cuComplex* G, int N);
  void qvar  (float* G, int N);
  
 private:
  const int batch_size_;
  const int mem_size_;
  dim3 dG, dB;
  Grids     * grids_ ;
  cuComplex * tmp    ;
  cufftHandle gradperp_plan_R2C;
  cufftHandle gradperp_plan_C2R;
  cufftHandle gradperp_plan_dxC2R;
  cufftHandle gradperp_plan_dyC2R;
};
