#pragma once
#include "grids.h"
#include "moments.h"

class GradParallel {
 public:
  
  GradParallel() {};
  GradParallel(Grids* grids, bool abs=false);
  ~GradParallel();

  virtual void eval(Moments* m);
  virtual void eval(cuComplex* m, cuComplex* res);
  virtual void fft_only(cuComplex* m, cuComplex* res, int dir);
 
 private:
  Grids* grids_;
  
  cufftHandle gradpar_plan_forward;
  cufftHandle gradpar_plan_inverse;
};

class GradParallelLocal : public GradParallel {
 public:
  GradParallelLocal(Grids* grids, bool abs=false);
  ~GradParallelLocal() {};

  void eval(Moments* m);
  void eval(cuComplex* m, cuComplex* res);
 private:
  Grids* grids_;
  const bool abs_;

  dim3 dimGrid, dimBlock;
};
