#pragma once
#include "grids.h"
#include "moments.h"
#include "cufftXt.h"
#include "cufft.h"

class MomentsG; // Forward Declaration

class GradParallel {
 public:
  GradParallel() {};
  virtual ~GradParallel() {};

  virtual void dz(MomentsG* G)=0;
  virtual void dz(cuComplex* m, cuComplex* res)=0;
  virtual void abs_dz(cuComplex* m, cuComplex* res)=0;
  virtual void fft_only(cuComplex* m, cuComplex* res, int dir) {};
};

class GradParallelPeriodic : public GradParallel {
 public:
  GradParallelPeriodic(Grids* grids);
  ~GradParallelPeriodic();

  void dz(MomentsG* G);
  void dz(cuComplex* m, cuComplex* res);
  void abs_dz(cuComplex* m, cuComplex* res);
  void fft_only(cuComplex* m, cuComplex* res, int dir);
  
 private:
  Grids* grids_;
  
  cufftHandle dz_plan_forward;
  cufftHandle abs_dz_plan_forward;
  cufftHandle dz_plan_inverse;

};

class GradParallelLinked : public GradParallel {
 public:
  GradParallelLinked(Grids* grids, int jtwist);
  ~GradParallelLinked();

  void dz(MomentsG* G);
  void dz(cuComplex* m, cuComplex* res);
  void abs_dz(cuComplex* m, cuComplex* res);
  void linkPrint();
  void identity(MomentsG* G); // for testing

 private:
  Grids *grids_;
  
  int get_nClasses(int *idxRight, int *idxLeft, int *linksR, int *linksL, int *n_k, int naky, int ntheta0, int jshift0);
  void get_nLinks_nChains(int *nLinks, int *nChains, int *n_k, int nClasses, int naky, int ntheta0);
  void kFill(int nClasses, int *nChains, int *nLinks, int **ky, int **kx, int *linksL, int *linksR, int *idxRight, int naky, int ntheta0);
  void set_callbacks();
  void clear_callbacks();
  
  int nClasses;
  int *nLinks, *nChains;
  int **ikxLinked_h, **ikyLinked_h;
  int **ikxLinked, **ikyLinked;
  float **kzLinked;
  cuComplex **G_linked;

  cufftHandle* dz_plan_forward;
  cufftHandle* dz_plan_inverse;
  cufftHandle* dz_plan_forward_singlemom;
  cufftHandle* abs_dz_plan_forward_singlemom;
  cufftHandle* dz_plan_inverse_singlemom;
  dim3 *dG, *dB;
};

class GradParallelLocal : public GradParallel {
 public:
  GradParallelLocal(Grids* grids);
  ~GradParallelLocal() {};

  void dz(MomentsG* G);
  void dz(cuComplex* m, cuComplex* res);
  void abs_dz(cuComplex* m, cuComplex* res);
 private:
  Grids* grids_;

  dim3 dG, dB;
};

class GradParallel1D {
 public:
  GradParallel1D(Grids* grids);
  ~GradParallel1D();
  void dz1D(float* b); 

 private:
  Grids* grids_;
  
  cufftHandle dz_plan_forward;
  cufftHandle dz_plan_inverse;
  
  cuComplex *b_complex;
};
