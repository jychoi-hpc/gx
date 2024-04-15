#pragma once
#include "grids.h"
#include "moments.h"
#include "parameters.h"
#include "fields.h"
#include "cufftXt.h"
#include "cufft.h"
#include "device_funcs.h"

class GradParallel {
 public:
  GradParallel() {};
  virtual ~GradParallel() {};

  virtual void dz(MomentsG* G, MomentsG* res, bool accumulate=false)=0;
  virtual void dz(cuComplex* m, cuComplex* res, bool accumulate=false)=0;
  virtual void hyperz(MomentsG* G, MomentsG* res, float nu, bool accumulate=false) {printf("hyperz not yet implemented for this class\n"); exit(1);};
  virtual void dz2(MomentsG* G) {printf("dz2 not yet implemented for this class\n"); exit(1);};
  virtual void dz2(cuComplex* m, cuComplex* res) {printf("dz2 not yet implemented for this class\n"); exit(1);};
  virtual void zft(MomentsG* G)=0;
  virtual void zft(cuComplex* m, cuComplex* res)=0;
  virtual void dealias(MomentsG* G) {};
  virtual void dealias(cuComplex* f) {};
  virtual void applyBCs(MomentsG* G, MomentsG* GRhs, Fields* f, float* kperp2, double dt) {};
  
  virtual void zft_inverse(MomentsG* G)=0;
  //  virtual void zft_inverse(cuComplex* m, cuComplex* res)=0;
  virtual void abs_dz(MomentsG* G, MomentsG* res, bool accumulate=false)=0;
  virtual void abs_dz(cuComplex* m, cuComplex* res, bool accumulate=false)=0;
  virtual void fft_only(cuComplex* m, cuComplex* res, int dir) {};
};

class GradParallelPeriodic : public GradParallel {
 public:
  GradParallelPeriodic(Grids* grids);
  ~GradParallelPeriodic();

  void dealias(MomentsG* G);
  void dealias(cuComplex* f);
  void  dz(MomentsG* G, MomentsG* res, bool accumulate=false);   
  void  dz(cuComplex* m, cuComplex* res, bool accumulate=false);
  void dz2(MomentsG* G);  void dz2(cuComplex* m, cuComplex* res);
  void zft(MomentsG* G);  void zft(cuComplex* m, cuComplex* res);

  void zft_inverse(MomentsG* G);
  //  void zft_inverse(cuComplex* m, cuComplex* res);
  
  void abs_dz(MomentsG* G, MomentsG* res, bool accumulate=false) {};
  void abs_dz(cuComplex* m, cuComplex* res, bool accumulate=false);
  void fft_only(cuComplex* m, cuComplex* res, int dir);
  dim3 dGd, dBd, dGf, dBf;
  
 private:
  Grids * grids_ ;
  
  cufftHandle zft_plan_forward;  cufftHandle dz_plan_forward;  cufftHandle dz2_plan_forward;
  cufftHandle zft_plan_inverse;  cufftHandle dz_plan_inverse;
  cufftHandle abs_dz_plan_forward;
};

class GradParallelLinked : public GradParallel {
 public:
  GradParallelLinked(Parameters* pars, Grids* grids);
  ~GradParallelLinked();

  void dealias(MomentsG* G);
  void dealias(cuComplex* f);
  void dz(MomentsG* G, MomentsG* res, bool accumulate=false);     void dz(cuComplex* m, cuComplex* res, bool accumulate=false);
  void hyperz(MomentsG* G, MomentsG* res, float nu, bool accumulate=false);
  void dz2(MomentsG* G);    void dz2(cuComplex* m, cuComplex* res);
  void zft(MomentsG* G);   void zft(cuComplex* m, cuComplex* res);
  void applyBCs(MomentsG* G, MomentsG* GRhs, Fields* f, float* kperp2, double dt);

  void zft_inverse(MomentsG* G);
  //  void zft_inverse(cuComplex* m, cuComplex* res);
  
  void abs_dz(MomentsG* G, MomentsG* res, bool accumulate=false);
  void abs_dz(cuComplex* m, cuComplex* res, bool accumulate=false);
  void linkPrint();
  void identity(MomentsG* G); // for testing

 private:
  Parameters * pars_;
  Grids * grids_ ;
  
  int get_nClasses(int *idxRight, int *idxLeft, int *linksR, int *linksL, int *n_k, int naky, int ntheta0, int jshift0);
  void get_nLinks_nChains(int *nLinks, int *nChains, int *n_k, int nClasses, int naky, int ntheta0);
  void kFill(int nClasses, int *nChains, int *nLinks, int **ky, int **kx, int *linksL, int *linksR, int *idxRight, int naky, int ntheta0);
  void set_callbacks();
  void clear_callbacks();
  
  int nClasses;
  int * nLinks  ;
  int * nChains ;
  int **ikxLinked_h, **ikyLinked_h;
  int **ikxLinked, **ikyLinked;
  float **kzLinked;
  cuComplex **G_linked;

  cufftHandle * zft_plan_forward;  cufftHandle * dz_plan_forward;  cufftHandle * dz2_plan_forward; 
  cufftHandle * zft_plan_inverse;  cufftHandle * dz_plan_inverse;
  cufftHandle * hyperz_plan_forward; cufftHandle * hyperz_plan_inverse;

  cufftHandle * zft_plan_forward_singlemom;
  cufftHandle * zft_plan_inverse_singlemom;

  cufftHandle * dz_plan_forward_singlemom;
  cufftHandle * dz_plan_inverse_singlemom;
  cufftHandle * abs_dz_plan_forward;
  cufftHandle * dz2_plan_forward_singlemom;
  cufftHandle * abs_dz_plan_forward_singlemom;
  dim3 * dG;
  dim3 * dB;
};

class GradParallelNTFT : public GradParallel {
 public:
  GradParallelNTFT(Parameters* pars, Grids* grids);
  ~GradParallelNTFT();

  void dealias(MomentsG* G);
  void dealias(cuComplex* f);
  void dz(MomentsG* G, MomentsG* res, bool accumulate=false);     void dz(cuComplex* m, cuComplex* res, bool accumulate=false);
  void hyperz(MomentsG* G, MomentsG* res, float nu, bool accumulate=false);
  void dz2(MomentsG* G);    void dz2(cuComplex* m, cuComplex* res);
  void zft(MomentsG* G);   void zft(cuComplex* m, cuComplex* res);
  void applyBCs(MomentsG* G, MomentsG* GRhs, Fields* f, float* kperp2, double dt);

  void zft_inverse(MomentsG* G);
  //  void zft_inverse(cuComplex* m, cuComplex* res);
  
  void abs_dz(MomentsG* G, MomentsG* res, bool accumulate=false);
  void abs_dz(cuComplex* m, cuComplex* res, bool accumulate=false);
  void linkPrint();
  void identity(MomentsG* G); // for testing

 private:
  Parameters * pars_;
  Grids * grids_ ;
  
  int get_mode_nums_ntft(int *mode_nums, int nz, int naky, int nakx, int jtwist, int *m0, int nyc, float *ky);
  int get_nClasses_ntft(int *mode_size, int *mode_size_ref, int *mode_nums, int *nExtra, int naky, int nakx, int nz, int mode, int nLinks_max, int nLinks_min);
  void get_nChains_nLinks_ntft(int *mode_size, int *nLinks, int *nChains, int nClasses, int nakx, int naky, int nLinks_min, int mode);
  void kFill_ntft(int nClasses, int *nChains, int *nLinks, int *nExtra, int **ikyNTFT, int **neg_ikxdzNTFT, int naky, int nakx, int jtwist, int nz, int mode, int *mode_size_ref, int *mode_nums, int nx, int *m0, int nyc);
  int calc_idx0(int idx, int nshift, int nakx);
  void set_callbacks();
  void clear_callbacks();

  int nClasses;
  int * nLinks  ;
  int * nChains ;
  int * nExtra;
  int **ikxLinked_h, **ikyLinked_h;
  int **ikxLinked, **ikyLinked;
  float **kzLinked;
  cuComplex **G_linked;
  int * mode_nums;
  int mode;
  int * mode_size;
  int * mode_size_ref;

  cufftHandle * zft_plan_forward;  cufftHandle * dz_plan_forward;  cufftHandle * dz2_plan_forward;
  cufftHandle * zft_plan_inverse;  cufftHandle * dz_plan_inverse;
  cufftHandle * hyperz_plan_forward; cufftHandle * hyperz_plan_inverse;

  cufftHandle * zft_plan_forward_singlemom;
  cufftHandle * zft_plan_inverse_singlemom;

  cufftHandle * dz_plan_forward_singlemom;
  cufftHandle * dz_plan_inverse_singlemom;
  cufftHandle * abs_dz_plan_forward;
  cufftHandle * dz2_plan_forward_singlemom;
  cufftHandle * abs_dz_plan_forward_singlemom;

  dim3 * dG;
  dim3 * dB;
};

class GradParallelLocal : public GradParallel {
 public:
  GradParallelLocal(Grids* grids);
  ~GradParallelLocal() {};

  void dz(MomentsG* G, MomentsG* res, bool accumulate=false);
  void dz(cuComplex* m, cuComplex* res, bool accumulate=false);
  void dz2(MomentsG* G);
  void dz2(cuComplex* m, cuComplex* res);
  void zft(MomentsG* G);
  void zft(cuComplex* m, cuComplex* res);

  void zft_inverse(MomentsG* G);
  //  void zft_inverse(cuComplex* m, cuComplex* res);
  
  void abs_dz(MomentsG* G, MomentsG* res, bool accumulate=false) {};
  void abs_dz(cuComplex* m, cuComplex* res, bool accumulate=false);
 private:
  Grids * grids_ ;

  dim3 dG, dB;
  float kpar;
  float mkpar2;
};

class GradParallel1D {
 public:
  GradParallel1D(Grids* grids);
  ~GradParallel1D();
  void dz1D(float* b); 

 private:
  Grids * grids_ ;
  
  cufftHandle dz_plan_forward; cufftHandle dz2_plan_forward; 
  cufftHandle dz_plan_inverse;

  cuComplex * b_complex ;
};
