#pragma once
#include "laguerre_transform.h"
#include "grids.h"
#include "geometry.h"
#include "grad_perp.h"
#include "moments.h"
#include "fields.h"
#include "reductions.h"
#include "device_funcs.h"
#include "species.h"

class Nonlinear {
 public:
  virtual ~Nonlinear() {};
  virtual void nlps(MomentsG* G, Fields* f, MomentsG* G_res) = 0;
  virtual void get_max_frequency(Fields *f, double *wmax) {};
  GradPerp* get_grad_perp_f() {return grad_perp_f;}
  float* get_fXY() {return fXY;}

 protected:
  GradPerp *grad_perp_f;
  float *fXY;
};

class Nonlinear_GK : public Nonlinear {
 public:
  Nonlinear_GK(Parameters* pars, Grids* grids, Geometry* geo);
  ~Nonlinear_GK();

  void nlps(MomentsG* G, Fields* f, MomentsG* G_res);
  void get_max_frequency(Fields *f, double *wmax);
  void qvar(cuComplex* G, int N);
  void qvar(float* G, int N);
  
 private:

  int nBatch;
  size_t Size; 
  bool ks, vp;
  dim3 dGk, dBk, dGx, dBx, dGx_single, dBx_single;
  dim3 dGx_ntft, dBx_ntft, dGx_single_ntft, dBx_single_ntft, dGphi_ntft, dBphi_ntft;
  float cfl_x_inv, cfl_y_inv;
  double dt_cfl;

  Parameters        * pars_           ;
  Grids             * grids_          ;  
  Geometry          * geo_            ;
  
  Reduction<float>         * red             ; 
  LaguerreTransform * laguerre        ;
  LaguerreTransform * laguerre_single ;
  GradPerp          * grad_perp_G     ;
  GradPerp          * grad_perp_G_single ;
  GradPerp          * grad_perp_J0f ;

  MomentsG * NL_apar;
  cuComplex * tmp_c   ;
  float * dG          ;
  float * dg_dx       ;
  float * dg_dy       ;
  float * val1        ;
  float * Gy          ;
  float * dJ0phi_dx   ;
  float * dJ0phi_dy   ;
  float * dphi_dy     ;
  float * dJ0apar_dx ;
  float * dJ0apar_dy ;
  float * dphi        ;
  float * dchi        ;
  float * g_res       ;
  float vmax_x[1]     ;
  float vmax_y[1]     ;
  cuComplex * J0phi   ;
  cuComplex * J0apar ;
  cuComplex * iKxG    ;
  cuComplex * iKxG_single;

  void     *red_d_temp_storage;
  size_t   red_temp_storage_bytes;

  cudaStream_t G_stream = 0;
  cudaStream_t f_stream = 0;
  cudaEvent_t grad_perp_f_finished;
};

class Nonlinear_KREHM : public Nonlinear {
 public:
  Nonlinear_KREHM(Parameters* pars, Grids* grids);
  ~Nonlinear_KREHM();

  void nlps(MomentsG* G, Fields* f, MomentsG* G_res);
  void get_max_frequency(Fields *f, double *wmax);
  
 private:

  int nBatch;
  dim3 dGk, dBk, dGx, dBx, dGx_single, dBx_single;
  float cfl_x_inv, cfl_y_inv;
  double dt_cfl;

  Parameters        * pars_           ;
  Grids             * grids_          ;  
  
  Reduction<float>         * red             ; 
  GradPerp          * grad_perp_G     ;

  float * dg_dx       ;
  float * dg_dy       ;
  float * dphi_dx     ;
  float * dphi_dy     ;
  float * dchi_dx     ;
  float * dchi_dy     ;
  float * dG;
  cuComplex * tmp_c;
  MomentsG * G_tmp;

  float * val1        ;
  float vmax_x[1]     ;
  float vmax_y[1]     ;

  float rho_s;
  float d_e;
};

class Nonlinear_cetg : public Nonlinear {
 public:
  Nonlinear_cetg(Parameters* pars, Grids* grids);
  ~Nonlinear_cetg();

  void nlps(MomentsG* G, Fields* f, MomentsG* G_res);
  void get_max_frequency(Fields *f, double *wmax);
  
 private:

  int nBatch;
  dim3 dGk, dBk, dGx, dBx, dGx_single, dBx_single;
  float cfl_x_inv, cfl_y_inv;
  double dt_cfl;

  Parameters        * pars_           ;
  Grids             * grids_          ;  
  
  Reduction<float>         * red             ; 
  GradPerp          * grad_perp_G     ;

  float * dg_dx       ;
  float * dg_dy       ;
  float * dphi_dx     ;
  float * dphi_dy     ;
  float * dG;
  cuComplex * tmp_c;
  MomentsG * G_tmp;

  float * val1        ;
  float vmax_x[1]     ;
  float vmax_y[1]     ;

};

class Nonlinear_KS : public Nonlinear {
 public:
  Nonlinear_KS(Parameters* pars, Grids* grids);
  ~Nonlinear_KS();

  void nlps(MomentsG* G, Fields* f, MomentsG* G_res);
  void qvar(cuComplex* G, int N);
  void qvar(float* G, int N);
  
 private:

  int nBatch;
  dim3 dGx, dBx;

  Parameters        * pars_           ;
  Grids             * grids_          ;  
  
  GradPerp          * grad_perp_G     ;

  float * Gy          ;
  float * dg_dy       ;
  float * g_res       ;
};

class Nonlinear_VP : public Nonlinear {
 public:
  Nonlinear_VP(Parameters* pars, Grids* grids);
  ~Nonlinear_VP();

  void nlps(MomentsG* G, Fields* f, MomentsG* G_res);
  void qvar(cuComplex* G, int N);
  void qvar(float* G, int N);
  
 private:

  int nBatch;
  dim3 dGx, dBx;

  Parameters        * pars_           ;
  Grids             * grids_          ;  
  
  GradPerp          * grad_perp_G     ;

  float * Gy          ;
  float * dphi_dy     ;
  float * g_res       ;
};
