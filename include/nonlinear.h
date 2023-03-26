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
  virtual double cfl(Fields *f, double dt_max) = 0;
  virtual void get_max_frequency(Fields *f, double *wmax) {};
};

class Nonlinear_GK : public Nonlinear {
 public:
  Nonlinear_GK(Parameters* pars, Grids* grids, Geometry* geo);
  ~Nonlinear_GK();

  void nlps(MomentsG* G, Fields* f, MomentsG* G_res);
  double cfl(Fields *f, double dt_max);
  void get_max_frequency(Fields *f, double *wmax);
  void qvar(cuComplex* G, int N);
  void qvar(float* G, int N);
  
 private:

  int nBatch;
  size_t Size; 
  bool ks, vp;
  dim3 dGk, dBk, dGx, dBx, dGx_single, dBx_single;
  float cfl_x_inv, cfl_y_inv;
  double dt_cfl;

  Parameters        * pars_           ;
  Grids             * grids_          ;  
  Geometry          * geo_            ;
  
  Red               * red             ; 
  LaguerreTransform * laguerre        ;
  LaguerreTransform * laguerre_single ;
  GradPerp          * grad_perp_G     ;
  GradPerp          * grad_perp_G_single ;
  GradPerp          * grad_perp_J0f ;
  GradPerp          * grad_perp_f   ;

  MomentsG * G_tmp;
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
};

class Nonlinear_KREHM : public Nonlinear {
 public:
  Nonlinear_KREHM(Parameters* pars, Grids* grids);
  ~Nonlinear_KREHM();

  void nlps(MomentsG* G, Fields* f, MomentsG* G_res);
  double cfl(Fields *f, double dt_max);
  
 private:

  int nBatch;
  dim3 dGk, dBk, dGx, dBx;
  float cfl_x_inv, cfl_y_inv;
  double dt_cfl;

  Parameters        * pars_           ;
  Grids             * grids_          ;  
  
  Red               * red             ; 
  GradPerp          * grad_perp       ;

  float * dg_dx       ;
  float * dg_dy       ;
  float * dphi_dx     ;
  float * dphi_dy     ;
  float * dchi_dx     ;
  float * dchi_dy     ;
  float * tmp_r;
  cuComplex * tmp_c;

  float * val1        ;
  float vPhi_max_x[1]     ;
  float vPhi_max_y[1]     ;
  float vA_max_x[1]     ;
  float vA_max_y[1]     ;

  float rho_s;
  float d_e;
};

class Nonlinear_KS : public Nonlinear {
 public:
  Nonlinear_KS(Parameters* pars, Grids* grids);
  ~Nonlinear_KS();

  void nlps(MomentsG* G, Fields* f, MomentsG* G_res);
  double cfl(Fields *f, double dt_max);
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
  double cfl(Fields *f, double dt_max);
  void qvar(cuComplex* G, int N);
  void qvar(float* G, int N);
  
 private:

  int nBatch;
  dim3 dGx, dBx;

  Parameters        * pars_           ;
  Grids             * grids_          ;  
  
  GradPerp          * grad_perp_G     ;
  GradPerp          * grad_perp_f   ;

  float * Gy          ;
  float * dphi_dy     ;
  float * g_res       ;
};
