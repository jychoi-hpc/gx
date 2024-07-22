#pragma once
#include "cufft.h"
#include "device_funcs.h"
#include "fields.h"
#include "moments.h"
#include "grad_parallel.h"
#include "closures.h"
#include "get_error.h"

class Linear {
 public:
  virtual ~Linear() {};
  virtual void rhs(MomentsG* G, Fields* f, MomentsG* GRhs, double dt) = 0;
  virtual void get_max_frequency(double *wmax) {};
  // conservation terms
  cuComplex * upar_bar      ;
  cuComplex * uperp_bar     ;
  cuComplex * t_bar         ;
};

class Linear_GK : public Linear {
public:
  Linear_GK(Parameters* pars, Grids* grids, Geometry* geo); 
  ~Linear_GK();

  void rhs(MomentsG* G, Fields* f, MomentsG* GRhs, double dt);
  void get_max_frequency(double* wmax);

  //  int zderiv(MomentsG *G);

  dim3 dimGrid, dimBlock, dG, dB, dGs, dBs, dimGridh, dimBlockh, dB_all, dG_all;
  int sharedSize;
  
 private:
  Geometry       * geo_     ;
  Parameters     * pars_    ;
  Grids          * grids_   ;  
  GradParallel   * grad_par ;
  Closures       * closures ;
  MomentsG       * tmpG ;


  // Hammett-Belli hyper
  cuComplex * df            ;
  cuComplex * favg          ;
  float     * s01           ;
  float     * s10           ;
  float     * s11           ;
  float     * vol_fac       ; 

  float volDenom;
  size_t maxSharedSize;
  
};

class Linear_KREHM : public Linear {
public:
  Linear_KREHM(Parameters* pars, Grids* grids, Geometry* geo); 
  ~Linear_KREHM();

  void rhs(MomentsG* G, Fields* f, MomentsG* GRhs, double dt);
  void get_max_frequency(double* wmax);

  dim3 dimGrid, dimBlock, dG, dB, dGs, dBs, dimGridh, dimBlockh, dB_all, dG_all;
  int sharedSize;
  
 private:

  Geometry       * geo_     ;
  Parameters     * pars_    ;
  Grids          * grids_   ;  
  GradParallel   * grad_par ;
  Closures       * closures ;

  float rho_s;
  float d_e;
  float nu_ei;
};

