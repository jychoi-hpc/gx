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
  bool ks;
  bool vp;

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

class Linear_cetg : public Linear {
public:
  Linear_cetg(Parameters* pars, Grids* grids, Geometry* geo); 
  ~Linear_cetg();

  void rhs(MomentsG* G, Fields* f, MomentsG* GRhs, double dt);
  void get_max_frequency(double* wmax);

  dim3 dGs, dBs;
  
 private:

  Geometry       * geo_     ;
  Parameters     * pars_    ;
  Grids          * grids_   ;  
  GradParallel   * grad_par ;

  float Z_ion;
  float tau_bar;
  float c1, c2, c3, C12, C23;
  
};

class Linear_KS : public Linear {
public:
  Linear_KS(Parameters* pars, Grids* grids); 
  ~Linear_KS();

  void rhs(MomentsG* G, Fields* f, MomentsG* GRhs, double dt);

  dim3 dG, dB;
  
 private:

  Parameters     * pars_    ;
  Grids          * grids_   ;  
};

class Linear_VP : public Linear {
public:
  Linear_VP(Parameters* pars, Grids* grids); 
  ~Linear_VP();

  void rhs(MomentsG* G, Fields* f, MomentsG* GRhs, double dt);

  dim3 dG, dB;
  
 private:

  Parameters     * pars_    ;
  Grids          * grids_   ;  
};
