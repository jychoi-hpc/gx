#pragma once

#include "grids.h"
#include "parameters.h"
#include "moments.h"
#include "grad_parallel.h"
#include "geometry.h"
#include "device_funcs.h"
#include "get_error.h"
#include "smith_par_closure.h"

class Closures {
 public:
  virtual ~Closures() {};
  virtual void apply_closures(MomentsG* G, MomentsG* GRhs) = 0;
};

class Beer42 : public Closures {
 public:
  Beer42(Parameters* pars, Grids* grids, Geometry* geo, GradParallel* grad_par);
  ~Beer42();
  void apply_closures(MomentsG* G, MomentsG* GRhs);

 private:
  float gpar_;
  Grids        * grids_   ;
  Parameters   * pars_    ;
  GradParallel * grad_par_;
  float        * omegad_  ;
  cuComplex    * tmp      ;

  // closure coefficients
  float Beta_par;
  float D_par;
  float D_perp;
  cuComplex * nu ;

  dim3 dimGrid, dimBlock;
};

class SmithPerp : public Closures {
 public: 
  SmithPerp(Parameters* pars, Grids* grids, Geometry* geo);
  ~SmithPerp();
  void apply_closures(MomentsG* G, MomentsG* GRhs);

 private:
  Grids      * grids_  ;
  Parameters * pars_   ;
  float      * omegad_ ;
  
  // closure coefficent array, to be allocated
  cuComplex * Aclos_   ;
  int q_;

  dim3 dimGrid, dimBlock; 
};

class SmithPar : public Closures {
 public: 
  SmithPar(Parameters* pars, Grids* grids, Geometry* geo, GradParallel* grad_par);
  ~SmithPar();
  void apply_closures(MomentsG* G, MomentsG* GRhs);

 private:
  dim3 dimGrid, dimBlock;  
  int q_;
  float gpar_;
  Grids        * grids_   ;
  Parameters   * pars_    ;
  GradParallel * grad_par_;
  cuComplex    * tmp      ;
  cuComplex    * tmp_abs  ;

  // closure coefficent array
  cuComplex * a_coefficients_ ;

  // closure array
  cuComplex * clos ;
};
