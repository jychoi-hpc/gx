#pragma once

#include "parameters.h"
#include "grids.h"
#include "geometry.h"
#include "fields.h"
#include "moments.h"
#include "device_funcs.h"
#include "get_error.h"
#include "nccl.h"

class Solver {
 public:
  virtual ~Solver() {};
  virtual void fieldSolve(MomentsG** G, Fields* fields) = 0;
};

class Solver_GK : public Solver {
 public:
  Solver_GK(Parameters* pars, Grids* grids, Geometry* geo);
  ~Solver_GK();
  
  void fieldSolve(MomentsG** G, Fields* fields);
  void svar(cuComplex* f, int N);
  void svar(float* f, int N);
  
  cuComplex * nbar ;
  cuComplex * nbar_tmp ;
  cuComplex * jparbar ;
  cuComplex * jperpbar ;

private:

  void zero(cuComplex* f);
  
  dim3 dG, dB, dg, db;
  int count;

  float * phiavgdenom ;
  float * ampereParFac;
  float * qneutFacPhi;
  float * qneutFacBpar;
  float * amperePerpFacPhi;
  float * amperePerpFacBpar;
  cuComplex * tmp ;

  // local private copies
  Parameters * pars_  ;
  Grids      * grids_ ;
  Geometry   * geo_   ;

};

class Solver_KREHM : public Solver {
 public:
  Solver_KREHM(Parameters* pars, Grids* grids);
  ~Solver_KREHM();
  
  void fieldSolve(MomentsG** G, Fields* fields);
  
  cuComplex * nbar ;

private:

  dim3 dG, dB, dg, db;

  cuComplex * tmp ;

  // local private copies
  Parameters * pars_  ;
  Grids      * grids_ ;
  Geometry   * geo_   ;
};

class Solver_VP : public Solver {
 public:
  Solver_VP(Parameters* pars, Grids* grids);
  ~Solver_VP();
  
  void fieldSolve(MomentsG** G, Fields* fields);
  void svar(cuComplex* f, int N);
  void svar(float* f, int N);
  

private:

  void zero(cuComplex* f);
  
  dim3 dG, dB;


  // local private copies
  Parameters * pars_  ;
  Grids      * grids_ ;
};
