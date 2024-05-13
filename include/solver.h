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
  virtual float find_max_qneutDenom_inv() = 0;
  virtual float* getQneutDenom() {return nullptr;};
  virtual float* getAmpereParFac() {return nullptr;};
  virtual float* getQneutFacBpar() {return nullptr;};
  virtual float* getAmperePerpFacPhi() {return nullptr;};
  virtual float* getAmperePerpFacBpar() {return nullptr;};
  virtual float* getBparDenom() {return nullptr;};
  virtual float* get_max_qneutFacPhi_inv() {return nullptr;};
  virtual float* get_max_qneutFacBpar_inv() {return nullptr;};
  virtual float* get_max_ampereParFac_inv() {return nullptr;};
  virtual float* get_max_amperePerpFacPhi_inv() {return nullptr;};
  virtual float* get_max_amperePerpFacBpar_inv() {return nullptr;};
  virtual void set_equilibrium_current(MomentsG* G, Fields* fields) {};
};

class Solver_GK : public Solver {
 public:
  Solver_GK(Parameters* pars, Grids* grids, Geometry* geo);
  ~Solver_GK();
  
  void fieldSolve(MomentsG** G, Fields* fields);
  float  find_max_qneutDenom_inv();
  void svar(cuComplex* f, int N);
  void svar(float* f, int N);
  float* getQneutDenom() {return qneutFacPhi;};
  float* getAmpereParFac() {return ampereParFac;};
  float* getQneutFacBpar() {return qneutFacBpar;};
  float* getAmperePerpFacPhi() {return amperePerpFacPhi;};
  float* getAmperePerpFacBpar() {return amperePerpFacBpar;};

  float* get_max_qneutFacPhi_inv() {return max_qneutFacPhi_inv;};
  float* get_max_ampereParFac_inv() {return max_ampereParFac_inv;};
  float* get_max_qneutFacBpar_inv() {return max_qneutFacBpar_inv;};
  float* get_max_amperePerpFacBphi_inv() {return max_amperePerpFacPhi_inv;};
  float* get_max_amperePerpFacBpar_inv() {return max_amperePerpFacBpar_inv;};
  float* getBparDenom() {return BparDenom;};


  cuComplex * nbar ;
  cuComplex * nbar_tmp ;
  cuComplex * jparbar ;
  cuComplex * jperpbar ;

private:

  void zero(cuComplex* f);
  
  dim3 dG, dB, dg, db;
  int count;
  
  float * max_qneutFacPhi_inv;
  float * max_qneutFacBpar_inv;
  float * max_ampereParFac_inv;
  float * max_amperePerpFacPhi_inv;
  float * max_amperePerpFacBpar_inv;

  float * phiavgdenom ;
  float * qneutDenom;
  float * ampereParFac;
  float * qneutFacPhi;
  float * qneutFacBpar;
  float * amperePerpFacPhi;
  float * amperePerpFacBpar;
  float * BparDenom;
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
  void set_equilibrium_current(MomentsG* G, Fields* fields);
  float find_max_qneutDenom_inv();

  cuComplex * nbar ;

private:

  dim3 dG, dB, dg, db;
  int count;

  cuComplex * tmp ;
  cuComplex *moms, *density, *current;

  // local private copies
  Parameters * pars_  ;
  Grids      * grids_ ;
  Geometry   * geo_   ;
};

class Solver_cetg : public Solver {
 public:
  Solver_cetg(Parameters* pars, Grids* grids);
  ~Solver_cetg();
  
  void fieldSolve(MomentsG** G, Fields* fields);
  float find_max_qneutDenom_inv();

private:

  dim3 dG, dB, dg, db;
  int count;

  cuComplex *moms, *density;

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
  float find_max_qneutDenom_inv();


private:

  void zero(cuComplex* f);
  
  dim3 dG, dB;


  // local private copies
  Parameters * pars_  ;
  Grids      * grids_ ;
};
