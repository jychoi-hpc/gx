#pragma once
#include "device_funcs.h"
#include "fields.h"
#include "moments.h"

class ExB {
 public:
  //virtual ~ExB() {};
  void flow_shear_shift(MomentsG* G, Fields* f, double dt);
};

class ExB_GK : public ExB {
public:
  ExB_GK(Parameters* pars, Grids* grids, Geometry* geo); 
  //~ExB_GK();

  void flow_shear_shift(MomentsG* G, Fields* f, double dt);

  dim3 dimGrid, dimBlock, dG, dB, dGs, dBs, dimGridh, dimBlockh, dB_all, dG_all;
  int sharedSize;
  
 private:

  Geometry       * geo_     ;
  Parameters     * pars_    ;
  Grids          * grids_   ;  
  MomentsG       * GRhs_par ;

  dim3 dGk, dBk;
  int nt1;
  
};

