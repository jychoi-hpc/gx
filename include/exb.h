#pragma once
#include "device_funcs.h"
#include "fields.h"
#include "moments.h"

class exb {
 public:
  virtual ~exb() {};
  virtual void flow_shear_shift(MomentsG* G, Fields* f, double dt) = 0;
};

class exb_GK : public exb {
public:
  exb_GK(Parameters* pars, Grids* grids, Geometry* geo); 
  ~exb_GK();

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

