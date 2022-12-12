#pragma once
#include "device_funcs.h"
#include "fields.h"
#include "moments.h"
#include "geometry.h"

class ExB {
  public:
    virtual ~ExB() {};
    virtual void flow_shear_shift(MomentsG* G, Fields* f, double dt) = 0;
};

class ExB_GK : public ExB {
  public:
    ExB_GK(Parameters* pars, Grids* grids, Geometry* geo); 
    ~ExB_GK();

    void flow_shear_shift(MomentsG* G, Fields* f, double dt);

  private:

    Geometry       * geo_     ;
    Parameters     * pars_    ;
    Grids          * grids_   ;  
    MomentsG       * GRhs_par ;

    dim3 dGk, dBk;
    int nt1, dimBlockfield, dimGridfield;
};
