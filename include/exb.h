#pragma once
#include "device_funcs.h"
#include "fields.h"
#include "moments.h"
#include "geometry.h"

class ExB {
  public:
    virtual ~ExB() {};
    virtual void flow_shear_shift(Fields* f, double dt) = 0;
    virtual void flow_shear_g_shift(MomentsG* G) = 0;
};

class ExB_GK : public ExB {
  public:
    ExB_GK(Parameters* pars, Grids* grids, Geometry* geo); 
    ~ExB_GK();
    void flow_shear_shift(Fields* f, double dt);
    void flow_shear_g_shift(MomentsG* G);

  private:
    Geometry       * geo_     ;
    Parameters     * pars_    ;
    Grids          * grids_   ;
    MomentsG       * GRhs_par ;
    dim3 dGk, dBk;
    dim3 dimGrid_xy, dimBlock_xy, dimGrid_xyz, dimBlock_xyz, dimGrid_xyzlm, dimBlock_xyzlm;
    int nt1, dimBlockfield, dimGridfield;
    Fields * fieldsTmp;
    MomentsG  * gTmp;
};
