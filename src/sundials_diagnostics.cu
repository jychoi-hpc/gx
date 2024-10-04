#include "diagnostic_classes.h"

SundialsErrWgt::SundialsErrWgt(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf)
 : MomentsDiagnostic(pars, grids, geo, nonlinear, ncdf, "SundialsErrWgt")  // call base class constructor
{
  if(grids_->iproc != 0) skipWrite = true; // We do reductions, so only write from proc0

}

void SundialsErrWgt::calculate(MomentsG** G, Fields* fields, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    CP_TO_CPU(f_h + is*grids_->NxNycNz, G[is]->G(0,0), sizeof(cuComplex)*grids_->NxNycNz);

    // Just leave the XY stuff zeroed
  }
}
