#include "diagnostic_classes.h"

class SundialsErrWgt : public MomentsDiagnostic {
 public:
  SundialsErrWgt(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf);
  void calculate(MomentsG** G, Fields* f, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d);
};

