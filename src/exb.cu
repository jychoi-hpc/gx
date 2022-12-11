#include "exb.h"

//=======================================
// Linear_GK
// object for handling linear terms in GK
//=======================================
exb::exb(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo), closures(nullptr) 
{
}

exb::~exb()
{
  //if (closures) delete closures;
  //if (favg)       cudaFree(favg);
}

void exb::flow_shear_shift(MomentsG* G, Fields* f, float* kx_shift, int* jump, double dt)
{
  // shift moments and fields in kx to account for ExB shear
  kxs_phase_shift<<<dimGrid,dimBlock>>>(kx_shift, jump, ky, x, phasefac g_exb, dt);
  // update geometry
  update_geo<<<dimGrid,dimBlock>>>(kx_shift, ky, cv_d, gb_d, kperp2,
                           cv, cv0, gb, gb0, omegad,
                           gds2, gds21, gds22, bmagInv, shat, jump);
  // shift phi
  field_shift<<<dimGrid,dimBlock>>>(f->phi,jump);
  // shift apar and bpar
  // if apar, bpar terms...
  field_shift<<<dimGrid,dimBlock>>>(f->apar,jump);
  field_shift<<<dimGrid,dimBlock>>>(f->bpar,jump);
  // shift dist function
  field_shift<<<dimGrid,dimBlock>>>(G,jump);
}
