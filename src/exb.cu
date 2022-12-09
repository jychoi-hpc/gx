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

void exb::flow_shear_shift(cuComplex *Phi, float* kx_shift, int* jump, double dt)
{
  // shift moments and fields in kx to account for ExB shear
  kx_phase_shift<<<dimGrid,dimBlock>>>(kx_shift,jump,ky,g_exb,dt);
  // shift phi
  field_shift<<<dimGrid,dimBlock>>>(Phi,jump);
  // shift apar and bpar
  field_shift<<<dimGrid,dimBlock>>>(Apar,jump);
  field_shift<<<dimGrid,dimBlock>>>(Bpar,jump);
  // shift dist function
  field_shift<<<dimGrid,dimBlock>>>(G,jump);
}
