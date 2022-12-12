#include "exb.h"

//=======================================
// exb
// object for handling flow shear terms.
//=======================================
ExB_GK::ExB_GK(Parameters* pars_, Grids* grids, Geometry* geo) :
  pars_(pars_), grids_(grids), geo_(geo)
{
  //unsigned int nxkyz = grids_->NxNycNz;
  //unsigned int nlag = grids_->Nj;

  int nbx = min(32, grids_->NxNycNz);   int ngx = 1 + (grids_->NxNycNz-1)/nbx;
  int nby = min(16, grids_->Nj);   int ngy = 1 + (grids_->Nj-1)/nby;

  dBk = dim3(nbx, nby, 1);
  dGk = dim3(ngx, ngy, 1);

  // Dimensions for the field_shift kernels.
  nt1 = pars_->i_share;   nb1 = 1 + (nbx-1)/nt1;

  // JFP: this is likely wrong.
  dimBlockfield = nt1;
  dimGridfield = nb1;
}

ExB_GK::~ExB_GK()
{
  //if (closures) delete closures;
  //if (favg)       cudaFree(favg);
}

void ExB_GK::flow_shear_shift(MomentsG* G, Fields* f, double dt)
{
  // shift moments and fields in kx to account for ExB shear
  kxstar_phase_shift<<<grids_->NxNyc,nt1>>>(grids_->kxstar, grids_->kxbar_ikx, grids_->ky, grids_->x, grids_->phasefac, pars_->g_exb, dt, pars_->x0);
  // update geometry
  geo_shift<<<grids_->NxNycNz,nt1>>>(grids_->kxstar, grids_->ky, geo_->cv_d, geo_->gb_d, geo_->kperp2,
                           geo_->cv, geo_->cv0, geo_->gb, geo_->gb0, geo_->omegad,
                           geo_->gds2, geo_->gds21, geo_->gds22, geo_->bmagInv, pars_->shat);
  // shift fields
  field_shift<<<dimGridfield,dimBlockfield>>>(f->phi,grids_->kxbar_ikx);
  if (pars_->fapar > 0.) field_shift<<<<dimGridfield,dimBlockfield>>>(f->apar,grids_->kxbar_ikx);
  //if (pars_->fbpar > 0.) field_shift<<<dimGridfield,dimBlockfield>>>(f->bpar,kxbar_ikx); // JFP: note: to update once we have bpar.
  // shift dist function, batching in m.
  for(int m=grids_->m_lo; m<grids_->m_up; m++) {
    int m_local = m - grids_->m_lo;
    field_shift <<< dGk, dBk >>> (G->Gm(m_local),kxbar_ikx);

}
