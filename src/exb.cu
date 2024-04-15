#include "exb.h"
//=======================================
// exb
// object for handling flow shear terms.
//=======================================
ExB_GK::ExB_GK(Parameters* pars_, Grids* grids, Geometry* geo) :
  pars_(pars_), grids_(grids), geo_(geo), phi_tmp(nullptr), g_tmp(nullptr)
{ 
  checkCuda(cudaMalloc(&phi_tmp, sizeof(cuComplex)*grids_->NxNycNz));
  checkCuda(cudaMalloc(&g_tmp  , sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl*grids_->Nm));

  // Nx*Nyc kernels
  dimBlock_xy = dim3(32,16);
  dimGrid_xy  = dim3(1+(grids->Nyc-1)/dimBlock_xy.x, 1+(grids->Nx-1)/dimBlock_xy.y);
  
  // Nx*Nyc*Nz kernels
  dimBlock_xyz = dim3(32, 4, 4);
  dimGrid_xyz  = dim3(1+(grids->Nyc-1)/dimBlock_xyz.x, 1+(grids->Nx-1)/dimBlock_xyz.y, 1+(grids->Nz-1)/dimBlock_xyz.z);

  // Nx*Nyc*Nz*Nl*Nm kernels
  int nn1 = grids_->Nyc;             int nt1 = min(nn1, 16);    int nb1 = (nn1-1)/nt1 + 1;
  int nn2 = grids_->Nx*grids_->Nz;   int nt2 = min(nn2, 16);    int nb2 = (nn2-1)/nt2 + 1;
  int nn3 = grids_->Nm*grids_->Nl;   int nt3 = min(nn3,  4);    int nb3 = (nn3-1)/nt3 + 1;
  
  dimBlock_xyzlm = dim3(nt1, nt2, nt3);
  dimGrid_xyzlm  = dim3(nb1, nb2, nb3);
  checkCuda(cudaGetLastError());

  CP_TO_GPU (grids_->x, grids_->x_h, sizeof(float)*grids_->Nx); //find a better place for this? only need to do it once for exb/ntft // JMH

}
ExB_GK::~ExB_GK()
{
  if (phi_tmp)  cudaFree(phi_tmp);
  if (g_tmp)    cudaFree(g_tmp);
}
void ExB_GK::flow_shear_shift(Fields* f, double dt) // this is called once per timestep
{
  // update kxstar terms and phasefactor
  kxstar_phase_shift<<<dimGrid_xy, dimBlock_xy>>>(grids_->kxstar, grids_->kxbar_ikx_new, grids_->kxbar_ikx_old, grids_->ky, grids_->x, grids_->phasefac_exb, grids_->phasefacminus_exb, pars_->g_exb, dt, pars_->x0, pars_->ExBshear_phase);

  // update geometry
  if (pars_->nonTwist) {
    geo_shift_ntft<<<dimGrid_xyz, dimBlock_xyz>>>(grids_->kxstar, grids_->ky, geo_->cv_d, geo_->gb_d, geo_->kperp2,
                             geo_->cvdrift, geo_->cvdrift0, geo_->gbdrift, geo_->gbdrift0, geo_->omegad,
                             geo_->gds2, geo_->gds21, geo_->gds22, geo_->bmagInv, pars_->shat,
			     geo_->ftwist, geo_->deltaKx, geo_->m0, pars_->x0);
    if (!pars_->linear) iKx_shift_ntft <<<dimGrid_xyz, dimBlock_xyz>>>(grids_->iKx, pars_->g_exb, dt, grids_->ky);
  } else { 
    geo_shift<<<dimGrid_xyz, dimBlock_xyz>>>(grids_->kxstar, grids_->ky, geo_->cv_d, geo_->gb_d, geo_->kperp2,
                             geo_->cvdrift, geo_->cvdrift0, geo_->gbdrift, geo_->gbdrift0, geo_->omegad,
                             geo_->gds2, geo_->gds21, geo_->gds22, geo_->bmagInv, pars_->shat);
  }
  // update fields
  CP_TO_GPU (phi_tmp, f->phi,  sizeof(cuComplex)*grids_->NxNycNz);
  field_shift<<<dimGrid_xyz,dimBlock_xyz>>>    (f->phi, phi_tmp, grids_->kxbar_ikx_new, grids_->kxbar_ikx_old, pars_->g_exb);
}

void ExB_GK::flow_shear_g_shift(MomentsG* G) // this is called for each G used in the timestepping scheme per timestep
{
  // update G
  CP_TO_GPU (g_tmp,   G->G(),  sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl*grids_->Nm);
  g_shift <<< dimGrid_xyzlm, dimBlock_xyzlm>>> (G->G(), g_tmp, grids_->kxbar_ikx_new, grids_->kxbar_ikx_old, pars_->g_exb);
}
