#pragma once
#include "netcdf.h"
#include "parameters.h"
#include "grids.h"
#include "device_funcs.h"
#include "get_error.h"

class MomentsG {
 public:
  MomentsG(Parameters* pars, Grids* grids, int is=-1);
  ~MomentsG();

  // accessor function to get pointer to specific l,m,s of G array
  // calling with two arguments checks bounds for (l,m) in 2D
  cuComplex* G(int l, int m) {
    assert(l<grids_->Nl && "Invalid moment requested: l out of bounds");
    assert(m<grids_->Nm+grids_->m_ghost && m>=-grids_->m_ghost && "Invalid moment requested: m out of bounds");
    return &G_lm[grids_->NxNycNz*(l + grids_->Nl*(m+grids_->m_ghost))]; // note shift by m_ghost! 
    // glm[ky, kx, z]
  }
  
  // calling with one argument checks bounds in 1D i.e. (0,Nmoms)
  // really the no-argument version is a specialisation of the 1-argument one, not the 2-argument one hence the default
  // value is here and not on the 2-argument declaration
  cuComplex* G(int moment_idx = 0) {
    assert( moment_idx >= 0 && moment_idx < grids_->Nmoms && "Invalid moment requested: moment index out of bounds" );
    return &G_lm[grids_->NxNycNz*(moment_idx + grids_->Nl * grids_->m_ghost)]; // note shift by m_ghost! 
    // glm[ky, kx, z]
  }

  cuComplex * Gm(int m_loc) {   return G(0,m_loc);   }

  // accessor to G array including ghosts (never used) 
  cuComplex * Gghost(int l=0, int m=0) {
    return &G_lm[grids_->NxNycNz*(l + grids_->Nl*m)];
  }

  void update_tprim(double time);
  void qvar (int N);
  void apply_mask(void);
  void initVP(double* time);
  void initialConditions(double* time=nullptr);
  void restart_write(int nc, int id);
  void restart_read(double* time);
  
  void rescale(float * phi_max);  
  
  void add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2);
  void add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2, double c3, MomentsG* G3);
  void add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2, double c3, MomentsG* G3,
		  double c4, MomentsG* G4);
  void add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2, double c3, MomentsG* G3,
		  double c4, MomentsG* G4, double c5, MomentsG* G5);

  void scale(double scalar);
  void scale(cuComplex scalar);
  void mask(void);
  void set_zero(void);

  void reality(int ngz);

  void sync(bool sync = false);
  void syncNCCL(bool sync);
  void syncMPI();
  
  inline void copyFrom(MomentsG* source) {
    cudaMemcpy(this->Gghost(), source->Gghost(), grids_->size_G, cudaMemcpyDeviceToDevice);
  }
 
  dim3 dimGrid, dimBlock, dG_all, dB_all;

  // pointer to pars_->species_h[is] struct
  specie* species;

  cuComplex * dens_ptr;
  cuComplex * upar_ptr;
  cuComplex * tpar_ptr;
  cuComplex * tprp_ptr;
  cuComplex * qpar_ptr;
  cuComplex * qprp_ptr;

  cudaStream_t syncStream;
 
 private:
  cuComplex  * G_lm   ;
  Grids      * grids_ ;
  Parameters * pars_  ;
  int is_glob_;
};
