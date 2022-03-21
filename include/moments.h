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
  // calling with no arguments gives pointer to beginning of G_lm
  cuComplex* G(int l=0, int m=0) {
    return &G_lm[grids_->NxNycNz*(l + grids_->Nl*m)];
    // glm[ky, kx, z]
  }
  
  cuComplex * Gm(int m) {   return G(0,m);   }

  void qvar (int N);
  void apply_mask(void);
  void initVP(double* time);
  void initialConditions(double* time);
  void restart_write(double* time);
  void restart_read(double* time);
  
  void getH(cuComplex* J0phi);
  void getG(cuComplex* J0phi);

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

 //  void dz(MomentsG* G);
  void reality(int ngz);
  
  inline void copyFrom(MomentsG* source) {
    cudaMemcpy(this->G(), source->G(), grids_->size_G, cudaMemcpyDeviceToDevice);
  }
 
  dim3 dimGrid, dimBlock, dG_all, dB_all;

  // pointer to pars_->species_h[is] struct
  const specie* species;

  cuComplex * dens_ptr;
  cuComplex * upar_ptr;
  cuComplex * tpar_ptr;
  cuComplex * tprp_ptr;
  cuComplex * qpar_ptr;
  cuComplex * qprp_ptr;
 
 private:
  cuComplex  * G_lm   ;
  Grids      * grids_ ;
  Parameters * pars_  ;
};
