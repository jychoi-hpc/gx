#pragma once
#include "netcdf.h"
#include "parameters.h"
#include "grids.h"
#include "device_funcs.h"
#include "get_error.h"

class MomentsG {
 public:
  MomentsG(Parameters* pars, Grids* grids);
  ~MomentsG();

  // accessor function to get pointer to specific l,m,s of G array
  // calling with no arguments gives pointer to beginning of G_lm
  cuComplex* G(int l=0, int m=0, int s=0) {
    //    return &G_lm[grids_->NxNycNz*l + grids_->NxNycNz*grids_->Nl*m + grids_->NxNycNz*grids_->Nmoms*s];
    return &G_lm[grids_->NxNycNz*(l + grids_->Nl*(m + grids_->Nm*s))];
    // glm[ky, kx, z]
  }
  float* tz(int s=0) {return &tzs[s];}
  float* zt(int s=0) {return &zts[s];}
  float* nt(int s=0) {return &nts[s];}
  float* as(int s=0) {return &aps[s];}
  float* r2(int s=0) {return &r2s[s];}
  float* nz(int s=0) {return &nzs[s];}
  float* qn(int s=0) {return &qns[s];}
  float* amp(int s=0) {return &amps[s];}
  float* nu(int s=0) {return &nu_ss[s];}
  float* vt(int s=0) {return &vts[s];}
  float* tp(int s=0) {return &tps[s];}
  float* up(int s=0) {return &ups[s];}
  float* fp(int s=0) {return &fps[s];}
  int*   ty(int s=0) {return &typ[s];}
  
  cuComplex * Gm(int m, int s=0) {   return G(0,m,s);   }

  void update_tprim(double time);
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

  cuComplex ** dens_ptr;
  cuComplex ** upar_ptr;
  cuComplex ** tpar_ptr;
  cuComplex ** tprp_ptr;
  cuComplex ** qpar_ptr;
  cuComplex ** qprp_ptr;
 
 private:
  cuComplex  * G_lm   ;
  Grids      * grids_ ;
  Parameters * pars_  ;

  float * tzs = nullptr;
  float * zts = nullptr;
  float * nts = nullptr;
  float * aps = nullptr;
  float * r2s = nullptr;
  float * nzs = nullptr;
  float * qns = nullptr;
  float * amps = nullptr;
  float * nu_ss = nullptr;
  float * vts = nullptr;
  float * tps = nullptr;
  float * ups = nullptr;
  float * fps = nullptr;
  int   * typ = nullptr;
};
