#pragma once
#include "device_funcs.h"
#include "parameters.h"
#include "grids.h"
#include "moments.h"
#include "fields.h"
#include "ncdf.h"
#include "grad_parallel.h"
#include "grad_perp.h"
#include "reservoir.h"
#include "trinity_interface.h"

class Diagnostics {
 public:
  virtual ~Diagnostics() {};
  virtual bool loop(MomentsG* G, Fields* fields, double dt, int counter, double time) = 0 ;
  virtual void finish(MomentsG* G, Fields* fields, double time) = 0;  
  //virtual void write_init(MomentsG* G, Fields* f) = 0;
};

class Diagnostics_GK : public Diagnostics {
 public:
  Diagnostics_GK(Parameters *pars, Grids *grids, Geometry *geo);
  ~Diagnostics_GK();

  bool loop(MomentsG* G, Fields* fields, double dt, int counter, double time) ;
  void finish(MomentsG* G, Fields* fields, double time);  
  //void write_init(MomentsG* G, Fields* f);
  void copy_fluxes_to_trinity(trin_fluxes_struct *tfluxes);

private:
  float* P2(int s=0) {return &P2s[grids_->NxNycNz*s];}

  int ndiag; 
  int ikx_local, iky_local, iz_local;
  dim3 dG_spectra, dB_spectra, dG_all, dB_all, dbp, dgp; //, dG_scale, dB_scale;
  dim3 dGk, dBk;
  bool checkstop();
 
  float Dks; 
  float fluxDenom; float * flux_fac; 
  float  volDenom; float * vol_fac ;
  float * kvol_fac;
  
  cuComplex valphi;

  Parameters   * pars_         ;
  Grids        * grids_        ;
  Geometry     * geo_          ;  

  GradPerp     * grad_perp     ; 
  GradParallel * grad_par      ;
  Fields       * fields_old    ;
  NetCDF_ids   * id            ;
  Reservoir    * rc            ;
  
  float        * G2            ;
  float        * P2s           ;
  float        * Phi2          ;
  float        * val           ;
  cuComplex    * omg_d         ;
  cuComplex    * tmp_omg_h     ;
  cuComplex    * t_bar         ;
  cuComplex    * favg          ;
  cuComplex    * df            ;
  cuComplex    * amom_d        ;
  cuComplex    * vEk           ;
  float        * phi_max       ; 

  float *gy_d, *gy_h;
  double *ry_h;
  
  void print_omg (cuComplex *W);
  void get_rh    (Fields* f);
  //  void reduce2z  (float* fk, cuComplex* f);
  void print_growth_rates_to_screen (cuComplex *w);

  void write_Wtot  (float   Wh, bool endrun);
  //  void pzt(MomentsG* G, Fields* f);

  char stopfilename_[2000];
};

class Diagnostics_KREHM : public Diagnostics {
 public:
  Diagnostics_KREHM(Parameters *pars, Grids *grids);
  ~Diagnostics_KREHM();

  bool loop(MomentsG* G, Fields* fields, double dt, int counter, double time) ;
  void finish(MomentsG* G, Fields* fields, double time);  
  //void write_init(MomentsG* G, Fields* f);
  void copy_fluxes_to_trinity(trin_fluxes_struct *tfluxes);

private:
  float* P2(int s=0) {return &P2s[grids_->NxNycNz*s];}

  int ndiag; 
  int ikx_local, iky_local, iz_local;
  dim3 dG_spectra, dB_spectra, dG_all, dB_all, dbp, dgp; //, dG_scale, dB_scale;
  dim3 dGk, dBk;
  bool checkstop();
 
  float Dks; 
  float fluxDenom; float * flux_fac; 
  float  volDenom; float * vol_fac ;
  float * kvol_fac;
  
  cuComplex valphi;

  Parameters   * pars_         ;
  Grids        * grids_        ;
  Geometry     * geo_          ;  

  GradPerp     * grad_perp     ; 
  GradParallel * grad_par      ;
  Fields       * fields_old    ;
  NetCDF_ids   * id            ;
  Reservoir    * rc            ;
  
  float        * G2            ;
  float        * P2s           ;
  float        * Phi2          ;
  float        * val           ;
  cuComplex    * omg_d         ;
  cuComplex    * tmp_omg_h     ;
  cuComplex    * t_bar         ;
  cuComplex    * favg          ;
  cuComplex    * df            ;
  cuComplex    * amom_d        ;
  cuComplex    * vEk           ;
  float        * phi_max       ; 

  float *gy_d, *gy_h;
  double *ry_h;
  
  void print_omg (cuComplex *W);
  void get_rh    (Fields* f);
  //  void reduce2z  (float* fk, cuComplex* f);
  void print_growth_rates_to_screen (cuComplex *w);

  void write_Wtot  (float   Wh, bool endrun);
  //  void pzt(MomentsG* G, Fields* f);

  char stopfilename_[2000];
};
