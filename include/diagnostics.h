#pragma once
#include "device_funcs.h"
#include "parameters.h"
#include "grids.h"
#include "moments.h"
#include "linear.h"
#include "nonlinear.h"
#include "fields.h"
#include "ncdf.h"
#include "grad_parallel.h"
#include "grad_perp.h"
#include "reservoir.h"
#include "diagnostic_classes.h"
#include "spectra_calc.h"
#include <memory>

using namespace std;

class Diagnostics {
 public:
  virtual ~Diagnostics() {};
  virtual bool loop(MomentsG** G, Fields* fields, double dt, int counter, double time) = 0 ;
  virtual void finish(MomentsG** G, Fields* fields, double time) = 0;  
  void restart_write(MomentsG** G, double *time);
  bool checkstop();
  void print_growth_rates_to_screen (cuComplex *w);

 protected:
  Parameters   * pars_         ;
  Grids        * grids_        ;
  char stopfilename_[2000];

};

class Diagnostics_GK : public Diagnostics {
 public:
  Diagnostics_GK(Parameters *pars, Grids *grids, Geometry *geo, Linear *linear, Nonlinear *nonlinear);
  ~Diagnostics_GK();

  bool loop(MomentsG** G, Fields* fields, double dt, int counter, double time) ;
  void finish(MomentsG** G, Fields* fields, double time);  

private:
 
  float * tmpf;
  cuComplex * tmpC;
  float * tmpG;
  Geometry     * geo_          ;
  GradPerp     * grad_perp     ; 
  GradParallel * grad_par      ;
  Fields       * fields_old    ;
  MomentsG     ** G_old    ;
  NetCDF       * ncdf_         ;
  NetCDF       * ncdf_big_         ;
  Reservoir    * rc            ;
  AllSpectraCalcs * allSpectra_;
  Linear * linear_;
  Nonlinear * nonlinear_;

  void get_rh    (Fields* f);

  vector<unique_ptr<SpectraDiagnostic>> spectraDiagnosticList;
  GrowthRateDiagnostic *growthRateDiagnostic;
  vector<unique_ptr<MomentsDiagnostic>> momentsDiagnosticList;
  FieldsDiagnostic *fieldsDiagnostic;
  FieldsXYDiagnostic *fieldsXYDiagnostic;
};

class Diagnostics_KREHM : public Diagnostics {
 public:
  Diagnostics_KREHM(Parameters *pars, Grids *grids, Geometry *geo, Linear *linear, Nonlinear *nonlinear);
  ~Diagnostics_KREHM();

  bool loop(MomentsG** G, Fields* fields, double dt, int counter, double time) ;
  void finish(MomentsG** G, Fields* fields, double time);  

private:
  float * tmpf;
  cuComplex * tmpC;
  float * tmpG;
  Geometry     * geo_          ;
  Fields       * fields_old    ;
  NetCDF       * ncdf_         ;
  NetCDF       * ncdf_big_         ;
  AllSpectraCalcs * allSpectra_;
  Linear * linear_;
  Nonlinear * nonlinear_;

  void get_rh    (Fields* f);

  vector<unique_ptr<SpectraDiagnostic>> spectraDiagnosticList;
  GrowthRateDiagnostic *growthRateDiagnostic;
  vector<unique_ptr<MomentsDiagnostic>> momentsDiagnosticList;
  FieldsDiagnostic *fieldsDiagnostic;
  FieldsXYDiagnostic *fieldsXYDiagnostic;
};

class Diagnostics_cetg : public Diagnostics {
 public:
  Diagnostics_cetg(Parameters *pars, Grids *grids, Geometry *geo);
  ~Diagnostics_cetg();

  bool loop(MomentsG** G, Fields* fields, double dt, int counter, double time) ;
  void finish(MomentsG** G, Fields* fields, double time);  

private:
  float* P2(int s=0) {return &P2s[grids_->NxNycNz*s];}
  float* G2(int s=0) {return &G2s[grids_->NxNycNz*s];}

  int ndiag; 
  int ikx_local, iky_local, iz_local;
  dim3 dG_spectra, dB_spectra, dG_all, dB_all, dbp, dgp; 
  dim3 dGk, dBk;
 
  float fluxDenom; float * flux_fac; 
  float  volDenom; float * vol_fac ;
  
  Geometry     * geo_          ;  
  Fields       * fields_old    ;
  NetCDF_ids   * id            ;
  
  float        * G2s           ;
  float        * P2s           ;
  cuComplex    * omg_d         ;
  cuComplex    * tmp_omg_h     ;
  cuComplex    * vEk           ;

  void print_omg (cuComplex *W);

};
