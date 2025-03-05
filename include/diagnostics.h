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
//#include "reservoir.h"
#include "diagnostic_classes.h"
#include "spectra_calc.h"
#include <memory>

using namespace std;

class Diagnostics {
 public:
  virtual ~Diagnostics() {};
  virtual bool loop(MomentsG** G, Fields* fields, double dt, int counter, double time) = 0 ;
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
  Diagnostics_GK(Parameters *pars, Grids *grids, Geometry *geo, Linear *linear, Nonlinear *nonlinear, NetCDF* ncdf);
  ~Diagnostics_GK();

  bool loop(MomentsG** G, Fields* fields, double dt, int counter, double time) ;

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
  //Reservoir    * rc            ;
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
  Diagnostics_KREHM(Parameters *pars, Grids *grids, Geometry *geo, Linear *linear, Nonlinear *nonlinear, NetCDF* ncdf);
  ~Diagnostics_KREHM();

  bool loop(MomentsG** G, Fields* fields, double dt, int counter, double time) ;

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
