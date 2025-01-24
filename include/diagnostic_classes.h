#pragma once
#include "parameters.h"
#include "grids.h"
#include "geometry.h"
#include "linear.h"
#include "nonlinear.h"
#include "moments.h"
#include "fields.h"
#include "ncdf.h"
#include "reductions.h"
#include "spectra_calc.h"
#include "device_funcs.h"
#include "netcdf_par.h"
#include "netcdf.h"
#include "get_error.h"
#include <string>

using namespace std;

// a SpectraDiagnostic is a diagnostic for which 
// spectra can be computed, i.e. a diagnostic quantity 
// that can be summed over various dimensions.
// for example, |Phi|**2 is a diagnostic quantity that 
// can be summed over various indices to get
// e.g. |Phi|**2(kx), |Phi|**2(ky), etc. 
// the output is real-valued
class SpectraDiagnostic {
 public:
  SpectraDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf);
  ~SpectraDiagnostic() {};
  virtual void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf) = 0;
  virtual void set_dt_data(MomentsG** G_old, Fields* f_old, float dt) {};
 protected:
  void add_spectra(SpectraCalc *spectra);
  void write_spectra(float* data);
  void set_kernel_dims();

  vector<SpectraCalc*> spectraList;
  vector<int> spectraIds;
  string varname;
  int nc_group, nc_type;
  bool isMoments;
  bool skipWrite = false;
  dim3 dG, dB;
  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;
  NetCDF* ncdf_;
  string description = "";
};

// The following classes each define a particular diagnostic quantity

// |Phi|**2
class Phi2Diagnostic : public SpectraDiagnostic {
 public:
  Phi2Diagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

// |Phi(ky=0)|**2
class Phi2ZonalDiagnostic : public SpectraDiagnostic {
 public:
  Phi2ZonalDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

// |Apar|**2
class Apar2Diagnostic : public SpectraDiagnostic {
 public:
  Apar2Diagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

// Wphi = (1-Gamma0(b_s))|Phi|**2
class WphiDiagnostic : public SpectraDiagnostic {
 public:
  WphiDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

// Wapar = |k_perp Apar|**2
class WaparDiagnostic : public SpectraDiagnostic {
 public:
  WaparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

// Wg = |G|^2
class WgDiagnostic : public SpectraDiagnostic {
 public:
  WgDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

class WphiKrehmDiagnostic : public SpectraDiagnostic {
 public:
  WphiKrehmDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

class WaparKrehmDiagnostic : public SpectraDiagnostic {
 public:
  WaparKrehmDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

// HeatFlux (Q)
class HeatFluxDiagnostic : public SpectraDiagnostic {
 public:
  HeatFluxDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

class HeatFluxESDiagnostic : public SpectraDiagnostic {
 public:
  HeatFluxESDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

class HeatFluxAparDiagnostic : public SpectraDiagnostic {
 public:
  HeatFluxAparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

class HeatFluxBparDiagnostic : public SpectraDiagnostic {
 public:
  HeatFluxBparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

// ParticleFlux (Gamma)
class ParticleFluxDiagnostic : public SpectraDiagnostic {
 public:
  ParticleFluxDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

class ParticleFluxESDiagnostic : public SpectraDiagnostic {
 public:
  ParticleFluxESDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

class ParticleFluxAparDiagnostic : public SpectraDiagnostic {
 public:
  ParticleFluxAparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

class ParticleFluxBparDiagnostic : public SpectraDiagnostic {
 public:
  ParticleFluxBparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

// TurbulentHeating (H)
class TurbulentHeatingDiagnostic : public SpectraDiagnostic {
 public:
  TurbulentHeatingDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Linear* linear, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
  void set_dt_data(MomentsG** G_old, Fields* f_old, float dt);

 private:
  Linear *linear_;
  MomentsG** G_old_;
  Fields* f_old_;
  float dt_;
};

class GrowthRateDiagnostic {
 public:
  GrowthRateDiagnostic(Parameters* pars, Grids* grids, NetCDF* ncdf);
  ~GrowthRateDiagnostic();
  void calculate_and_write(Fields* f, Fields* f_old, double dt);
 private:
  void dealias_and_reorder(cuComplex* fold, float* fnew);

  int ndim, N, Nwrite;
  int dims[6];
  size_t count[6] = {0};
  size_t start[6] = {0};
  int varid;

  string varname;
  int nc_group, nc_type;
  dim3 dG, dB;
  Parameters* pars_;
  Grids* grids_;
  NetCDF* ncdf_;

  cuComplex *omg_d, *omg_h;
  float *cpu;
};

class FieldsDiagnostic {
 public:
  FieldsDiagnostic(Parameters* pars, Grids* grids, NetCDF* ncdf);
  ~FieldsDiagnostic();
  void calculate_and_write(Fields* f);
 private:
  void dealias_and_reorder(cuComplex* fold, float* fnew);

  int ndim, N, Nwrite;
  int dims[6];
  size_t count[6] = {0};
  size_t start[6] = {0};
  int varids[3];

  string varnames[3];
  int nc_group, nc_type;
  dim3 dG, dB;
  Parameters* pars_;
  Grids* grids_;
  NetCDF* ncdf_;

  cuComplex *f_h;
  float *cpu;
};

class FieldsXYDiagnostic {
 public:
  FieldsXYDiagnostic(Parameters* pars, Grids* grids, Nonlinear* nonlinear, NetCDF* ncdf);
  ~FieldsXYDiagnostic();
  void calculate_and_write(Fields* f);
 private:
  void dealias_and_reorder(float* fold, float* fnew);

  int ndim, N, Nwrite;
  int dims[4];
  size_t count[4] = {0};
  size_t start[4] = {0};
  int varids[3];

  string varnames[3];
  int nc_group, nc_type;
  dim3 dG, dB;
  Parameters* pars_;
  Grids* grids_;
  NetCDF* ncdf_;
  Nonlinear* nonlinear_;
  GradPerp* grad_perp_;

  float *fXY;
  float *f_h;
  float *cpu;
};

class MomentsDiagnostic {
 public:
  MomentsDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf, string varname);
  ~MomentsDiagnostic() {
    free(f_h);
    free(cpu);
    if(pars_->nonlinear_mode) {
      free(fXY_h);
      free(cpuXY);
    }
  }
  void calculate_and_write(MomentsG** G, Fields* f, cuComplex* tmp);
 protected:
  //virtual void calculate(MomentsG** G, Fields* f, cuComplex* f_h, cuComplex* tmp_d) = 0;
  virtual void calculate(MomentsG** G, Fields* f, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d) = 0;
  void dealias_and_reorder(cuComplex* fold, float* fnew);
  void dealias_and_reorder_XY(float* fold, float* fnew);

  int ndim, N, Nwrite;
  int dims[6];
  size_t count[6] = {0};
  size_t start[6] = {0};
  size_t dummy_count[6] = {0};
  size_t dummy_start[6] = {0};
  int varid;

  int ndimXY, NXY;
  int dimsXY[5];
  size_t countXY[5] = {0};
  size_t startXY[5] = {0};
  size_t dummy_countXY[5] = {0};
  size_t dummy_startXY[5] = {0};
  int varidXY;

  int nc_group, nc_type;
  dim3 dG, dB;
  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;
  NetCDF* ncdf_;
  Nonlinear* nonlinear_;
  GradPerp* grad_perp_;

  cuComplex *f_h;
  float *cpu;

  float *fXY;
  float *fXY_h;
  float *cpuXY;
  bool skipWrite;
};

class DensityDiagnostic : public MomentsDiagnostic {
 public:
  DensityDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf);
  void calculate(MomentsG** G, Fields* f, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d);
};

class UparDiagnostic : public MomentsDiagnostic {
 public:
  UparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf);
  void calculate(MomentsG** G, Fields* f, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d);
};

class TparDiagnostic : public MomentsDiagnostic {
 public:
  TparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf);
  void calculate(MomentsG** G, Fields* f, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d);
};

class TperpDiagnostic : public MomentsDiagnostic {
 public:
  TperpDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf);
  void calculate(MomentsG** G, Fields* f, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d);
};

class ParticleDensityDiagnostic : public MomentsDiagnostic {
 public:
  ParticleDensityDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf);
  void calculate(MomentsG** G, Fields* f, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d);
};

class ParticleUparDiagnostic : public MomentsDiagnostic {
 public:
  ParticleUparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf);
  void calculate(MomentsG** G, Fields* f, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d);
};

class ParticleUperpDiagnostic : public MomentsDiagnostic {
 public:
  ParticleUperpDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf);
  void calculate(MomentsG** G, Fields* f, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d);
};

class ParticleTempDiagnostic : public MomentsDiagnostic {
 public:
  ParticleTempDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf);
  void calculate(MomentsG** G, Fields* f, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d);
};
