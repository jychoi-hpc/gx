#pragma once

#include "parameters.h"
#include "grids.h"
#include "moments.h"
#include "fields.h"
#include "grad_parallel.h"

class Diagnostics {
 public:
  Diagnostics(Parameters *pars, Grids *grids, Geometry *geo);
  ~Diagnostics();

  bool loop_diagnostics(Moments* moms, Fields* fields, float dt, int counter, float time) ;
  void final_diagnostics(Moments* moms, Fields* fields);

  void writeGridFile(const char* filename); // MFM
  void writeMomOrField(cuComplex* m, const char* filename);
  void writeMomOrFieldKpar(cuComplex* m, const char* filename);
  void writeHLspectrum(cuComplex* ghl, int ikx=-100, int iky=-100);
  void writeGeo();
  void writeGrowthRates();
  void writeTimeHistory(cuComplex* f, float time, int i, int j, int k, FILE* out);
  

 private:
  Fields *fields_old;
  GradParallel* grad_parallel;

  cuDoubleComplex *growth_rates, *growth_rates_h;

  cuComplex *m_h;
  cuComplex *res;

  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;

  int maxThreadsPerBlock_;
  dim3 dimGrid_xy, dimBlock_xy;

  void print_growth_rates_to_screen();

  void HLspectrum(cuComplex* ghl, int ikx=-100, int iky=-100);

  bool checkstop();
 
  float* hlspectrum;
 
  float fluxDenom;

  bool mask;

  char stopfilename_[2000];
  FILE* timefile;
};
