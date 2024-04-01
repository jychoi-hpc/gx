#pragma once
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_hermite.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#include "moments.h"
#include "grids.h"
#include <cublas_v2.h>

class HermiteTransform {
 public:

  HermiteTransform(Grids* grids, float vmax=-1.0);
  ~HermiteTransform();
  
  double * get_toGrid() {return toGrid;}
  double * get_toSpectral() {return toSpectral;}
  double * get_roots() {return roots;}
  float get_scale_fac() {return scale_fac;}
  float get_vmax() {return vmax;}

 private:
  Grids * grids_     ; 
  double * toGrid     ;
  double * toSpectral ;
  double * roots      ;
  const int M;
  float vmax;
  float scale_fac;

  void initTransforms(double* toGrid, double* toSpectral, double* roots);
};


