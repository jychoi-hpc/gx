#ifndef GEO_H
#define GEO_H

#include "cufft.h"
#include "parameters.h"
#include "grids.h"
#include "grad_parallel.h" // MFM
#include "device_funcs.h"
#include "get_error.h"

// class GradParallel; // Forward Declaration

class Geometry {
 public:
  Geometry();
  ~Geometry();
  
  float * z_h ;
  float * gbdrift_h ;
  float * grho_h ;
  float * cvdrift_h ;
  float * bmag_h ;
  float * bmagInv_h ;
  float * bgrad_h ;
  float * gds2_h ;
  float * gds21_h ;
  float * gds22_h ;
  float * cvdrift0_h ;
  float * gbdrift0_h ;
  float * jacobian_h ;

  float * z ;
  float * gbdrift ;
  float * grho ;
  float * cvdrift ;
  float * bmag ;
  float * bmagInv ;
  float * bgrad ;
  float * gds2 ;
  float * gds21 ;
  float * gds22 ;
  float * cvdrift0 ;
  float * gbdrift0 ;
  float * jacobian ;

  float * gradpar_arr ;
  float * Rplot ;
  float * Zplot ;
  float * aplot ;
  float * Xplot ;
  float * Yplot ;
  float * Rprime ;
  float * Zprime ;
  float * aprime ;
  float * deltaFL ;

  float drhodpsi;
  float gradpar;
  float bi;
  float aminor;
  float shat;
  bool zero_shat_; 
  
  cuComplex * bmag_complex ;
  float * bgrad_temp ;

  // operator arrays
  float * kperp2 ;
  float * omegad ; // still used in closures. Should be cleaned out. 
  float * cv_d ;
  float * gb_d ; 
  float * kperp2_h ;
  
  void initializeOperatorArrays(Grids* grids);
  void calculate_bgrad(Grids* grids); // MFM

  bool operator_arrays_allocated_;
};

class S_alpha_geo : public Geometry {
 public:
  S_alpha_geo(Parameters* parameters, Grids* grids);

};

class Gs2_geo : public Geometry {
 public:
  Gs2_geo();
};

class geo_nc : public Geometry {
 public:
  geo_nc(Parameters* parameters, Grids* grids);
};

// MFM
class File_geo : public Geometry {
 public:
  File_geo(Parameters* parameters, Grids* grids);

};

//void set_geometry(input_parameters_struct * pars, grids_struct * grids, geometry_coefficents_struct * geo, struct gx_parameters_struct * gxpars);

//void copy_geo_arrays_to_device(geometry_coefficents_struct * geo, geometry_coefficents_struct * geo_h, input_parameters_struct * pars, int Nz);


#endif
