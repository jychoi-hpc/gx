#pragma once
#include "parameters.h"
#include "grids.h"
#include "cufft.h"

class Fields {
  public:
  Fields(Parameters *pars, Grids *grids);
  ~Fields();
  
  cuComplex * phi    ;
  cuComplex * phi_h  ;
  cuComplex * apar   ;
  cuComplex * apar_h ;
  cuComplex * apar_ext   ;
  cuComplex * apar_ext_h ;
  float *apar_ext_realspace_h;
  float *apar_ext_realspace;
  cuComplex * bpar   ;
  cuComplex * bpar_h ;

  cuComplex * ne ;
  cuComplex * ue ;
  cuComplex * Te ;

  cuComplex * ne_h ;
  cuComplex * ue_h ;
  cuComplex * Te_h ;

  void print_phi(void);
  void print_apar(void);
  void print_bpar(void);
  void rescale(float * phi_max);

  inline void copyPhiFrom(Fields* source) {
    cudaMemcpy(phi, source->phi, size_, cudaMemcpyDeviceToDevice);
  }
  inline void copyAparFrom(Fields* source) {
    cudaMemcpy(apar, source->apar, size_, cudaMemcpyDeviceToDevice);
  }
  inline void copyBparFrom(Fields* source) {
    cudaMemcpy(bpar, source->bpar, size_, cudaMemcpyDeviceToDevice);
  }

  inline void copyFrom( Fields* source ) {
    copyPhiFrom( source );
    if( pars_->fapar > 0.0 )
      copyAparFrom( source );
    if( pars_->fbpar > 0.0 )
      copyBparFrom( source );
  }

private:
  const size_t size_;
  const size_t sizeReal_;
  int N;
  Parameters * pars_  ;
  Grids * grids_ ;
};
