
#include "grad_perp.h"
#include "get_error.h"

exb_kernel::exb_kernel(Grids* grids, int batch_size, int mem_size, float* phasefac, float* minusphasefac) // phasefac is a function of ky and time.
  : grids_(grids), batch_size_(batch_size), mem_size_(mem_size), tmp(nullptr)
{

}

// Updates kx star and phasefac for flow shear.
void exb_kernel::kxshift(float* kx_shift, int* jump, float* ky,float* xgrid, float* phasefac float g_exb, double dt) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();

  float dkx = (float) 1./X0_d;
  
  if(idy<ny/2+1) {
    // Idea is that we track the difference between kx_star = kx(t=0) - ky gamma_E time and kx_bar = the nearest kx on grid. We need this for the phase factor in the FFT. Additionally, jump tells us how to shift ikx in the function shiftField.
    kx_shift[idy] = kx_shift[idy] - ky[idy]*g_exb*dt;      // 
    jump[idy] = roundf(kx_shift[idy]/dkx);                 //roundf() is C equivalent of f90 nint(). jump*dkx gives the closest kx on the grid, which is kxbar.
    kx_shift[idy] = kx_shift[idy] - jump[idy]*dkx; // kx_star - kx_bar, which multiplied by x, is the phase.
    phasefac[idy] = kx_shift[idy]*xgrid; // this depends on both iky and idx. Should I calculate phase here?
  }
}
// Subtleties: 1 extra padding in ky, normalization for theta0 and gexb, not letting kx go to ±inf, restarting kperp, updating kperp, kperp and kx at different Runge-Kutta timesteps

void exb_kernel::shiftField(cuComplex* field, int* jump)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy(); 
  unsigned int idz = get_idz();
  
  if(idx<nx && idy<(ny/2+1) && idz<nz) {
    unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
    
    int ikx_shifted = get_ikx(idx) - jump[idy];
    
    //if field is sheared beyond resolution or mask, set field to zero
    if( ikx_shifted > (nx-1)/3 || ikx_shifted < -(nx-1)/3 ) {
      field[index].x = 0.;
      field[index].y = 0.;
    }
    else {
      unsigned int idx_shifted;
      if(ikx_shifted < 0)       
        idx_shifted = ikx_shifted + nx;
      else idx_shifted = ikx_shifted;	
    
      unsigned int index_shifted = idy + (ny/2+1)*idx_shifted + nx*(ny/2+1)*idz;
      
      field[index] = field[index_shifted];	
	
    }
  }
}
