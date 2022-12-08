/*__device__ int ikx_indexed(int idx) {
  if( idx<(2*(nx/3)+1)/2+1 )
    return idx + (2*(nx/3)+1)/2+1;  //is this right???
  else
    return idx - (2*(nx/3)+1)/2;
}*/


__global__ void kxshift(float* kx_shift, int* jump, float* ky, float g_exb, float avgdt) 
{
  unsigned int idy = get_idy();
  
  float dkx = (float) 1./X0_d;
  
  if(idy<ny/2+1) {
    kx_shift[idy] = kx_shift[idy] - ky[idy]*g_exb*avgdt;      // this is kx star
    jump[idy] = roundf(kx_shift[idy]/dkx);                 //roundf() is C equivalent of f90 nint()
    kx_shift[idy] = kx_shift[idy] - jump[idy]*dkx; // this is kbar
    phasefac[idy] = kx_shift - kx_bar;
  }
}

// Subtleties: 1 extra padding in ky, normalization for theta0 and gexb, not letting kx go to ±inf, restarting kperp, updating kperp, kperp and kx at different Runge-Kutta timesteps

__global__ void shiftField(cuComplex* field, int* jump)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy(); 
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idx<nx && idy<(ny/2+1) && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      int ikx_shifted = get_ikx(idx) - jump[idy];
      
      //if field is sheared beyond resolution or mask, set field to zero
      if( ikx_shifted > (nx-1)/3 || ikx_shifted < -(nx-1)/3 ) {
        field[index].x = 0.;
	field[index].y = 0.;
      }
      //otherwise 	
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
  
}


/*

__global__ void shiftField(cuComplex* field, int* jump) 
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy(); 
  unsigned int idz = get_idz();
  
  int nx_unmasked = 2*(nx/3)+1
  int ny_unmasked = (ny-1)/3+1
  
  if(nz<=zthreads) {
    if(idy>0 && idy<ny_unmasked && idz<nz) {     	      
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(jump[idy] < 0) {
        if(idx<nx_unmasked+jump[idy]) {
          
	  int index_shifted = ikx_indexed(idx-jump[idy]);
	  
	  ikx_indexed(idx) - jump[idy]
	  
	  field[index] = field[idy + (ny/2+1)*idx_from + nx*(ny/2+1)*idz];
	}
	if(idx>=nx_unmasked+jump[idy] && idx<nx_unmasked) {
	  field[idy + (ny/2+1)*idx + nx*(ny/2+1)*idz].x = 0;
	  field[idy + (ny/2+1)*ikx_indexed(idx) + nx*(ny/2+1)*idz].y = 0;
	}
      }
      if(jump[idy] > 0) {
        if(idx<jump[idy]) {
          field[idy + (ny/2+1)*ikx_indexed(idx) + nx*(ny/2+1)*idz].x = 0;
	  field[idy + (ny/2+1)*ikx_indexed(idx) + nx*(ny/2+1)*idz].y = 0;
	}
	if(idx>=jump[idy] && idx<(2*(nx/3)+1)) {
	  int idx_to = ikx_indexed(idx);
	  int idx_from = ikx_indexed(idx-jump[idy]);

	  field[idy + (ny/2+1)*idx_to + nx*(ny/2+1)*idz] = field[idy + (ny/2+1)*idx_from + nx*(ny/2+1)*idz];
	}
      }
    }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
     if(idy>0 && idy<(ny-1)/3 && idz<zthreads) {     	
        unsigned int IDZ = idz + i*zthreads;
	
	if(jump[idy] < 0) {
          if(idx<(2*(nx/3)+1)+jump[idy]) {
            int idx_to = ikx_indexed(idx);
	    int idx_from = ikx_indexed(idx-jump[idy]);

	    field[idy + (ny/2+1)*idx_to + nx*(ny/2+1)*IDZ] = field[idy + (ny/2+1)*idx_from + nx*(ny/2+1)*IDZ];
	  }
	  if(idx>=(2*(nx/3)+1)+jump[idy] && idx<(2*(nx/3)+1)) {
	    field[idy + (ny/2+1)*ikx_indexed(idx) + nx*(ny/2+1)*IDZ].x = 0;
	    field[idy + (ny/2+1)*ikx_indexed(idx) + nx*(ny/2+1)*IDZ].y = 0;
	  }
	}
	if(jump[idy] > 0) {
          if(idx<jump[idy]) {
            field[idy + (ny/2+1)*ikx_indexed(idx) + nx*(ny/2+1)*IDZ].x = 0;
	    field[idy + (ny/2+1)*ikx_indexed(idx) + nx*(ny/2+1)*IDZ].y = 0;
	  }
	  if(idx>=jump[idy] && idx<(2*(nx/3)+1)) {
	    int idx_to = ikx_indexed(idx);
	    int idx_from = ikx_indexed(idx-jump[idy]);

	    field[idy + (ny/2+1)*idx_to + nx*(ny/2+1)*IDZ] = field[idy + (ny/2+1)*idx_from + nx*(ny/2+1)*IDZ];
	  }
	}
      }
    }
  }    
    
}*/	
      
    
      
