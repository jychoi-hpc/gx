#include "device_funcs.h"
#include "cuda_constants.h"
#include "grids.h"

__device__ unsigned int get_id1(void) {return __umul24(blockIdx.x,blockDim.x)+threadIdx.x;}
__device__ unsigned int get_id2(void) {return __umul24(blockIdx.y,blockDim.y)+threadIdx.y;}
__device__ unsigned int get_id3(void) {return __umul24(blockIdx.z,blockDim.z)+threadIdx.z;}


// use stirling's approximation
__host__ __device__ float factorial(int m) {
  if(m <2) return 1.;
  if(m==2) return 2.;
  if(m==3) return 6.;
  if(m==4) return 24.;
  if(m==5) return 120.;
  if(m==6) return 720.;
  else return sqrtf(2.*M_PI*m)*powf(m,m)*expf(-m)*(1.+1./(12.*m)+1./(288.*m*m));
}

__host__ __device__ float Jflr(int l, float b, bool enforce_JL_0) {
  if (l>30) return 0.; // protect against underflow for single precision evaluation

  if (l<0) return 0.;

  else if (l>=nl && enforce_JL_0) return 0;

  else return (1./factorial(l))*pow(-0.5*b, l)*expf(-b/2.);
}

__host__ __device__ float g0(float b) {

  float tol = 1.e-7;
  float tk, b2, b2sq;
  float g, x, xi, err;

  if (b < tol) {return 1.0;}

  b2 = 0.5 * b;
  b2sq = b2 * b2;
#ifdef __CUDA_ARCH__
  tk = __expf(-b);
#else
  tk = exp(-b);
#endif
  g = tk;

  x = 1.;
  err = 1.;
  
  while (err > tol) {
    xi = 1./x;
    tk  = tk * b2sq * xi * xi;
    g += tk;
    x  += 1.;
    err = abs(tk/g);
  }
  
  if(g<tol) g=tol; 
  return g;

}

__host__ __device__ float g1(float b) {

  float tol = 1.e-7;
  float tk, b2, b2sq;
  float g, x, xi, xp1i, err;

  if (b < tol) {return 0.0;}

  b2 = 0.5 * b;
  b2sq = b2 * b2;
#ifdef __CUDA_ARCH__
  tk = __expf(-b) * b2;
#else
  tk = exp(-b) * b2;
#endif
  g = tk;

  x = 1.;
  err = 1.;
  
  while (err > tol) {
    xi = 1./x;
    xp1i=1./(1.+x);
    tk  = tk * b2sq * xi * xp1i;
    g += tk;
    x += 1.;
    err = abs(tk/g);
  }
  
  if(g<tol) g=tol; 
  return g;

}

__host__ __device__ float sgam0 (float b) {return sqrt(g0(b));}

__host__ __device__ bool operator>(cuComplex f, cuComplex g)
{
  return f.x*f.x+f.y*f.y > g.x*g.x+g.y*g.y;
}

__host__ __device__ bool operator<(cuComplex f, cuComplex g)
{
  return f.x*f.x+f.y*f.y < g.x*g.x+g.y*g.y;
}


__host__ __device__ cuComplex operator+(cuComplex f, cuComplex g) 
{
  return cuCaddf(f,g);
} 

__host__ __device__ cuComplex operator-(cuComplex f, cuComplex g)
{
  return cuCsubf(f,g);
}  

__host__ __device__ cuComplex operator-(cuComplex f)
{
  cuComplex zero;
  zero.x = 0.;
  zero.y = 0.;
  return cuCsubf(zero,f);
}  
/*
__host__ __device__ cuComplex operator*(float scaler, cuComplex f) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}

__host__ __device__ cuComplex operator*(cuComplex f, float scaler) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}
*/
__host__ __device__ cuComplex operator*(float scaler, cuComplex f) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}

__host__ __device__ cuComplex operator*(cuComplex f, float scaler) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}

__host__ __device__ cuComplex operator*(cuComplex f, cuComplex g)
{
  return cuCmulf(f,g);
}

__host__ __device__ cuDoubleComplex operator*(cuDoubleComplex f, cuDoubleComplex g)
{
  return cuCmul(f,g);
}

__host__ __device__ cuComplex operator/(cuComplex f, float scaler)
{
  cuComplex result;
  result.x = f.x / scaler;
  result.y = f.y / scaler;
  return result;
}

__host__ __device__ cuComplex operator/(cuComplex f, cuComplex g) 
{
  return cuCdivf(f,g);
}

__host__ __device__ cuDoubleComplex operator/(cuDoubleComplex f, cuDoubleComplex g) 
{
  return cuCdiv(f,g);
}

__host__ __device__ int get_ikx(int idx) {

  if (idx < nx/2+1)
    return idx;
  else
    return idx-nx;
}

__device__ bool not_fixed_eq(int idxyz) {
  int idxy_fixed = iky_fixed + ikx_fixed*nyc; // should be idx_fixed??
  if ( idxyz%(nx*nyc) == idxy_fixed )
    return false;
  else
    return true;
}

// MM // Introduced __host__ __device__ functions to the add_scaled_singlemom_x kernels

__host__ __device__ void add_scaled_singlemom(cuComplex* res, double c1, cuComplex* m1, double c2, cuComplex* m2, double c3, cuComplex* m3, unsigned int idxyz) { 
  if(idxyz<nx*nyc*nz) res[idxyz] = c1*m1[idxyz] + c2*m2[idxyz] + c3*m3[idxyz];
}

__host__ __device__ void add_scaled_singlemom(cuComplex* res, cuComplex c1, cuComplex* m1, cuComplex c2, cuComplex* m2, unsigned int idxyz) { 
  if(idxyz<nx*nyc*nz) res[idxyz] = c1*m1[idxyz] + c2*m2[idxyz];
}

__host__ __device__ void add_scaled_singlemom(cuComplex* res, double c1, cuComplex* m1, double c2, cuComplex* m2, unsigned int idxyz) { 
  if(idxyz<nx*nyc*nz) res[idxyz] = c1*m1[idxyz] + c2*m2[idxyz];
}

__global__ void add_scaled_singlemom_kernel(cuComplex* res, double c1, cuComplex* m1, double c2, cuComplex* m2)
{
  unsigned int idxyz = get_id1();
  add_scaled_singlemom(res, c1, m1, c2, m2, idxyz);
}

__global__ void add_scaled_singlemom_kernel(cuComplex* res, double c1, cuComplex* m1, double c2, cuComplex* m2, double c3, cuComplex* m3)
{
  unsigned int idxyz = get_id1();
  add_scaled_singlemom(res, c1, m1, c2, m2, c3, m3, idxyz);
}

__global__ void add_scaled_singlemom_kernel(cuComplex* res, cuComplex c1, cuComplex* m1, cuComplex c2, cuComplex* m2)
{
  unsigned int idxyz = get_id1();
  add_scaled_singlemom(res, c1, m1, c2, m2, idxyz);
}

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2)
{
  // should this be masked? tested for un/masked?
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
	          + c2 * m2[ig];
        }
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, bool eqfix)
{
  unsigned int idxyz = get_id1();
  if(not_fixed_eq(idxyz) && idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
	          + c2 * m2[ig];
        }
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2,
				  double c3, cuComplex* m3)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
	          + c2 * m2[ig]
	          + c3 * m3[ig];
        }
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2,
				  double c3, cuComplex* m3, bool eqfix)
{
  unsigned int idxyz = get_id1();
  if(not_fixed_eq(idxyz) && idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
	          + c2 * m2[ig]
	          + c3 * m3[ig];
        }
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
	          + c2 * m2[ig] 
	          + c3 * m3[ig]
	          + c4 * m4[ig];
        }
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4, bool eqfix)
{
  unsigned int idxyz = get_id1();
  if(not_fixed_eq(idxyz) && idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
	          + c2 * m2[ig] 
	          + c3 * m3[ig]
	          + c4 * m4[ig];
        }
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4,
				  double c5, cuComplex* m5)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
   	          + c2 * m2[ig] 
	          + c3 * m3[ig]
	          + c4 * m4[ig] 
	          + c5 * m5[ig];
        }
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4,
				  double c5, cuComplex* m5, bool eqfix)
{
  unsigned int idxyz = get_id1();
  if(not_fixed_eq(idxyz) && idxyz<nx*nyc*nz) { 
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
   	          + c2 * m2[ig] 
	          + c3 * m3[ig]
	          + c4 * m4[ig] 
	          + c5 * m5[ig];
        }
      }
    }
  }
}

__global__ void scale_kernel(cuComplex* res, cuComplex* mom, double scalar)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = scalar*mom[ig];
        }
      }
    }
  }
}

__global__ void scale_kernel(cuComplex* res, cuComplex* mom, cuComplex scalar)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = scalar*mom[ig];
        }
      }
    }
  }
}

__host__ __device__ void scale_singlemom(cuComplex* res, cuComplex* mom, cuComplex scalar, unsigned int idxyz)
{
  if(idxyz<nx*nyc*nz) res[idxyz] = scalar*mom[idxyz];
}

__global__ void scale_singlemom_kernel(cuComplex* res, cuComplex* mom, cuComplex scalar)
{
  unsigned int idxyz = get_id1();
  scale_singlemom(res, mom, scalar, idxyz);
}

__global__ void reality_kernel(cuComplex* g) 
{
  for(int s = 0; s < nspecies; s++) {
    for (int lm = threadIdx.z; lm < nm*nl;     lm += blockDim.z) {
      for (int idz = threadIdx.y; idz<nz;     idz += blockDim.y) {
        for (int idx = threadIdx.x; idx<(nx-1)/3+1; idx += blockDim.x) {
          int ig  = nyc*(idx    + nx*idz) + nx*nyc*nz*lm + nx*nyc*nz*nl*nm*s; 
          int ig2 = nyc*(nx-idx + nx*idz) + nx*nyc*nz*lm + nx*nyc*nz*nl*nm*s; 
	  //	  if (ig<nx*nyc*nz*nl*nm*s && ig2<nx*nyc*nz*nl*nm*s) {
	    if(idx==0) {
	      g[ig].x = 0.;
	      g[ig].y = 0.;
	    } else if(idx!=0) {
	      g[ig2].x = g[ig].x;
	      g[ig2].y = -g[ig].y;
	    }
	    //	  }
	}
      }
    }
  }
}

__global__ void reality_singlemom_kernel(cuComplex* mom) 
{
  for (int idz = threadIdx.y; idz<nz; idz+=blockDim.y) {
    for (int idx = threadIdx.x; idx<(nx-1)/3+1; idx+=blockDim.x) {
      unsigned int index  = nyc*(idx    + nx*idz);
      unsigned int index2 = nyc*(nx-idx + nx*idz);
      if(idx==0) {
	mom[index].x = 0.;
	mom[index].y = 0.;
      } else if(idx!=0) {
        mom[index2].x =  mom[index].x;
        mom[index2].y = -mom[index].y;
      }
    }
  }
}

__host__ __device__ bool unmasked(int idx, int idy) {
  int ikx = get_ikx(idx);
  if ( !(idx==0 && idy==0)
       && idy <  (ny-1)/3 + 1
       && idx <   nx                 // both indices must be in range 
       && ikx <  (nx-1)/3 + 1
       && ikx > -(nx-1)/3 - 1
       )
    return true;
  else
    return false;
}

// not the opposite of unmasked b/c indices could simply be out of range
__host__ __device__ bool masked(int idx, int idy) {
  int ikx = get_ikx(idx);
  if ( ( (idx==0 && idy==0) || idy > (ny-1)/3  || ikx > (nx-1)/3 || ikx < -(nx-1)/3 )
       && idx < nx  // index should be in range to be actively masked
       && idy < ny) // index should be in range to be actively masked
    return true;
  else
    return false;
}

__global__ void calc_bgrad(float* bgrad, float* bgrad_temp, float* bmag, float scale)
{
  // BD This will fail for Nz > 1024
  unsigned int idz = get_id1();
  if (idz < nz) bgrad[idz] = ( bgrad_temp[idz] / bmag[idz] ) * scale;
}

__global__ void init_kperp2(float* kperp2, float* kx, float* ky,
			    float* gds2, float* gds21, float* gds22,
			    float* bmagInv, float shat) 
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  float shatInv = 1./shat; // Needs a test for zero

  if (unmasked(idx, idy) && idz < nz) { 
    unsigned int idxyz = idy + nyc*idx + nx*nyc*idz;
    kperp2[idxyz] = ( ky[idy] * ( ky[idy] * gds2[idz] 
                      + 2. * kx[idx] * shatInv * gds21[idz]) 
                      + pow( kx[idx] * shatInv, 2) * gds22[idz] ) 
                      * pow( bmagInv[idz], 2); 
  }
}

__global__ void init_omegad(float* omegad, float* cv_d, float* gb_d, float* kx, float* ky,
			    float* cv, float* gb, float* cv0, float* gb0, float shat) 
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  float shatInv = 1./shat; // BD this needs an exception for shat = 0.

  // cv/gb is the y-directed part of the curvature / grad-B drift
  // cv0/gb0 is the part proportional to the theta_0, aka the x-directed component

  if ( unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = idy + nyc*idx + nx*nyc*idz;
    cv_d[idxyz] = ky[idy] * cv[idz] + kx[idx] * shatInv * cv0[idz] ;     
    gb_d[idxyz] = ky[idy] * gb[idz] + kx[idx] * shatInv * gb0[idz] ;
    omegad[idxyz] = cv_d[idxyz] + gb_d[idxyz];
  }
}

# define H_(XYZ, L, M, S) g[(XYZ) + nx*nyc*nz*(L) + nx*nyc*nz*nl*(M) + nx*nyc*nz*nl*nm*(S)] + Jflr(L,b_s)*phi_
# define G_(XYZ, L, M, S) g[(XYZ) + nx*nyc*nz*(L) + nx*nyc*nz*nl*(M) + nx*nyc*nz*nl*nm*(S)] // H = G, except for m = 0
// C = C(H) but H and G are the same function for all m!=0. Our main array defines g so the correction to produce
// H is only appropriate for m=0. In other words, the usage here is basically handling the delta_{m0} terms
// in a clumsy way
__global__ void Tbar(cuComplex* t_bar, cuComplex* g, cuComplex* phi, float *kperp2)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    cuComplex phi_ = phi[idxyz];
    int index = idxyz;
    t_bar[index] = make_cuComplex(0., 0.);
    float b_s = kperp2[idxyz]; // only species=0, assumes zt, rho2 = 1
    for(int l=0; l<nl; l++) {
      // energy conservation correction for nlaguerre = 1
      if (nl == 1) {
	t_bar[index] = t_bar[index] + sqrtf(2.)*Jflr(l,b_s)*G_(idxyz, l, 2, 0);
      } else {
	t_bar[index] = t_bar[index] + sqrtf(2.)/3.*Jflr(l,b_s)*G_(idxyz, l, 2, 0)
	  + 2./3.*( l*Jflr(l-1,b_s) + 2.*l*Jflr(l,b_s) + (l+1)*Jflr(l+1,b_s) )*H_(idxyz, l, 0, 0);
      }
    }
  }
}

// =============================================================================
// Kernels initially found in linear.cu

// main kernel function for calculating RHS
# define S_G(L, M) s_g[sidxyz + (sDimx)*(L) + (sDimx)*(sDimy)*(M)]
__global__ void rhs_linear(cuComplex *g, cuComplex* phi, 
	cuComplex* upar_bar, cuComplex* uperp_bar, cuComplex* t_bar,
	float* kperp2, float* cv_d, float*gb_d, float* bgrad, float* ky, specie* species,
	cuComplex* rhs_par, cuComplex* rhs)
{
  extern __shared__ cuComplex s_g[]; // aliased below by macro S_G, defined above

  unsigned int idxyz = threadIdx.x + blockIdx.x*blockDim.x;
  const unsigned int idy = idxyz % nyc; 
  const unsigned int idx = idxyz / nyc % nx;
  if(idxyz<nx*nyc*nz && unmasked(idx, idy)) {
    const unsigned int sidxyz = threadIdx.x;
    // these modulo operations are expensive... better way to get these indices?
    //    const unsigned int idy = idxyz % (nx*nyc) % nyc;// BD there is extra here, right?
    const unsigned int idz = idxyz / (nx*nyc);
    
    // shared memory blocks of size blockDim.x * (nl+2) * (nm+4)
    const int sDimx = blockDim.x;
    const int sDimy = nl+2;
  
    // read these values into (hopefully) register memory. 
    // local to each thread (i.e. each idxyz).
    // since idxyz is linear, these accesses are coalesced.
    const cuComplex phi_ = phi[idxyz];
  
    // all threads in a block will likely have same value of idz, so they will be reading same value of bgrad[idz].
    // if bgrad were in shared memory, would have bank conflicts.
    // no bank conflicts for reading from global memory though. 
    const float bgrad_ = bgrad[idz];  
  
    // this is coalesced?
    const cuComplex iky_ = make_cuComplex(0., ky[idy]); 
  
   //#pragma unroll
   for(int is=0; is<nspecies; is++) { // might be a better way to handle species loop here...
    specie s = species[is];

    // species-specific constants
    const float vt_ = s.vt;
    const float zt_ = s.zt;
    const float nu_ = s.nu_ss; 
    const float tprim_ = s.tprim;
    const float fprim_ = s.fprim;
    const float b_s = kperp2[idxyz] * s.rho2;
    const cuComplex icv_d_s = 2. * s.tz * make_cuComplex(0., cv_d[idxyz]);
    const cuComplex igb_d_s = 2. * s.tz * make_cuComplex(0., gb_d[idxyz]);

    // conservation terms (species-specific)
    cuComplex upar_bar_  =  upar_bar[idxyz + is*nx*nyc*nz];
    cuComplex uperp_bar_ = uperp_bar[idxyz + is*nx*nyc*nz];
    cuComplex t_bar_     =     t_bar[idxyz + is*nx*nyc*nz];
  
    // read tile of g into shared mem
    // each thread in the block reads in multiple values of l and m
    for (int m = threadIdx.z; m < nm; m += blockDim.z) {
     for (int l = threadIdx.y; l < nl; l += blockDim.y) {
      int globalIdx = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*is; 
      int sl = l + 1;
      int sm = m + 2;
      S_G(sl, sm) = g[globalIdx];
     }
    }
  
    // this syncthreads is not necessary unless ghosts require information from interior cells
    //    __syncthreads();
  
    // set up ghost cells in m (for all l's)
    for (int l = threadIdx.y; l < nl; l += blockDim.y) {
      int sl = l + 1;
      int sm = threadIdx.z + 2;
      if(sm < 4) {
        // set ghost to zero at low m
        S_G(sl, sm-2) = make_cuComplex(0., 0.);
  
        // set ghost with closures at high m
        S_G(sl, sm+nm) = make_cuComplex(0., 0.);
      }
    }
  
    // set up ghost cells in l (for all m's)
    for (int m = threadIdx.z; m < nm+2; m += blockDim.z) {
      int sm = m + 1; // this takes care of corners...
      int sl = threadIdx.y + 1;
      if(sl < 2) {
        // set ghost to zero at low l
        S_G(sl-1, sm) = make_cuComplex(0., 0.);
  
        // set ghost with closures at high l
        S_G(sl+nl, sm) = make_cuComplex(0., 0.);
      }
    }
  
    __syncthreads();
  
    // stencil (on non-ghost cells)
    for (int m = threadIdx.z; m < nm; m += blockDim.z) {
     for (int l = threadIdx.y; l < nl; l += blockDim.y) {
      int globalIdx = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*is; 
      int sl = l + 1; // offset to get past ghosts
      int sm = m + 2; // offset to get past ghosts
  
      // need to calculate parallel terms separately because need to take derivative via fft 
      rhs_par[globalIdx] = -vt_*( sqrtf(m+1)*S_G(sl,sm+1) + sqrtf(m)*S_G(sl,sm-1) );
  
      // remaining terms
      rhs[globalIdx] = 
       - vt_ * bgrad_ * ( - sqrtf(m+1)*(l+1)*S_G(sl,sm+1) - sqrtf(m+1)* l   *S_G(sl-1,sm+1)  
                          + sqrtf(m  )* l   *S_G(sl,sm-1) + sqrtf(m  )*(l+1)*S_G(sl+1,sm-1) )
  
	- icv_d_s * ( sqrtf((m+1)*(m+2))*S_G(sl,sm+2) + 2.*m*S_G(sl,sm) + sqrtf(m*(m-1))*S_G(sl,sm-2) )
	- igb_d_s * (              (l+1)*S_G(sl+1,sm) + 2.*(l+1)*S_G(sl,sm)          + l*S_G(sl-1,sm) )
  
	- nu_ * ( b_s + 2*l + m ) * ( S_G(sl,sm) );
  
      // add potential, drive, and conservation terms in low hermite moments
      if(m==0) {
        rhs[globalIdx] = rhs[globalIdx] + phi_ * (
            Jflr(l-1,b_s)*(      -l *igb_d_s * zt_ +           tprim_  *l  * iky_ )
	  + Jflr(l,  b_s)*( -2*(l+1)*igb_d_s * zt_ + (fprim_ + tprim_*2*l) * iky_ )
	  + Jflr(l+1,b_s)*(   -(l+1)*igb_d_s * zt_ )
	  + Jflr(l+1,b_s,false)*                              tprim_*(l+1) * iky_ )
	  + nu_ * sqrtf(b_s) * ( Jflr(l, b_s) + Jflr(l-1, b_s) ) * uperp_bar_
	  + nu_ * 2. * ( l*Jflr(l-1,b_s) + 2.*l*Jflr(l,b_s) + (l+1)*Jflr(l+1,b_s) ) * t_bar_ 
	  - nu_ * ( b_s + 2*l ) * Jflr(l, b_s) * phi_ * zt_ ;
	/* Add this last line in because for the m=0 part, w/o worrying about conservation terms:
	
	   dG/dt = - nu ( b + 2*l) H  << equation we are supposed to be solving 
	  = - nu ( b + 2*l) (G + J0 phi Z/T)
	  = - nu ( b + 2*l) G - nu ( b + 2*l) J0 phi Z/T  << equation in our variables
	*/
      }

      if(m==1) {
        rhs_par[globalIdx] = rhs_par[globalIdx] - Jflr(l,b_s)*phi_ * zt_ * vt_;

        rhs[globalIdx] = rhs[globalIdx] - phi_ * (
	          l*Jflr(l,b_s) + (l+1)*Jflr(l+1,b_s) ) * bgrad_ * vt_ * zt_
      		+ nu_ * Jflr(l,b_s) * upar_bar_;
      }
      if(m==2) {
        rhs[globalIdx] = rhs[globalIdx] + phi_ *
	          Jflr(l,b_s) * (-2*icv_d_s * zt_ + tprim_ * iky_)/sqrtf(2) 
		+ nu_ * sqrtf(2) * Jflr(l,b_s) * t_bar_;
      }  
     } // l loop
    } // m loop
  
   } // species loop
  } // idxyz < NxNycNz
}

# define Hzt_(XYZ, L, M, S) g[(XYZ) + nx*nyc*nz*(L) + nx*nyc*nz*nl*(M) + nx*nyc*nz*nl*nm*(S)] + Jflr(L,b_s)*phi_*zt_
//# define G_(XYZ, L, M, S) g[(XYZ) + nx*nyc*nz*(L) + nx*nyc*nz*nl*(M) + nx*nyc*nz*nl*nm*(S)] // H = G, except for m = 0
// C = C(H) but H and G are the same function for all m!=0. Our main array defines g so the correction to produce
// H is only appropriate for m=0. In other words, the usage here is basically handling the delta_{m0} terms
// in a clumsy way
__global__ void conservation_terms(cuComplex* upar_bar, cuComplex* uperp_bar, cuComplex* t_bar,
				   cuComplex* g, cuComplex* phi, float *kperp2, specie* species)
{
  unsigned int idxyz = get_id1();

  if(idxyz<nx*nyc*nz) {
    cuComplex phi_ = phi[idxyz];
    for(int is=0; is<nspecies; is++) {
      const float zt_ = species[is].zt;
      int index = idxyz + nx*nyc*nz*is;
      upar_bar[index] = make_cuComplex(0., 0.);
      uperp_bar[index] = make_cuComplex(0., 0.);
      t_bar[index] = make_cuComplex(0., 0.);
      float b_s = kperp2[idxyz]*species[is].rho2;
      // sum over l
      for(int l=0; l<nl; l++) {
        upar_bar[index] = upar_bar[index] + Jflr(l,b_s)*G_(idxyz, l, 1, is);
        // Hzt_(...) is defined by macro above. Only use H here for m=0. Confusing!
        uperp_bar[index] = uperp_bar[index] + (Jflr(l,b_s) + Jflr(l-1,b_s))*Hzt_(idxyz, l, 0, is);

        // energy conservation correction for nlaguerre = 1
        if (nl == 1) {
            t_bar[index] = t_bar[index] + sqrtf(2.)*Jflr(l,b_s)*G_(idxyz, l, 2, is);
        } else {
            t_bar[index] = t_bar[index] + sqrtf(2.)/3.*Jflr(l,b_s)*G_(idxyz, l, 2, is)
		    + 2./3.*( l*Jflr(l-1,b_s) + 2.*l*Jflr(l,b_s) + (l+1)*Jflr(l+1,b_s) )*Hzt_(idxyz, l, 0, is);
        }
      }
      uperp_bar[index] = uperp_bar[index]*sqrtf(b_s);
    }
  }
}

__global__ void hypercollisions(cuComplex* g, float nu_hyper_l, float nu_hyper_m, int p_hyper_l, int p_hyper_m, cuComplex* rhs) {
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    float scaled_nu_hyp_l = (float) nl * nu_hyper_l;
    float scaled_nu_hyp_m = (float) nm * nu_hyper_m; // scaling appropriate for curvature. Too big for slab
    for(int is=0; is<nspecies; is++) { 
    for (int m = threadIdx.z; m < nm; m += blockDim.z) {
     for (int l = threadIdx.y; l < nl; l += blockDim.y) {
      int globalIdx = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*is; 
      if(m>2 || l>1) {
        rhs[globalIdx] = rhs[globalIdx] -
  	   (scaled_nu_hyp_l*pow((float) l/nl, (float) p_hyper_l)
	   +scaled_nu_hyp_m*pow((float) m/nm, p_hyper_m))*g[globalIdx];
      }
     }
    }
   }
  }
}
// end of kernels from linear.cu
// =============================================================================



// =============================================================================
// beginning of kernels originally in closures.cu

# define LM(L, M) idxyz + nx*nyc*nz*(L) + nx*nyc*nz*nl*(M)
__global__ void beer_toroidal_closures(cuComplex* g, cuComplex* gRhs, float* omegad, cuComplex* nu)
{
  unsigned int idxyz = get_id1();

  if(idxyz<nx*nyc*nz) {

    const cuComplex iomegad = make_cuComplex(0., omegad[idxyz]);
    const float abs_omegad = abs(omegad[idxyz]);

    gRhs[LM(0,2)] = gRhs[LM(0,2)]
      - sqrtf(2)*abs_omegad*( nu[1].x*sqrtf(2)*g[LM(0,2)] + nu[2].x*g[LM(1,0)] )
      - sqrtf(2)* iomegad * ( nu[1].y*sqrtf(2)*g[LM(0,2)] + nu[2].y*g[LM(1,0)] );

    gRhs[LM(1,0)] = gRhs[LM(1,0)]
      - 2.*abs_omegad*( nu[3].x*sqrtf(2)*g[LM(0,2)] + nu[4].x*g[LM(1,0)] )
      - 2.* iomegad * ( nu[3].y*sqrtf(2)*g[LM(0,2)] + nu[4].y*g[LM(1,0)] );

    gRhs[LM(0,3)] = gRhs[LM(0,3)]
      - 1./sqrtf(6)*abs_omegad*( nu[5].x*g[LM(0,1)] + nu[6].x*sqrtf(6)*g[LM(0,3)] + nu[7].x*g[LM(1,1)] )
      - 1./sqrtf(6)* iomegad * ( nu[5].y*g[LM(0,1)] + nu[6].y*sqrtf(6)*g[LM(0,3)] + nu[7].y*g[LM(1,1)] );

    gRhs[LM(1,1)] = gRhs[LM(1,1)]
      - abs_omegad*( nu[8].x*g[LM(0,1)] + nu[9].x*sqrtf(6)*g[LM(0,3)] + nu[10].x*g[LM(1,1)] )
      -  iomegad * ( nu[8].y*g[LM(0,1)] + nu[9].y*sqrtf(6)*g[LM(0,3)] + nu[10].y*g[LM(1,1)] );
  }

}

__global__ void smith_perp_toroidal_closures(cuComplex* g, cuComplex* gRhs, float* omegad, cuComplex* Aclos, int q)
{
  unsigned int idxyz = get_id1();
  
  if(idxyz<nx*nyc*nz) {

    const cuComplex iomegad = make_cuComplex(0., omegad[idxyz]);
    const cuComplex abs_omegad = make_cuComplex(abs(omegad[idxyz]),0.);

    int L = nl - 1;

    // apply closure to Lth laguerre equation for all hermite moments
    for(int m=0; m<nm; m++) {
      // calculate closure expression as sum of lower laguerre moments
      cuComplex clos = make_cuComplex(0.,0.);
      for(int l=L; l>=nl-q; l--) {
        clos = clos + (abs_omegad*Aclos[L-l].y + iomegad*Aclos[L-l].x)*g[LM(l,m)];
      }

      gRhs[LM(L,m)] = gRhs[LM(L,m)] - (L+1)*clos;
    }
  }

}
// end of kernels from closures.cu
// =============================================================================


// =============================================================================
// kernels from grad_parallel.cu

__device__ void i_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  float *kz = (float*) kzData;
  unsigned int idz = offset / (nx*nyc);
  cuComplex Ikz = make_cuComplex(0., kz[idz]);
  ((cuComplex*)dataOut)[offset] = Ikz*element/nz;    
}

__device__ void abs_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  float *kz = (float*) kzData;
  unsigned int idz = offset / (nx*nyc);
  ((cuComplex*)dataOut)[offset] = abs(kz[idz])*element/nz;
}

__device__ void i_kz_1d(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  float *kz = (float*) kzData;
  unsigned int idz = offset;
  cuComplex Ikz = make_cuComplex(0., kz[idz]);
  ((cuComplex*)dataOut)[offset] = Ikz*element/nz;
}
extern __managed__ cufftCallbackStoreC i_kz_callbackPtr = i_kz;
extern __managed__ cufftCallbackStoreC i_kz_1d_callbackPtr = abs_kz;
extern __managed__ cufftCallbackStoreC abs_kz_callbackPtr = i_kz_1d;
// =============================================================================


// kernels for grad_parallel_linked.cu
__device__ void i_kzLinked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  /*
  // Could do it this way: 
  // Passed: nLinks; we know nz
  unsigned int nzL = nz*nLinks;
  unsigned int idz = offset % (nzL);
  float zpnLinv = (float) 1./zp*nLinks;
  float kz;
  int j = idz % (nzL/2+1)     
  if (idz < nzL/2+1) {
    kz = (float) idz * zpnLinv;
  } else {
    int idzs = idz-nzL;
    kz = (float) idzs * zpnLinv;
  }
  cuComplex Ikz = make_cuComplex(0., kz);
  float normalization = (float) 1./nzL;
  ((cuComplex*)dataOut)[offset] = Ikz*element*normalization;
  */
  float *kz = (float*) kzData;
  int nLinks = (int) lrintf(1./(zp*kz[1]));
  unsigned int idz = offset % (nz*nLinks);
  cuComplex Ikz = make_cuComplex(0., kz[idz]);
  float normalization = (float) 1./(nz*nLinks);
  ((cuComplex*)dataOut)[offset] = Ikz*element*normalization;
}

__device__ void abs_kzLinked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  float *kz = (float*) kzData;
  int nLinks = (int) lrintf(1./(zp*kz[1]));
  unsigned int idz = offset % (nz*nLinks);
  float normalization = (float) 1./(nz*nLinks);
  ((cuComplex*)dataOut)[offset] = abs(kz[idz])*element*normalization;
}

__global__ void init_kzLinked(float* kz, int nLinks)
{
  for(int i=0; i<nz*nLinks; i++) {
    if(i<nz*nLinks/2+1) {
      kz[i] = (float) i/(zp*nLinks);
    } else {
      kz[i] = (float) (i-nz*nLinks)/(zp*nLinks);
    }
  }
}

__global__ void linkedCopy(cuComplex* G, cuComplex* G_linked, int nLinks, int nChains, int* ikx, int* iky, int nMoms)
{
  unsigned int idz = get_id1();
  unsigned int idk = get_id2();
  unsigned int idlm = get_id3();

  if(idz<nz && idk<nLinks*nChains && idlm<nMoms) {
    unsigned int idlink = idz + nz*idk + nz*nLinks*nChains*idlm;
    unsigned int globalIdx = iky[idk] + nyc*ikx[idk] + idz*nx*nyc + idlm*nx*nyc*nz;

    // NRM: seems hopeless to make these accesses coalesced. how bad is it?
    G_linked[idlink] = G[globalIdx];
  }
}

__global__ void linkedCopyBack(cuComplex* G_linked, cuComplex* G, int nLinks, int nChains, int* ikx, int* iky, int nMoms)
{
  unsigned int idz = get_id1();
  unsigned int idk = get_id2();
  unsigned int idlm = get_id3();

  if(idz<nz && idk<nLinks*nChains && idlm<nMoms) {
    unsigned int idlink = idz + nz*idk + nz*nLinks*nChains*idlm;
    unsigned int globalIdx = iky[idk] + nyc*ikx[idk] + idz*nx*nyc + idlm*nx*nyc*nz;

    G[globalIdx] = G_linked[idlink];
  }
}

extern __managed__ cufftCallbackStoreC i_kzLinked_callbackPtr = i_kzLinked;
extern __managed__ cufftCallbackStoreC abs_kzLinked_callbackPtr = abs_kzLinked;
// =============================================================================

// previously from grad_perp.cu
__device__ cuComplex i_kx(void *dataIn, size_t offset, void *kxData, void *sharedPtr)
{
  float *kx = (float*) kxData;
  unsigned int idx = offset / nyc % nx;
  cuComplex Ikx = make_cuComplex(0., kx[idx]);
  return Ikx*((cuComplex*)dataIn)[offset];
}

__device__ cuComplex i_ky(void *dataIn, size_t offset, void *kyData, void *sharedPtr)
{
  float *ky = (float*) kyData;
  unsigned int idy = offset % nyc; 
  cuComplex Iky = make_cuComplex(0., ky[idy]);
  return Iky*((cuComplex*)dataIn)[offset];
}

__device__ void mask_and_scale(void *dataOut, size_t offset, cufftComplex element, void *data, void * sharedPtr)
{
  unsigned int idx = offset / nyc % nx;
  unsigned int idy = offset % nyc; 
  if (masked(idx, idy)) {
    ((cuComplex*)dataOut)[offset].x = 0.;
    ((cuComplex*)dataOut)[offset].y = 0.;
  } else {
    // scale
    ((cuComplex*)dataOut)[offset] = element/(nx*ny);
  }
}

extern __managed__ cufftCallbackLoadC i_kx_callbackPtr = i_kx;
extern __managed__ cufftCallbackLoadC i_ky_callbackPtr = i_ky;
extern __managed__ cufftCallbackStoreC mask_and_scale_callbackPtr = mask_and_scale;

// =============================================================================

// previously defined in nonlinear.cu
__global__ void J0phiToGrid(cuComplex* J0phi, cuComplex* phi, float* kperp2,
			    float* muB, float rho2_s)
{
  unsigned int idxyz = get_id1();
  unsigned int J = (3*nl/2-1);

  if(idxyz<nx*nyc*nz) {
    for (int j = threadIdx.y; j < J; j += blockDim.y) {
      J0phi[idxyz + nx*nyc*nz*j] = j0f(sqrtf(2. * muB[j] * kperp2[idxyz]*rho2_s)) * phi[idxyz];
    }
  }
}

__global__ void bracket(float* g_res, float* dg_dx, float* dJ0phi_dy,
			float* dg_dy, float* dJ0phi_dx, float kxfac)
{
  unsigned int idxyz = get_id1();
  unsigned int J = (3*nl/2-1);

  if(idxyz<nx*ny*nz) {
    for (int j = threadIdx.y; j < J; j += blockDim.y) {
      unsigned int ig = idxyz + nx*ny*nz*j;

      g_res[ig] = ( dg_dx[ig] * dJ0phi_dy[ig] - dg_dy[ig] * dJ0phi_dx[ig] ) * kxfac;

    }
  }
}
// =============================================================================

// previously in diagnostics.cu
__global__ void growthRates(cuComplex *phi, cuComplex *phiOld, double dt, cuComplex *omega)
{ // MM //
  unsigned int idxy = get_id1();
  cuComplex i_dt = make_cuComplex(0., (float) 1./dt);
  int J = nx*nyc;
  int IG = (int) nz/2 ;
  
  if ( idxy<J && idxy > 0) {
    if (abs(phi[idxy+J*IG].x)!=0 && abs(phi[idxy+J*IG].y)!=0) {
      cuComplex ratio = phi[ idxy + J*IG ] / phiOld[ idxy + J*IG ];
      
      cuComplex logr;
      logr.x = (float) log(cuCabsf(ratio));
      logr.y = (float) atan2(ratio.y,ratio.x);
      omega[idxy] = logr*i_dt;
    } else {
      omega[idxy].x = 0.;
      omega[idxy].y = 0.;
    }
  }
}

// sum over ky, kz, z with flux surface average
__global__ void volume_average(float* res, cuComplex* f, cuComplex* g, float* jacobian, float fluxDenomInv, int ikx, int iky) {
  // reduction code follows https://github.com/parallel-forall/code-samples/blob/master/posts/parallel_reduction_with_shfl/device_reduce_atomic.h
  // device_reduce_atomic_kernel
  float sum = 0.;
  cuComplex fg;
  for(int idxyz=blockIdx.x*blockDim.x+threadIdx.x;idxyz<nx*nyc*nz;idxyz+=blockDim.x*gridDim.x) {
    unsigned int idy = idxyz % nyc; 
    unsigned int idx = idxyz % (nx*nyc) / nyc; 
    unsigned int idz = idxyz / (nx*nyc);
    float fac=2.;
    if(idy==0) fac = 1.0;

    if(ikx<0 && iky<0) { // default: sum over all k's
      if(idy>0 || idx>0) {
        fg = cuConjf(f[idxyz])*g[idxyz]*jacobian[idz]*fac*fluxDenomInv;
        sum += fg.x;
      }
    } else {
      if(idy==iky && idx==ikx) {
        fg = cuConjf(f[idxyz])*g[idxyz]*jacobian[idz]*fac*fluxDenomInv;
        sum += fg.x;
      }
    }
  }
  atomicAdd(res,sum);
}

__global__ void get_pzt (float* primary, float* secondary, float* tertiary, cuComplex* phi, cuComplex* tbar)
{
  float Psum = 0.;
  float Zsum = 0.;
  float Tsum = 0.;
  
  cuComplex P2, T2;
  for (int idxyz=blockIdx.x*blockDim.x+threadIdx.x;idxyz<nx*nyc*nz;idxyz+=blockDim.x*gridDim.x) {

    unsigned int idy = idxyz % nyc; 
    unsigned int idx = idxyz / nyc % nx;

    if ( unmasked(idx, idy)) {

      // Ultimately:
      // For this xyz, get:

      // phi_zonal(kx) = 0.
      // phi_zonal(kx) = int dz Phi (kx, ky=0, z)
      // secondary = sum phi_zonal**2 (kx)
      //
      // primary = 0.
      // primary = sum Phi(kx=0, ky, z)**2
      //
      // tertiary = 0.
      // tertiary = sum (Phi(kx, ky=0, z)**2 ) - secondary

      // for now:
      // secondary = sum Phi (kx, ky=0, z)**2
      // primary   = sum tbar(kx=0, ky, z)**2
      // tertiary  = sum tbar(kx!=0,ky, z)**2

      // Caution: missing all geometry 
      
      float fac = 2.;
      if(idy==0) fac = 1.0;
      
      P2 = cuConjf(phi[idxyz])*phi[idxyz]*fac;
      T2 = cuConjf(tbar[idxyz])*tbar[idxyz]*fac; // assumes main species only

      if (idx==0) {
	Psum = T2.x;   atomicAdd(primary, Psum);       // P2
      } else {
	Tsum = T2.x;   atomicAdd(tertiary, Tsum);       // T2
      }
            
      if (idy==0) {
	Zsum = P2.x;   atomicAdd(secondary,Zsum);       // Z2
      }
    }
  }
}

# define G0s_(XYZ, L, M) g[(XYZ) + nx*nyc*nz*(L) + nx*nyc*nz*nl*(M)]
__global__ void heat_flux(float* qflux, cuComplex* phi, cuComplex* g, float* ky, 
                          float* jacobian, float fluxDenomInv, float *kperp2, float rho2_s, 
                          int ikx, int iky)
{
  float sum = 0.;
  cuComplex fg;
  for(int idxyz=blockIdx.x*blockDim.x+threadIdx.x;idxyz<nx*nyc*nz;idxyz+=blockDim.x*gridDim.x) {
    //  float sum = 0.; // This was being zeroed out before the loop. It should be zero for each trip since we atomicadd each trip

    unsigned int idy = idxyz % nyc; 
    unsigned int idx = idxyz / nyc % nx;

    if ( unmasked(idx, idy) && idy > 0) {

      unsigned int idz = idxyz / (nx*nyc);
      cuComplex vE_r = make_cuComplex(0., ky[idy]) * phi[idxyz];
      
      float b_s = kperp2[idxyz]*rho2_s;
      
      // sum over l
      cuComplex p_bar = make_cuComplex(0.,0.);
      for(int l=0; l<nl; l++) {
	// G0s_(...) is defined by macro above
	p_bar = p_bar + 1./sqrtf(2.)*Jflr(l,b_s)*G0s_(idxyz, l, 2)
	  + ( l*Jflr(l-1,b_s) + (2.*l+1.5)*Jflr(l,b_s) + (l+1)*Jflr(l+1,b_s) )*G0s_(idxyz, l, 0);
      }
      
      float fac = 2.;
      if(idy==0) fac = 1.0;
      
      if(ikx<0 && iky<0) { // default: sum over all relevant k's
	fg = cuConjf(vE_r)*p_bar*jacobian[idz]*fac*fluxDenomInv;
	sum = fg.x;
      } else { // single mode specified by ikx, iky
	if(idy==iky && idx==ikx) {
	  fg = cuConjf(vE_r)*p_bar*jacobian[idz]*fac*fluxDenomInv;
	  sum = fg.x;
	}
      }
      atomicAdd(qflux,sum);
    }
  }
}
// =============================================================================

// previously in smith_par_closure.cu
/* Kernel to cast cuDoubleComplex array to a cuComplex array (calculation is done with double precision
   and then converted to single precision for use in the simulation */
__global__ void castDoubleToFloat(cuDoubleComplex *array_d, cuComplex *array_f, int size) {
  for (int i = 0; i < size; i++) array_f[i] = cuComplexDoubleToFloat(array_d[i]);
}
// =============================================================================

// previously in forcing.cu
__global__ void stirring_kernel(cuComplex force, cuComplex *moments, int forcing_index) {
    moments[forcing_index] = moments[forcing_index] + force;
}

void generate_random_numbers(float *random_real, float *random_imag, float forcing_amp_, float dt) {

  // Box-Muller transform to generate random normal variables
  float ran_amp = ( (float) rand()) / ((float) RAND_MAX + 1.0 );

  // dt term in timestepper scheme accounted for in amp
  float amp = sqrt(abs(forcing_amp_*dt*log(ran_amp)));
  float phase = M_PI*(2.0*( (float) rand()) / ((float) RAND_MAX + 1.0 ) -1.0);

  *random_real = amp*cos(phase);
  *random_imag = amp*sin(phase);
}
// =============================================================================

// previously in grids.cu
__global__ void kInit(float* kx, float* ky, float* kz, float X0, float Y0, int Zp) 
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  if(id<nyc) { 
    ky[id] = (float) id/Y0;
  }
  if(id<nx/2+1) {
    kx[id] = (float) id/X0;
  } else if (id<nx) {
    kx[id] = (float) (id - nx)/X0;
  }
  if(id<(nz/2+1)) {
    kz[id] = (float) id/Zp;
  } else if(id<nz) {
    kz[id] = (float) (id - nz)/Zp;
  }
}
// =============================================================================