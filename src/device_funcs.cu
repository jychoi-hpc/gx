#include "device_funcs.h"
#include "cuda_constants.h"
#include <stdio.h>
 
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

__device__ float Jflr(int l, float b, bool enforce_JL_0) {
  if (l>30) return 0.; // protect against underflow for single precision evaluation

  if (l<0) return 0.;
  else if (l>=nl && enforce_JL_0) return 0;
  else return 1./factorial(l)*pow(-0.5*b, l)*expf(-b/2.); // Assumes <J_0> = exp(-b/2)
}

__host__ __device__ float g0(float b) {

  float tol = 1.e-7;
  float tk, b2, b2sq;
  float g, x, xi, err;

  if (b < tol) {return 1.0;}

  b2 = 0.5 * b;
  b2sq = b2 * b2;
  tk = expf(-b);
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
  tk = expf(-b) * b2;
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
__host__ __device__ bool operator>(cuComplex f, cuComplex g) { return f.x*f.x+f.y*f.y > g.x*g.x+g.y*g.y; }
__host__ __device__ bool operator<(cuComplex f, cuComplex g) { return f.x*f.x+f.y*f.y < g.x*g.x+g.y*g.y; }
__host__ __device__ cuComplex operator+(cuComplex f, cuComplex g) { return cuCaddf(f,g); }
__host__ __device__ cuComplex operator-(cuComplex f, cuComplex g) { return cuCsubf(f,g); }

__host__ __device__ cuComplex operator-(cuComplex f)
{
  cuComplex zero;
  zero.x = 0.; zero.y = 0.;
  return cuCsubf(zero,f);
}  

__host__ __device__ cuComplex operator*(float scale, cuComplex f) 
{
  cuComplex result;
  result.x = scale*f.x;
  result.y = scale*f.y;
  return result;
}

__host__ __device__ cuComplex operator*(cuComplex f, float scale) 
{
  cuComplex result;
  result.x = scale*f.x;
  result.y = scale*f.y;
  return result;
}

__host__ __device__ cuComplex operator*(cuComplex f, cuComplex g) { return cuCmulf(f,g); }
__host__ __device__ cuDoubleComplex operator*(cuDoubleComplex f, cuDoubleComplex g) { return cuCmul(f,g); }

__host__ __device__ cuComplex operator/(cuComplex f, float scale)
{
  cuComplex result;
  result.x = f.x / scale;
  result.y = f.y / scale;
  return result;
}

__host__ __device__ cuComplex operator/(cuComplex f, cuComplex g) { return cuCdivf(f,g); }
__host__ __device__ cuDoubleComplex operator/(cuDoubleComplex f, cuDoubleComplex g) { return cuCdiv(f,g); }

__device__ int get_ikx(int idx) {
  if (idx < nx/2+1)
    return idx;
  else
    return idx-nx;
}

__device__ bool not_fixed_eq(int idxyz) {
  int idxy_fixed = iky_fixed + ikx_fixed*nyc;
  if ( idxyz%(nx*nyc) == idxy_fixed )
    return false;
  else
    return true;
}

__global__ void add_scaled_singlemom_kernel(cuComplex* res, double c1, cuComplex* m1, double c2, cuComplex* m2)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) res[idxyz] = c1*m1[idxyz] + c2*m2[idxyz];
}

__global__ void add_scaled_singlemom_kernel(cuComplex* res, double c1, cuComplex* m1, double c2, cuComplex* m2, double c3, cuComplex* m3)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) res[idxyz] = c1*m1[idxyz] + c2*m2[idxyz] + c3*m3[idxyz];
}

__global__ void add_scaled_singlemom_kernel(cuComplex* res, cuComplex c1, cuComplex* m1, cuComplex c2, cuComplex* m2)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) res[idxyz] = c1*m1[idxyz] + c2*m2[idxyz];
}

/*

!eqfix            false  true    false   true     
not_fixed_eq      true   true    false   false 
 
action            true   true    false    true

bool proceed == (!eqfix || not_fixed_eq);
*/

__global__ void add_scaled_kernel(cuComplex* res,
				  const double c1, cuComplex* m1,
				  const double c2, cuComplex* m2, bool neqfix = true)
{
  unsigned int idxy = get_id1(); 
  if(idxy < nx*nyc) {
    if (neqfix || not_fixed_eq(idxy)) {
      
      unsigned int idz  = get_id2();
      if (idz < nz) {
	
	unsigned int idslm = get_id3(); 
	unsigned int ig = idxy + nx*nyc*idz + nx*nyc*nz*idslm;
	
	res[ig] = c1 * m1[ig] + c2 * m2[ig];
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res,
				  const double c1, cuComplex* m1,
				  const double c2, cuComplex* m2,
				  const double c3, cuComplex* m3, bool neqfix = true)
{
  unsigned int idxy = get_id1(); 
  if(idxy < nx*nyc) {
    if (neqfix || not_fixed_eq(idxy)) {
      
      unsigned int idz  = get_id2();
      if (idz < nz) {
	
	unsigned int idslm = get_id3(); 
	unsigned int ig = idxy + nx*nyc*idz + nx*nyc*nz*idslm;
	
	res[ig] = c1 * m1[ig] + c2 * m2[ig] + c3 * m3[ig];
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res, 
				  const double c1, cuComplex* m1,
				  const double c2, cuComplex* m2, 
				  const double c3, cuComplex* m3,
				  const double c4, cuComplex* m4, bool neqfix = true)
{
  unsigned int idxy = get_id1(); 
  if(idxy < nx*nyc) {
    if (neqfix || not_fixed_eq(idxy)) {
      
      unsigned int idz  = get_id2();
      if (idz < nz) {
	
	unsigned int idslm = get_id3(); 
	unsigned int ig = idxy + nx*nyc*idz + nx*nyc*nz*idslm;
	
	res[ig] = c1 * m1[ig] + c2 * m2[ig] + c3 * m3[ig] + c4 * m4[ig];
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res, 
				  const double c1, cuComplex* m1,
				  const double c2, cuComplex* m2, 
				  const double c3, cuComplex* m3,
				  const double c4, cuComplex* m4,
				  const double c5, cuComplex* m5, bool neqfix = true)
{
  unsigned int idxy = get_id1(); 
  if(idxy < nx*nyc) {
    if (neqfix || not_fixed_eq(idxy)) {
      
      unsigned int idz  = get_id2();
      if (idz < nz) {
	
	unsigned int idslm = get_id3(); 
	unsigned int ig = idxy + nx*nyc*idz + nx*nyc*nz*idslm;
	
	res[ig] = c1 * m1[ig] + c2 * m2[ig] + c3 * m3[ig] + c4 * m4[ig] + c5 * m5[ig];
      }
    }
  }
}


__global__ void scale_kernel(cuComplex* res, cuComplex* mom, const double scalar)
{
  unsigned int idxy = get_id1(); 
  if (idxy < nx*nyc) {
    unsigned int idz  = get_id2();
    if (idz < nz) {
      
      unsigned int idslm = get_id3(); 
      unsigned int ig = idxy + nx*nyc*idz + nx*nyc*nz*idslm;
      
      res[ig] = scalar*mom[ig];
    }
  }
}


__global__ void scale_kernel(cuComplex* res, cuComplex* mom, const cuComplex scalar)
{
  unsigned int idxy = get_id1(); 
  if (idxy < nx*nyc) {
    unsigned int idz  = get_id2();
    if (idz < nz) {
      
      unsigned int idslm = get_id3(); 
      unsigned int ig = idxy + nx*nyc*idz + nx*nyc*nz*idslm;
      
      res[ig] = scalar*mom[ig];
    }
  }
}

__global__ void scale_singlemom_kernel(cuComplex* res, cuComplex* mom, const cuComplex scalar)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) res[idxyz] = scalar*mom[idxyz];
}

__global__ void reality_kernel(cuComplex* g) 
{
  int idx = get_id1();
  int idz = get_id2();
  int idslm = get_id3();
  
  if (idx < (nx-1)/3+1 && idz < nz) {

    int ig  = nyc*(idx    + nx*idz) + nx*nyc*nz*idslm;
    int ig2 = nyc*(nx-idx + nx*idz) + nx*nyc*nz*idslm;
    
    if(idx==0) {
      g[ig].x = 0.;
      g[ig].y = 0.;
    } else {
      g[ig2].x =  g[ig].x;
      g[ig2].y = -g[ig].y;
    }
  }
}

__device__ float Jfac(int il, float b)
{
  return il*Jflr(il-1, b) + (2*il + 1.5)*Jflr(il, b) + (il+1)*Jflr(il+1, b);
}

__device__ bool unmasked(int idx, int idy) {
  int ikx = get_ikx(idx);
  if ( !(idx==0 && idy==0)
       && idy <  (ny-1)/3 + 1
       && idx <   nx                 // both indices must be in range 
       && ikx <  (nx-1)/3 + 1
       && ikx > -(nx-1)/3 - 1)
    return true;
  else
    return false;
}

// not the opposite of unmasked b/c indices could simply be out of range
__device__ bool masked(int idx, int idy) {
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
  // should re-do this with real space/wavenumber indexing and using unmasked function
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    cuComplex phi_ = phi[idxyz];
    //    int index = idxyz;
    t_bar[idxyz] = make_cuComplex(0., 0.);
    float b_s = kperp2[idxyz]; // only species=0, assumes zt, rho2 = 1
    for(int l=0; l<nl; l++) {
      // energy conservation correction for nlaguerre = 1
      if (nl == 1) {
	t_bar[idxyz] = t_bar[idxyz] + sqrtf(2.)*Jflr(l,b_s)*G_(idxyz, l, 2, 0);
      } else {
	t_bar[idxyz] = t_bar[idxyz] + sqrtf(2.)/3.*Jflr(l,b_s)*G_(idxyz, l, 2, 0)
	  + 2./3.*( l*Jflr(l-1,b_s) + 2.*l*Jflr(l,b_s) + (l+1)*Jflr(l+1,b_s) )*H_(idxyz, l, 0, 0);
      }
    }
  }
}

__global__ void growthRates(cuComplex *phi, cuComplex *phiOld, double dt, cuComplex *omega)
{
  unsigned int idxy = get_id1();
  cuComplex i_dt = make_cuComplex(0., (float) 1./dt);
  int J = nx*nyc;

  if (idxy < J) {
    int IG = (int) nz/2 ;
    
    int idx = idxy / nyc % nx;
    int idy = idxy % nyc;
    
    if (unmasked(idx, idy)) {
      if (abs(phi[idxy+J*IG].x)!=0 && abs(phi[idxy+J*IG].y)!=0) {
	cuComplex ratio = phi[ idxy + J*IG ] / phiOld[ idxy + J*IG ];
	
	cuComplex logr;
	logr.x = (float) log(cuCabsf(ratio));
	logr.y = (float) atan2(ratio.y,ratio.x);
	omega[idxy] = logr*i_dt;
	if (isnan(omega[idxy].x)) {omega[idxy].x = 0.; omega[idxy].y = 0.0;}
      } else {
	omega[idxy].x = 0.;
	omega[idxy].y = 0.;
      }
    }
  }
}

__global__ void J0phiToGrid(cuComplex* J0phi, cuComplex* phi, float* kperp2,
			    float* muB, float rho2_s)
{
  unsigned int idxyz = get_id1();
  unsigned int idj = get_id2();
  if(idxyz<nx*nyc*nz && idj<nj) {
    unsigned int ig = idxyz + nx*nyc*nz*idj;
    J0phi[ig] = j0f(sqrtf(2. * muB[idj] * kperp2[idxyz]*rho2_s)) * phi[idxyz];
  }
}

__global__ void castDoubleToFloat (cuDoubleComplex *array_d, cuComplex *array_f, int size) {
  for (int i = 0; i < size; i++) array_f[i] = cuComplexDoubleToFloat(array_d[i]);
}

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

__managed__ cufftCallbackLoadC i_kx_callbackPtr = i_kx;
__managed__ cufftCallbackLoadC i_ky_callbackPtr = i_ky;
__managed__ cufftCallbackStoreC mask_and_scale_callbackPtr = mask_and_scale;

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

__managed__ cufftCallbackStoreC i_kz_callbackPtr = i_kz;
__managed__ cufftCallbackStoreC i_kz_1d_callbackPtr = i_kz_1d;
__managed__ cufftCallbackStoreC abs_kz_callbackPtr = abs_kz;

__global__ void acc(float *a, float *b) {a[0] = a[0] + b[0];}

__global__ void bracket(float* g_res, float* dg_dx, float* dJ0phi_dy,
			float* dg_dy, float* dJ0phi_dx, float kxfac)
{
  unsigned int idxyz = get_id1();
  unsigned int idj = get_id2();

  if(idxyz<nx*ny*nz && idj<nj) {
    unsigned int ig = idxyz + nx*ny*nz*idj;
    g_res[ig] = ( dg_dx[ig] * dJ0phi_dy[ig] - dg_dy[ig] * dJ0phi_dx[ig] ) * kxfac;
    
  }
}

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

__global__ void stirring_kernel(cuComplex force, cuComplex *moments, int forcing_index)
{ moments[forcing_index] = moments[forcing_index] + force; }

__global__ void vol_summand(float *rmom, cuComplex* f, cuComplex* g, float* jacobian, float fluxDenomInv)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  unsigned int idxyz = idy + nyc*idx + nx*nyc*idz;

  if (idy<nyc && idx<nx && idz<nz) {
    if (unmasked(idx, idy)) {      
      cuComplex fg;  
      float fac=2.;
      if(idy==0) fac = 1.0;
      
      fg = cuConjf(f[idxyz])*g[idxyz]*jacobian[idz]*fac*fluxDenomInv;
      rmom[idxyz] = fg.x;
    } else {
      rmom[idxyz] = 0.;
    }
  }
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

# define Gh_(XYZ, L, M) g[(XYZ) + nx*nyc*nz*(L) + nx*nyc*nz*nl*(M)]
__global__ void heat_flux_summand(float* rmom, cuComplex* phi, cuComplex* g, float* ky, 
				  float* jacobian, float fluxDenomInv, float *kperp2, float rho2_s)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  cuComplex fg;

  unsigned int idxyz = idy + nyc*idx + nx*nyc*idz;
  if (idy<nyc && idx<nx && idz<nz) { 
    if (unmasked(idx, idy) && idy > 0) {    
      
      cuComplex vE_r = make_cuComplex(0., ky[idy]) * phi[idxyz];
    
      float b_s = kperp2[idxyz]*rho2_s;
    
      // sum over l
      cuComplex p_bar = make_cuComplex(0.,0.);
      for(int il=0; il<nl; il++) {
	p_bar = p_bar + Jfac(il, b_s)*Gh_(idxyz, il, 0) + rsqrtf(2.)*Jflr(il, b_s)*Gh_(idxyz, il, 2);
      }
    
      float fac = 2. * fluxDenomInv * jacobian[idz];

      fg = cuConjf(vE_r)*p_bar*fac;
      rmom[idxyz] = fg.x;

    } else {
      rmom[idxyz] = 0.;
    }
  }
}

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

__global__ void real_space_density(cuComplex* nbar, cuComplex* g, float *kperp2, specie *species) 
{
  unsigned int idxyz = get_id1();

  if(idxyz<nx*nyc*nz) {
    nbar[idxyz] = make_cuComplex(0., 0.);
    int idy = idxyz % nyc;
    int idx = idxyz / nyc % nx;
    if ( unmasked(idx, idy) ) {
      for(int is=0; is<nspecies; is++) {
	const float b_s = kperp2[idxyz]*species[is].rho2;
	for(int l=0; l<nl; l++) {
	  // sum over l for m=0
	  // Each thread does the full Laguerre sum for a particular (kx, ky, z) element
	  //
	  // This needs a factor of species[is].dens, I believe BD
	  nbar[idxyz] = nbar[idxyz] + Jflr(l,b_s)*g[idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*nm*is];
	}	
      }
    }
  }
}

__global__ void qneutAdiab_part1(cuComplex* PhiAvgNum_tmp, cuComplex* nbar,
				 float* kperp2, float* jacobian, specie* species, float ti_ov_te)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  if ( unmasked(idx, idy) && idz < nz) {
  
    unsigned int index = idy + nyc*idx + nx*nyc*idz;
    float pfilter2 = 0.;
    
    // BD This species stuff will need to be changed when species are shared over GPUs. 
    for(int is=0; is<nspecies; is++) {
      specie s = species[is];
      pfilter2 += s.dens*s.z*s.zt*( 1. - g0(kperp2[index]*s.rho2) );
    }
    
    PhiAvgNum_tmp[index] = ( nbar[index] / (ti_ov_te + pfilter2 ) ) * jacobian[idz];
  }
}


__global__ void qneutAdiab_part2(cuComplex* Phi, cuComplex* PhiAvgNum_tmp, cuComplex* nbar,
				 float* PhiAvgDenom, float* kperp2, float* jacobian,
				 specie* species, float ti_ov_te)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();
  
  if ( unmasked(idx, idy) && idz < nz) {

    unsigned int index = idy + nyc*idx + nx*nyc*idz;
    unsigned int idxy  = idy + nyc*idx;

    float pfilter2 = 0.;
    
    // BD multiple GPUs will require change here
    for(int is=0; is<nspecies; is++) {
      specie s = species[is];
      pfilter2 += s.dens*s.z*s.zt*( 1. - g0(kperp2[index]*s.rho2) );
    }

    // This is okay because PhiAvgNum_zSum is local to each thread
    cuDoubleComplex PhiAvgNum_zSum;
    PhiAvgNum_zSum.x = (double) 0.;
    PhiAvgNum_zSum.y = (double) 0.;

    for(int i=0; i<nz; i++) {
      PhiAvgNum_zSum.x = (double) PhiAvgNum_zSum.x + PhiAvgNum_tmp[idxy + i*nx*nyc].x;
      PhiAvgNum_zSum.y = (double) PhiAvgNum_zSum.y + PhiAvgNum_tmp[idxy + i*nx*nyc].y;
    }
    
    cuDoubleComplex PhiAvg;	
    if (idy == 0 && idx != 0) {
      PhiAvg.x = PhiAvgNum_zSum.x/( (double) PhiAvgDenom[idx] );
      PhiAvg.y = PhiAvgNum_zSum.y/( (double) PhiAvgDenom[idx] ); 
    } else {
      PhiAvg.x = 0.; PhiAvg.y = 0.;
    }

    //    Phi[index].x = (( nbar[index].x  ) / (ti_ov_te + pfilter2) ) * jacobian[idz];
    //    Phi[index].y = (( nbar[index].y  ) / (ti_ov_te + pfilter2) ) * jacobian[idz];    
    Phi[index].x = 0.0*( nbar[index].x + ti_ov_te*PhiAvg.x ) / (ti_ov_te + pfilter2);
    Phi[index].y = 0.0*( nbar[index].y + ti_ov_te*PhiAvg.y ) / (ti_ov_te + pfilter2);
  }
}

__global__ void calc_phiavgdenom(float* PhiAvgDenom, float* kperp2,
				 float* jacobian, specie* species, float ti_ov_te)
{   
  unsigned int idx = get_id1();
  
  if ( idx > 0 && idx < nx) {
    int ikx = get_ikx(idx);
    if (ikx <  (nx-1)/3+1 && ikx > -(nx-1)/3-1) {

      float pfilter2;
      PhiAvgDenom[idx] = 0.;  

      for(int i=0; i<nz; i++) {
	pfilter2 = 0.;
	for (int is=0; is<nspecies; is++) {
	  specie s = species[is];
	  pfilter2 += s.dens*s.z*s.zt*(1. - g0(kperp2[(idx + i*nx)*nyc]*s.rho2));
	}	
	PhiAvgDenom[idx] = PhiAvgDenom[idx] + jacobian[i] * pfilter2 / (ti_ov_te + pfilter2);
      }
    }
  }
}

// Is this really set up correctly? 
__global__ void add_source(cuComplex* f, float source)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();
  
  if ( unmasked(idx, idy) && idz < nz) {
    unsigned int index = idy + nyc*idx + nx*nyc*idz;
    f[index].x = f[index].x + source;
  }
}

__global__ void qneutAdiab(cuComplex* Phi, cuComplex* nbar,
			   float* kperp2, float* jacobian,
			   specie* species, float ti_ov_te)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  // BD all these qneut routines now assume we are working with the dealiased grids only
  if ( unmasked(idx, idy) && idz < nz) {
    
    unsigned int index = idy + nyc*idx + nx*nyc*idz;
    float pfilter2 = 0.;
    
    // BD Check this for parallel correctness when nspecies > 1
    for(int is=0; is<nspecies; is++) {
      specie s = species[is];
      pfilter2 += s.dens*s.z*s.zt*( 1. - g0(kperp2[index]*s.rho2) );
    }
    
    //    Phi[index].x = ( nbar[index].x / (ti_ov_te + pfilter2 ) ) * jacobian[idz]; // what is this factor of jacobian[idz]?
    //    Phi[index].y = ( nbar[index].y / (ti_ov_te + pfilter2 ) ) * jacobian[idz];

    Phi[index].x = 0.0*( nbar[index].x / (ti_ov_te + pfilter2 ) );
    Phi[index].y = 0.0*( nbar[index].y / (ti_ov_te + pfilter2 ) );
  }
}

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

__managed__ cufftCallbackStoreC i_kzLinked_callbackPtr = i_kzLinked;
__managed__ cufftCallbackStoreC abs_kzLinked_callbackPtr = abs_kzLinked;

__global__ void linkedCopy(cuComplex* G, cuComplex* G_linked,
			   int nLinks, int nChains, int* ikx, int* iky, int nMoms)
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

__global__ void linkedCopyBack(cuComplex* G_linked, cuComplex* G,
			       int nLinks, int nChains, int* ikx, int* iky, int nMoms)
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

// main kernel function for calculating RHS
# define S_G(L, M) s_g[sidxyz + (sDimx)*(L) + (sDimx)*(sDimy)*(M)]
__global__ void rhs_linear(cuComplex* g, cuComplex* phi, cuComplex* upar_bar, cuComplex* uperp_bar, cuComplex* t_bar,
			   float* kperp2, float* cv_d, float* gb_d, float* bgrad, float* ky, specie* species,
			   cuComplex* rhs_par, cuComplex* rhs)
{
  extern __shared__ cuComplex s_g[]; // aliased below by macro S_G, defined above

  //  unsigned int idxyz = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int idxyz = get_id1();
  const unsigned int idy = idxyz % nyc; 
  const unsigned int idx = idxyz / nyc % nx;
  if(idxyz<nx*nyc*nz && unmasked(idx, idy)) {
    const unsigned int sidxyz = threadIdx.x;
    // these modulo operations are expensive... better way to get these indices?
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
    // blockIdx for y and z and both unity in the kernel invocation
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
    // blockIdx.y is unity in the kernel invocation
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
    // blockIdx.z is unity in the kernel invocation
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
    // blockIdx for y and z are unity in the kernel invocation
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

__global__ void hypercollisions(cuComplex* g, float nu_hyper_l, float nu_hyper_m, int p_hyper_l, int p_hyper_m, cuComplex* rhs) {
  unsigned int idxyz = get_id1();

  if(idxyz<nx*nyc*nz) {
    float scaled_nu_hyp_l = (float) nl * nu_hyper_l;
    float scaled_nu_hyp_m = (float) nm * nu_hyper_m; // scaling appropriate for curvature. Too big for slab
    for(int is=0; is<nspecies; is++) { 
    // blockIdx for y and z are unity in the kernel invocation      
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

# define Hc_(XYZ, L, M, S) g[(XYZ) + nx*nyc*nz*(L) + nx*nyc*nz*nl*(M) + nx*nyc*nz*nl*nm*(S)] + Jflr(L,b_s)*phi_*zt_
# define Gc_(XYZ, L, M, S) g[(XYZ) + nx*nyc*nz*(L) + nx*nyc*nz*nl*(M) + nx*nyc*nz*nl*nm*(S)] // H = G, except for m = 0
// C = C(H) but H and G are the same function for all m!=0. Our main array defines g so the correction to produce
// H is only appropriate for m=0. In other words, the usage here is basically handling the delta_{m0} terms
// in a clumsy way
__global__ void conservation_terms(cuComplex* upar_bar, cuComplex* uperp_bar, cuComplex* t_bar,
				   cuComplex* g, cuComplex* phi, float *kperp2, specie* species)
{
  unsigned int idxyz = get_id1();

  // can this kernel be moved into the cub framework?

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
        upar_bar[index] = upar_bar[index] + Jflr(l,b_s)*Gc_(idxyz, l, 1, is);
        // Hc_(...) is defined by macro above. Only use H here for m=0. Confusing!
        uperp_bar[index] = uperp_bar[index] + (Jflr(l,b_s) + Jflr(l-1,b_s))*Hc_(idxyz, l, 0, is);

        // energy conservation correction for nlaguerre = 1
        if (nl == 1) {
            t_bar[index] = t_bar[index] + sqrtf(2.)*Jflr(l,b_s)*Gc_(idxyz, l, 2, is);
        } else {
            t_bar[index] = t_bar[index] + sqrtf(2.)/3.*Jflr(l,b_s)*Gc_(idxyz, l, 2, is)
		    + 2./3.*( l*Jflr(l-1,b_s) + 2.*l*Jflr(l,b_s) + (l+1)*Jflr(l+1,b_s) )*Hc_(idxyz, l, 0, is);
        }
      }
      uperp_bar[index] = uperp_bar[index]*sqrtf(b_s);
    }
  }
}

__global__ void Hmom(cuComplex * G, float* hmom_real, float* hmom_imag, int m, int forcing_index)
{
    hmom_real[m] = G[forcing_index].x;
    hmom_imag[m] = G[forcing_index].y;
}
