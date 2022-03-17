#include <stdio.h>
#include "device_funcs.h"

__device__ __constant__ int nx, ny, nyc, nz, nspecies, nm, nl, nj, zp, ikx_fixed, iky_fixed;

void setdev_constants(int Nx, int Ny, int Nyc, int Nz, int Nspecies, int Nm, int Nl, int Nj, int Zp, int ikxf, int ikyf)
{
  cudaMemcpyToSymbol ( nx,        &Nx,        sizeof(int));
  cudaMemcpyToSymbol ( ny,        &Ny,        sizeof(int));
  cudaMemcpyToSymbol ( nyc,       &Nyc,       sizeof(int));
  cudaMemcpyToSymbol ( nz,        &Nz,        sizeof(int));
  cudaMemcpyToSymbol ( nspecies,  &Nspecies,  sizeof(int));
  cudaMemcpyToSymbol ( nm,        &Nm,        sizeof(int));
  cudaMemcpyToSymbol ( nl,        &Nl,        sizeof(int));
  cudaMemcpyToSymbol ( nj,        &Nj,        sizeof(int));
  cudaMemcpyToSymbol ( zp,        &Zp,        sizeof(int));
  cudaMemcpyToSymbol ( ikx_fixed, &ikxf,      sizeof(int));
  cudaMemcpyToSymbol ( iky_fixed, &ikyf,      sizeof(int));
  
}

// threadIdx == ithread
// blockDim == nthread
// blockIdx == iblock
//
// dimGrid == nblock
// dimBlock == nthread
// 
// loops are therefore executed in parallel (for each dimension) and can
// be thought of as something like 
// for (int iblock = 0; iblock < nblock; iblock++) {
//   for (int ithread = 0; ithread < nthread; ithread++) {
//     int idx = ithread + iblock * nthread ;
//     f[idx] = ...
//   }
//  }
// as long as one keeps track of memory correctly -- meaning,
// each idx is assumed to control its own memory.
// If you want to couple idx's you want to use shared memory
// which is shared for each iblock of nthreads in the straightforward case
// and can be updated safely only with explicit synchronization.
//
// This way of thinking only works correctly with the understanding that
// block.x, block.y, block.z are one block from the point of view of what
// can be synchronized inside of a kernel. This is probably why Nvidia
// made the semantics of cuda the way they did, but it is confusing
// if one is accustomed to thinking of ithread, nthread as going together
// and iblock, nblock going together, since in the cuda semantics, one
// should naturally think of a grid of blocks, with coding that provides
// ithread and iblock as built ins. 
//
// To make things easier,
// define get_id1, get_id2, get_id3, which return
// idx1 (for the .x loops)
// idx2 (for the .y loops)
// idx3 (for the .z loops)
// 

__device__ unsigned int get_id1(void) {return __umul24(blockIdx.x,blockDim.x)+threadIdx.x;}
__device__ unsigned int get_id2(void) {return __umul24(blockIdx.y,blockDim.y)+threadIdx.y;}
__device__ unsigned int get_id3(void) {return __umul24(blockIdx.z,blockDim.z)+threadIdx.z;}

// use stirling's approximation
__host__ __device__ float factorial(int m) {
  if (m <2) return 1.;
  if (m==2) return 2.;
  if (m==3) return 6.;
  if (m==4) return 24.;
  if (m==5) return 120.;
  if (m==6) return 720.;
  else return sqrtf(2.*M_PI*m)*powf(m,m)*expf(-m)*(1.+1./(12.*m)+1./(288.*m*m));
}

__device__ float Jflr(const int l, const float b, bool enforce_JL_0) {
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
  
  if (g<tol) g=tol; 
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
  
  if (g<tol) g=tol; 
  return g;

}

__host__ __device__ float sgam0 (float b) {return sqrt(g0(b));}
__host__ __device__ bool operator>(cuComplex f, cuComplex g) { return f.x*f.x+f.y*f.y > g.x*g.x+g.y*g.y; }
__host__ __device__ bool operator<(cuComplex f, cuComplex g) { return f.x*f.x+f.y*f.y < g.x*g.x+g.y*g.y; }
__host__ __device__ cuComplex operator+(cuComplex f, cuComplex g) { return cuCaddf(f,g); }
__host__ __device__ cuComplex operator-(cuComplex f, cuComplex g) { return cuCsubf(f,g); }

__host__ __device__ cuComplex operator+(float f, cuComplex g)
{
  cuComplex result;
  result.x = f + g.x;
  result.y = g.y;
  return result;
}

__host__ __device__ cuComplex operator+(cuComplex g, float f)
{
  cuComplex result;
  result.x = f + g.x;
  result.y = g.y;
  return result;
}

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

__global__ void getPhi (cuComplex *phi, cuComplex *G, float* ky)
{
  unsigned int idy = get_id1();
  if (idy > 0 && idy < 1 + (ny-1)/3) {

    float ky2inv = 1./(ky[idy]*ky[idy]);

    phi[idy] = G[idy]*ky2inv;
  }
  if (idy == 0) {
    phi[idy].x = 0.; phi[idy].y = 0.;
  }
}

__global__ void phiSolve_krehm (cuComplex *phi, cuComplex *G0, float* kx, float* ky, float rho_i)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();
  if ( unmasked(idx, idy) && idz < nz ) { 
    unsigned int idxyz = idy + nyc*(idx + nx*idz); 

    float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];
    float gam0 = g0(kperp2*rho_i*rho_i/2.);

    phi[idxyz] = G0[idxyz]*rho_i*rho_i/2./(gam0 - 1.);
  }
}

__global__ void aparSolve_krehm (cuComplex *apar, cuComplex *G1, float* kx, float* ky, float rho_s, float d_e)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();
  if ( unmasked(idx, idy) && idz < nz ) { 
    unsigned int idxyz = idy + nyc*(idx + nx*idz); 

    float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];

    apar[idxyz] = -G1[idxyz]*rho_s*d_e/(1. + d_e*d_e*kperp2);
  }
}

__global__ void rhs_lin_vp(const cuComplex *G, const cuComplex* phi, cuComplex* GRhs, float* ky,
			   bool closure, float nu, float nuh, int alpha, int alpha_h)
{
  unsigned int idy = get_id1();
  if (idy < 1 + (ny-1)/3) {
    unsigned int m = get_id2();
    if (m < nm) {
      unsigned int ig  = idy + nyc*m;
      unsigned int mm1 = idy + nyc*(m-1);
      unsigned int mp1 = idy + nyc*(m+1);
      cuComplex Iky = make_cuComplex(0., ky[idy]);
      float k2 =    ky[idy]     *ky[idy];
      float k2max = ky[(ny-1)/3]*ky[(ny-1)/3];
      float k2norm = k2/k2max; 
      
      if (m < nm-1) {GRhs[ig] =            G[mp1] * sqrtf(m+1);
	if (m >  0)  GRhs[ig] = GRhs[ig] + G[mm1] * sqrtf(m);
	if (m == 1)  GRhs[ig] = GRhs[ig] + phi[idy];
	
          	     GRhs[ig] = - Iky * GRhs[ig];
	
      } else {
	             GRhs[ig] =            G[mm1] * sqrtf(m); 
        if (closure) GRhs[ig] = GRhs[ig] + G[mm1] * sqrtf(m+1);

	             GRhs[ig] = - Iky * GRhs[ig];
		     
        if (closure) GRhs[ig] = GRhs[ig] - ky[idy] * 2. * sqrtf(nm) * G[ig];
      }
      
      if ((nu > 0) && (m > 2)) {
	             GRhs[ig] = GRhs[ig] - nu  * pow(m, alpha) * G[ig];
      }
      if (nuh > 0)   GRhs[ig] = GRhs[ig] - nuh * pow(k2norm, alpha_h) * G[ig];
    }
  }
}

__global__ void rhs_ks(const cuComplex *G, cuComplex *GRhs, float *ky, float eps)
{
  unsigned int idy = get_id1();
  if (idy < (ny-1)/3 + 1) {
    float k2 = ky[idy]*ky[idy];    float lin = (1.0+eps)*k2 - k2*k2;

    GRhs[idy] = lin * G[idy];
  }
}

__global__ void add_section(cuComplex *res, const cuComplex *tmp, int ntot)
{
  unsigned int i = get_id1();
  if (i < ntot) res[i] = res[i] + tmp[i];
}

__global__ void add_scaled_singlemom_kernel(cuComplex* res,
					    double c1, const cuComplex* m1,
					    double c2, const cuComplex* m2)
{
  unsigned int idxyz = get_id1();
  if (idxyz < nx*nyc*nz) res[idxyz] = c1*m1[idxyz] + c2*m2[idxyz];
}

__global__ void add_scaled_singlemom_kernel(cuComplex* res,
					    double c1, const cuComplex* m1,
					    double c2, const cuComplex* m2,
					    double c3, const cuComplex* m3)
{
  unsigned int idxyz = get_id1();
  if (idxyz < nx*nyc*nz) res[idxyz] = c1*m1[idxyz] + c2*m2[idxyz] + c3*m3[idxyz];
}

__global__ void add_scaled_singlemom_kernel(cuComplex* res,
					    cuComplex c1, const cuComplex* m1,
					    cuComplex c2, const cuComplex* m2)
{
  unsigned int idxyz = get_id1();
  if (idxyz < nx*nyc*nz) res[idxyz] = c1*m1[idxyz] + c2*m2[idxyz];
}

/*

!eqfix            false  true    false   true     
not_fixed_eq      true   true    false   false 
 
action            true   true    false    true

bool proceed == (!eqfix || not_fixed_eq);
*/

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2, bool neqfix = true)
{
  unsigned int idxy = get_id1(); 
  if (idxy < nx*nyc) {
    if (neqfix || not_fixed_eq(idxy)) {
      
      unsigned int idz  = get_id2();
      if (idz < nz) {
	
	unsigned int idslm = get_id3(); 
	unsigned int ig = idxy + nx*nyc*(idz + nz*idslm);
	
	res[ig] = c1 * m1[ig] + c2 * m2[ig];
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2,
				  double c3, const cuComplex* m3, bool neqfix = true)
{
  unsigned int idxy = get_id1(); 
  if (idxy < nx*nyc) {
    if (neqfix || not_fixed_eq(idxy)) {
      
      unsigned int idz  = get_id2();
      if (idz < nz) {
	
	unsigned int idslm = get_id3(); 
	unsigned int ig = idxy + nx*nyc*(idz + nz*idslm);
	
	res[ig] = c1 * m1[ig] + c2 * m2[ig] + c3 * m3[ig];
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2, 
				  double c3, const cuComplex* m3,
				  double c4, const cuComplex* m4, bool neqfix = true)
{
  unsigned int idxy = get_id1(); 
  if (idxy < nx*nyc) {
    if (neqfix || not_fixed_eq(idxy)) {
      
      unsigned int idz  = get_id2();
      if (idz < nz) {
	
	unsigned int idslm = get_id3(); 
	unsigned int ig = idxy + nx*nyc*(idz + nz*idslm);
	
	res[ig] = c1 * m1[ig] + c2 * m2[ig] + c3 * m3[ig] + c4 * m4[ig];
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2, 
				  double c3, const cuComplex* m3,
				  double c4, const cuComplex* m4,
				  double c5, const cuComplex* m5, bool neqfix = true)
{
  unsigned int idxy = get_id1(); 
  if (idxy < nx*nyc) {
    if (neqfix || not_fixed_eq(idxy)) {
      
      unsigned int idz  = get_id2();
      if (idz < nz) {
	
	unsigned int idslm = get_id3(); 
	unsigned int ig = idxy + nx*nyc*(idz + nz*idslm);
	
	res[ig] = c1 * m1[ig] + c2 * m2[ig] + c3 * m3[ig] + c4 * m4[ig] + c5 * m5[ig];
      }
    }
  }
}

__global__ void init_Fake_G (float* g)
{
  unsigned int idy = get_id1();
  if (idy < ny) {
    float theta = 2.*M_PI*idy/((float) ny);
    g[idy] = cosf(theta);
  }
}

__global__ void update_Fake_G (float* g, int i)
{
  unsigned int idy = get_id1();
  if (idy < ny) {
    float theta = 2.*M_PI*idy/((float) ny);
    g[idy] = cosf(2.*theta - 2.*M_PI*i/32.);
  }
}

__global__ void setval(float* f, float val, int N)
{
  unsigned int n = get_id1();
  if (n<N) f[n] = val;
}

__global__ void setval(double* f, double val, int N)
{
  unsigned int n = get_id1();
  if (n<N) f[n] = val;
}

__global__ void setval(cuComplex* f, cuComplex val, int N)
{
  unsigned int n = get_id1();
  if (n<N) f[n] = val;
}

__global__ void demote(float* G, double* dG, int N)
{
  unsigned int n = get_id1();
  if (n<N) G[n] = (float) dG[n];
}

__global__ void promote(double* dG, float* G, float* Gnoise, int N)
{
  unsigned int n = get_id1();
  if (n<N) dG[n] = (double) G[n]*Gnoise[n];
}

__global__ void promote(double* dG, float* G, int N)
{
  unsigned int n = get_id1();
  if (n<N) dG[n] = (double) G[n];
}

__global__ void getr2(double* r2, double* r, int N)
{
  unsigned int n = get_id1();
  if (n<N) r2[n] = (n%2 == 0) ? r[n] : r[n]*r[n];
}

__global__ void getV(double* V, double* dG, double* r2, int M, int N) 
{
  unsigned int m = get_id1();
  if (m<M) {
    unsigned int n = get_id2();
    if (n<N) V[m + M*n] +=  dG[m] * r2[n];
  }
}

__global__ void getB(double* W, double beta, int N)
{
  int n = get_id1();
  if (n<N) W[n+N*n] += beta;
}

__global__ void setI(double* Id, int N)
{
  int n = get_id1();
  if (n<N) Id[n + N*n] = 1.;
}

__global__ void setA(double* A, double fac, int N)
{
  unsigned int n = get_id1();
  if (n<N) A[n] *= (double) fac;
}

__global__ void getW(double* W, double* r2, int N) 
{
  int n1 = get_id1();
  if (n1<N) {
    int n2 = get_id2();
    if (n2<n1+1) {
      double r2n1 = r2[n1];
      double r2n2 = r2[n2];
      W[n1 + N*n2] += r2n1 * r2n2;
      if (n2<n1) W[n2 + N*n1] = W[n1 + N*n2];
    }
  }
}

__global__ void copyV(float* P, double* V, int N)
{
  unsigned int n = get_id1();
  if (n<N) P[n] = (float) V[n];
}

__global__ void WinG(double* res, double* Win, double* dG, int Q, int M)
{
  unsigned int q = get_id1();
  if (q<Q) {
    unsigned int m = get_id2();
    if (m<M) {
      unsigned int n = q + Q*m;
      res[n] = Win[n] * dG[m];
    }
  }
}

__global__ void update_state(double* res, double* A, double* x, int K, int N)
{
  unsigned int n = get_id1();
  if (n<N) {
    for (int k=0; k<K; k++) res[n] +=  A[k + K*n] * x[k + K*n];
    res[n] = tanh(res[n]);
  }
}

__global__ void myPrep(double* x, double* r, int* col, int NK)
{
  unsigned int i = get_id1();
  if (i < NK) {
    x[i] = r[ col[i] ];
  }
}

__global__ void mySpMV(double* x2, double* xy, double* y2,
		       double* y, double* x, double* A, double* r, int K, int N)
{
  unsigned int n = get_id1();
  if (n < N) {
    y[n] = A[K*n] * x[K*n];
    for (int k=1; k<K; k++) y[n] += A[k + K*n] * x[k + K*n];
    y2[n] = y[n] * y[n];  //printf("y2[%d] = %e \n",n,y2[n]);
    xy[n] = r[n] * y[n];  //printf("xy[%d] = %e \n",n,xy[n]);
    x2[n] = r[n] * r[n];  //printf("x2[%d] = %e \n",n,x2[n]);
  }  
}

__global__ void eig_residual(double* y, double* A, double* x, double* R,
			     double* r2, double eval, int K, int N)
{
  unsigned int n = get_id1();
  if (n < N) {
    y[n] = A[K*n] * x[K*n];
    for (int k=1; k<K; k++) y[n] += A[k + K*n] * x[k + K*n];
    //    printf("eval = %e \n", eval);
    double res = eval*R[n] - y[n];
    r2[n] = pow(res,2);
    //    printf("r2[%d] = %e \n",n,r2[n]);
  }
}

__global__ void est_eval(double eval, double *fLf, double* f2) {eval = fLf[0]/f2[0];}


__global__ void inv_scale_kernel(double* res, const double* f, const double* scalar, int N)
{
  unsigned int n = get_id1();
  if (n < N) {
    res[n] = f[n]/sqrt(scalar[0]);
    //    printf("scalar = %e \t res[%d] = %e \n",scalar[0], n, res[n]);
  }
}

__global__ void scale_kernel(cuComplex* res, double scalar)
{
  unsigned int idxy = get_id1(); 
  if (idxy < nx*nyc) {
    unsigned int idz  = get_id2();
    if (idz < nz) {
      
      unsigned int idslm = get_id3(); 
      unsigned int ig = idxy + nx*nyc*(idz + nz*idslm);
      
      res[ig] = scalar*res[ig];
    }
  }
}

__global__ void scale_kernel(cuComplex* res, const cuComplex scalar)
{
  unsigned int idxy = get_id1(); 
  if (idxy < nx*nyc) {
    unsigned int idz  = get_id2();
    if (idz < nz) {
      
      unsigned int idslm = get_id3(); 
      unsigned int ig = idxy + nx*nyc*(idz + nz*idslm);
      
      res[ig] = scalar*res[ig];
    }
  }
}

__global__ void scale_singlemom_kernel(cuComplex* res, cuComplex* mom, cuComplex scalar)
{
  unsigned int idxyz = get_id1();
  if (idxyz < nx*nyc*nz) res[idxyz] = scalar*mom[idxyz];
}

__global__ void scale_singlemom_kernel(cuComplex* res, cuComplex* mom, float scalar)
{
  unsigned int idxyz = get_id1();
  if (idxyz < nx*nyc*nz) res[idxyz] = scalar*mom[idxyz];
}

__global__ void reality_kernel(cuComplex* g, int N) 
{
  unsigned int idx = get_id1();
  unsigned int idz = get_id2();
  unsigned int idlms = get_id3();
  
  if (idx < (nx-1)/3+1 && idz < nz && idlms < N) {

    unsigned int ig  = nyc*(   idx + nx*(idz + nz*idlms));
    unsigned int ig2 = nyc*(nx-idx + nx*(idz + nz*idlms));

    if (idx==0) {
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
  if ( idx < nx           // index should be in range to be actively masked
    && idy < ny           // index should be in range to be actively masked
       && ( (idx==0 && idy==0) || idy > (ny-1)/3  || ikx > (nx-1)/3 || ikx < -(nx-1)/3 ))
    return true;
  else
    return false;
}

__global__ void Hkernel (cuComplex *G, cuComplex* J0phi)
{
  // For the m=0 components (all values of ky, kx, z, l, s) add J0phi * zt_
  // G = G + J0phi*zt_   ... if m = 0
}
__global__ void Gkernel (cuComplex *G, cuComplex* J0phi)
{
  // For the m=0 components (all values of ky, kx, z, l, s) subtract J0phi * zt_
  // G = G - J0phi*zt_   ... if m = 0
}

__global__ void maskG(cuComplex* g)
{
  unsigned int idxy = get_id1();
  if (idxy < nx*nyc) {
    unsigned int idy = idxy % nyc;
    unsigned int idx = idxy / nyc; 
    if (masked(idx, idy)) {
      unsigned int idz = get_id2();
      unsigned int idslm = get_id3();
      if ((idz < nz) && (idslm < nl*nm*nspecies)) {
	unsigned int globalIdx = idxy + nyc*nx*(idz + nz*idslm);
	g[globalIdx] = make_cuComplex(0., 0.);
      }
    }
  }  
}

__global__ void calc_bgrad(float* bgrad, const float* bgrad_temp, const float* bmag, float scale)
{
  unsigned int idz = get_id1();
  if (idz < nz) bgrad[idz] = ( bgrad_temp[idz] / bmag[idz] ) * scale;
}

__global__ void init_kxs(float* kxs, float* kx, float* th0)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  if (unmasked(idx, idy)) {
    kxs[idy+nyc*idx] = kx[idx]; // should read this from a file if this is a restarted case
  }
}

__global__ void update_kxs(float* kxs, float* dth0)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  if (unmasked(idx, idy)) {
    //    kxs[idy + nyc*idx]  =
  }
}

__global__ void update_geo(float* kxs, float* ky, float* cv_d, float* gb_d, float* kperp2,
			   float* cv, float* cv0, float* gb, float* gb0, float* omegad, 
			   float* gds2, float* gds21, float* gds22, float* bmagInv, float shat)
{

  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  float shatInv = 1./shat; // Needs a test for zero
  
  if (idy>0 && unmasked(idx, idy) && idz < nz) { 
    unsigned int idxyz = idy + nyc*(idx + nx*idz);
    kperp2[idxyz] = ( ky[idy] * ( ky[idy] * gds2[idz] + 2. * kxs[idy+nyc*idx] * shatInv * gds21[idz]) 		       
			+ pow( kxs[idy+nyc*idx] * shatInv, 2) * gds22[idz] ) * pow( bmagInv[idz], 2);
    
    cv_d[idxyz] = ky[idy] * cv[idz] + kxs[idy+nyc*idx] * shatInv * cv0[idz] ;     
    gb_d[idxyz] = ky[idy] * gb[idz] + kxs[idy+nyc*idx] * shatInv * gb0[idz] ;
    omegad[idxyz] = cv_d[idxyz] + gb_d[idxyz];
  }
}

__global__ void init_kperp2(float* kperp2, const float* kx, const float* ky,
			    const float* gds2, const float* gds21, const float* gds22,
			    const float* bmagInv, float shat) 
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  float shatInv = 1./shat; // Needs a test for zero

  if (unmasked(idx, idy) && idz < nz) { 
    unsigned int idxyz = idy + nyc*(idx + nx*idz);
    kperp2[idxyz] = ( ky[idy] * ( ky[idy] * gds2[idz] 
                      + 2. * kx[idx] * shatInv * gds21[idz]) 
                      + pow( kx[idx] * shatInv, 2) * gds22[idz] ) 
                      * pow( bmagInv[idz], 2);
  }
}

__global__ void init_omegad(float* omegad, float* cv_d, float* gb_d, const float* kx, const float* ky,
			    const float* cv, const float* gb, const float* cv0, const float* gb0, float shat) 
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  float shatInv = 1./shat; // BD this needs an exception for shat = 0.

  // cv/gb is the y-directed part of the curvature / grad-B drift
  // cv0/gb0 is the part proportional to the theta_0, aka the x-directed component

  if ( unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = idy + nyc*(idx + nx*idz);
    cv_d[idxyz] = ky[idy] * cv[idz] + kx[idx] * shatInv * cv0[idz] ;     
    gb_d[idxyz] = ky[idy] * gb[idz] + kx[idx] * shatInv * gb0[idz] ;
    omegad[idxyz] = cv_d[idxyz] + gb_d[idxyz];
  }
}

# define H_(XYZ, L, M, S) (g[(XYZ) + nx*nyc*nz*((L) + nl*((M) + nm*(S)))] + Jflr(L,b_s)*phi_)
# define G_(XYZ, L, M, S)  g[(XYZ) + nx*nyc*nz*((L) + nl*((M) + nm*(S)))]
// H = G, except for m = 0
// C = C(H) but H and G are the same function for all m!=0. Our main array defines g so the correction to produce
// H is only appropriate for m=0. In other words, the usage here is basically handling the delta_{m0} terms
// in a clumsy way
__global__ void Tbar(cuComplex* t_bar, const cuComplex* g, const cuComplex* phi, const float *kperp2) // crude diagnostic
{
  // should re-do this with real space/wavenumber indexing and using unmasked function
  unsigned int idxyz = get_id1();
  if (idxyz < nx*nyc*nz) {
    cuComplex phi_ = phi[idxyz];
    //    int index = idxyz;
    t_bar[idxyz] = make_cuComplex(0., 0.);
    float b_s = kperp2[idxyz]; // only species=0, assumes zt, rho2 = 1 !! bug
    for (int l=0; l < nl; l++) {
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

__global__ void growthRates(const cuComplex *phi, const cuComplex *phiOld, double dt, cuComplex *omega)
{
  unsigned int idxy = get_id1();
  cuComplex i_dt = make_cuComplex(0., (float) 1./dt);
  unsigned int J = nx*nyc;

  if (idxy < J) {
    int IG = (int) nz/4 ;
    
    int idy = idxy % nyc;
    int idx = idxy / nyc; // % nx;
    
    if (unmasked(idx, idy)) {
      if (abs(phi[idxy+J*IG].x)!=0 && abs(phi[idxy+J*IG].y)!=0) {
	cuComplex ratio = phi[ idxy + J*IG ] / phiOld[ idxy + J*IG ];
	
	cuComplex logr;
	logr.x = (float) log(cuCabsf(ratio));
	logr.y = (float) atan2(ratio.y,ratio.x);
	omega[idxy] = logr*i_dt;
	if (isnan(omega[idxy].x)) {omega[idxy].x = -77777.; omega[idxy].y = -77777.;}
      } else {
	omega[idxy].x = -99999.;
	omega[idxy].y = -99999.;
      }
    }
  }
}

__global__ void J0phiToGrid(cuComplex* J0phi, const cuComplex* phi, const float* kperp2,
			    const float* muB, const float rho2_s)
{
  unsigned int idxyz = get_id1();
  unsigned int idj = get_id2();
  if (idxyz < nx*nyc*nz && idj < nj) {
    unsigned int ig = idxyz + nx*nyc*nz*idj;
    J0phi[ig] = j0f(sqrtf(2. * muB[idj] * kperp2[idxyz]*rho2_s)) * phi[idxyz];
  }
}

__global__ void castDoubleToFloat (const cuDoubleComplex *array_d, cuComplex *array_f, int size) {
  for (unsigned int i = 0; i < size; i++) array_f[i] = cuComplexDoubleToFloat(array_d[i]);
}

__global__ void ddx (cuComplex *res, cuComplex *f, float *kx)
{
  unsigned int idy = get_id1();
  if (idy < nyc) {
    unsigned int idx = get_id2();
    if (idx < nx) {
      if (unmasked(idx, idy)) {
	unsigned int idz = get_id3();
	if (idz < nz) {
	  cuComplex Ikx = make_cuComplex(0., kx[idx]);
	  unsigned int ig = idy + nyc*(idx + nx*idz);
	  res[ig] = Ikx*f[ig]; 
	}
      }
    }
  }
}

__global__ void ddy (cuComplex *res, cuComplex *f, float *ky)
{
  unsigned int idy = get_id1();
  if (idy < nyc) {
    unsigned int idx = get_id2();
    if (idx < nx) {
      if (unmasked(idx, idy)) {
	unsigned int idz = get_id3();
	if (idz < nz) {
	  cuComplex Iky = make_cuComplex(0., ky[idy]);
	  unsigned int ig = idy + nyc*(idx + nx*idz);
	  res[ig] = Iky*f[ig];
	}
      }
    }
  }
}

__global__ void mddy (cuComplex *res, cuComplex *f, float *ky)
{
  unsigned int idy = get_id1();
  if (idy < nyc) {
    unsigned int idx = get_id2();
    if (idx < nx) {
      if (unmasked(idx, idy)) {
	unsigned int idz = get_id3();
	if (idz < nz) {
	  cuComplex Iky = make_cuComplex(0., ky[idy]);
	  unsigned int ig = idy + nyc*(idx + nx*idz);
	  res[ig] = -Iky*f[ig];
	}
      }
    }
  }
}

__global__ void d2x (cuComplex *res, cuComplex *f, float *kx)
{
  unsigned int idy = get_id1();
  if (idy < nyc) {
    unsigned int idx = get_id2();
    if (idx < nx) {
      if (unmasked(idx, idy)) {
	unsigned int idz = get_id3();
	if (idz < nz) {
	  cuComplex Ikx = make_cuComplex(0., kx[idx]);
	  unsigned int ig = idy + nyc*(idx + nx*idz);
	  res[ig] = Ikx*Ikx*f[ig];
	}
      }
    }
  }
}

__device__ cuComplex i_kx(void *dataIn, size_t offset, void *kxData, void *sharedPtr)
{
  float *kx = (float*) kxData;
  unsigned int idx = offset / nyc % nx;
  cuComplex Ikx = make_cuComplex(0., kx[idx]);
  return Ikx*((cuComplex*)dataIn)[offset];
}

__device__ cuComplex i_kxs(void *dataIn, size_t offset, void *kxsData, void *sharedPtr)
{
  float *kxs = (float*) kxsData;
  unsigned int idxy = offset % (nyc*nx);
  cuComplex Ikxs = make_cuComplex(0., kxs[idxy]);
  return Ikxs*((cuComplex*)dataIn)[offset];
}

__device__ cuComplex i_ky(void *dataIn, size_t offset, void *kyData, void *sharedPtr)
{
  float *ky = (float*) kyData;
  unsigned int idy = offset % nyc; 
  cuComplex Iky = make_cuComplex(0., ky[idy]);
  return Iky*((cuComplex*)dataIn)[offset];
}

// for ExB shear, still need to take care of the phase factors associated with kx grid misses
__device__ void mask_and_scale(void *dataOut, size_t offset, cufftComplex element, void *data, void * sharedPtr)
{
  //  unsigned int idz = offset / (nyc*nx) % nz;
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

__managed__ cufftCallbackLoadC i_kxs_callbackPtr = i_kxs;
__managed__ cufftCallbackLoadC i_kx_callbackPtr = i_kx;
__managed__ cufftCallbackLoadC i_ky_callbackPtr = i_ky;
__managed__ cufftCallbackStoreC mask_and_scale_callbackPtr = mask_and_scale;

// Multiplies by i kz / Nz 
__device__ void i_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  float *kz = (float*) kzData;
  unsigned int idz = offset / (nx*nyc);
  cuComplex Ikz = make_cuComplex(0., kz[idz]);
  ((cuComplex*)dataOut)[offset] = Ikz*element/nz;    
}

__device__ void zfts(void *dataOut, size_t offset, cufftComplex element, void *data, void *sharedPtr)
{
  ((cuComplex*)dataOut)[offset] = element/nz;    
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

__managed__ cufftCallbackStoreC zfts_callbackPtr = zfts;
__managed__ cufftCallbackStoreC i_kz_callbackPtr = i_kz;
__managed__ cufftCallbackStoreC i_kz_1d_callbackPtr = i_kz_1d;
__managed__ cufftCallbackStoreC abs_kz_callbackPtr = abs_kz;

__global__ void acc(float *a, const float *b)
{a[0] = a[0] + b[0];}

__global__ void nlvp(float *res, const float *Gy, const float *dphi)
{
  unsigned int idy = get_id1();
  if (idy < ny) {
    unsigned int m = get_id2();
    if (m > 0 && m < nm) {
      unsigned int ig  = idy + ny*m;
      unsigned int mm1 = idy + ny*(m-1);
      res[ig] = - sqrtf(m) * dphi[idy] * Gy[mm1];
    }
  }
}

__global__ void nlks(float *res, const float *Gy, const float *dG)
{
  unsigned int idy = get_id1();
  if (idy < ny) res[idy] = - Gy[idy] * dG[idy];
}

__global__ void kz_dealias (cuComplex *G, int *kzm, int LMS)
{
  unsigned int idxy = get_id1();
  if (idxy < nx*nyc) {
    int idy = idxy % nyc;
    int idx = idxy / nyc;
    if (unmasked(idx, idy)) {
      unsigned int idz = get_id2();
      if (idz < nz) {
	unsigned int idlms = get_id3();
	if (idlms < LMS) {
	  unsigned int ig = idxy + nx*nyc*(idz + nz*idlms);
	  if (kzm[idz] == 0) {
	    G[ig].x = 0.; 
	    G[ig].y = 0.;
	  }
	}
      }
    }
  }
}

__global__ void bracket(float* g_res, const float* dg_dx, const float* dJ0phi_dy,
			const float* dg_dy, const float* dJ0phi_dx, float kxfac)
{
  unsigned int idxyz = get_id1();
  unsigned int idj = get_id2();

  if (idxyz < nx*ny*nz && idj < nj) {
    unsigned int ig = idxyz + nx*ny*nz*idj;
    g_res[ig] = ( dg_dx[ig] * dJ0phi_dy[ig] - dg_dy[ig] * dJ0phi_dx[ig] ) * kxfac;
    
  }
}

# define LM(L, M, S) idxyz + nx*nyc*nz*((L) + nl*((M) + nm*(S)))
__global__ void beer_toroidal_closures(const cuComplex* g, cuComplex* gRhs,
				       const float* omegad,
				       const cuComplex* nu,
				       const float* tz)
{
  unsigned int idxyz = get_id1();

  if (idxyz < nx*nyc*nz) {
    const cuComplex iomegad = make_cuComplex(0., omegad[idxyz]);
    const float abs_omegad = abs(omegad[idxyz]);

    for (int is=0; is < nspecies; is++) {
      const float tz_ = tz[is];
      
      const cuComplex iwd_s = iomegad * tz_;
      const float awd_s = abs_omegad * tz_;

      gRhs[LM(0,2,is)] = gRhs[LM(0,2,is)]
	- sqrtf(2) * awd_s * ( nu[1].x*sqrtf(2)*g[LM(0,2,is)] + nu[2].x*g[LM(1,0,is)] )
	- sqrtf(2) * iwd_s * ( nu[1].y*sqrtf(2)*g[LM(0,2,is)] + nu[2].y*g[LM(1,0,is)] );
      
      gRhs[LM(1,0,is)] = gRhs[LM(1,0,is)]
	- 2. * awd_s * ( nu[3].x*sqrtf(2)*g[LM(0,2,is)] + nu[4].x*g[LM(1,0,is)] )
	- 2. * iwd_s * ( nu[3].y*sqrtf(2)*g[LM(0,2,is)] + nu[4].y*g[LM(1,0,is)] );
      
      gRhs[LM(0,3,is)] = gRhs[LM(0,3,is)]
	- 1./sqrtf(6) * awd_s * ( nu[5].x*g[LM(0,1,is)] + nu[6].x*sqrtf(6)*g[LM(0,3,is)] + nu[7].x*g[LM(1,1,is)] )
	- 1./sqrtf(6) * iwd_s * ( nu[5].y*g[LM(0,1,is)] + nu[6].y*sqrtf(6)*g[LM(0,3,is)] + nu[7].y*g[LM(1,1,is)] );
      
      gRhs[LM(1,1,is)] = gRhs[LM(1,1,is)]
	- awd_s * ( nu[8].x*g[LM(0,1,is)] + nu[9].x*sqrtf(6)*g[LM(0,3,is)] + nu[10].x*g[LM(1,1,is)] )
	- iwd_s * ( nu[8].y*g[LM(0,1,is)] + nu[9].y*sqrtf(6)*g[LM(0,3,is)] + nu[10].y*g[LM(1,1,is)] );
    }    
  }
}

__global__ void smith_perp_toroidal_closures(const cuComplex* g, cuComplex* gRhs,
					     const float* omegad, const cuComplex* Aclos, int q, const float* tz)
{
  unsigned int idxyz = get_id1();
  
  if (idxyz < nx*nyc*nz) {

    const cuComplex    iomegad = make_cuComplex(0., omegad[idxyz]);
    const float     abs_omegad = abs(omegad[idxyz]);
    
    for (int is=0; is < nspecies; is++) {
      const float tz_ = tz[is];

      const cuComplex iwd_s =    iomegad * tz_;
      const float     awd_s = abs_omegad * tz_;

      int L = nl - 1;
      
      // apply closure to Lth laguerre equation for all hermite moments
      for (int m=0; m < nm; m++) {

	// calculate closure expression as sum of lower laguerre moments
	cuComplex clos = make_cuComplex(0.,0.);
	for (int l=L; l>=nl-q; l--) clos = clos + (awd_s * Aclos[L-l].y + iwd_s * Aclos[L-l].x)*g[LM(l,m,is)];

	gRhs[LM(L,m,is)] = gRhs[LM(L,m,is)] - (L+1)*clos;
      }
    }
  }
}
 
__global__ void stirring_kernel(const cuComplex force, cuComplex *moments, int forcing_index)
{
  moments[forcing_index] = moments[forcing_index] + force; }

__global__ void yzavg(float *vE, float *vEavg, float *vol_fac)
{
  unsigned int idx = get_id1();
  if (idx < nx) {
    float avg = 0.;
    for (int idy = 0; idy<ny; idy++) {
      for (int idz = 0; idz<nz; idz++) {
	unsigned int ig = idy + ny*(idx + nx*idz);
	avg += vE[ig]*vol_fac[idz];
      }
    }
    float fac = 1./((float) ny);
    vEavg[idx] = avg*fac;
  }
}

__global__ void xytranspose(float *in, float *out)
{
  // Transpose to accommodate ncview
  unsigned int idy = get_id1();
  if (idy < ny) {
    unsigned int idx = get_id2();
    if (idx < nx) {
      out[idx + nx*idy] = in[idy + ny*idx];
    }
  }
}

__global__ void fieldlineaverage(cuComplex *favg, cuComplex *df, const cuComplex *f, const float *volJac)
{

  unsigned int idxy = get_id1();
  if (idxy < nx*nyc) {
    unsigned int idy = idxy % nyc;
    unsigned int idx = idxy / nyc;

    favg[idx] = make_cuComplex(0., 0.); 

    // calculate <<f>> 
    if (idy == 0 && unmasked(idx, idy)) {
      for (int idz = 0; idz<nz; idz++) {
	favg[idx] = favg[idx] + f[idxy + nx*nyc*idz] * volJac[idz];
      }
      for (int idz = 0; idz<nz; idz++) {
	df[idxy + nx*nyc*idz] = f[idxy + nx*nyc*idz] - favg[idx];
      }
    }
    
    if (idy > 0 && unmasked(idx, idy)) {
      for (int idz = 0; idz<nz; idz++) df[idxy + nx*nyc*idz] = f[idxy + nx*nyc*idz];
    } 

    if (masked(idx, idy)) {
      for (int idz = 0; idz<nz; idz++) df[idxy + nx*nyc*idz] = make_cuComplex(0., 0.);
    }
  }
}

__global__ void W_summand(float *G2, const cuComplex* g, const float* volJac, const float *nts) 
{
  unsigned int idxy = get_id1(); 
  if (idxy < nx*nyc) {
    unsigned int idz = get_id2();
    if (idz < nz) {
      unsigned int idslm = get_id3();
      unsigned int ig = idxy + nx*nyc*(idz + nz*idslm);
      unsigned int is = idslm / (nm*nl);

      unsigned int idy = idxy % nyc;
      unsigned int idx = idxy / nyc;// % nx;
      const float nt_ = nts[is];
      cuComplex fg;
      if (unmasked(idx, idy)) {

	float fac = 2.0;
	if (idy==0) fac = 1.0;
	fg = cuConjf(g[ig]) * g[ig] * volJac[idz] * fac;
	G2[ig] = 0.5 * fg.x * nt_;
      } else {
	G2[ig] = 0.;
      }
    }
  }
}

__global__ void vol_summand(float *rmom, const cuComplex* f, const cuComplex* g, const float* volJac)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  unsigned int idxyz = idy + nyc*(idx + nx*idz);

  if (idy < nyc && idx < nx && idz < nz) {
    if (unmasked(idx, idy)) {      
      cuComplex fg;  
      float fac=2.;
      if (idy==0) fac = 1.0;
      
      fg = cuConjf( f[idxyz] ) * g[idxyz] * volJac[idz] * fac;

      rmom[idxyz] = fg.x;
    } else {
      rmom[idxyz] = 0.;
    }
  }
}

__global__ void get_pzt(float* primary, float* secondary, float* tertiary, const cuComplex* phi, const cuComplex* tbar)
{
  float Psum = 0.;
  float Zsum = 0.;
  float Tsum = 0.;
  
  cuComplex P2, T2;
  for (unsigned int idxyz = blockIdx.x*blockDim.x+threadIdx.x;idxyz < nx*nyc*nz; idxyz += blockDim.x*gridDim.x) {

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
      if (idy==0) fac = 1.0;
      
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

__global__ void rescale_kernel(cuComplex* f, float* phi_max, int N)
{
  unsigned int idxy  = get_id1();
  unsigned int idz   = get_id2();
  unsigned int idlms = get_id3();
  
  if (idxy < nyc*nx && idz < nz && idlms < N) {
    float fac = 1./phi_max[idxy];
    unsigned int ig = idxy + nyc*nx*(idz + nz*idlms);
    f[ig] = fac * f[ig];
  }
}

// the following is only useful for linear runs
__global__ void maxPhi(float* phi_max, const cuComplex *phi)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  cuComplex tmp;
  float pmax = 0.0;
  if (idy < nyc && idx < nx) {
    if (unmasked(idx, idy)) {
      if (idy == 0) {
	phi_max[idy + nyc*idx] = 1.0;
	return;
      }
      for (int idz = 0; idz<nz; idz++) {
	unsigned int idxyz = idy + nyc*(idx + nx*idz);
	tmp = cuConjf( phi[idxyz] ) * phi[idxyz];
	pmax = max(pmax, tmp.x);
      }
      phi_max[idy + nyc*idx] = sqrtf(pmax);
    } else {
      phi_max[idy + nyc*idx] = 1.0; // avoid dumb problems with masked modes
    }
  }
}

__global__ void Wphi_scale(float* p2, float alpha)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  unsigned int idxyz = idy + nyc*(idx + nx*idz);

  if (idy < nyc && idx < nx && idz < nz) { 
    if (unmasked(idx, idy)) {
      p2[idxyz] *= alpha;
    } else {
      p2[idxyz] = 0.;
    }
  }
}

__global__ void Wphi2_summand(float *p2, const cuComplex *phi, const float *volJac)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  unsigned int idxyz = idy + nyc*(idx + nx*idz);

  if (idy < nyc && idx < nx && idz < nz) { 
    if (unmasked(idx, idy)) {    
      cuComplex tmp;
      float fac=2.;
      if (idy==0) fac = 1.0;

      tmp = cuConjf( phi[idxyz] ) * phi[idxyz] *fac * volJac[idz] ;
      p2[idxyz] = 0.5 * tmp.x;

    } else {
      p2[idxyz] = 0.;
    }
  }
}

__global__ void Wphi_summand(float* p2, const cuComplex* phi, const float* volJac, const float* kperp2, float rho2_s)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  unsigned int idxyz = idy + nyc*(idx + nx*idz);

  if (idy < nyc && idx < nx && idz < nz) { 
    if (unmasked(idx, idy)) {    
      cuComplex tmp;
      float fac=2.;
      if (idy==0) fac = 1.0;

      float b_s = kperp2[idxyz]*rho2_s;

      tmp = cuConjf( phi[idxyz] ) * ( 1.0 - g0(b_s) ) * phi[idxyz] * fac * volJac[idz];
      p2[idxyz] = 0.5 * tmp.x;

    } else {
      p2[idxyz] = 0.;
    }
  }
}

__global__ void Wphi_summand_krehm(float* p2, const cuComplex* phi, const float* volJac, const float* kx, const float* ky, float rho_i)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  unsigned int idxyz = idy + nyc*(idx + nx*idz);

  if (idy < nyc && idx < nx && idz < nz) { 
    if (unmasked(idx, idy)) {    
      cuComplex tmp;
      float fac=2.;
      if (idy==0) fac = 1.0;

      float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];
      float gam0 = g0(kperp2*rho_i*rho_i/2.);

      tmp = cuConjf( phi[idxyz] ) * ( 1.0 - gam0 ) * phi[idxyz] * fac * volJac[idz];
      p2[idxyz] = 0.5 * tmp.x;

    } else {
      p2[idxyz] = 0.;
    }
  }
}

# define Gh_(XYZ, L, M) g[(XYZ) + nx*nyc*nz*((L) + nl*(M))]
__global__ void heat_flux_summand(float* qflux, const cuComplex* phi, const cuComplex* g, const float* ky, 
				  const float* flxJac, const float *kperp2, float rho2_s, float pres)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  cuComplex fg;

  unsigned int idxyz = idy + nyc*(idx + nx*idz);
  if (idy < nyc && idx < nx && idz < nz) { 
    if (unmasked(idx, idy) && idy > 0) {    
      
      cuComplex vE_r = make_cuComplex(0., ky[idy]) * phi[idxyz];
    
      float b_s = kperp2[idxyz]*rho2_s;
    
      // sum over l
      cuComplex p_bar = make_cuComplex(0.,0.);

      for (int il=0; il < nl; il++) {
	p_bar = p_bar + Jfac(il, b_s)*Gh_(idxyz, il, 0) + rsqrtf(2.)*Jflr(il, b_s)*Gh_(idxyz, il, 2);
      }
    
      fg = cuConjf(vE_r) * p_bar * 2. * flxJac[idz];
      qflux[idxyz] = fg.x * pres;

    } else {
      qflux[idxyz] = 0.;
    }
  }
}

__global__ void part_flux_summand(float* pflux, const cuComplex* phi, const cuComplex* g, const float* ky, 
				  const float* flxJac, const float *kperp2, float rho2_s, float n_s)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  cuComplex fg;

  unsigned int idxyz = idy + nyc*(idx + nx*idz);
  if (idy < nyc && idx < nx && idz < nz) { 
    if (unmasked(idx, idy) && idy > 0) {    
      
      cuComplex vE_r = make_cuComplex(0., ky[idy]) * phi[idxyz];
    
      float b_s = kperp2[idxyz]*rho2_s;
    
      // sum over l
      cuComplex n_bar = make_cuComplex(0.,0.);

      for (int il=0; il < nl; il++) {
	n_bar = n_bar + Jflr(il, b_s)*Gh_(idxyz, il, 0);
      }
    
      fg = cuConjf(vE_r) * n_bar * 2. * flxJac[idz];
      pflux[idxyz] = fg.x * n_s;

    } else {
      pflux[idxyz] = 0.;
    }
  }
}

__global__ void kInit(float* kx, float* ky, float* kz, int* kzm, float* kzp, const float X0, const float Y0, const int Zp, bool dealias_kz) 
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  if (id < nyc) {
    ky[id] = (float) id/Y0; if (false) printf("ky[%d] = %f \t ",id, ky[id]);
  }
  if (id < nx/2+1) {
    kx[id] = (float) id/X0; if (false) printf("kx[%d] = %f \t ",id, kx[id]);
  } else if (id < nx) {
    kx[id] = (float) (id - nx)/X0; if (false) printf("kx[%d] = %f \t ", id, kx[id]);
  }
  if (id < (nz/2+1)) {
    kz[id] = (float) id/Zp; if (false) printf("kz[%d] = %f \n ", id, kz[id]);
  } else if (id < nz) {
    kz[id] = (float) (id - nz)/Zp; if (false) printf("kz[%d] = %f \n ", id, kz[id]);
  }

  if (id < (nz-1)/3+1)                     {kzm[id] = 1; kzp[id] = kz[id];}
  if (dealias_kz) {
    if (id > (nz-1)/3 && id < nz - (nz-1)/3) {kzm[id] = 0; kzp[id] = 0.;}
  } else {
    if (id > (nz-1)/3 && id < nz - (nz-1)/3) {kzm[id] = 0; kzp[id] = kz[id];}
  }
  if (id-nz > -(1 + (nz-1)/3) && id < nz)  {kzm[id] = 1; kzp[id] = kz[id];}
  
  if (false) printf("\n");

}


__global__ void ampere(cuComplex* Apar,
		       const cuComplex* g,
		       const float* kperp2,
		       const float* rho2s,
		       const float* as,
		       const float* amps,
		       const float beta)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  if ( unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = idy + nyc*(idx + nx*idz); // spatial index
    
    cuComplex jpar;    jpar = make_cuComplex(0., 0.);
        
    float denom = kperp2[idxyz];
    for (int is=0 ; is < nspecies; is++) {
      const float b_s = kperp2[idxyz] * rho2s[is];
      const float j_ = as[is]; // as = n_s*z_s*vt_s*beta_ref/2
      const float amp_ = amps[is]; // amps = n_s*z_s^2/m_s*beta_ref/2
      float g0_s = 0.;
      for (int l=0; l < nl; l++) {
	unsigned int m = 1; // only m=1 components needed here
	unsigned int ig = idxyz + nx*nyc*nz*(l + nl*(m + nm*is));
	const float Jl = Jflr(l, b_s);
	jpar = jpar + j_ * Jl * g[ig];
	g0_s += Jl*Jl;
      }
      // this term is needed when using formulation without dA/dt
      denom += amp_ * g0_s;
    }        
    Apar[idxyz] = jpar / denom;
  }
}

__global__ void real_space_density(cuComplex* nbar, const cuComplex* g, const float *kperp2,
				   const float *rho2s, const float *nzs)
{
  unsigned int idy = get_id1();
  if (idy < nyc) {
    unsigned int idx = get_id2();
    if (idx < nx) {
      if (unmasked(idx, idy)) {
	unsigned int idz = get_id3();
	unsigned int idxyz = idy + nyc*(idx + nx*idz);
	for (int is=0; is < nspecies; is++) {
	  const float b_s = kperp2[idxyz] * rho2s[is];
	  for (int l=0; l < nl; l++) {
	    unsigned int ig = idxyz + nx*nyc*nz*(l + nl*nm*is);
	    nbar[idxyz] = nbar[idxyz] + Jflr(l, b_s) * g[ig] * nzs[is];
	  }
	}
      }
    }
  }
}

__global__ void qneut(cuComplex* Phi, const cuComplex* g, const float* kperp2,
		      const float* rho2s, const float* qn, const float* nzs)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  if ( unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = idy + nyc*(idx + nx*idz); 
    
    cuComplex nbar;    nbar = make_cuComplex(0., 0.);
    float denom = 0.;
        
    for (int is=0 ; is < nspecies; is++) {
      const float b_s = kperp2[idxyz] * rho2s[is];
      const float qn_ = qn[is]; // qn = n_s*z_s^2/T_s
      const float nz_ = nzs[is];

      float g0_s = 0.;
      for (int l=0; l < nl; l++) {
	unsigned int m = 0; // only m=0 components are needed here
	unsigned int ig = idxyz + nx*nyc*nz*(l + nl*(m + nm*is));
	const float Jl = Jflr(l, b_s);
	nbar = nbar + Jl * g[ig] * nz_;
	g0_s += Jl*Jl;
      }
      denom += qn_ * ( 1. - g0_s );
    }    
    
    Phi[idxyz] = nbar / denom;    
  }
}

/*
========================

Given 

phi = <<phi>> + D      (1)

we have

nbar = tau (phi - <<phi>>) + (1-G) phi

or 

nbar = tau phi + (1-G) phi - tau << phi >>.          (2) 

Use (1) to rewrite this as 

nbar = tau <<phi>> + tau D + (1-G) <<phi>> + (1-G) D - tau <<phi>>

nbar = tau D + (1-G) <<phi>> + (1-G) D         

<<phi>> = [ nbar - tau D ] / (1-G).

This is fine, but solve for D to make it useful:

D = nbar/(tau + 1 - G) + (1-G)/(tau + 1 - G) <<phi>>

Since by definition <<D>> = 0, we have 

<<nbar/(tau + 1 - G)>> = << (1-G)/(tau+1-G)>> <<phi>>

so <<phi>> = <<nbar/(tau + 1 - G)>> / << (1-G)/(tau+1-G) >>.    (3)

Use (2) with (3): 

nbar = (tau + 1 - G) phi - tau <<phi>>

phi = (nbar + tau <<phi>>) / (tau + 1 - G)  (4)

Eq. (4) is the result. Note that there is an implied sum over 
species everywhere (1-G) appears, and that nbar is the actual 
density from summing over all the ions at fixed r. 


The denominator of the RHS of Eq. (3) is precalculated when 
the solver object is instantiated using the calc_phiavgdenom
kernel here.

The <<...>> averages below can be off by a constant as they 
end up in ratios.

The variables in parts 1 and 2 of the quasineutrality solver 
for iphi00 == 2 are:

====

nbar = tau phi - tau <<phi>> + (1-G) phi

phi = [ nbar + tau <<phi>> ] / (tau + 1 - G)

phi = nbar / (tau + 1 - G) + tau <<nbar/(tau+1-G)>> / <<(1-G)/(tau + 1 - G)>>

PhiAvgNum_tmp is integrand for second term on RHS:

PhiAvgNum_tmp = J nbar / (tau + 1 - G)

PhiAvgNum_zSum = int dz PhiAvgNum_tmp == <<nbar/(tau+1-G)>>

PhiAvgDenom == <<(1-G)/(tau + 1 - G)>>

PhiAvg == <<nbar/(tau+1-G)>> / <<(1-G)/(tau + 1 - G)>> == <<phi>>

Phi = nbar / (tau + 1 - G) + tau PhiAvg / (tau + 1 - G)

=====================
*/

/*
This proposed substitute routine fails because pfilter2 is not thread-local memory

__global__ void qneut_fieldlineaveraged(cuComplex *Phi, const cuComplex *nbar, const float *PhiAvgDenom, 
					const float *kperp2, const float *jacobian,
					const specie *species, const float tau_fac, float *pfilter2)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();

  if ( unmasked(idx, idy)) {
    unsigned int idxy  = idy + nyc*idx;

    cuDoubleComplex PhiAvg; // this is thread-local
    PhiAvg.x = (double) 0.;
    PhiAvg.y = (double) 0.;
   
    for (int idz=0; idz < nz; idz++) {
      pfilter2[idz] = 0.;
      for (int is=0; is < nspecies; is++) {
	specie s = species[is];
	pfilter2[idz] += s.dens*s.z*s.zt*( 1. - g0(kperp2[idxy + idz*nx*nyc]*s.rho2) );
      }
    }
    
    if (idy == 0) { 
      for (int idz=0; idz < nz; idz++) {
	int idxyz = idxy + idz*nx*nyc;	
	PhiAvg.x = (double) PhiAvg.x + ( nbar[idxyz].x / (tau_fac + pfilter2[idz] ) ) * jacobian[idz];
	PhiAvg.y = (double) PhiAvg.y + ( nbar[idxyz].y / (tau_fac + pfilter2[idz] ) ) * jacobian[idz];
      }      
      PhiAvg.x = PhiAvg.x/( (double) PhiAvgDenom[idx] );
      PhiAvg.y = PhiAvg.y/( (double) PhiAvgDenom[idx] );      
    }
    
    for (int idz=0; idz < nz; idz++) {
      int idxyz = idxy + idz*nx*nyc;
      Phi[idxyz].x = ( nbar[idxyz].x + tau_fac * PhiAvg.x ) / (tau_fac + pfilter2[idz]);
      Phi[idxyz].y = ( nbar[idxyz].y + tau_fac * PhiAvg.y ) / (tau_fac + pfilter2[idz]);
    }
  }
  
  // No need for the memset to zero if we do this:
  if ( masked(idx, idy)) {
    for (int idz=0; idz < nz; idz++) {
      unsigned int idxyz = idy + nyc*idx + nx*nyc*idz;
      Phi[idxyz].x = 0.;
      Phi[idxyz].y = 0.;
    }
  }
}
*/
__global__ void qneutAdiab_part1(cuComplex* PhiAvgNum_tmp, const cuComplex* nbar,
				 const float* kperp2, const float* jacobian,
				 const float* rho2s, const float* qns, float tau_fac)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  if ( unmasked(idx, idy) && idz < nz) {
  
    unsigned int idxyz = idy + nyc*(idx + nx*idz);
    float pfilter2 = 0.;
    
    for (int is=0; is < nspecies; is++) {
      float b_s = kperp2[idxyz] * rho2s[is]; 
      pfilter2 += qns[is] * ( 1. - g0(b_s) );
    }
    
    PhiAvgNum_tmp[idxyz] = ( nbar[idxyz] / (tau_fac + pfilter2 ) ) * jacobian[idz];
  }
}


__global__ void qneutAdiab_part2(cuComplex* Phi, const cuComplex* PhiAvgNum_tmp, const cuComplex* nbar,
				 const float* PhiAvgDenom, const float* kperp2,
				 const float* rho2s, const float* qns, float tau_fac)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();
  
  if ( unmasked(idx, idy) && idz < nz) {

    unsigned int idxyz = idy + nyc*(idx + nx*idz);
    unsigned int idxy  = idy + nyc*idx;

    float pfilter2 = 0.;
    
    for (int is=0; is < nspecies; is++) {
      float b_s = kperp2[idxyz] * rho2s[is]; 
      pfilter2 += qns[is]*( 1. - g0(b_s) );
    }

    // This is okay because PhiAvgNum_zSum is local to each thread
    cuDoubleComplex PhiAvgNum_zSum;
    PhiAvgNum_zSum.x = (double) 0.;
    PhiAvgNum_zSum.y = (double) 0.;

    // inefficient
    for (int i=0; i < nz; i++) {
      PhiAvgNum_zSum.x = (double) PhiAvgNum_zSum.x + PhiAvgNum_tmp[idxy + i*nx*nyc].x; // numerator of Eq. 3
      PhiAvgNum_zSum.y = (double) PhiAvgNum_zSum.y + PhiAvgNum_tmp[idxy + i*nx*nyc].y;
    }
    
    cuDoubleComplex PhiAvg;	
    if (idy == 0 && idx != 0) {
      PhiAvg.x = PhiAvgNum_zSum.x/( (double) PhiAvgDenom[idx] ); // Eq. 3
      PhiAvg.y = PhiAvgNum_zSum.y/( (double) PhiAvgDenom[idx] ); 
    } else {
      PhiAvg.x = 0.; PhiAvg.y = 0.;
    }

    Phi[idxyz].x = ( nbar[idxyz].x + tau_fac * PhiAvg.x ) / (tau_fac + pfilter2); // Eq 4
    Phi[idxyz].y = ( nbar[idxyz].y + tau_fac * PhiAvg.y ) / (tau_fac + pfilter2);
  }
}

__global__ void calc_phiavgdenom(float* PhiAvgDenom, const float* kperp2, const float* jacobian,
				 const float* rho2s, const float* qns, float tau_fac)
{   
  unsigned int idx = get_id1();
  
  if ( idx > 0 && idx < nx) {
    int ikx = get_ikx(idx);
    if (ikx <  (nx-1)/3+1 && ikx > -(nx-1)/3-1) {

      float pfilter2;
      PhiAvgDenom[idx] = 0.;  

      for (int idz=0; idz < nz; idz++) {
	pfilter2 = 0.;
	for (int is=0; is < nspecies; is++) {
	  int idy = 0;
	  int idxyz = idy + nyc*(idx + idz*nx);
	  float b_s = kperp2[idxyz] * rho2s[is];
	  pfilter2 += qns[is] * (1. - g0(b_s));
	}	
	PhiAvgDenom[idx] = PhiAvgDenom[idx] + jacobian[idz] * pfilter2 / (tau_fac + pfilter2);
      }
    }
  }
}

// Is this really set up correctly? 
__global__ void add_source(cuComplex* f, const float source)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();
  
  if ( unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = idy + nyc*(idx + nx*idz);
    f[idxyz].x = f[idxyz].x + source;
  }
}

__global__ void qneutAdiab(cuComplex* Phi, const cuComplex* nbar,
			   const float* kperp2, const float* rho2s, const float* qns, 
			   float tau_fac)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  if ( unmasked(idx, idy) && idz < nz) {
    
    unsigned int idxyz = idy + nyc*(idx + nx*idz);
    float pfilter2 = 0.;
    
    for (int is=0; is < nspecies; is++) {
      float b_s = kperp2[idxyz] * rho2s[is] ; 
      pfilter2 += qns[is] * ( 1. - g0(b_s));
    }
    
    Phi[idxyz].x = ( nbar[idxyz].x / (tau_fac + pfilter2 ) );  
    Phi[idxyz].y = ( nbar[idxyz].y / (tau_fac + pfilter2 ) );  
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

__device__ void zfts_Linked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  float *kz  = (float*) kzData;
  int nLinks = (int) lrintf(1./(zp*kz[1]));
  float normalization = (float) 1./(nz*nLinks);
  ((cuComplex*)dataOut)[offset] = element*normalization;
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
  for (int i=0; i < nz*nLinks; i++) {
    if (i < nz*nLinks/2+1) {
      kz[i] = (float) i/(zp*nLinks);
    } else {
      kz[i] = (float) (i-nz*nLinks)/(zp*nLinks);
    }
  }
}

__managed__ cufftCallbackStoreC  zfts_Linked_callbackPtr = zfts_Linked;
__managed__ cufftCallbackStoreC   i_kzLinked_callbackPtr = i_kzLinked;
__managed__ cufftCallbackStoreC abs_kzLinked_callbackPtr = abs_kzLinked;

__global__ void linkedCopy(const cuComplex* G, cuComplex* G_linked,
			   int nLinks, int nChains, const int* ikx, const int* iky, int nMoms)
{
  unsigned int idz  = get_id1();
  unsigned int idk  = get_id2();
  unsigned int idlm = get_id3();

  if (idz < nz && idk < nLinks*nChains && idlm < nMoms) {
    unsigned int idlink = idz + nz*(idk + nLinks*nChains*idlm);
    unsigned int globalIdx = iky[idk] + nyc*(ikx[idk] + nx*(idz + nz*idlm));
    // NRM: seems hopeless to make these accesses coalesced. how bad is it?
    G_linked[idlink] = G[globalIdx];
  }
}

__global__ void linkedCopyBack(const cuComplex* G_linked, cuComplex* G,
			       int nLinks, int nChains, const int* ikx, const int* iky, int nMoms)
{
  unsigned int idz = get_id1();
  unsigned int idk = get_id2();
  unsigned int idlm = get_id3();

  if (idz < nz && idk < nLinks*nChains && idlm < nMoms) {
    unsigned int idlink = idz + nz*(idk + nLinks*nChains*idlm);
    //    unsigned int globalIdx = iky[idk] + nyc*ikx[idk] + idz*nx*nyc + idlm*nx*nyc*nz;
    unsigned int globalIdx = iky[idk] + nyc*(ikx[idk] + nx*(idz + nz*idlm));

    G[globalIdx] = G_linked[idlink];
  }
}

__global__ void streaming_rhs(const cuComplex* g, const cuComplex* phi, const cuComplex* apar, const float* kperp2, const float* rho2s, 
			      const float gradpar, const float* vt, const float* zt, cuComplex* rhs_par)
{
  unsigned int idy  = get_id1();
  unsigned int idx  = get_id2();
  unsigned int idzl = get_id3();
  if ((idy < nyc) && (idx < nx) && unmasked(idx, idy) && (idzl < nz*nl)) {

    int m = 0;       // m = 0 case
    for (int is = 0; is < nspecies; is++) {
      const float vt_ = vt[is];
      unsigned int globalIdx = idy + nyc*( idx + nx*(idzl + nz*nl*(m   + nm * is)));
      unsigned int mp1       = idy + nyc*( idx + nx*(idzl + nz*nl*(m+1 + nm * is)));
      rhs_par[globalIdx] = -vt_ * sqrtf(m+1) * g[mp1] * gradpar;
    }
        
    m = nm - 1;     // m = nm-1 case
    for (int is = 0; is < nspecies; is++) {
      const float vt_ = vt[is];
      unsigned int globalIdx = idy + nyc*( idx + nx*(idzl + nz*nl*(m   + nm * is)));
      unsigned int mm1       = idy + nyc*( idx + nx*(idzl + nz*nl*(m-1 + nm * is)));
      rhs_par[globalIdx] = -vt_ * sqrtf(m) * g[mm1]  * gradpar;
    }
    
    for (int m = 1; m < nm-1; m++) {
      for (int is = 0; is < nspecies; is++) {
	const float vt_ = vt[is];
	unsigned int globalIdx = idy + nyc*( idx + nx*(idzl + nz*nl*(m   + nm * is)));	
	unsigned int mp1       = idy + nyc*( idx + nx*(idzl + nz*nl*(m+1 + nm * is)));
	unsigned int mm1       = idy + nyc*( idx + nx*(idzl + nz*nl*(m-1 + nm * is)));
	
	rhs_par[globalIdx] = -vt_ * (sqrtf(m+1)*g[mp1] + sqrtf(m)*g[mm1]) * gradpar;
      }
    }

    // field terms
    unsigned int idz = idzl % nz;     
    unsigned int l   = idzl / nz;
    unsigned int idxyz = idy + nyc*(idx + nx*idz);
    const cuComplex phi_ = phi[idxyz];
    const cuComplex apar_ = apar[idxyz];
    for (int is = 0; is < nspecies; is++) {
      const float vt_ = vt[is];
      const float zt_ = zt[is]; 
      const float b_s = rho2s[is] * kperp2[idxyz];
      
      m = 1;          // m = 1 has Phi term
      if (nm > 1) {
        unsigned int globalIdx = idy + nyc*( idx + nx*(idzl + nz*nl*(m + nm * is)));
	rhs_par[globalIdx] = rhs_par[globalIdx] - Jflr(l, b_s) * phi_ * zt_ * vt_ * gradpar;
      }

      // the following Apar terms are only needed in the formulation without dA/dt
      m = 0;          // m = 0 has Apar term
      unsigned int globalIdx = idy + nyc*( idx + nx*(idzl + nz*nl*(m + nm * is)));
      rhs_par[globalIdx] = rhs_par[globalIdx] - Jflr(l, b_s) * apar_ * zt_ * vt_ * vt_ * gradpar;

      m = 2;          // m = 2 has Apar term
      if (nm > 2) {
	unsigned int globalIdx = idy + nyc*( idx + nx*(idzl + nz*nl*(m + nm * is)));
	rhs_par[globalIdx] = rhs_par[globalIdx] - sqrtf(2.) * Jflr(l, b_s) * apar_ * zt_ * vt_ * vt_ * gradpar;
      }
    }    
  }
}

// main kernel function for calculating RHS
# define S_H(L, M) s_h[sidxyz + (sDimx)*(L) + (sDimx)*(sDimy)*(M)]
__global__ void rhs_linear(const cuComplex* g, const cuComplex* phi, const cuComplex* apar,
			   const cuComplex* upar_bar, const cuComplex* uperp_bar, const cuComplex* t_bar,
			   const float* kperp2, const float* cv_d, const float* gb_d, const float* bgrad,
			   const float* ky, const float* vt, const float* zt, const float* tz, const float* nzs, const float* as,
			   const float* nu_ss, const float* tprim, const float* uprim, const float* fprim,
			   const float* rho2s, const int* typs, cuComplex* rhs, bool hegna)  // bb6126 - hegna test
{
  extern __shared__ cuComplex s_h[]; // aliased below by macro S_H, defined above
  
  //  unsigned int idxyz = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int idxyz = get_id1();
  const unsigned int idy = idxyz % nyc; 
  const unsigned int idx = idxyz / nyc % nx;
  if (idxyz < nx*nyc*nz && unmasked(idx, idy)) {
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
    const cuComplex apar_ = apar[idxyz];
  
    // all threads in a block will likely have same value of idz, so they will be reading same value of bgrad[idz].
    // if bgrad were in shared memory, would have bank conflicts.
    // no bank conflicts for reading from global memory though. 
    const float bgrad_ = bgrad[idz];  
  
    // this is coalesced
    const cuComplex iky_ = make_cuComplex(0., ky[idy]); 

    unsigned int nR = nyc * nx * nz;
    
   for (int is=0; is < nspecies; is++) { // might be a better way to handle species loop here...
     //     specie s = species[is];
     
     // species-specific constants
     const float vt_ = vt[is];
     const float zt_ = zt[is];
     const float tz_ = tz[is];
     const float nz_ = nzs[is];
     const float nu_ = nu_ss[is]; 
     const float tprim_ = tprim[is];
     const float uprim_ = uprim[is];
     const float fprim_ = fprim[is];
     const float kperp2_ = kperp2[idxyz];
     const float b_s = kperp2_ * rho2s[is];
     float nuei_ = 0.0;
     float as_i = 1.0;
     float nzvt_i = 1.0;
     // for electrons, account for e-i collisions
     if(typs[is] == 1) {
       nuei_ = nu_;
       // get as = z*n*vt*beta/2 from first ion species (assume this is main ions)
       for(int j=0; j<nspecies; j++) {
         if(typs[j] == 0) {as_i = as[j]; nzvt_i = nzs[j]*vt[j]; break;}
       }
     }
     
     const cuComplex icv_d_s = 2. * tz_ * make_cuComplex(0., cv_d[idxyz]);
     const cuComplex igb_d_s = 2. * tz_ * make_cuComplex(0., gb_d[idxyz]);

     // conservation terms (species-specific)
     cuComplex upar_bar_  =  upar_bar[idxyz + is * nR]; 
     cuComplex uperp_bar_ = uperp_bar[idxyz + is * nR];
     cuComplex t_bar_     =     t_bar[idxyz + is * nR];
     
     // read tile of g into shared mem
     // each thread in the block reads in multiple values of l and m
     // blockIdx for y and z and both of size unity in the kernel invocation
     for (int m = threadIdx.z; m < nm; m += blockDim.z) {
       for (int l = threadIdx.y; l < nl; l += blockDim.y) {
	 unsigned int globalIdx = idxyz + nR*(l + nl*(m + nm*is));
	 int sl = l + 1;
	 int sm = m + 2;
	 S_H(sl, sm) = g[globalIdx];
	 // add phi term for m=0 to change g into h
	 if (m==0) S_H(sl, sm) = S_H(sl, sm) + zt_*Jflr(l, b_s)*phi_;
	 // add apar term for m=1 (this is only needed in the formulation without dA/dt)
	 if (m==1) S_H(sl, sm) = S_H(sl, sm) - zt_*vt_*Jflr(l, b_s)*apar_;
       }
     }
      
     // this syncthreads is not necessary unless ghosts require information from interior cells
     //     __syncthreads();
     
     // set up ghost cells in m (for all l's)
     // blockIdx.y is of size unity in the kernel invocation
     for (int l = threadIdx.y; l < nl; l += blockDim.y) {
       int sl = l + 1;
       int sm = threadIdx.z + 2;
       if (sm < 4) {
	 // set ghost to zero at low m
	 S_H(sl, sm-2) = make_cuComplex(0., 0.);
	 
	 // set ghost with closures at high m
	 S_H(sl, sm+nm) = make_cuComplex(0., 0.);
       }
     }
     
     // set up ghost cells in l (for all m's)
     // blockIdx.z is unity in the kernel invocation
     for (int m = threadIdx.z; m < nm+2; m += blockDim.z) {
       int sm = m + 1; // this takes care of corners...
       int sl = threadIdx.y + 1;
       if (sl < 2) {
	 // set ghost to zero at low l
	 S_H(sl-1, sm) = make_cuComplex(0., 0.);
	 
	 // set ghost with closures at high l
	 S_H(sl+nl, sm) = make_cuComplex(0., 0.);
       }
     }
     
     __syncthreads();
     
     // stencil (on non-ghost cells)
     // blockIdx for y and z are unity in the kernel invocation
     for (int m = threadIdx.z; m < nm; m += blockDim.z) {
       for (int l = threadIdx.y; l < nl; l += blockDim.y) {
	 unsigned int globalIdx = idxyz + nR*(l + nl*(m + nm*is));
	 int sl = l + 1; // offset to get past ghosts
	 int sm = m + 2; // offset to get past ghosts
  
	 rhs[globalIdx] = rhs[globalIdx] 
	   - vt_ * bgrad_ * ( - sqrtf(m+1)*(l+1)*S_H(sl,sm+1) - sqrtf(m+1)* l   *S_H(sl-1,sm+1)  
                              + sqrtf(m  )* l   *S_H(sl,sm-1) + sqrtf(m  )*(l+1)*S_H(sl+1,sm-1) )
  
	   - icv_d_s * ( sqrtf((m+1)*(m+2))*S_H(sl,sm+2) + (2*m+1)*S_H(sl,sm) + sqrtf(m*(m-1))*S_H(sl,sm-2) )
	   - igb_d_s * (              (l+1)*S_H(sl+1,sm) + (2*l+1)*S_H(sl,sm)              + l*S_H(sl-1,sm) )
	   
	   - (nu_ + nuei_) * ( b_s + 2*l + m ) * ( S_H(sl,sm) );

	 // add drive and conservation terms in low hermite moments
	 if (m==0) {
	   rhs[globalIdx] = rhs[globalIdx] 
            + iky_ * phi_ * (
               Jflr(l-1,b_s)*l*tprim_
	     + Jflr(l,  b_s)*(fprim_ + 2*l*tprim_)
	     + Jflr(l+1,b_s,false)*(l+1)*tprim_ 
	    )
	    + nu_ * sqrtf(b_s) * ( Jflr(l, b_s) + Jflr(l-1, b_s) ) * uperp_bar_
	    + nu_ * 2. * ( l*Jflr(l-1,b_s) + 2.*l*Jflr(l,b_s) + (l+1)*Jflr(l+1,b_s) ) * t_bar_; 
	 }

	 // bb6126 - hegna test

	 if (m==1) {
	   cuComplex upar_bar_i = (nspecies>1 && as_i>0) ? kperp2_*apar_/as_i - nz_*vt_*upar_bar_/(nzvt_i) : make_cuComplex(0.,0.);

	   rhs[globalIdx] = rhs[globalIdx] 
            + vt_ * iky_ * apar_ * (
               Jflr(l-1,b_s)*l*tprim_
	     + Jflr(l,  b_s)*(fprim_ + (2*l+1)*tprim_)
	     + Jflr(l+1,b_s,false)*(l+1)*tprim_ 
	    )
      	    + Jflr(l,b_s) * (nu_*upar_bar_ + nuei_*upar_bar_i)
            + phi_ * Jflr(l,b_s) * uprim_ * iky_ / vt_; // need to set uprim_ more carefully; this is a placeholder
	 }
	 if (m==2) {
	   rhs[globalIdx] = rhs[globalIdx] + iky_*phi_*Jflr(l,b_s)/sqrtf(2.)*tprim_ 
	      + nu_ * sqrtf(2.) * Jflr(l,b_s) * t_bar_;
	 }  

	 if (m==3) {
	   rhs[globalIdx] = rhs[globalIdx] 
            + vt_ * iky_ * apar_ * sqrtf(3./2.) * tprim_ * Jflr(l,b_s);
         }
       } // l loop
     } // m loop
  
   } // species loop
  } // idxyz < NxNycNz
}

__global__ void rhs_linear_krehm(const cuComplex* g, const cuComplex* phi, const cuComplex* apar, 
			  const float nu_ei, const float rhos, const float de, cuComplex* rhs_par)
{
  unsigned int idy  = get_id1();
  unsigned int idx  = get_id2();
  unsigned int idz = get_id3();
  const float rhos_ov_de = rhos/de;
  if ((idy < nyc) && (idx < nx) && unmasked(idx, idy) && (idz < nz)) {

    int m = 0;       // m = 0 case
    unsigned int globalIdx = idy + nyc*( idx + nx*(idz + nz*(m  )));
    unsigned int mp1       = idy + nyc*( idx + nx*(idz + nz*(m+1)));
    rhs_par[globalIdx] = -rhos_ov_de * sqrtf(m+1) * g[mp1];
    
    m = nm - 1;     // m = nm-1 case
    globalIdx = idy + nyc*( idx + nx*(idz + nz*(m  )));
    unsigned int mm1       = idy + nyc*( idx + nx*(idz + nz*(m-1)));
    rhs_par[globalIdx] = -rhos_ov_de * sqrtf(m) * g[mm1];
    
    for (int m = 1; m < nm-1; m++) {
       globalIdx = idy + nyc*( idx + nx*(idz + nz*(m  )));	
       mp1       = idy + nyc*( idx + nx*(idz + nz*(m+1)));
       mm1       = idy + nyc*( idx + nx*(idz + nz*(m-1)));
       
       rhs_par[globalIdx] = -rhos_ov_de * (sqrtf(m+1)*g[mp1] + sqrtf(m)*g[mm1]);
       // collision term
       if(m!=2) rhs_par[globalIdx] = rhs_par[globalIdx] - nu_ei*m*g[globalIdx];
    }

    // additional field terms in m<=2 equations
    unsigned int idxyz = idy + nyc*(idx + nx*idz);
    const cuComplex phi_ = phi[idxyz];
    const cuComplex apar_ = apar[idxyz];

    m = 0;          
    globalIdx = idy + nyc*( idx + nx*(idz + nz*(m)));
    rhs_par[globalIdx] = rhs_par[globalIdx] - apar_/(de*de);

    m = 1;          
    if (nm > 1) {
      globalIdx = idy + nyc*( idx + nx*(idz + nz*(m)));
      rhs_par[globalIdx] = rhs_par[globalIdx] + phi_/(rhos*de) - nu_ei*apar_/(rhos*de);
    }    

    m = 2;         
    if (nm > 2) {
      globalIdx = idy + nyc*( idx + nx*(idz + nz*(m)));
      rhs_par[globalIdx] = rhs_par[globalIdx] - sqrtf(2)*apar_/(de*de);
    }    
  }
}





__global__ void hyperdiff(const cuComplex* g, const float* kx, const float* ky,
			  float nu_hyper, float D_hyper, cuComplex* rhs) {

  unsigned int idxyz = get_id1();

  if (idxyz < nx*nyc*nz) {
    unsigned int idy = idxyz % nyc;
    unsigned int idx = (idxyz / nyc) % nx;
    if (unmasked(idx, idy)) {	
      float kxmax = kx[(nx-1)/3];
      float kymax = ky[(ny-1)/3];
      float k2s = 1./pow((kxmax*kxmax + kymax*kymax), nu_hyper);      
      float Dfac = D_hyper*k2s*pow((kx[idx]*kx[idx] + ky[idy]*ky[idy]), nu_hyper);
      
      unsigned int l = get_id2();
      if (l<nl) {
	unsigned int m = get_id3();
	if (m<nm) {
	  for (int is=0; is < nspecies; is++) {
	    unsigned int ig = idxyz + nx*nyc*nz*(l + nl*(m + nm*is));
	    rhs[ig] = rhs[ig] - Dfac * g[ig];
	  }
	}
      }
    }
  }
}

__global__ void hypercollisions(const cuComplex* g, const float nu_hyper_l, const float nu_hyper_m,
				const int p_hyper_l, const int p_hyper_m, const float* vt, cuComplex* rhs) {
  unsigned int idxyz = get_id1();
  
  if (idxyz < nx*nyc*nz) {
    for (int is=0; is < nspecies; is++) { 
      float scaled_nu_hyp_l = (float) nl * nu_hyper_l;
      float scaled_nu_hyp_m = (float) nm * nu_hyper_m; // scaling appropriate for curvature. Too big for slab
      // blockIdx for y and z are unity in the kernel invocation      
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
	for (int l = threadIdx.y; l < nl; l += blockDim.y) {
	  int globalIdx = idxyz + nx*nyc*nz*(l + nl*(m + nm*is)); 
	  if (m>2 || l>1) {
	    rhs[globalIdx] = rhs[globalIdx] -
	      (scaled_nu_hyp_l*pow((float) l/nl, (float) p_hyper_l)
	       +scaled_nu_hyp_m*pow((float) m/nm, p_hyper_m))*g[globalIdx];
	  }
	}
      }
    }
  }
}

__global__ void get_s1 (float* s10, float* s11, const float* kx, const float* ky, const cuComplex* df, float w_osc)
{
  // non-zonal shearing zonal:    
  // S10(z) = { << sum_(kx, ky!=0) 2 ky**4 |phi_1|**2 >> }          (sums ky!=0, so * 2)
  //                            
  // non-zonal shearing non-zonal:
  // S11(z) = { sum_(kx, ky!=0) 2 [(kx**2 + ky**2)**2 |phi_1|**2] } (sums ky!=0, so * 2)
  //
  unsigned int idz = get_id1();
  if (idz < nz) {
    s10[idz] = 0.;
    s11[idz] = 0.;
    for (int idy = 1; idy < nyc; idy++) {
      for (int idx = 0; idx < nx; idx++) {
	if (unmasked(idx, idy)) {
	  unsigned int idxyz = idy + nyc*(idx + nx*idz);
	  float kp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];
	  float df2 = df[idxyz].x*df[idxyz].x + df[idxyz].y*df[idxyz].y;

	  s10[idz] += 2. * powf(ky[idy], 4) * df2; 
	  s11[idz] += 2. * powf(kp2, 2)     * df2;
	}
      }      
    }
    s10[idz] = 0.5 * (-w_osc + sqrtf(powf(w_osc, 2) + 2 * s10[idz]));
    s11[idz] = 0.5 * (-w_osc + sqrtf(powf(w_osc, 2) + 2 * s11[idz]));
  }
}

__global__ void get_s01 (float* s01, const cuComplex* favg, const float* kx, const float w_osc) {
  s01[0] = 0.;
  for (int idx = 0; idx < nx; idx++) {
    s01[0] += pow(kx[idx], 4) * (favg[idx].x*favg[idx].x + favg[idx].y*favg[idx].y);
  }
  s01[0] = 0.5 * (-w_osc + sqrtf(powf(w_osc, 2) + 2 * s01[0]));
}

__global__ void HB_hyper (const cuComplex* G, const float* s01, const float* s10, const float* s11,
			  const float* kx, const float* ky, const float D_HB, const int p_HB, cuComplex* RHS)
{
  unsigned int idy = get_id1();
  if (idy < nyc) {
    unsigned int idxz = get_id2();
    if (idxz < nx*nz) {
      unsigned int idx = idxz % nx;
      unsigned int idz = idxz / nx;
      if (unmasked(idx, idy)) {
	unsigned int idlms = get_id3();
	if (idlms < nl*nm*nspecies) {
	  unsigned int ig = idy + nyc*(idxz + nx*nz*idlms);

	  float kxmax = kx[(nx-1)/3];
	  float kymax = ky[(ny-1)/3];
	  float kpmax2 = kxmax*kxmax + kymax*kymax;
	  float kp2 = (kx[idx]*kx[idx] + ky[idy]*ky[idy])/kpmax2;
	  
	  float D10 = D_HB * powf(kx[idx]/kxmax, 4);
	  float D01 = D_HB * powf(kx[idx]/kxmax, 4) * ky[idy]/kymax;
	  float D11 = D_HB * powf(kp2, p_HB);
	  
	  float sfac = (idy == 0) ? s10[idz] * D10 : s11[idz] * D11 + s01[0] * D01;
	  RHS[ig] = RHS[ig] - sfac * G[ig];
	}
      }
    }
  }
}

# define Hc_(XYZ, L, M, S) (g[(XYZ) + nx*nyc*nz*((L) + nl*((M) + nm*(S)))] + Jflr(L,b_s)*phi_*zt_)
# define H1c_(XYZ, L, M, S) (g[(XYZ) + nx*nyc*nz*((L) + nl*((M) + nm*(S)))] - Jflr(L,b_s)*apar_*zt_*vt_)
# define Gc_(XYZ, L, M, S)  g[(XYZ) + nx*nyc*nz*((L) + nl*((M) + nm*(S)))]
// H = G, except for m = 0
// C = C(H) but H and G are the same function for all m!=0. Our main array defines g so the correction to produce
// H is only appropriate for m=0. In other words, the usage here is basically handling the delta_{m0} terms
// in a clumsy way
__global__ void conservation_terms(cuComplex* upar_bar, cuComplex* uperp_bar, cuComplex* t_bar,
				   const cuComplex* g, const cuComplex* phi, const cuComplex* apar, const float *kperp2,
				   const float* zt, const float* rho2s, const float* vt)
{
  unsigned int idxyz = get_id1();

  if (idxyz < nx*nyc*nz) {
    cuComplex phi_ = phi[idxyz];
    cuComplex apar_ = apar[idxyz];
    for (int is=0; is < nspecies; is++) {
      const float zt_ = zt[is];
      const float vt_ = vt[is];
      unsigned int index = idxyz + nx*nyc*nz*is;

      upar_bar[index]  = make_cuComplex(0., 0.);
      uperp_bar[index] = make_cuComplex(0., 0.);
      t_bar[index]     = make_cuComplex(0., 0.);
      
      float b_s = kperp2[idxyz] * rho2s[is];
      // sum over l
      for (int l=0; l < nl; l++) {

        // Hc_(...) is defined by macro above. Only use here for m=0. 
	uperp_bar[index] = uperp_bar[index] + (Jflr(l,b_s) + Jflr(l-1,b_s))*Hc_(idxyz, l, 0, is);

        // H1c_(...) is defined by macro above. Only use here for m=1.
        upar_bar[index] = upar_bar[index] + Jflr(l,b_s)*H1c_(idxyz, l, 1, is);

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

// uperp_bar(ky, kx, z, s) = sqrt(b(s)) * sum_l [Jflr(ky, kx, z, l,  b(s)) + Jflr(ky, kx, z, l-1, b(s))] *
//                                                 [g(ky, kx, z, l, 0, is) + Jflr(ky, kx, z, l, b(s)) phi(ky, kx, z, s)) 
// Could form H (ky, kx, z, l, 0, s)
// and        J (ky, kx, z, l, s)
// and        J'(ky, kx, z, l, s)   (where l == l-1 and J'(-1) == 0)
// and then this is a reduction over l.
// Ah, actually we should define J'' == J + J' and build and store that
// J'' has size: Nyc Nx Nz Nl Nspecies
//
// We should define the summands: One kernel to calculate H. One kernel to build J'' (at the beginning of the run only)
// and then this would be a multiplication element-wise
//
// or should we just recalculate J'' on the fly every time? There are factorials and exponentials. Probably not?
// Let's store it. So there should be a kernel to build J'' in the constructor of Linear.
// Then the job for a given timestep would be to build the summand sqrt(b(s)) J'' H
// and then perform a tensor reduction
//
// upar_bar(ky, kx, z, s) = sum Jflr(ky, kx, z, l, b(s)) * g(ky, kx, z, l, 1, s)
// which is again a reduction over l. Build the summand, do a tensor reduction.
// 
// tpar_bar works exactly like upar_bar.
// tperp_bar works like uperp, except now we work with
// J''' == l J(l-1) * 2l J(l) + (l+1) J(l+1)
// We should keep H around for both uperp and tperp
// finally, we take t_bar to be a weighted sum of tpar_bar and tperp_bar.
