#include <stdio.h>
#include "device_funcs.h"

__device__ __constant__ int nx, ny, nyc, nz, nspecies, nm, nl, nj, zp, ikx_fixed, iky_fixed, is_lo, is_up, m_lo, m_up, m_ghost, nm_glob;

void setdev_constants(int Nx,
		      int Ny,
		      int Nyc,
		      int Nz,
		      int Nspecies,
		      int Nm,
		      int Nl,
		      int Nj,
		      int Zp,
		      int ikxf,
		      int ikyf,
		      int is_lo_in,
		      int is_up_in,
		      int m_lo_in,
		      int m_up_in,
		      int m_ghost_in,
		      int Nm_glob)
{
  cudaMemcpyToSymbol ( nx,        &Nx,         sizeof(int));
  cudaMemcpyToSymbol ( ny,        &Ny,         sizeof(int));
  cudaMemcpyToSymbol ( nyc,       &Nyc,        sizeof(int));
  cudaMemcpyToSymbol ( nz,        &Nz,         sizeof(int));
  cudaMemcpyToSymbol ( nspecies,  &Nspecies,   sizeof(int));
  cudaMemcpyToSymbol ( nm,        &Nm,         sizeof(int));
  cudaMemcpyToSymbol ( nm_glob,   &Nm_glob,    sizeof(int));
  cudaMemcpyToSymbol ( nl,        &Nl,         sizeof(int));
  cudaMemcpyToSymbol ( nj,        &Nj,         sizeof(int));
  cudaMemcpyToSymbol ( zp,        &Zp,         sizeof(int));
  cudaMemcpyToSymbol ( ikx_fixed, &ikxf,       sizeof(int));
  cudaMemcpyToSymbol ( iky_fixed, &ikyf,       sizeof(int));
  cudaMemcpyToSymbol ( is_lo,     &is_lo_in,   sizeof(int));
  cudaMemcpyToSymbol ( is_up,     &is_up_in,   sizeof(int));
  cudaMemcpyToSymbol ( m_lo,      &m_lo_in,    sizeof(int));
  cudaMemcpyToSymbol ( m_up,      &m_up_in,    sizeof(int));
  cudaMemcpyToSymbol ( m_ghost,   &m_ghost_in, sizeof(int));
  
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
// i is the x-index
// j is the y-index
// k is the z-index
// For compatibility with cuFFT layouts, the y-data is first, followed by x-data and then z-data
__device__ unsigned int get_idxyz(unsigned int i, unsigned int j, unsigned int k) {return j + nyc*(i + nx*k);}


// use Stirling's approximation
__host__ __device__ float factorial(int m) {
  if (m <2) return 1.;
  if (m==2) return 2.;
  if (m==3) return 6.;
  if (m==4) return 24.;
  if (m==5) return 120.;
  if (m==6) return 720.;
  else return sqrtf(2.*M_PI*m)*powf(m,m)*expf(-m)*(1.+1./(12.*m)+1./(288.*m*m));
}

// enforce_JL_0 is an optional argument, true by default.
__device__ float Jflr(const int l, const float b, bool enforce_JL_0) {
  if (l>30) return 0.; // protect against underflow for single precision evaluation
  else if (l<0) return 0.;
  else if (l>=nl && enforce_JL_0) return 0;
  else return 1./factorial(l)*pow(-0.5*b, l)*expf(-b/2.); // Assumes <J_0> = exp(-b/2)
}

__device__ float JflrA(const int l, const float b)
{
  return l*Jflr(l-1, b) + 2*l*Jflr(l, b) + (l+1)*Jflr(l+1, b);
}

__device__ float JflrB(const int l, const float b, bool enforce_JL_0) {
  if (l>=nl && enforce_JL_0) return 0;
  else return Jflr(l, b) + Jflr(l-1, b);
}

__device__ float Jfac(const int l, const float b)
{
  return 1.5*Jflr(l, b) + JflrA(l, b);
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
    err = fabsf(tk/g);
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
    err = fabsf(tk/g);
  }
  
  if (g<tol) g=tol; 
  return g;

}

__host__ __device__ float sgam0 (float b) {return sqrtf(g0(b));}
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

__global__ void phiSolve_cetg (cuComplex *phi, cuComplex *G0, float tau_bar)
{
  idXYZ;

  if ( unmasked(idx, idy) && idz < nz ) { 
    unsigned int idxyz = get_idxyz(idx, idy, idz);

    phi[idxyz] = -G0[idxyz] * tau_bar;
  }
}

__global__ void phiSolve_krehm (cuComplex *phi, cuComplex *G0, float* kx, float* ky, float rho_i)
{
  idXYZ;
    
  if ( unmasked(idx, idy) && idz < nz ) { 
    unsigned int idxyz = get_idxyz(idx, idy, idz);

    float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];
    float gam0 = g0(kperp2*rho_i*rho_i/2.);

    phi[idxyz] = G0[idxyz]*rho_i*rho_i/2./(gam0 - 1.);
  }
}

__global__ void aparSolve_krehm (cuComplex *apar, cuComplex *G1, float* kx, float* ky, float rho_s, float d_e)
{
  idXYZ;
    
  if ( unmasked(idx, idy) && idz < nz ) { 
    unsigned int idxyz = get_idxyz(idx, idy, idz);

    float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];

    apar[idxyz] = -G1[idxyz]*rho_s*d_e/(1. + d_e*d_e*kperp2);
  }
}

__global__ void equilibrium_current_krehm (cuComplex *G1, float* kx, float* ky, float rho_s, float d_e, cuComplex* apar_ext)
{
  idXYZ;
  
  if ( unmasked(idx, idy) && idz < nz ) { 
    unsigned int idxyz = get_idxyz(idx, idy, idz);

    float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];

    G1[idxyz] = G1[idxyz] - (1. + d_e*d_e*kperp2)*apar_ext[idxyz]/(rho_s*d_e);
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

__global__ void abs(float *f, int N)
{
  unsigned int i = get_id1();
  if (i < N) f[i] = fabsf(f[i]);
}

__global__ void add_section(cuComplex *res, const cuComplex *tmp, int ntot)
{
  unsigned int i = get_id1();
  if (i < ntot) res[i] = res[i] + tmp[i];
}

__global__ void add_scaled_singlemom_kernel(float* res,
					    double c1, const float* m1,
					    double c2, const float* m2)
{
  unsigned int idxyz = get_id1();
  if (idxyz < nx*ny*nz) res[idxyz] = c1*m1[idxyz] + c2*m2[idxyz];
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
  unsigned int idz  = get_id2();
  unsigned int idlm = get_id3();

  if (idxy < nx*nyc && idz < nz && idlm < nl*nm) {
    if (neqfix || not_fixed_eq(idxy)) {
      
      unsigned int ig = idxy + nx*nyc*(idz + nz*idlm);
      
      res[ig] = c1 * m1[ig] + c2 * m2[ig];
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2,
				  double c3, const cuComplex* m3, bool neqfix = true)
{
  unsigned int idxy = get_id1(); 
  unsigned int idz  = get_id2();
  unsigned int idlm = get_id3(); 
  if (idxy < nx*nyc && idz < nz && idlm < nl*nm) {
    if (neqfix || not_fixed_eq(idxy)) {
      
      unsigned int ig = idxy + nx*nyc*(idz + nz*idlm);
      
      res[ig] = c1 * m1[ig] + c2 * m2[ig] + c3 * m3[ig];
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
  unsigned int idz  = get_id2();
  unsigned int idlm = get_id3(); 
  if (idxy < nx*nyc && idz < nz && idlm < nl*nm) {
    if (neqfix || not_fixed_eq(idxy)) {
      
      unsigned int ig = idxy + nx*nyc*(idz + nz*idlm);
      
      res[ig] = c1 * m1[ig] + c2 * m2[ig] + c3 * m3[ig] + c4 * m4[ig];
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
  unsigned int idz  = get_id2();
  unsigned int idlm = get_id3(); 
  if (idxy < nx*nyc && idz < nz && idlm < nl*nm) {
    if (neqfix || not_fixed_eq(idxy)) {
      
      unsigned int ig = idxy + nx*nyc*(idz + nz*idlm);
	
      res[ig] = c1 * m1[ig] + c2 * m2[ig] + c3 * m3[ig] + c4 * m4[ig] + c5 * m5[ig];
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
  unsigned int n = get_id2();
  if (m<M && n<N) V[m + M*n] +=  dG[m] * r2[n];
}

__global__ void getB(double* W, double beta, int N)
{
  unsigned int n = get_id1();
  if (n<N) W[n+N*n] += beta;
}

__global__ void setI(double* Id, int N)
{
  unsigned int n = get_id1();
  if (n<N) Id[n + N*n] = 1.;
}

__global__ void setA(double* A, double fac, int N)
{
  unsigned int n = get_id1();
  if (n<N) A[n] *= (double) fac;
}

__global__ void getW(double* W, double* r2, int N) 
{
  unsigned int n1 = get_id1();
  unsigned int n2 = get_id2();
  if (n1 < N && n2 < n1 + 1) {
    double r2n1 = r2[n1];
    double r2n2 = r2[n2];
    W[n1 + N*n2] += r2n1 * r2n2;
    if (n2<n1) W[n2 + N*n1] = W[n1 + N*n2];
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
  unsigned int m = get_id2();
  if (q<Q && m<M) {
    unsigned int n = q + Q*m;
    res[n] = Win[n] * dG[m];
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
  if (i < NK) x[i] = r[ col[i] ];
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
  if (n < N) res[n] = f[n]/sqrt(scalar[0]);
}

__global__ void scale_kernel(cuComplex* res, double scalar)
{
  unsigned int idxy = get_id1(); 
  unsigned int idz  = get_id2();
  unsigned int idlm = get_id3(); 
  if (idxy < nx*nyc && idz < nz && idlm < nl*nm) {
    unsigned int ig = idxy + nx*nyc*(idz + nz*idlm);
    
    res[ig] = scalar*res[ig];
  }
}

__global__ void scale_kernel(cuComplex* res, const cuComplex scalar)
{
  unsigned int idxy = get_id1(); 
  unsigned int idz  = get_id2();
  unsigned int idlm = get_id3(); 
  if (idxy < nx*nyc && idz < nz && idlm < nl*nm) {
    unsigned int ig = idxy + nx*nyc*(idz + nz*idlm);
    
    res[ig] = scalar*res[ig];
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

__device__ bool masked(int idx, int idy) {
  int ikx = get_ikx(idx);
  if ( (idx < nx)
       && (idy < ny)
       && ( (idx==0 && idy==0)
	    || idy > (ny-1)/3
	    || ikx > (nx-1)/3
	    || ikx < -(nx-1)/3 )
       )
    return true;
  else
    return false;
}

__global__ void maskG(cuComplex* g)
{
  unsigned int idxy = get_id1();
  unsigned int idz  = get_id2();
  unsigned int idlm = get_id3();

  if (idxy < nx*nyc) {
    unsigned int idy = idxy % nyc;
    unsigned int idx = idxy / nyc;
    if (masked(idx, idy) && idz < nz && idlm < nl*nm) {
      
      unsigned int globalIdx = idxy + nyc*nx*(idz + nz*idlm);
      g[globalIdx] = make_cuComplex(0., 0.);
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
  if (unmasked(idx, idy)) kxs[idy+nyc*idx] = kx[idx]; // should read this from a file if this is a restarted case
}

__global__ void update_kxs(float* kxs, float* dth0)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  if (unmasked(idx, idy)) {
    //    kxs[idy + nyc*idx]  =
  }
}

__global__ void update_geo(float* kxs,
			   float* ky,
			   float* cv_d,
			   float* gb_d,
			   float* kperp2,
			   float* cv,
			   float* cv0,
			   float* gb,
			   float* gb0,
			   float* omegad, 
			   float* gds2,
			   float* gds21,
			   float* gds22,
			   float* bmagInv,
			   float shat)
{
  idXYZ;
  
  if (idy>0 && unmasked(idx, idy) && idz < nz) { 
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    unsigned int idxy  = idy + nyc*idx;

    if (shat == 0.0) {
      kperp2[idxyz] = ( ky[idy] * ( ky[idy] * gds2[idz] + 2. * kxs[idxy] * gds21[idz]) 
			+ pow( kxs[idxy], 2) * gds22[idz] ) * pow( bmagInv[idz], 2);
    
      cv_d[idxyz] = ky[idy] * cv[idz] + kxs[idxy] * cv0[idz] ;     
      gb_d[idxyz] = ky[idy] * gb[idz] + kxs[idxy] * gb0[idz] ;
      omegad[idxyz] = cv_d[idxyz] + gb_d[idxyz];

    } else {
      float shatInv = 1./shat; 
    
      kperp2[idxyz] = ( ky[idy] * ( ky[idy] * gds2[idz] + 2. * kxs[idxy] * shatInv * gds21[idz])
			+ pow( kxs[idxy] * shatInv, 2) * gds22[idz] ) * pow( bmagInv[idz], 2);
      
      cv_d[idxyz] = ky[idy] * cv[idz] + kxs[idxy] * shatInv * cv0[idz] ;     
      gb_d[idxyz] = ky[idy] * gb[idz] + kxs[idxy] * shatInv * gb0[idz] ;
      omegad[idxyz] = cv_d[idxyz] + gb_d[idxyz];
    }
  }
}

// note: kperp2 = kperp**2 / B**2   (because it is kperp**2 rho**2)
__global__ void init_kperp2(float* kperp2,
			    const float* kx,
			    const float* ky,
			    const float* gds2,
			    const float* gds21,
			    const float* gds22,
			    const float* bmagInv,
			    const float  shat) 
{
  idXYZ;
  
  if (unmasked(idx, idy) && idz < nz) { 
    unsigned int idxyz = get_idxyz(idx, idy, idz);

    if (shat == 0.0) {
      kperp2[idxyz] = ( ky[idy] * ( ky[idy] * gds2[idz] 
		      + 2. * kx[idx]       * gds21[idz] ) 
		      + pow( kx[idx], 2)   * gds22[idz] )
	              * pow( bmagInv[idz], 2);
    } else {	      
      float shatInv = 1./shat; 

      kperp2[idxyz] = ( ky[idy] * ( ky[idy] * gds2[idz] 
                      + 2. * kx[idx] * shatInv * gds21[idz] ) 
                      + pow( kx[idx] * shatInv, 2) * gds22[idz] ) 
                      * pow( bmagInv[idz], 2);
    }
  }
}

__global__ void init_omegad(float* omegad,
			    float* cv_d,
			    float* gb_d,
			    const float* kx,
			    const float* ky,
			    const float* cv,
			    const float* gb,
			    const float* cv0,
			    const float* gb0,
			    const float  shat) 
{
  idXYZ;
  
  // cv/gb is the y-directed part of the curvature / grad-B drift
  // cv0/gb0 is the part proportional to the theta_0, aka the x-directed component

  if ( unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);

    if (shat == 0.0) {

      cv_d[idxyz] = ky[idy] * cv[idz] + kx[idx] * cv0[idz] ;     
      gb_d[idxyz] = ky[idy] * gb[idz] + kx[idx] * gb0[idz] ;
      omegad[idxyz] = cv_d[idxyz] + gb_d[idxyz];
    } else {
      float shatInv = 1./shat; 
      cv_d[idxyz] = ky[idy] * cv[idz] + kx[idx] * shatInv * cv0[idz] ;     
      gb_d[idxyz] = ky[idy] * gb[idz] + kx[idx] * shatInv * gb0[idz] ;
      omegad[idxyz] = cv_d[idxyz] + gb_d[idxyz];
    }
  }
}

# define H_(XYZ, L, M, S) (g[(int) (XYZ) + nx*nyc*nz*((L) + nl*((M) + nm*(S)))] + Jflr(L,b_s)*phi_)
# define G_(XYZ, L, M, S)  g[(int) (XYZ) + nx*nyc*nz*((L) + nl*((M) + nm*(S)))]
// H = G, except for m = 0
// C = C(H) but H and G are the same function for all m!=0. Our main array defines g so the correction to produce
// H is only appropriate for m=0. In other words, the usage here is basically handling the delta_{m0} terms
// in a clumsy way
__global__ void Tbar(cuComplex* t_bar, const cuComplex* g, const cuComplex* phi, const float *kperp2) // crude diagnostic
{

  unsigned int idxyz = get_id1();
  unsigned int idy = idxyz % nyc;
  unsigned int idx = (idxyz / nyc) % nx;

  if (unmasked(idx, idy) && (idxyz < nx*nyc*nz)) {
    cuComplex phi_ = phi[idxyz];
    t_bar[idxyz] = make_cuComplex(0., 0.);
    float b_s = kperp2[idxyz]; // only species=0, assumes zt, rho2 = 1 !! bug
    for (int l=0; l < nl; l++) {
      // energy conservation correction for nlaguerre = 1
      if (nl == 1) {
	t_bar[idxyz] = t_bar[idxyz] + sqrtf(2.)*Jflr(l,b_s)*G_(idxyz, l, 2, 0);
      } else {
	t_bar[idxyz] = t_bar[idxyz] + sqrtf(2.)/3. * Jflr(l,b_s) * G_(idxyz, l, 2, 0)
	  + 2./3. * ( l * Jflr(l-1,b_s) + 2. * l * Jflr(l,b_s) + (l+1) * Jflr(l+1,b_s) ) * H_(idxyz, l, 0, 0);
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
    int IG = (int) nz/2 ;
    int id = idxy + J*IG;
    
    int idy = idxy % nyc;
    int idx = idxy / nyc;
    
    if (unmasked(idx, idy)) {
      if (abs(phi[id].x)!=0 && abs(phi[id].y)!=0) {
	cuComplex ratio = phi[id] / phiOld[id];
	
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
}

__global__ void J0fToGrid(cuComplex* J0f,
			  const cuComplex* f,
			  const float* kperp2,
			  const float* muB,
			  const float rho2_s,
			  const float fac)
{
  unsigned int idxyz = get_id1();
  unsigned int idy = idxyz % nyc;
  unsigned int idx = (idxyz / nyc) % nx;
  unsigned int idj = get_id2();
  if (unmasked(idx, idy) && idxyz < nx*nyc*nz && idj < nj) {
    unsigned int ig = idxyz + nx*nyc*nz*idj;
    J0f[ig] = j0f(sqrtf(2. * muB[idj] * kperp2[idxyz]*rho2_s)) * f[idxyz] * fac;
  }
}

__global__ void J0phiAndBparToGrid(cuComplex* J0phiB,
				   const cuComplex* phi,
				   const cuComplex* bpar,
				   const float* kperp2,
				   const float* muB,
				   const float rho2_s,
				   const float tz,
				   const float fphi,
				   const float fbpar)
{
  unsigned int idxyz = get_id1();
  unsigned int idy = idxyz % nyc;
  unsigned int idx = (idxyz / nyc) % nx;
  unsigned int idj = get_id2();
  if (unmasked(idx, idy) && idxyz < nx*nyc*nz && idj < nj) {
    unsigned int ig = idxyz + nx*nyc*nz*idj;
    float alpha = sqrtf(2. * muB[idj] * kperp2[idxyz]*rho2_s);
    float j1_over_alpha = alpha < 1e-8 ? 0.5 : j1f(alpha)/alpha;
    J0phiB[ig] = j0f(alpha) * phi[idxyz] * fphi + tz*2.*muB[idj]*j1_over_alpha * bpar[idxyz] * fbpar;
  }
}

__global__ void castDoubleToFloat (const cuDoubleComplex *array_d, cuComplex *array_f, int size) {
  for (unsigned int i = 0; i < size; i++) array_f[i] = cuComplexDoubleToFloat(array_d[i]);
}

__global__ void ddx (cuComplex *res, cuComplex *f, float *kx)
{
  idXYZ;
  
  if (unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);

    cuComplex Ikx = make_cuComplex(0., kx[idx]);
    
    res[idxyz] = Ikx * f[idxyz]; 
  }
}

__global__ void ddy (cuComplex *res, cuComplex *f, float *ky)
{
  idXYZ;
  
  if (unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);

    cuComplex Iky = make_cuComplex(0., ky[idy]);

    res[idxyz] = Iky * f[idxyz];
  }
}

__global__ void mddy (cuComplex *res, cuComplex *f, float *ky)
{
  idXYZ;
  
  if (unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    
    cuComplex Iky = make_cuComplex(0., ky[idy]);

    res[idxyz] = -Iky * f[idxyz];
  }
}

__global__ void d2x (cuComplex *res, cuComplex *f, float *kx)
{
  idXYZ;
  
  if (unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);

    cuComplex Ikx = make_cuComplex(0., kx[idx]);

    res[idxyz] = Ikx * Ikx * f[idxyz];
  }
}

__device__ cuComplex i_kx(void *dataIn, size_t offset, void *kxData, void *sharedPtr)
{
  unsigned int idx = offset / nyc % nx;

  float *kx = (float*) kxData;

  cuComplex Ikx = make_cuComplex(0., kx[idx]);

  return Ikx * ((cuComplex*)dataIn)[offset];
}

__device__ cuComplex i_kxs(void *dataIn, size_t offset, void *kxsData, void *sharedPtr)
{
  unsigned int idxy = offset % (nyc*nx);

  float *kxs = (float*) kxsData;

  cuComplex Ikxs = make_cuComplex(0., kxs[idxy]);

  return Ikxs * ((cuComplex*)dataIn)[offset];
}

__device__ cuComplex i_ky(void *dataIn, size_t offset, void *kyData, void *sharedPtr)
{
  unsigned int idy = offset % nyc; 

  float *ky = (float*) kyData;

  cuComplex Iky = make_cuComplex(0., ky[idy]);

  return Iky * ((cuComplex*)dataIn)[offset];
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

// callbacks for loads
__device__ cufftCallbackLoadC i_kxs_callbackPtr = i_kxs;
__device__ cufftCallbackLoadC i_kx_callbackPtr  = i_kx;
__device__ cufftCallbackLoadC i_ky_callbackPtr  = i_ky;

// callback for stores
__device__ cufftCallbackStoreC mask_and_scale_callbackPtr = mask_and_scale;

// Multiplies by i kz / Nz 
__device__ void i_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  unsigned int idz = offset / (nx*nyc);

  float *kz = (float*) kzData;

  cuComplex Ikz = make_cuComplex(0., kz[idz]);

  ((cuComplex*)dataOut)[offset] = Ikz * element/nz;    
}

// Multiplies by - kz**2 / Nz 
__device__ void mkz2(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  unsigned int idz = offset / (nx*nyc);

  float *kz = (float*) kzData;

  cuComplex Ikz = make_cuComplex(0., kz[idz]);

  ((cuComplex*)dataOut)[offset] = Ikz * Ikz * element/nz;    
}

// Multiplies by 1 / Nz
__device__ void zfts(void *dataOut, size_t offset, cufftComplex element, void *data, void *sharedPtr)
{
  ((cuComplex*)dataOut)[offset] = element/nz;    
}

// Multiplies by |kz| / Nz
__device__ void abs_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  unsigned int idz = offset / (nx*nyc);

  float *kz = (float*) kzData;

  ((cuComplex*)dataOut)[offset] = abs(kz[idz]) * element/nz;
}

// Multiplies by ikz / Nz
__device__ void i_kz_1d(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  unsigned int idz = offset;

  float *kz = (float*) kzData;

  cuComplex Ikz = make_cuComplex(0., kz[idz]);

  ((cuComplex*)dataOut)[offset] = Ikz * element/nz;
}

// Multiplies by -kz**2 / Nz
__device__ void mkz2_1d(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  unsigned int idz = offset;

  float *kz = (float*) kzData;

  cuComplex Ikz = make_cuComplex(0., kz[idz]);

  ((cuComplex*)dataOut)[offset] = Ikz * Ikz * element/nz;
}

// callbacks for stores
__device__ cufftCallbackStoreC zfts_callbackPtr    = zfts;
__device__ cufftCallbackStoreC i_kz_callbackPtr    = i_kz;
__device__ cufftCallbackStoreC mkz2_callbackPtr    = mkz2;
__device__ cufftCallbackStoreC i_kz_1d_callbackPtr = i_kz_1d;
__device__ cufftCallbackStoreC mkz2_1d_callbackPtr = mkz2_1d;
__device__ cufftCallbackStoreC abs_kz_callbackPtr  = abs_kz;

//__global__ void acc(float *a, const float *b)
//{a[0] = a[0] + b[0];}

// Nonlinear term in the vlasov-poisson model
__global__ void nlvp(float *res, const float *Gy, const float *dphi)
{
  unsigned int idy = get_id1();
  unsigned int m   = get_id2();
  if (idy < ny && m < nm) {
    unsigned int ig  = idy + ny*m;
    unsigned int mm1 = idy + ny*(m-1);
    res[ig] = - sqrtf(m) * dphi[idy] * Gy[mm1];
  }
}

// Nonlinear term in the KS model
__global__ void nlks(float *res, const float *Gy, const float *dG)
{
  unsigned int idy = get_id1();
  if (idy < ny) res[idy] = - Gy[idy] * dG[idy];
}

__global__ void kz_dealias (cuComplex *G, int *kzm, int LM)
{
  unsigned int idxy = get_id1();
  unsigned int idz  = get_id2();
  unsigned int idlm = get_id3();
  if (idxy < nx*nyc) {
    int idy = idxy % nyc;
    int idx = idxy / nyc;
    if (unmasked(idx, idy) && idz < nz && idlm < LM) {
      unsigned int ig = idxy + nx*nyc*(idz + nz*idlm);
      if (kzm[idz] == 0) {
	G[ig].x = 0.; 
	G[ig].y = 0.;
      }
    }
  }
}

__global__ void bracket(      float* __restrict__ g_res,
			const float* __restrict__ dg_dx,
			const float* __restrict__ dJ0phi_dy,
			const float* __restrict__ dg_dy,
			const float* __restrict__ dJ0phi_dx,
			float kxfac)
{
  unsigned int idxyz = get_id1();
  unsigned int idj   = get_id2();
  unsigned int idm   = get_id3();

  if (idxyz < nx*ny*nz && idj < nj && idm < nm) {
    unsigned int iphi = idxyz + nx*ny*nz*idj;
    unsigned int ig   = idxyz + nx*ny*nz*(idj + nj*idm);
    g_res[ig] = ( dg_dx[ig] * dJ0phi_dy[iphi] - dg_dy[ig] * dJ0phi_dx[iphi] ) * kxfac;
  }
}

__global__ void bracket_cetg(      float* __restrict__ g_res,
			     const float* __restrict__ dg_dx,
			     const float* __restrict__ dphi_dy,
			     const float* __restrict__ dg_dy,
			     const float* __restrict__ dphi_dx,
			     float kxfac)
{
  unsigned int idxyz = get_id1();

  if (idxyz < nx*ny*nz) {
    unsigned int iphi = idxyz;
    unsigned int ig = idxyz + nx*ny*nz;
    g_res[ig] = ( dg_dx[ig] * dphi_dy[iphi] - dg_dy[ig] * dphi_dx[iphi] ) * kxfac;
    
  }
}

# define LM(L, M) idxyz + nx*nyc*nz*((L) + nl*(M))
__global__ void beer_toroidal_closures(const cuComplex* g, cuComplex* gRhs,
				       const float* omegad,
				       const cuComplex* nu,
				       const float tz_)
{
  unsigned int idxyz = get_id1();

  if (idxyz < nx*nyc*nz) {
    const cuComplex iomegad = make_cuComplex(0., omegad[idxyz]);
    const float abs_omegad = abs(omegad[idxyz]);

    const cuComplex iwd_s = iomegad * tz_;
    const float awd_s = abs_omegad * tz_;

    gRhs[LM(0,2)] = gRhs[LM(0,2)]
      - sqrtf(2) * awd_s * ( nu[1].x*sqrtf(2)*g[LM(0,2)] + nu[2].x*g[LM(1,0)] )
      - sqrtf(2) * iwd_s * ( nu[1].y*sqrtf(2)*g[LM(0,2)] + nu[2].y*g[LM(1,0)] );
    
    gRhs[LM(1,0)] = gRhs[LM(1,0)]
      - 2. * awd_s * ( nu[3].x*sqrtf(2)*g[LM(0,2)] + nu[4].x*g[LM(1,0)] )
      - 2. * iwd_s * ( nu[3].y*sqrtf(2)*g[LM(0,2)] + nu[4].y*g[LM(1,0)] );
    
    gRhs[LM(0,3)] = gRhs[LM(0,3)]
      - 1./sqrtf(6) * awd_s * ( nu[5].x*g[LM(0,1)] + nu[6].x*sqrtf(6)*g[LM(0,3)] + nu[7].x*g[LM(1,1)] )
      - 1./sqrtf(6) * iwd_s * ( nu[5].y*g[LM(0,1)] + nu[6].y*sqrtf(6)*g[LM(0,3)] + nu[7].y*g[LM(1,1)] );
    
    gRhs[LM(1,1)] = gRhs[LM(1,1)]
      - awd_s * ( nu[8].x*g[LM(0,1)] + nu[9].x*sqrtf(6)*g[LM(0,3)] + nu[10].x*g[LM(1,1)] )
      - iwd_s * ( nu[8].y*g[LM(0,1)] + nu[9].y*sqrtf(6)*g[LM(0,3)] + nu[10].y*g[LM(1,1)] );
  }
}

__global__ void smith_perp_toroidal_closures(const cuComplex* g, cuComplex* gRhs,
					     const float* omegad, const cuComplex* Aclos, int q, const float tz_)
{
  unsigned int idxyz = get_id1();
  
  if (idxyz < nx*nyc*nz) {

    const cuComplex    iomegad = make_cuComplex(0., omegad[idxyz]);
    const float     abs_omegad = abs(omegad[idxyz]);
    
    const cuComplex iwd_s =    iomegad * tz_;
    const float     awd_s = abs_omegad * tz_;

    int L = nl - 1;
    
    // apply closure to Lth laguerre equation for all hermite moments
    for (int m=0; m < nm; m++) {

      // calculate closure expression as sum of lower laguerre moments
      cuComplex clos = make_cuComplex(0.,0.);
      for (int l=L; l>=nl-q; l--) clos = clos + (awd_s * Aclos[L-l].y + iwd_s * Aclos[L-l].x)*g[LM(l,m)];

      gRhs[LM(L,m)] = gRhs[LM(L,m)] - (L+1)*clos;
    }
  }
}
 
__global__ void stirring_kernel(const cuComplex force, cuComplex *moments, int forcing_index)
{
  moments[forcing_index] = moments[forcing_index] + force;
}

__global__ void kz_stirring_kernel(const cuComplex force, cuComplex *moments, int kx, int ky, int kz)
{
  unsigned int idz = get_id1();
  float z = idz/nz* 2*M_PI*zp;
  int forcing_index = ky + nyc*kx + nx*nyc*idz; 
  moments[forcing_index] = moments[forcing_index] + force*cosf(kz*z); 
}


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
  unsigned int idx = get_id2();
  if (idy < ny && idx < nx) out[idx + nx*idy] = in[idy + ny*idx];
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

__global__ void Wg_summand(float *G2, const cuComplex* g, const float* volJac, const float nt_) 
{
  unsigned int idxy = get_id1(); 
  unsigned int idz  = get_id2();
  unsigned int idlm = get_id3();
  if (idxy < nx*nyc && idz < nz && idlm < nm*nl) {
    unsigned int ig = idxy + nx*nyc*(idz + nz*idlm);

    unsigned int idy = idxy % nyc;
    unsigned int idx = idxy / nyc;
    if (unmasked(idx, idy)) {

      float fac = 2.0;
      if (idy==0) fac = 1.0;
      cuComplex fg = cuConjf(g[ig]) * g[ig] * volJac[idz] * fac;
      G2[ig] = 0.5 * fg.x * nt_;
    } else {
      G2[ig] = 0.;
    }
  }
}

__global__ void vol_summand(float *rmom, const cuComplex* f, const cuComplex* g, const float* volJac)
{
  idXYZ;
  
  if (idy < nyc && idx < nx && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    if (unmasked(idx, idy)) {      

      float fac=2.;
      if (idy==0) fac = 1.0;
      
      cuComplex fg = cuConjf( f[idxyz] ) * g[idxyz] * volJac[idz] * fac;

      rmom[idxyz] = fg.x;
    } else {
      rmom[idxyz] = 0.;
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

// the following is only used for some linear runs
__global__ void maxPhi(float* phi_max, const cuComplex *phi)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  float pmax = 0.0;
  if (idy < nyc && idx < nx) {
    if (unmasked(idx, idy)) {
      if (idy == 0) {
	phi_max[idy + nyc*idx] = 1.0;
	return;
      }
      for (int idz = 0; idz<nz; idz++) {
	unsigned int idxyz = get_idxyz(idx, idy, idz);

	cuComplex tmp = cuConjf( phi[idxyz] ) * phi[idxyz];
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
  idXYZ;

  if (idy < nyc && idx < nx && idz < nz) { 
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    if (unmasked(idx, idy)) {
      p2[idxyz] *= alpha;
    } else {
      p2[idxyz] = 0.;
    }
  }
}

__global__ void Phi2_summand(float *p2, const cuComplex *phi, const float *volJac)
{
  idXYZ;  

  if (idy < nyc && idx < nx && idz < nz) { 
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    if (unmasked(idx, idy)) {    
      float fac=2.;
      if (idy==0) fac = 1.0;

      cuComplex tmp = cuConjf( phi[idxyz] ) * phi[idxyz] *fac * volJac[idz] ;
      p2[idxyz] = 0.5 * tmp.x;

    } else {
      p2[idxyz] = 0.;
    }
  }
}

__global__ void Phi2_zonal_summand(float *p2, const cuComplex *phi, const float *volJac)
{
  idXYZ; 
  
  unsigned int idxyz = get_idxyz(idx, idy, idz);

  if (idy < nyc && idx < nx && idz < nz) { 
    if (unmasked(idx, idy)) {    
      cuComplex tmp;
      float fac;
      if (idy==0) fac = 1.0;
      else fac = 0.0;  // zero out non-zonal modes

      tmp = cuConjf( phi[idxyz] ) * phi[idxyz] *fac * volJac[idz] ;
      p2[idxyz] = 0.5 * tmp.x;

    } else {
      p2[idxyz] = 0.;
    }
  }
}

__global__ void Wphi_summand(float* p2, const cuComplex* phi, const float* volJac, const float* kperp2, float rho2_s)
{
  idXYZ;
  
  if (idy < nyc && idx < nx && idz < nz) { 
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    if (unmasked(idx, idy)) {    
      float fac=2.;
      if (idy==0) fac = 1.0;

      float b_s = kperp2[idxyz]*rho2_s;

      cuComplex tmp = cuConjf( phi[idxyz] ) * ( 1.0 - g0(b_s) ) * phi[idxyz] * fac * volJac[idz];
      p2[idxyz] = 0.5 * tmp.x;

    } else {
      p2[idxyz] = 0.;
    }
  }
}

__global__ void Wapar_summand(float* p2, const cuComplex* apar, const float* volJac, const float* kperp2, const float* bmag)
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

      float kp2 = kperp2[idxyz]*bmag[idz]*bmag[idz];

      tmp = cuConjf( apar[idxyz] ) * kp2 * apar[idxyz] * fac * volJac[idz];
      p2[idxyz] = 0.5 * tmp.x;

    } else {
      p2[idxyz] = 0.;
    }
  }
}

__global__ void Wphi_summand_krehm(float* p2,
                                   const cuComplex* phi,
                                   const float* volJac,
                                   const float* kx,
                                   const float* ky,
                                   float rho_i)
{
  idXYZ;
  
  if (idy < nyc && idx < nx && idz < nz) { 
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    if (unmasked(idx, idy)) {    
      float fac=2.;
      if (idy==0) fac = 1.0;

      float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];
      float gam0 = g0(kperp2*rho_i*rho_i/2.);

      cuComplex tmp = cuConjf( phi[idxyz] ) * ( 1.0 - gam0 ) * phi[idxyz] * fac * volJac[idz];
      p2[idxyz] = 0.5 * tmp.x * 2. / (rho_i * rho_i);

    } else {
      p2[idxyz] = 0.;
    }
  }
}

// WApar_summand_krehm
__global__ void Wapar_summand_krehm(float* p2,
				    const cuComplex* apar,
				    const cuComplex* apar_ext,
				    const float* volJac,
				    const float* kx,
				    const float* ky,
				    float rho_i)
{
  idXYZ;
  
  if (idy < nyc && idx < nx && idz<nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    if(unmasked(idx, idy)) {
      float fac=2.;
      if (idy==0) fac = 1.0;

      float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];
      cuComplex apar_perturb = apar[idxyz]; // - apar_ext[idxyz];
      cuComplex tmp = kperp2 * cuConjf( apar_perturb ) * apar_perturb * fac;
      p2[idxyz] = 0.5 * tmp.x;
    } else {
      p2[idxyz] = 0.;
    }
  }
}

# define Gh_(XYZ, L, M) g[(int) (XYZ) + nx*nyc*nz*((L) + nl*(M-m_lo))]
__global__ void heat_flux_summand(float* qflux,
                                  const cuComplex* phi,
                                  const cuComplex* apar,
                                  const cuComplex* bpar,
                                  const cuComplex* g,
                                  const float* ky,
				  const float* flxJac,
                                  const float *kperp2,
                                  float rho2_s,
                                  float pres,
                                  float vts,
                                  float tzs)
{
  idXYZ;
  
  if (idy < nyc && idx < nx && idz < nz) { 
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    if (unmasked(idx, idy) && m_lo == 0) {    
      
      cuComplex vPhi_r = make_cuComplex(0., ky[idy]) * phi[idxyz];
      cuComplex vA_r   = make_cuComplex(0., ky[idy]) * apar[idxyz];
      cuComplex vB_r   = make_cuComplex(0., ky[idy]) * bpar[idxyz];
    
      float b_s = kperp2[idxyz]*rho2_s;
    
      // sum over l
      cuComplex p_bar = make_cuComplex(0.,0.);
      cuComplex q_bar = make_cuComplex(0.,0.);
      cuComplex qB_bar = make_cuComplex(0.,0.);

      for (int il=0; il < nl; il++) {
	p_bar = p_bar + Jfac(il, b_s)*Gh_(idxyz, il, 0) + rsqrtf(2.)*Jflr(il, b_s)*Gh_(idxyz, il, 2);
	q_bar = q_bar + Jfac(il, b_s)*Gh_(idxyz, il, 1) + Jflr(il, b_s)*(sqrtf(1.5)*Gh_(idxyz, il, 3)+ Gh_(idxyz, il, 1));
	qB_bar = qB_bar + (Jfac(il, b_s)+Jfac(il-1,b_s))*Gh_(idxyz, il, 0) + rsqrtf(2.)*JflrB(il, b_s)*Gh_(idxyz, il, 2);
      }
    
      cuComplex fg = (cuConjf(vPhi_r) * p_bar - vts * cuConjf(vA_r) * q_bar + tzs * cuConjf(vB_r) * qB_bar) * 2. * flxJac[idz];
      qflux[idxyz] = fg.x * pres;

    } else {
      qflux[idxyz] = 0.;
    }
  }
}

__global__ void heat_flux_ES_summand(float* qflux, const cuComplex* phi, const cuComplex* g, const float* ky, 
				  const float* flxJac, const float *kperp2, float rho2_s, float pres, float vts)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  cuComplex fg;

  unsigned int idxyz = idy + nyc*(idx + nx*idz);
  if (idy < nyc && idx < nx && idz < nz) { 
    if (unmasked(idx, idy) && idy > 0 && m_lo == 0) {    
      
      cuComplex vPhi_r = make_cuComplex(0., ky[idy]) * phi[idxyz];
    
      float b_s = kperp2[idxyz]*rho2_s;
    
      // sum over l
      cuComplex p_bar = make_cuComplex(0.,0.);

      for (int il=0; il < nl; il++) {
	p_bar = p_bar + Jfac(il, b_s)*Gh_(idxyz, il, 0) + rsqrtf(2.)*Jflr(il, b_s)*Gh_(idxyz, il, 2);
      }
    
      fg = (cuConjf(vPhi_r) * p_bar) * 2. * flxJac[idz];
      qflux[idxyz] = fg.x * pres;

    } else {
      qflux[idxyz] = 0.;
    }
  }
}

__global__ void heat_flux_Apar_summand(float* qflux, const cuComplex* apar, const cuComplex* g, const float* ky, 
				  const float* flxJac, const float *kperp2, float rho2_s, float pres, float vts)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  cuComplex fg;

  unsigned int idxyz = idy + nyc*(idx + nx*idz);
  if (idy < nyc && idx < nx && idz < nz) { 
    if (unmasked(idx, idy) && idy > 0 && m_lo == 0) {    
      
      cuComplex vA_r = make_cuComplex(0., ky[idy]) * apar[idxyz];
    
      float b_s = kperp2[idxyz]*rho2_s;
    
      // sum over l
      cuComplex q_bar = make_cuComplex(0.,0.);

      for (int il=0; il < nl; il++) {
	q_bar = q_bar + Jfac(il, b_s)*Gh_(idxyz, il, 1) + Jflr(il, b_s)*(sqrtf(1.5)*Gh_(idxyz, il, 3)+ Gh_(idxyz, il, 1));
      }
    
      fg = (- vts * cuConjf(vA_r) * q_bar) * 2. * flxJac[idz];
      qflux[idxyz] = fg.x * pres;

    } else {
      qflux[idxyz] = 0.;
    }
  }
}

__global__ void heat_flux_Bpar_summand(float* qflux, const cuComplex* bpar, const cuComplex* g, const float* ky, 
				  const float* flxJac, const float *kperp2, float rho2_s, float pres, float tzs)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  cuComplex fg;

  unsigned int idxyz = idy + nyc*(idx + nx*idz);
  if (idy < nyc && idx < nx && idz < nz) { 
    if (unmasked(idx, idy) && idy > 0 && m_lo == 0) {    
      
      cuComplex vB_r = make_cuComplex(0., ky[idy]) * bpar[idxyz];
    
      float b_s = kperp2[idxyz]*rho2_s;
    
      // sum over l
      cuComplex qB_bar = make_cuComplex(0.,0.);

      for (int il=0; il < nl; il++) {
	qB_bar = qB_bar + (Jfac(il, b_s)+Jfac(il-1,b_s))*Gh_(idxyz, il, 0) + rsqrtf(2.)*JflrB(il, b_s)*Gh_(idxyz, il, 2);
      }
    
      fg = (tzs * cuConjf(vB_r) * qB_bar) * 2. * flxJac[idz];
      qflux[idxyz] = fg.x * pres;

    } else {
      qflux[idxyz] = 0.;
    }
  }
}

__global__ void particle_flux_summand(float* pflux,
                                      const cuComplex* phi,
                                      const cuComplex* apar,
                                      const cuComplex* bpar,
                                      const cuComplex* g,
                                      const float* ky,
				      const float* flxJac,
                                      const float *kperp2,
                                      float rho2_s,
                                      float n_s,
                                      float vts,
                                      float tzs)
{
  idXYZ; 
  
  if (idy < nyc && idx < nx && idz < nz && m_lo==0) { 
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    if (unmasked(idx, idy) && idy > 0) {    
      
      cuComplex vPhi_r = make_cuComplex(0., ky[idy]) * phi[idxyz];
      cuComplex vA_r = make_cuComplex(0., ky[idy]) * apar[idxyz];
      cuComplex vB_r = make_cuComplex(0., ky[idy]) * bpar[idxyz];
    
      float b_s = kperp2[idxyz]*rho2_s;
    
      // sum over l
      cuComplex n_bar = make_cuComplex(0.,0.);
      cuComplex u_bar = make_cuComplex(0.,0.);
      cuComplex uB_bar = make_cuComplex(0.,0.);

      for (int il=0; il < nl; il++) {
	n_bar = n_bar + Jflr(il, b_s)*Gh_(idxyz, il, 0);
	u_bar = u_bar + Jflr(il, b_s)*Gh_(idxyz, il, 1);
	uB_bar = uB_bar + JflrB(il, b_s)*Gh_(idxyz, il, 0);
      }
    
      cuComplex fg = (cuConjf(vPhi_r) * n_bar - vts*cuConjf(vA_r)*u_bar + tzs*cuConjf(vB_r)*uB_bar) * 2. * flxJac[idz];
      pflux[idxyz] = fg.x * n_s;

    } else {
      pflux[idxyz] = 0.;
    }
  }
}

__global__ void turbulent_heating_summand(float* heat, const cuComplex* phi, const cuComplex* apar, const cuComplex* bpar, 
                                          const cuComplex* phi_old, const cuComplex* apar_old, const cuComplex* bpar_old, 
                                          const cuComplex* g, const cuComplex* g_old, const float* volJac, const float* kperp2, const specie sp, float dt)
{
  idXYZ; 
  
  if (idy < nyc && idx < nx && idz < nz && m_lo==0 && unmasked(idx, idy)) { 
    unsigned int idxyz = get_idxyz(idx, idy, idz);

    const cuComplex phi_ = phi[idxyz];
    const cuComplex apar_ = apar[idxyz];
    const cuComplex bpar_ = bpar[idxyz];
    const cuComplex phi_old_ = phi_old[idxyz];
    const cuComplex apar_old_ = apar_old[idxyz];
    const cuComplex bpar_old_ = bpar_old[idxyz];
    const cuComplex dPhidt = (phi_ - phi_old_)/dt;
    const cuComplex dAdt = (apar_ - apar_old_)/dt;
    const cuComplex dBdt = (bpar_ - bpar_old_)/dt;

    const float rho2_ = sp.rho2;
    const float vt_ = sp.vt;
    const float zt_ = sp.zt;
    const float tz_ = sp.tz;
    const float nz_ = sp.nz;
    const float b_s = kperp2[idxyz]*rho2_;
    
    // sum over l
    cuComplex n_bar = make_cuComplex(0.,0.);
    cuComplex u_bar = make_cuComplex(0.,0.);
    cuComplex uB_bar = make_cuComplex(0.,0.);

    for (int il=0; il < nl; il++) {
      unsigned int idxyz_l0 = idxyz + nx*nyc*nz*(il + nl*(0 - m_lo));
      unsigned int idxyz_l1 = idxyz + nx*nyc*nz*(il + nl*(1 - m_lo));
      cuComplex H_l0 = g_old[idxyz_l0] + zt_*Jflr(il, b_s)*phi_ + JflrB(il, b_s)*bpar_;
      cuComplex H_l1 = g_old[idxyz_l1] - zt_*vt_*Jflr(il, b_s)*apar_;

      n_bar = n_bar + Jflr(il, b_s)*H_l0;
      u_bar = u_bar + Jflr(il, b_s)*H_l1;
      uB_bar = uB_bar + JflrB(il, b_s)*H_l0;
    }
    
    float fac = 2.0;
    if (idy==0) fac = 1.0;
    cuComplex fg = (cuConjf(dPhidt) * n_bar - vt_*cuConjf(dAdt)*u_bar + tz_*cuConjf(dBdt)*uB_bar) * fac * volJac[idz];
    heat[idxyz] = fg.x * nz_;
  }
}

__global__ void calc_n_bar(cuComplex* n_bar, cuComplex* g, const cuComplex* phi, const cuComplex* bpar, const float *kperp2, const specie sp)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();
  unsigned int idxyz = idy + nyc*(idx + nx*idz);

  if (idy < nyc && idx < nx && idz < nz && m_lo==0 && unmasked(idx,idy) && idy>0) { 
    const float b_ = kperp2[idxyz]*sp.rho2;
    const float zt_ = sp.zt;
    n_bar[idxyz] = make_cuComplex(0.,0.);
    cuComplex phi_ = phi[idxyz];
    cuComplex bpar_ = bpar[idxyz];
    for (int il=0; il < nl; il++) {
      cuComplex H0 = Gh_(idxyz, il, 0) + zt_*Jflr(il, b_)*phi_ + JflrB(il, b_)*bpar_;
      n_bar[idxyz] = n_bar[idxyz] + Jflr(il, b_)*H0;
    }
  }
}

__global__ void calc_upar_bar(cuComplex* upar_bar, cuComplex* g, const cuComplex* apar, const float *kperp2, const specie sp)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();
  unsigned int idxyz = idy + nyc*(idx + nx*idz);

  if (idy < nyc && idx < nx && idz < nz && m_lo==0 && unmasked(idx,idy) && idy>0) { 
    const float b_ = kperp2[idxyz]*sp.rho2;
    const float zt_ = sp.zt;
    const float vt_ = sp.vt;
    upar_bar[idxyz] = make_cuComplex(0.,0.);
    cuComplex apar_ = apar[idxyz];
    for (int il=0; il < nl; il++) {
      cuComplex H1 = Gh_(idxyz, il, 1) - zt_*vt_*Jflr(il, b_)*apar_;
      upar_bar[idxyz] = upar_bar[idxyz] + Jflr(il, b_)*H1;
    }
  }
}

__global__ void calc_uperp_bar(cuComplex* uperp_bar, cuComplex* g, const cuComplex* phi, const cuComplex* bpar, const float *kperp2, const specie sp)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();
  unsigned int idxyz = idy + nyc*(idx + nx*idz);

  if (idy < nyc && idx < nx && idz < nz && m_lo==0 && unmasked(idx,idy) && idy>0) { 
    const float b_ = kperp2[idxyz]*sp.rho2;
    const float zt_ = sp.zt;
    uperp_bar[idxyz] = make_cuComplex(0.,0.);
    cuComplex phi_ = phi[idxyz];
    cuComplex bpar_ = bpar[idxyz];
    for (int il=0; il < nl; il++) {
      cuComplex H0 = Gh_(idxyz, il, 0) + zt_*Jflr(il, b_)*phi_ + JflrB(il, b_)*bpar_;
      uperp_bar[idxyz] = uperp_bar[idxyz] + JflrB(il, b_)*H0;
    }
    uperp_bar[idxyz] = uperp_bar[idxyz]*sqrtf(b_);
  }
}

__global__ void calc_T_bar(cuComplex* T_bar, cuComplex* g, const cuComplex* phi, const cuComplex* bpar, const float *kperp2, const specie sp)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();
  unsigned int idxyz = idy + nyc*(idx + nx*idz);

  if (idy < nyc && idx < nx && idz < nz && m_lo==0 && unmasked(idx,idy) && idy>0) { 
    const float b_ = kperp2[idxyz]*sp.rho2;
    const float zt_ = sp.zt;
    T_bar[idxyz] = make_cuComplex(0.,0.);
    cuComplex phi_ = phi[idxyz];
    cuComplex bpar_ = bpar[idxyz];
    for (int l=0; l < nl; l++) {
      cuComplex H0 = Gh_(idxyz, l, 0) + zt_*Jflr(l, b_)*phi_ + JflrB(l, b_)*bpar_;
      cuComplex H2 = Gh_(idxyz, l, 2);
      T_bar[idxyz] = T_bar[idxyz] + JflrA(l, b_)*H0 + rsqrtf(2.)*Jflr(l, b_)*H2;
    }
    T_bar[idxyz] = T_bar[idxyz]*2./3.;
  }
}

__global__ void kInit(float* kx,
                      float* ky,
                      float* kz,
                      int* kzm,
                      float* kzp,
                      const float X0,
                      const float Y0,
                      const int Zp,
                      bool dealias_kz) 
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  if (id < nyc) ky[id] = (float) id/Y0;
  
  if (id < nx/2+1) {
    kx[id] = (float) id/X0; 
  } else if (id < nx) {
    kx[id] = (float) (id - nx)/X0; 
  }
  
  if (id < (nz/2+1)) {
    kz[id] = (float) id/Zp; 
  } else if (id < nz) {
    kz[id] = (float) (id - nz)/Zp; 
  }

  if (id < (nz-1)/3+1) {
    kzm[id] = 1;
    kzp[id] = kz[id];
  }

  if (dealias_kz) {
    if (id > (nz-1)/3 && id < nz - (nz-1)/3) {
      kzm[id] = 0;
      kzp[id] = 0.;
    }
  } else {
    if (id > (nz-1)/3 && id < nz - (nz-1)/3) {
      kzm[id] = 0;
      kzp[id] = kz[id];
    }
  }

  if (id-nz > -(1 + (nz-1)/3) && id < nz) {
    kzm[id] = 1;
    kzp[id] = kz[id];
  }
  
  if (false) {
    printf("ky[%d] = %f \t ",id, ky[id]);
    printf("kx[%d] = %f \t ",id, kx[id]);
    printf("kz[%d] = %f \n ",id, kz[id]);
    printf("\n");
  }
}

__global__ void real_space_density(cuComplex* nbar, const cuComplex* g, const float *kperp2, const specie sp)
{
  idXYZ;
  
  if (unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);

    const float b_s = kperp2[idxyz] * sp.rho2;
    for (unsigned int l=0; l < nl; l++) {
      unsigned int m = 0; // only m=0 components needed here
      unsigned int m_local = m - m_lo;
      unsigned int ig = idxyz + nx*nyc*nz*(l + nl*m_local);
      nbar[idxyz] = nbar[idxyz] + Jflr(l, b_s) * g[ig] * sp.nz;
    }
  }
}

__global__ void real_space_par_current(cuComplex* jbar, const cuComplex* g, const float *kperp2, const specie sp)
{
  idXYZ;
  
  if (unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    const float b_s = kperp2[idxyz] * sp.rho2;
    for (unsigned int l=0; l < nl; l++) {
      unsigned int m = 1; // only m=1 components needed here
      unsigned int m_local = m - m_lo;
      unsigned int ig = idxyz + nx*nyc*nz*(l + nl*m_local);
      // jparfac = beta_ref/2*Z_s*n_s*v_ts
      jbar[idxyz] = jbar[idxyz] + Jflr(l, b_s) * g[ig] * sp.jparfac;
    }
  }
}

__global__ void real_space_perp_current(cuComplex* jbar, const cuComplex* g,
					const float *kperp2, const float *bmagInv, const specie sp)
{
  idXYZ; 
  
  if (unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    const float b_s = kperp2[idxyz] * sp.rho2;
    for (unsigned int l=0; l < nl; l++) {
      unsigned int m = 0; // only m=0 components needed here
      unsigned int m_local = m - m_lo;
      unsigned int ig = idxyz + nx*nyc*nz*(l + nl*m_local);
      // jperpfac = -beta_ref/2*n_s*T_s
      jbar[idxyz] = jbar[idxyz] + JflrB(l, b_s) * bmagInv[idz]*bmagInv[idz] * g[ig] * sp.jperpfac;
    }
  }
}

__global__ void qneut(cuComplex* Phi, const cuComplex* nbar, const float* denom, float fphi)
{
  idXYZ; 
  
  if ( unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    
    Phi[idxyz] = fphi * nbar[idxyz] / denom[idxyz];    
  }
}

__global__ void ampere_apar(cuComplex* apar, cuComplex* jbar, float* denom, float fapar)
{
  idXYZ; 
  
  if ( unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    
    apar[idxyz] = fapar * jbar[idxyz] / denom[idxyz];
  }
}

__global__ void qneut_and_ampere_perp(cuComplex* Phi,
				      cuComplex* Bpar,
				      const cuComplex* SQ,
				      const cuComplex* SA, 
				      const float* QPhi,
				      const float* QB,
				      const float* APhi,
				      const float* AB,
				      const float fphi,
				      const float fbpar)
{
  idXYZ; 
  
  if ( unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);

    float denom = QPhi[idxyz]*AB[idxyz] - QB[idxyz]*APhi[idxyz]; 
    Phi[idxyz]  = fphi * (AB[idxyz]*SQ[idxyz] - QB[idxyz]*SA[idxyz])/denom;
    Bpar[idxyz]  = fbpar * (-APhi[idxyz]*SQ[idxyz] + QPhi[idxyz]*SA[idxyz])/denom;
  }
}

// compute qneutFacPhi  = sum_s z_s^2*n_s/tau_s*(1- sum_l J_l^2)
//         qneutFacBpar = -sum_s z_s*n_s*sum_l J_l*(J_l + J_{l-1})
//         ampereParFac = kperp2 + beta/2*sum_s z_s^2*n_s/m_s*sum_l J_l^2
//         amperePerpFacPhi  = beta/(2*B^2)*sum_s z_s*n_s*sum_l J_l*(J_l + J_{l-1})
//         amperePerpFacBpar = 1 + beta/(2*B^2)*sum_s n_s*t_s*sum_l (J_l + J_{l-1})^2
__global__ void sum_solverFacs(float* qneutFacPhi,
			       float* qneutFacBpar,
			       float* ampereParFac,
			       float* amperePerpFacPhi,
			       float* amperePerpFacBpar,
                               const float* kperp2,
			       const float* bmag,
			       const float* bmagInv,
			       const specie sp,
			       const float beta,
			       const bool first,
			       const float fapar,
			       const float fbpar,
			       bool long_wavelength_GK)
{
  idXYZ; 
  
  if ( unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);
        
    const float kperp2_ = kperp2[idxyz];
    float b_s;
    if (long_wavelength_GK) b_s = kperp2_ * sp.rho2_long_wavelength_GK;
    else b_s = kperp2_ * sp.rho2;

    float g0_s = 0.;
    float g01_s = 0.;
    float g11_s = 0.;
    for (int l=0; l < nl; l++) {
      const float Jl = Jflr(l, b_s);
      const float JlB = JflrB(l, b_s);
      g0_s += Jl*Jl;
      g01_s += Jl*JlB;
      g11_s += JlB*JlB;
    }

    if (long_wavelength_GK) {
      qneutFacPhi[idxyz] += sp.nz*sp.zt * b_s;
    } else {
      qneutFacPhi[idxyz] += sp.nz*sp.zt * ( 1. - g0_s );
    }

    if(fapar>0.) {
      // since kperp2 == kperp**2/B**2, need to multiply by bmag**2 to get kperp**2 for Ampere's law
      if (first) ampereParFac[idxyz] = kperp2_*bmag[idz]*bmag[idz]; 
      ampereParFac[idxyz] += sp.nz*sp.z/sp.mass*beta/2. * g0_s ;
    }

    if(fbpar>0.) {
      qneutFacBpar[idxyz] += -sp.nz * g01_s;
      amperePerpFacPhi[idxyz] += sp.nz*beta/2. * bmagInv[idz]*bmagInv[idz] * g01_s;
      if(first) amperePerpFacBpar[idxyz] = 1.;
      amperePerpFacBpar[idxyz] += sp.nt*beta/2. * bmagInv[idz]*bmagInv[idz] * g11_s;
    }
  }
}


__global__ void qneutAdiab_part1(cuComplex* PhiAvgNum_tmp, const cuComplex* nbar,
				 const float* jacobian, const float* qneutDenom, float tau_fac)
{
  idXYZ; 
  
  if ( unmasked(idx, idy) && idz < nz) {
  
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    float pfilter2 = qneutDenom[idxyz];
    
    PhiAvgNum_tmp[idxyz] = ( nbar[idxyz] / (tau_fac + pfilter2 ) ) * jacobian[idz];
  }
}


__global__ void qneutAdiab_part2(cuComplex* Phi,
				 const cuComplex* PhiAvgNum_tmp,
				 const cuComplex* nbar,
				 const float* PhiAvgDenom,
				 const float* qneutDenom,
				 float tau_fac,
				 float fphi)
{
  idXYZ; 
  
  if ( unmasked(idx, idy) && idz < nz) {

    unsigned int idxyz = get_idxyz(idx, idy, idz);
    unsigned int idxy  = idy + nyc*idx;

    float pfilter2 = qneutDenom[idxyz];

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

    Phi[idxyz].x = fphi * ( nbar[idxyz].x + tau_fac * PhiAvg.x ) / (tau_fac + pfilter2); // Eq 4
    Phi[idxyz].y = fphi * ( nbar[idxyz].y + tau_fac * PhiAvg.y ) / (tau_fac + pfilter2);
  }
}

__global__ void calc_phiavgdenom(float* PhiAvgDenom, const float* jacobian,
				 const float* qneutDenom, float tau_fac)
{   
  unsigned int idx = get_id1();
  unsigned int idy = 0;

  if ( unmasked(idx, idy)) {
  
    PhiAvgDenom[idx] = 0.;  
    
    for (unsigned int idz=0; idz < nz; idz++) {
      
      unsigned int idxyz = get_idxyz(idx, idy, idz);
      float pfilter2 = qneutDenom[idxyz];
      PhiAvgDenom[idx] = PhiAvgDenom[idx] + jacobian[idz] * pfilter2 / (tau_fac + pfilter2);
    }
  }
}

__global__ void add_source(cuComplex* f, const float source)
{
  idXYZ; 
  
  if ( unmasked(idx, idy) && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    f[idxyz].x = f[idxyz].x + source;
  }
}

__global__ void qneutAdiab(cuComplex* Phi, const cuComplex* nbar,
			   const float* qneutDenom, float tau_fac, float fphi)
{
  idXYZ; 
  
  if ( unmasked(idx, idy) && idz < nz) {
    
    unsigned int idxyz = get_idxyz(idx, idy, idz);

    Phi[idxyz].x = fphi * ( nbar[idxyz].x / (tau_fac + qneutDenom[idxyz] ) );  
    Phi[idxyz].y = fphi * ( nbar[idxyz].y / (tau_fac + qneutDenom[idxyz] ) );  
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

__device__ void mkz2_Linked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
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
  ((cuComplex*)dataOut)[offset] = Ikz*Ikz*element*normalization;
}

__device__ void zfts_Linked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  float *kz  = (float*) kzData;
  int nLinks = (int) lrintf(1./(zp*kz[1]));
  float normalization = (float) 1./(nz*nLinks);
  ((cuComplex*)dataOut)[offset] = element*normalization;
}

__device__ void hyperkzLinked(void *dataOut, size_t offset, cufftComplex element, void *hyperData, void *sharedPtr)
{
  int *data = (int*) hyperData;
  int nLinks = data[0];
  int p_hyper_z = data[1];
  unsigned int nzL = nz*nLinks;
  unsigned int idz = offset % (nzL);
  float zpnLinv = (float) 1./zp*nLinks;
  float hypkz;
  if (idz < nzL/2+1) {
    hypkz = powf((float) idz * zpnLinv / (nzL/2 * zpnLinv), p_hyper_z);
  } else {
    int idzs = idz-nzL;
    hypkz = powf((float) idzs * zpnLinv / (nzL/2 * zpnLinv), p_hyper_z);
  }
  float normalization = (float) 1./nzL;
  ((cuComplex*)dataOut)[offset] = -hypkz*element*normalization;
}

__device__ void abs_kzLinked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  float *kz = (float*) kzData;
  int nLinks = (int) lrintf(1./(zp*kz[1]));
  unsigned int idz = offset % (nz*nLinks);
  float normalization = (float) 1./(nz*nLinks);
  ((cuComplex*)dataOut)[offset] = abs(kz[idz])*element*normalization;
}

__global__ void init_kzLinked(float* kz, int nLinks, bool dealias_kz)
{
  int nzL = nz*nLinks;
  for (int i=0; i < nzL; i++) {
    if (i < nzL/2+1) {
      kz[i] = (float) i/(zp*nLinks);
    } else {
      kz[i] = (float) (i-nzL)/(zp*nLinks);
    }
    if (dealias_kz) {
      if (i > (nzL-1)/3 && i < nzL - (nzL-1)/3) {kz[i] = 0.0;}
    }
  }
}

__global__ void init_hyperkzLinked(float* hyperkz, int nLinks, bool dealias_kz, int p_hyper_z)
{
  int nzL = nz*nLinks;
  for (int i=0; i < nzL; i++) {
    if (i < nzL/2+1) {
      hyperkz[i] = pow((float) i/(zp*nLinks), p_hyper_z);
    } else {
      hyperkz[i] = pow((float) (i-nzL)/(zp*nLinks), p_hyper_z);
    }
    if (dealias_kz) {
      if (i > (nzL-1)/3 && i < nzL - (nzL-1)/3) {hyperkz[i] = 0.0;}
    }
  }
}

__device__ cufftCallbackStoreC  zfts_Linked_callbackPtr = zfts_Linked;
__device__ cufftCallbackStoreC   i_kzLinked_callbackPtr = i_kzLinked;
__device__ cufftCallbackStoreC   hyperkzLinked_callbackPtr = hyperkzLinked;
__device__ cufftCallbackStoreC  mkz2_Linked_callbackPtr = mkz2_Linked;
__device__ cufftCallbackStoreC abs_kzLinked_callbackPtr = abs_kzLinked;

__global__ void linkedCopy(const cuComplex* __restrict__ G,
			   cuComplex* __restrict__ G_linked,
			   int nLinks,
			   int nChains,
			   const int* __restrict__ ikx,
			   const int* __restrict__ iky,
			   int nMoms)
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

__global__ void linkedCopyBack(const cuComplex* __restrict__ G_linked,
			       cuComplex* __restrict__ G,
			       int nLinks,
			       int nChains,
			       const int* __restrict__ ikx,
			       const int* __restrict__ iky,
			       int nMoms)
{
  unsigned int idz  = get_id1();
  unsigned int idk  = get_id2();
  unsigned int idlm = get_id3();

  if (idz < nz && idk < nLinks*nChains && idlm < nMoms) {
    unsigned int idlink = idz + nz*(idk + nLinks*nChains*idlm);
    unsigned int globalIdx = iky[idk] + nyc*(ikx[idk] + nx*(idz + nz*idlm));
    G[globalIdx] = G_linked[idlink];
  }
}

__global__ void linkedAccumulateBack(const cuComplex* __restrict__ G_linked,
                                     cuComplex* __restrict__ G,
                                     int nLinks,
                                     int nChains,
                                     const int* __restrict__ ikx,
                                     const int* __restrict__ iky,
                                     int nMoms,
                                     float scale)
{
  unsigned int idz = get_id1();
  unsigned int idk = get_id2();
  unsigned int idlm = get_id3();

  if (idz < nz && idk < nLinks*nChains && idlm < nMoms) {
    unsigned int idlink = idz + nz*(idk + nLinks*nChains*idlm);
    unsigned int globalIdx = iky[idk] + nyc*(ikx[idk] + nx*(idz + nz*idlm));
    G[globalIdx] = G[globalIdx] + scale*G_linked[idlink];
  }
}

__global__ void dampEnds_linked(cuComplex* G,
                                cuComplex* phi,
                                cuComplex* apar,
                                cuComplex* bpar,
                                float* kperp2,
                                specie sp,
                                int nLinks,
                                int nChains,
                                const int* ikx,
                                const int* iky,
                                int nMoms,
                                cuComplex* GRhs,
                                float widthfrac,
                                float amp)
{
  unsigned int idz = get_id1();
  unsigned int idk = get_id2();
  unsigned int idlm = get_id3();

  if (idz < nz && idk < nLinks*nChains && idlm < nMoms) {
    unsigned int idzl = idz + nz*(idk % nLinks);
    unsigned int globalIdx = iky[idk] + nyc*(ikx[idk] + nx*(idz + nz*idlm));
    unsigned int idxyz = iky[idk] + nyc*(ikx[idk] + nx*idz);
    // unsigned int idxyz = get_idxyz(ikx[idk], iky[idk], idz);
    // Question: why are we using ikx and iky in the definition of idxyz? Surely this should be idx and idy?
    // Answer: in grad_parallel_linked, we use ky and kx to hold index values rather than k values. Confusing.
    
    float nu = 0.;
    // width = width of damping region in number of grid points 
    // set damping region width to 1/8 of extended domain (on either side)
    // widthfac = 1./8.;
    int width = (int) nz*nLinks*widthfrac;  
    float L = (float) 2*M_PI*zp*nLinks*widthfrac;
    float vmax = sqrtf(2*nm_glob); // estimate of max vpar on grid
    if (idzl <= width ) {
      float x = ((float) idzl)/width;
      nu = 1 - 2*x*x/(1+x*x*x*x);
    } else if (idzl >= nz*nLinks-width) {
      float x = ((float) nz*nLinks-idzl)/width;
      nu = 1 - 2*x*x/(1+x*x*x*x);
    }
    // only damp ends of non-zonal (ky>0) modes, since ky=0 modes should be periodic
    if(iky[idk]>0) {
      unsigned int idl = idlm % nl;
      unsigned int idm = idlm / nl;
      const float kperp2_ = kperp2[idxyz];
      const float zt_ = sp.zt;
      const float vt_ = sp.vt;
      const float rho2_ = sp.rho2;
      const float b_ = kperp2_ * rho2_;
      // the quantity we want to damp is h = g' + phi*FM - vpar*Apar*FM, so we need to adjust m=0 and m=1 with fields
      cuComplex H_ = G[globalIdx];
      if(idm+m_lo==0) H_ = H_ + zt_*Jflr(idl, b_)*phi[idxyz] + JflrB(idl, b_)*bpar[idxyz];
      if(idm+m_lo==1) H_ = H_ - zt_*vt_*Jflr(idl, b_)*apar[idxyz]; 
      GRhs[globalIdx] = GRhs[globalIdx] - nu*amp*H_;
      //GRhs[globalIdx] = GRhs[globalIdx] - 5.0*nu*vmax/L*H_;
    }
  }
}

__global__ void zeroEnds_linked(cuComplex* G, cuComplex* phi, cuComplex* apar, float* kperp2, specie sp,
			       int nLinks, int nChains, const int* ikx, const int* iky, int nMoms)
{
  unsigned int idz = get_id1();
  unsigned int idk = get_id2();
  unsigned int idlm = get_id3();

  if (idz < nz && idk < nLinks*nChains && idlm < nMoms) {
    unsigned int idzl = idz + nz*(idk % nLinks);
    unsigned int globalIdx = iky[idk] + nyc*(ikx[idk] + nx*(idz + nz*idlm));
    unsigned int idxyz = iky[idk] + nyc*(ikx[idk] + nx*idz);

    // only zero ends of non-zonal (ky>0) modes, since ky=0 modes should be periodic
    if(iky[idk]>0 && (idzl==0 || idzl==nz*nLinks)) {
      unsigned int idl = idlm % nl;
      unsigned int idm = idlm / nl;
      const float kperp2_ = kperp2[idxyz];
      const float b_ = kperp2_ * sp.rho2;
      G[globalIdx].x = 0.; G[globalIdx].y = 0.;
      if(idm+m_lo==0) G[globalIdx] = -sp.zt*Jflr(idl, b_)*phi[idxyz]; // this seems to cause numerical instability...
      if(idm+m_lo==1) G[globalIdx] = sp.zt*sp.vt*Jflr(idl, b_)*apar[idxyz];
    }

  }
}

__global__ void linkedFilterEnds(cuComplex* G, int ifilter,
				 int nLinks, int nChains, const int* ikx, const int* iky, int nMoms)
{
  unsigned int idz = get_id1();
  unsigned int idk = get_id2();
  unsigned int idlm = get_id3();

  if (idz < nz && idk < nLinks*nChains && idlm < nMoms) {
    unsigned int idzl = idz + nz*(idk % nLinks);
    unsigned int globalIdx = iky[idk] + nyc*(ikx[idk] + nx*(idz + nz*idlm));

    float filter = 1.;
    int width = 1;// nz*nLinks/ifilter;
    if (idzl <= width ) {
      float x = ((float) idzl)/width;
      filter = 2*x*x/(1+x*x*x*x);
    } else if (idzl >= nz*nLinks-width) {
      float x = ((float) nz*nLinks-idzl)/width;
      filter = 2*x*x/(1+x*x*x*x);
    }
    // only filter non-zonal (ky>0) modes, since ky=0 modes should be periodic
    if(iky[idk]>0) G[globalIdx] = filter*G[globalIdx];
  }
}

__global__ void streaming_rhs(const cuComplex* __restrict__ g,
			      const cuComplex* __restrict__ phi,
			      const cuComplex* __restrict__ apar,
			      const cuComplex* __restrict bpar,
			      const float* __restrict__ kperp2, 
			      const float gradpar,
			      const specie sp,
			      cuComplex* __restrict__ rhs_par)
{
  unsigned int idy  = get_id1();
  unsigned int idx  = get_id2();
  unsigned int idzl = get_id3();
  if (unmasked(idx, idy) && (idzl < nz*nl)) {
    unsigned int idz = idzl % nz;     
    unsigned int l   = idzl / nz;
    unsigned int idxyz = get_idxyz(idx, idy, idz);

    const cuComplex phi_  = phi[idxyz];
    const cuComplex apar_ = apar[idxyz];
    const cuComplex bpar_ = bpar[idxyz];

    const float b_s = sp.rho2 * kperp2[idxyz];
    const float zt_ = sp.zt;
    const float vt_ = sp.vt;
    int globalIdx;

    for (int m = m_lo; m < m_up; m++) {
      int m_local = m - m_lo;
      globalIdx = idy + nyc*( idx + nx*(idzl + nz*nl*(m_local)));	
      int mp1 = idy + nyc*( idx + nx*(idzl + nz*nl*(m_local+1)));
      int mm1 = idy + nyc*( idx + nx*(idzl + nz*nl*(m_local-1)));
      cuComplex gmp1 = make_cuComplex(0.,0.);
      cuComplex gmm1 = make_cuComplex(0.,0.);
      if(m>0) gmm1 = g[mm1];
      if(m<nm_glob-1) gmp1 = g[mp1];
      
      rhs_par[globalIdx] = rhs_par[globalIdx] -vt_ * (sqrtf(m+1)*gmp1 + sqrtf(m)*gmm1) * gradpar;
      
      // field terms
      if(m == 1) rhs_par[globalIdx] = rhs_par[globalIdx] - Jflr(l, b_s) * phi_ * zt_ * vt_ * gradpar
		   - JflrB(l, b_s) * bpar_ * vt_ * gradpar; // m = 1 has Phi & Bpar terms
      // the following Apar terms are only needed in the formulation without dA/dt
      if(m == 0) rhs_par[globalIdx] = rhs_par[globalIdx] + Jflr(l, b_s) * apar_ * zt_ * vt_ * vt_ * gradpar; // m = 0 has Apar term
      if(m == 2) rhs_par[globalIdx] = rhs_par[globalIdx] + sqrtf(2.) * Jflr(l, b_s) * apar_ * zt_ * vt_ * vt_ * gradpar; // m = 2 has Apar term
    }
  }
}

// main kernel function for calculating RHS
# define S_H(L, M) s_h[sidxyz + (sDimx)*(L) + (sDimx)*(sDimy)*(M)]
__global__ void rhs_linear(const cuComplex* __restrict__ g,
			   const cuComplex* __restrict__ phi,
			   const cuComplex* __restrict__ apar,
			   const cuComplex* __restrict__ bpar,
			   const cuComplex* __restrict__ upar_bar,
			   const cuComplex* __restrict__ uperp_bar,
			   const cuComplex* __restrict__ t_bar,
			   const float* __restrict__ kperp2,
			   const float* __restrict__ cv_d,
			   const float* __restrict__ gb_d,
			   const float* __restrict__ bmag,
			   const float* __restrict__ bgrad,
			   const float* __restrict__ ky,
			   const specie sp,
			   const specie sp_i,
			   cuComplex* __restrict__ rhs,
			   bool ei_colls)
{
  extern __shared__ cuComplex s_h[]; // aliased below by macro S_H, defined above
  
  const unsigned int idxyz = get_id1();
  const unsigned int idy = idxyz % nyc; 
  const unsigned int idx = idxyz / nyc % nx;
  const unsigned int idz = idxyz / (nx*nyc);
  if (unmasked(idx, idy) && idz < nz) {
    const unsigned int sidxyz = threadIdx.x;
    
    // shared memory blocks of size blockDim.x * (nl+2) * (nm+4)
    const int sDimx = blockDim.x;
    const int sDimy = nl+2;
  
    // read these values into (hopefully) register memory. 
    // local to each thread (i.e. each idxyz).
    // since idxyz is linear, these accesses are coalesced.
    const cuComplex phi_  = phi[idxyz];
    const cuComplex apar_ = apar[idxyz];
    const cuComplex bpar_ = bpar[idxyz];
  
    // all threads in a block will likely have same value of idz, so they will be reading same value of bgrad[idz].
    // if bgrad were in shared memory, would have bank conflicts.
    // no bank conflicts for reading from global memory though. 
    const float bmag_ = bmag[idz];
    const float bgrad_ = bgrad[idz];  
  
    // this is coalesced
    const cuComplex iky_ = make_cuComplex(0., ky[idy]); 

    unsigned int nR = nyc * nx * nz;
    
    // species-specific constants
    const float vt_ = sp.vt;
    const float zt_ = sp.zt;
    const float tz_ = sp.tz;
    const float nz_ = sp.nz;
    const float nu_ = sp.nu_ss; 
    const float tprim_ = sp.tprim;
    const float uprim_ = sp.uprim;
    const float fprim_ = sp.fprim;
    const float kperp2_ = kperp2[idxyz];
    const float b_s = kperp2_ * sp.rho2;
    float nuei_ = 0.0;
    float as_i = 1.0;
    float vt_i = 1.0;
    float nzvt_i = 1.0;
    // for electrons, account for e-i collisions
    if(sp.type == 1 && ei_colls) {
      if( nspecies > 1 )
	 nuei_ = sp_i.z * nu_;
      else
	 nuei_ = nu_;
      // get as = z*n*vt*beta/2 from first ion species (assume this is main ions)
      as_i = sp_i.jparfac; 
      vt_i = sp_i.vt;
      nzvt_i = sp_i.nz*sp_i.vt;
    }
    
    const cuComplex icv_d_s = tz_ * make_cuComplex(0., cv_d[idxyz]);
    const cuComplex igb_d_s = tz_ * make_cuComplex(0., gb_d[idxyz]);

    // conservation terms (species-specific)
    cuComplex upar_bar_  =  upar_bar[idxyz]; 
    cuComplex uperp_bar_ = uperp_bar[idxyz];
    cuComplex t_bar_     =     t_bar[idxyz];
    
    // read tile of g into shared mem
    // each thread in the block reads in multiple values of l and m
    // blockIdx for y and z and both of size unity in the kernel invocation
    int nm_shared = nm+2*m_ghost;
    if(m_ghost==0) nm_shared+=4;
    int nl_shared = nl+2;
    int ghost = m_ghost==0 ? 2 : m_ghost;
    for (int sm = threadIdx.z; sm < nm_shared; sm += blockDim.z) {
      for (int sl = threadIdx.y; sl < nl_shared; sl += blockDim.y) {
        int globalIdx = idxyz + nR*((sl-1) + nl*(sm-ghost));
        int l = sl-1;
        int m = sm-ghost+m_lo;
        if(m<0 || m>=nm_glob || l<0 || l>=nl) {
          S_H(sl, sm) = make_cuComplex(0., 0.);
        } else {
          S_H(sl, sm) = g[globalIdx];
          // add phi term for m=0 to change g into h
          if (m==0) S_H(sl, sm) = S_H(sl, sm) + zt_*Jflr(l, b_s)*phi_ + JflrB(l, b_s)*bpar_;
          // add apar term for m=1 (this is only needed in the formulation without dA/dt)
          if (m==1) S_H(sl, sm) = S_H(sl, sm) - zt_*vt_*Jflr(l, b_s)*apar_;
        }
      }
    }
     
    __syncthreads();
    
    // stencil (on non-ghost cells)
    // blockIdx for y and z are unity in the kernel invocation
    for (int m = threadIdx.z + m_lo; m < m_up; m += blockDim.z) {
      for (int l = threadIdx.y; l < nl; l += blockDim.y) {
        int m_local = m - m_lo;
        int globalIdx = idxyz + nR*(l + nl*m_local);
        int sl = l + 1;             // offset to get past ghosts
        int sm = m_local + m_ghost; // offset to get past ghosts
	if(m_ghost==0) sm+=2;
  
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
	    + Jflr(l+1,b_s)*(l+1)*tprim_ 
	   )
           + iky_/zt_ * bpar_ * (
              JflrB(l-1,b_s)*l*tprim_
	    + JflrB(l,  b_s)*(fprim_ + 2*l*tprim_)
	    + JflrB(l+1,b_s)*(l+1)*tprim_ 
	   )
	   + nu_ * sqrtf(b_s) * JflrB(l, b_s) * uperp_bar_
	   + ( nu_ + nuei_ ) * 2. * ( l*Jflr(l-1,b_s) + 2.*l*Jflr(l,b_s) + (l+1)*Jflr(l+1,b_s) ) * t_bar_; 
	}

	if (m==1) {
	  cuComplex upar_bar_i = (nspecies>1 && as_i>0) ? kperp2_*bmag_*bmag_*apar_/as_i - nz_*vt_*upar_bar_/(nzvt_i) : make_cuComplex(0.,0.);

	  rhs[globalIdx] = rhs[globalIdx] 
           - vt_ * iky_ * apar_ * (
              Jflr(l-1,b_s)*l*tprim_
	    + Jflr(l,  b_s)*(fprim_ + (2*l+1)*tprim_)
	    + Jflr(l+1,b_s)*(l+1)*tprim_ 
	   )
      	   + Jflr(l,b_s) * (nu_*upar_bar_ + nuei_*vt_i/vt_*upar_bar_i)
           + phi_ * Jflr(l,b_s) * uprim_ * iky_ / vt_; // need to set uprim_ more carefully; this is a placeholder
	}
	if (m==2) {
	  rhs[globalIdx] = rhs[globalIdx] + iky_*phi_*Jflr(l,b_s)/sqrtf(2.)*tprim_ 
	     + iky_/zt_*bpar_*JflrB(l, b_s)/sqrtf(2.)*tprim_ 
	     + ( nu_ + nuei_ ) * sqrtf(2.) * Jflr(l,b_s) * t_bar_;
	}  

	if (m==3) {
	  rhs[globalIdx] = rhs[globalIdx] 
           - vt_ * iky_ * apar_ * sqrtf(3./2.) * tprim_ * Jflr(l,b_s);
        }
      } // l loop
    } // m loop
  } // idxyz < NxNycNz
}

__global__ void rhs_linear_krehm(const cuComplex* g,
				 const cuComplex* phi,
				 const cuComplex* apar,
				 const cuComplex* apar_ext, 
				 const float nu_ei,
				 const float rhos,
				 const float de,
				 cuComplex* rhs_par)
{
  idXYZ; 

  if (unmasked(idx, idy) && (idz < nz)) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    unsigned int globalIdx;
    
    const float rhos_ov_de = rhos/de;
    const cuComplex phi_ = phi[idxyz];
    const cuComplex apar_ = apar[idxyz];

    for (int m = m_lo; m < m_up; m++) {
      int m_local = m - m_lo;
      globalIdx = idxyz + nyc*nx*nz * (m_local  );
      int mp1   = idxyz + nyc*nx*nz * (m_local+1);
      int mm1   = idxyz + nyc*nx*nz * (m_local-1);
      cuComplex gmp1 = make_cuComplex(0.,0.);
      cuComplex gmm1 = make_cuComplex(0.,0.);
      if(m>0) gmm1 = g[mm1];
      if(m<nm_glob-1) gmp1 = g[mp1];
      
      rhs_par[globalIdx] = rhs_par[globalIdx] -rhos_ov_de * (sqrtf(m+1)*gmp1 + sqrtf(m)*gmm1);
      
      // These field terms should be included only when the global value of m = 0, 1, 2.  Bug?
      // field terms
      if(m == 0) rhs_par[globalIdx] = rhs_par[globalIdx] - apar_/(de*de);  // m = 0 has Apar term      
      if(m == 1) rhs_par[globalIdx] = rhs_par[globalIdx] + phi_/(rhos*de); // m = 1 has Phi term      
      if(m == 2) rhs_par[globalIdx] = rhs_par[globalIdx] - sqrtf(2.) * apar_/(de*de); // m = 2 has Apar term
    }
  }
}

__global__ void krehm_collisions(const cuComplex* g,
				 const cuComplex* apar,
				 const cuComplex* apar_ext,
				 const float* kx,
				 const float* ky,
				 const float nu_ei,
				 const float rhos,
				 const float de,
				 cuComplex* rhs)
{
  idXYZ; 

  if (unmasked(idx, idy) && (idz < nz)) {
    const unsigned int idxyz = get_idxyz(idx, idy, idz);

    const cuComplex apar_     = apar[idxyz];
    const cuComplex apar_ext_ = apar_ext[idxyz];

    const float rhos_ov_de = rhos/de;
    const float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];

    for (unsigned int m = m_lo; m < m_up; m++) {
       unsigned int m_local = m - m_lo;
       unsigned int globalIdx = idxyz + nyc*nx*nz*m_local;
       
       // collision term
       if(m!=2) rhs[globalIdx] = rhs[globalIdx] - nu_ei*m*g[globalIdx];
       // for m=1, there are additional terms that add to the above
       if(m==1) rhs[globalIdx] = rhs[globalIdx] - nu_ei*apar_/(rhos*de) - nu_ei*de/rhos*kperp2*apar_ext_;
    }
  }
}

//
// Calculate terms proportional to kz**2
//
__global__ void rhs_diff_cetg(const cuComplex* density,
			      const cuComplex* temperature,
			      const cuComplex* phi,
			      const float gpar,
			      const float c1,
			      const float C12,
			      const float C23,
			      cuComplex* rhs_diff)
{
  idXYZ; 

  if (unmasked(idx, idy) && (idz < nz)) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);

    float gpar2 = gpar * gpar;
    
    rhs_diff[            idxyz] = gpar2/2. * c1 * (      density[idxyz] + C12 * temperature[idxyz] -       phi[idxyz]);
    rhs_diff[nx*nyc*nz + idxyz] = gpar2/3. * c1 * (C12 * density[idxyz] + C23 * temperature[idxyz] - C12 * phi[idxyz]); 
      
  }
}

//
// omega_star term appears in the temperature equation
//
__global__ void rhs_lin_cetg(const cuComplex* phi, const float* ky, cuComplex* rhs)
{
  idXYZ;
    
  if (unmasked(idx, idy) && (idz < nz)) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    unsigned int idxyzt = idxyz + nx*nyc*nz; 
    
    const cuComplex Iky = make_cuComplex(0., ky[idy]);
    
    rhs[idxyzt] = rhs[idxyzt] - 0.5 * Iky * phi[idxyz]; 

  }
}

__global__ void hyper_cetg(const cuComplex* g,
			   const float* kx,
			   const float* ky,
			   const float nu_hyper,
			   const float D_hyper,
			   cuComplex* rhs)
{
  idXYZ;
  
  if (unmasked(idx, idy) && (idz < nz)) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    
    float Dfac = D_hyper*pow((kx[idx]*kx[idx] + ky[idy]*ky[idy]), nu_hyper);
    
    for (int l = 0; l < 2; l++) {
      unsigned int ig = idxyz + nx*nyc*nz*l;
      rhs[ig] = rhs[ig] - Dfac * g[ig];
    }
  }
}

__global__ void hyperdiff(const cuComplex* g,
			  const float* kx,
			  const float* ky,
			  float nu_hyper,
			  float D_hyper,
			  cuComplex* rhs) 
{
  unsigned int idxyz   = get_id1();
  unsigned int l       = get_id2();
  unsigned int m_local = get_id3();
  
  if (idxyz < nx*nyc*nz && l<nl && m_local<nm) {
    unsigned int idy = idxyz % nyc;
    unsigned int idx = (idxyz / nyc) % nx;
    if (unmasked(idx, idy)) {	
      float kxmax = kx[(nx-1)/3];
      float kymax = ky[(ny-1)/3];
      float k2s = 1./powf((kxmax*kxmax + kymax*kymax), nu_hyper);      
      double Dfac = D_hyper*pow((double) (kx[idx]*kx[idx] + ky[idy]*ky[idy])/(kxmax*kxmax + kymax*kymax), nu_hyper);
      
      unsigned int ig = idxyz + nx*nyc*nz*(l + nl*m_local);
      rhs[ig] = rhs[ig] - Dfac * g[ig];
    }
  }
}

__global__ void hypercollisions(const cuComplex* g,
                                const float nu_hyper_l,
                                const float nu_hyper_m,
                                const float nu_hyper_lm,
                                const int p_hyper_l,
                                const int p_hyper_m,
                                const int p_hyper_lm,
                                cuComplex* rhs,
                                const float vt) 
{
  unsigned int idxyz = get_id1();
  
  unsigned int idy = idxyz % nyc;
  unsigned int idx = (idxyz / nyc) % nx;

  // I thought there was now a factor of |kz| in the Hermite hypercollisions?
  
  if ( unmasked(idx, idy) && idxyz < nx*nyc*nz) {
    float scaled_nu_hyp_l = (float) nl * nu_hyper_l;
    float scaled_nu_hyp_m = (float) nm_glob * nu_hyper_m; // scaling appropriate for curvature. Too big for slab

    unsigned int l = get_id2();                                                                
    if (l<nl) {
      unsigned int m = get_id3() + m_lo;
      if (m>=m_lo && m<m_up) {                                                                 
        int m_local = m - m_lo;
        int globalIdx = idxyz + nx*nyc*nz*(l + nl*m_local);                                    
        if (m>2 || l>1) { 
          rhs[globalIdx] = rhs[globalIdx] -
	    nu_hyper_lm*powf((float) (2*l + m)/(2*nl + nm_glob), p_hyper_lm)*g[globalIdx]
             - vt*(scaled_nu_hyp_l*powf((float) l/nl, (float) p_hyper_l)                              
             + scaled_nu_hyp_m*powf((float) m/nm_glob, (float) p_hyper_m))*g[globalIdx];                 
        }   
      }      
    }   
  }
}

__global__ void hypercollisions_kz(const cuComplex* g, const float nu, const int p, cuComplex* res) {
  unsigned int idxyz = get_id1();
  
  if (idxyz < nx*nyc*nz) {
    // blockIdx for y and z are unity in the kernel invocation      
    unsigned int l = get_id2();                                                                
    if (l<nl) {
      unsigned int m = get_id3() + m_lo;
      if (m>=m_lo && m<m_up) {                                                                 
        int m_local = m - m_lo;
        int globalIdx = idxyz + nx*nyc*nz*(l + nl*m_local);                                    
        if (m>2) { 
          res[globalIdx] = -nu*powf((float) m, p)*g[globalIdx];                 
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
	  unsigned int idxyz = get_idxyz(idx, idy, idz); 
	  float kp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];
	  float df2 = df[idxyz].x*df[idxyz].x + df[idxyz].y*df[idxyz].y;

	  s10[idz] += 2. * pow(ky[idy], 4) * df2; 
	  s11[idz] += 2. * pow(kp2, 2)     * df2;
	}
      }      
    }
    s10[idz] = 0.5 * (-w_osc + sqrtf(pow(w_osc, 2) + 2 * s10[idz]));
    s11[idz] = 0.5 * (-w_osc + sqrtf(pow(w_osc, 2) + 2 * s11[idz]));
  }
}

__global__ void get_s01 (float* s01, const cuComplex* favg, const float* kx, const float w_osc) {

  s01[0] = 0.;
  for (int idx = 0; idx < nx; idx++) {
    s01[0] += powf(fabsf(kx[idx]), 4) * (favg[idx].x*favg[idx].x + favg[idx].y*favg[idx].y);
  }
  s01[0] = 0.5 * (-w_osc + sqrtf(pow(w_osc, 2) + 2 * s01[0]));
}

// This hyperdiffusivity is not applied to Hermite ghost cells
 __global__ void HB_hyper (const cuComplex* G,
			  const float* s01,
			  const float* s10,
			  const float* s11,
			  const float* kx,
			  const float* ky,
			  const float D_HB,
			  const int p_HB,
			  cuComplex* RHS)
{
  unsigned int idy  = get_id1();
  unsigned int idxz = get_id2();
  unsigned int idlm = get_id3();
  if ( idy < nyc && idxz < nx*nz && idlm < nl*nm ) {
    unsigned int idx = idxz % nx;
    unsigned int idz = idxz / nx;
    if (unmasked(idx, idy)) {
      
      unsigned int ig = idy + nyc*(idxz + nx*nz*idlm);
      
      float kxmax = kx[(nx-1)/3];
      float kymax = ky[(ny-1)/3];
      float kpmax2 = kxmax*kxmax + kymax*kymax;
      float kp2 = (kx[idx]*kx[idx] + ky[idy]*ky[idy])/kpmax2;
	  
      float D10 = D_HB * pow(kx[idx]/kxmax, 4);
      float D01 = D_HB * pow(kx[idx]/kxmax, 4) * ky[idy]/kymax;
      float D11 = D_HB * pow(kp2, p_HB);
      
      float sfac = (idy == 0) ? s10[idz] * D10 : s11[idz] * D11 + s01[0] * D01;
      RHS[ig] = RHS[ig] - sfac * G[ig];
    }
  }
}
 
# define Hc_(XYZ, L, M) (g[(int) (XYZ) + nx*nyc*nz*((L) + nl*(M))] + Jflr(L,b_s)*phi_*zt_ + JflrB(L,b_s)*bpar_)
# define H1c_(XYZ, L, M) (g[(int) (XYZ) + nx*nyc*nz*((L) + nl*(M))] - Jflr(L,b_s)*apar_*zt_*vt_)
# define Gc_(XYZ, L, M)  g[(int) (XYZ) + nx*nyc*nz*((L) + nl*(M))]
// H = G, except for m = 0, 1, 2
__global__ void conservation_terms(cuComplex* upar_bar,
				   cuComplex* uperp_bar,
				   cuComplex* t_bar,
				   const cuComplex* g,
				   const cuComplex* phi,
				   const cuComplex* apar,
				   const cuComplex* bpar,
				   const float *kperp2,
				   const specie sp)
{
  unsigned int idxyz = get_id1();
  if (idxyz < nx*nyc*nz) {
    unsigned int idy = idxyz % nyc;
    unsigned int idx = (idxyz / nyc) % nx;
    
    if ( unmasked(idx, idy) ) {
      cuComplex phi_  = phi[idxyz];
      cuComplex apar_ = apar[idxyz];
      cuComplex bpar_ = bpar[idxyz];
      const float zt_ = sp.zt;
      const float vt_ = sp.vt;      
      
      upar_bar[idxyz]  = make_cuComplex(0., 0.);
      uperp_bar[idxyz] = make_cuComplex(0., 0.);
      t_bar[idxyz]     = make_cuComplex(0., 0.);
    
      float b_s = kperp2[idxyz] * sp.rho2;

      int m_glob;
      int m2 = 2 - m_lo; // local index corresponding to global m = 2
      int m0 = 0 - m_lo; // local index corresponding to global m = 0

      // sum over l
      for (int l=0; l < nl; l++) {
	
	// Hc_(...) is defined by macro above. Only use here for m=0. 
	m_glob = 0 + m_lo;
	if(m_glob == 0) uperp_bar[idxyz] = uperp_bar[idxyz] + (Jflr(l,b_s) + Jflr(l-1,b_s))*Hc_(idxyz, l, 0);
	
	// H1c_(...) is defined by macro above. Only use here for m=1.
	m_glob = 1 + m_lo;
	if(m_glob == 1) upar_bar[idxyz] = upar_bar[idxyz] + Jflr(l,b_s)*H1c_(idxyz, l, 1);
	
	// energy conservation correction for nlaguerre = 1
	m_glob = 2 + m_lo;
	if (nl == 1 && m_glob == 2) {
	  t_bar[idxyz] = t_bar[idxyz] + sqrtf(2.)*Jflr(l,b_s)*Gc_(idxyz, l, 2);
	} else if (nl > 1 && m0 >= -m_ghost && m2 < nm + m_ghost ) {
	  // t_bar needs to be correct on procs containing m=0 and m=2. 
	  // note: if nm = 1 (one Hermite per GPU), this will only work if m_ghost = 2, which is now enforced in grids.
	  // for m_ghost = 1 multi-gpu comms would be required for nm = 1...
	  t_bar[idxyz] = t_bar[idxyz] + sqrtf(2.)/3.*Jflr(l,b_s)*Gc_(idxyz, l, m2)
	    + 2./3.*( l*Jflr(l-1,b_s) + 2.*l*Jflr(l,b_s) + (l+1)*Jflr(l+1,b_s) )*Hc_(idxyz, l, m0);
	}
      }
      uperp_bar[idxyz] = uperp_bar[idxyz]*sqrtf(b_s);
    }
  }
}

__global__ void Wphi_summand_cetg(float* p2, const cuComplex* phi, const float* volJac)
{
  idXYZ;
  
  if (idx < nx && idy < nyc && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);
  
    if (unmasked(idx, idy)) {    

      float fac=2.;
      if (idy==0) fac = 1.0;

      cuComplex tmp = cuConjf( phi[idxyz] ) * phi[idxyz] * fac * volJac[idz];
      p2[idxyz] = 0.5 * tmp.x;

    } else {
      p2[idxyz] = 0.;
    }
  }
}

__global__ void heat_flux_summand_cetg(float* qflux,
				       const cuComplex* phi,
				       const cuComplex* g,
				       const float* ky, 
				       const float* flxJac,
				       float pres)
{
  idXYZ;
  
  if (idx < nx && idy < nyc && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    if (unmasked(idx, idy)) {    
      
      cuComplex vPhi_r = - make_cuComplex(0., ky[idy]) * phi[idxyz];
      
      cuComplex fg = (cuConjf(vPhi_r) * g[idxyz+nx*nyc*nz]) * 2. * flxJac[idz];
      qflux[idxyz] = fg.x * pres;

    } else {
      qflux[idxyz] = 0.;
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
