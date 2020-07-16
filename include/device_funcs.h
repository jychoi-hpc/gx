#pragma once
#include "cufft.h"
#include "cufftXt.h"
#include "species.h" // needed to add this
#include "cuda_constants.h"

__device__ unsigned int get_id1(void);
__device__ unsigned int get_id2(void);
__device__ unsigned int get_id3(void);

__host__ __device__ float factorial(int m);
__host__ __device__ float Jflr(int l, float b, bool enforce_JL_0=true);

__host__ __device__ float g0(float b);
__host__ __device__ float g1(float b);
__host__ __device__ float sgam0 (float b);

__host__ __device__ bool operator>(cuComplex f, cuComplex g);
__host__ __device__ bool operator<(cuComplex f, cuComplex g);


__host__ __device__ cuComplex operator+(cuComplex f, cuComplex g); 
__host__ __device__ cuComplex operator-(cuComplex f, cuComplex g);
__host__ __device__ cuComplex operator-(cuComplex f);

__host__ __device__ cuComplex operator*(float scaler, cuComplex f) ;
__host__ __device__ cuComplex operator*(cuComplex f, float scaler); 
__host__ __device__ cuComplex operator*(cuComplex f, cuComplex g);
__host__ __device__ cuDoubleComplex operator*(cuDoubleComplex f, cuDoubleComplex g);

__host__ __device__ cuComplex operator/(cuComplex f, float scaler);
__host__ __device__ cuComplex operator/(cuComplex f, cuComplex g) ;
__host__ __device__ cuDoubleComplex operator/(cuDoubleComplex f, cuDoubleComplex g) ;

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2);

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2,
				  double c3, cuComplex* m3);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4,
				  double c5, cuComplex* m5);

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, bool bdum);

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2,
				  double c3, cuComplex* m3, bool bdum);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4, bool bdum);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4,
				  double c5, cuComplex* m5, bool bdum);

__global__ void scale_kernel(cuComplex* res, cuComplex* m, double s);
__global__ void scale_kernel(cuComplex* res, cuComplex* m, cuComplex s);
__global__ void scale_singlemom_kernel(cuComplex* res, cuComplex* m, cuComplex s);

// Should we have some eqfix options in the singlemom_kernels? 

__global__ void add_scaled_singlemom_kernel(cuComplex* res,
					    double c1, cuComplex* m1,
					    double c2, cuComplex* m2);
__global__ void add_scaled_singlemom_kernel(cuComplex* res, 
					    double c1, cuComplex* m1,
					    double c2, cuComplex* m2,
					    double c3, cuComplex* m3);
__global__ void add_scaled_singlemom_kernel(cuComplex* res,
					    cuComplex c1, cuComplex* m1,
					    cuComplex c2, cuComplex* m2);

__global__ void reality_kernel(cuComplex* g);
__global__ void reality_singlemom_kernel(cuComplex* mom);

__host__ __device__ int get_ikx(int idx); // MM - 07/01/20
__host__ __device__ bool unmasked(int idx, int idy);
__host__ __device__ bool   masked(int idx, int idy);

__global__ void Tbar(cuComplex* t_bar, cuComplex* g, cuComplex* phi, float *kperp2);


// previously in linear.cu
__global__ void rhs_linear(cuComplex *g, cuComplex* phi, cuComplex* upar_bar,
                           cuComplex* uperp_bar, cuComplex* t_bar,
                           float* b, float* cv_d, float* gb_d, float* bgrad,
                           float* ky, specie* s, cuComplex* rhs_par, cuComplex* rhs);

__global__ void conservation_terms(cuComplex* upar_bar, cuComplex* uperp_bar,
                                   cuComplex* t_bar, cuComplex* G, cuComplex* phi,
                                   float *b, specie* species);

__global__ void hypercollisions(cuComplex* g, float nu_hyper_l, float nu_hyper_m,
                                int p_hyper_l, int p_hyper_m, cuComplex* rhs);
// -----------------------------------------------------------------------------


// previously in closures.cu
__global__ void beer_toroidal_closures(cuComplex* g, cuComplex* gRhs, float* omegad, cuComplex* nu);
__global__ void smith_perp_toroidal_closures(cuComplex* g, cuComplex* gRhs, float* omegad, cuComplex* Aclos, int q);
// -----------------------------------------------------------------------------


// previously in grad_parallel.cu
// needed to add declarations of the functions below since there was only a definition previously

__device__ void i_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void abs_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void i_kz_1d(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
// need to make the callback pointers extern to be able to be read from grad_xxx.cu
extern __managed__ cufftCallbackStoreC i_kz_callbackPtr;
extern __managed__ cufftCallbackStoreC i_kz_1d_callbackPtr;
extern __managed__ cufftCallbackStoreC abs_kz_callbackPtr;
// -----------------------------------------------------------------------------

// previously in grad_parallel_linked.cu
__global__ void linkedCopy(cuComplex* G, cuComplex* G_linked, int nLinks, int nChains, int* ikx, int* iky, int nMoms);
__global__ void linkedCopyBack(cuComplex* G_linked, cuComplex* G, int nLinks, int nChains, int* ikx, int* iky, int nMoms);
__device__ void i_kzLinked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void abs_kzLinked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__global__ void init_kzLinked(float* kz, int nLinks);

extern __managed__ cufftCallbackStoreC i_kzLinked_callbackPtr;
extern __managed__ cufftCallbackStoreC abs_kzLinked_callbackPtr;
// -----------------------------------------------------------------------------

// previously in grad_perp.cu
__device__ cuComplex i_kx(void *dataIn, size_t offset, void *kxData, void *sharedPtr);
__device__ cuComplex i_ky(void *dataIn, size_t offset, void *kyData, void *sharedPtr);
__device__ void mask_and_scale(void *dataOut, size_t offset, cufftComplex element, void *data, void * sharedPtr);

extern __managed__ cufftCallbackLoadC i_kx_callbackPtr;
extern __managed__ cufftCallbackLoadC i_ky_callbackPtr;
extern __managed__ cufftCallbackStoreC mask_and_scale_callbackPtr;
// -----------------------------------------------------------------------------

// previously in geometry.cu
__global__ void init_kperp2(float* kperp2, float* kx, float* ky, float* gds2, float* gds21, float* gds22, float* bmagInv, float shat) ;
__global__ void init_omegad(float* omegad, float* cv_d, float* gb_d, float* kx, float* ky, float* cv, float* gb, float* cv0, float* gb0, float shat) ;
__global__ void calc_bgrad(float* bgrad, float* bgrad_temp, float* bmag, float scale);
// -----------------------------------------------------------------------------

// previously in nonlinear.cu
__global__ void J0phiToGrid(cuComplex* J0phi, cuComplex* phi, float* b, float* muB, float rho2_s);
__global__ void bracket(float* g_res, float* dg_dx, float* dJ0phi_dy, float* dg_dy, float* dJ0Phi_dx, float kxfac);
// -----------------------------------------------------------------------------

// previously in diagnostics.cu
__global__ void growthRates(cuComplex *phi, cuComplex *phiOld, double dt, cuComplex *omega);
__global__ void volume_average(float* res, cuComplex* f, cuComplex* g, float* jacobian, float fluxDenomInv, int ikx=-1, int iky=-1);
__global__ void get_pzt (float* primary, float* secondary, float* tertiary, cuComplex* phi, cuComplex* tbar);
__global__ void heat_flux(float* qflux, cuComplex* phi, cuComplex* g, float* ky, float* jacobian, float fluxDenomInv, float *kperp2, float rho2_s, int ikx=-1, int iky=-1);
// -----------------------------------------------------------------------------

//previously in smith_par_closure.cu
__global__ void castDoubleToFloat (cuDoubleComplex *array_d, cuComplex *array_f, int size);
// -----------------------------------------------------------------------------

// previously in forcing.cu
__global__ void stirring_kernel(cuComplex force, cuComplex *moments, int forcing_index);
// -----------------------------------------------------------------------------

// previously in grids.cu
__global__ void kInit(float* kx, float* ky, float* kz, float X0, float Y0, int Zp);
// -----------------------------------------------------------------------------
