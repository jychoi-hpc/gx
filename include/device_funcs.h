#pragma once
#include "cufft.h"
#include "cufftXt.h"
#include "precision_types.h"

__device__ unsigned int get_id1(void);
__device__ unsigned int get_id2(void);
__device__ unsigned int get_id3(void);

__host__ __device__ float factorial(int m);
__device__ float Jflr(int l, float b, bool enforce_JL_0=true);
__device__ float Jfac(int l, float b);

__host__ __device__ float g0(float b);
__host__ __device__ float g1(float b);
__host__ __device__ float sgam0 (float b);

__host__ __device__ bool operator>(cuComplex f, cuComplex g);
__host__ __device__ bool operator<(cuComplex f, cuComplex g);

__host__ __device__ cuComplex operator+(cuComplex f, cuComplex g); 
__host__ __device__ cuComplex operator+(cuComplex g, float f);
__host__ __device__ cuComplex operator+(float f, cuComplex g);
__host__ __device__ cuComplex operator-(cuComplex f, cuComplex g);
__host__ __device__ cuComplex operator-(cuComplex f);

__host__ __device__ cuComplex operator*(float scaler, cuComplex f) ;
__host__ __device__ cuComplex operator*(cuComplex f, float scaler); 
__host__ __device__ cuComplex operator*(cuComplex f, cuComplex g);
__host__ __device__ cuDoubleComplex operator*(cuDoubleComplex f, cuDoubleComplex g);

__host__ __device__ cuComplex operator/(cuComplex f, float scaler);
__host__ __device__ cuComplex operator/(cuComplex f, cuComplex g) ;
__host__ __device__ cuDoubleComplex operator/(cuDoubleComplex f, cuDoubleComplex g) ;

__global__ void maskG(cuComplex* g);
__global__ void Hkernel (cuComplex* g, cuComplex* f);
__global__ void Gkernel (cuComplex* h, cuComplex* f);

void setdev_constants(int Nx, int Ny, int Nyc, int Nz, int Nspecies, int Nm, int Nl, int Nj, int Zp, int ikxf, int ikyf);

__global__ void acc_scaled_kernel(cuComplex* res,
				  double c1, const cuComplex* m1);

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2);

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2,
				  double c3, const cuComplex* m3);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2, 
				  double c3, const cuComplex* m3,
				  double c4, const cuComplex* m4);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2, 
				  double c3, const cuComplex* m3,
				  double c4, const cuComplex* m4,
				  double c5, const cuComplex* m5);

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2, bool bdum);

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2,
				  double c3, const cuComplex* m3, bool bdum);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2, 
				  double c3, const cuComplex* m3,
				  double c4, const cuComplex* m4, bool bdum);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2, 
				  double c3, const cuComplex* m3,
				  double c4, const cuComplex* m4,
				  double c5, const cuComplex* m5, bool bdum);

__global__ void promote(double* dG, float* G, int N);
__global__ void promote(double* dG, float* G, float* noise, int N);
__global__ void demote(float* G, double* dG, int N);
__global__ void getr2(double* r2, double* r, int N);
__global__ void copyV(double* P, double* V, int N);
__global__ void getV(double* V, double* G, double* R2, int M, int N);
__global__ void getW(double* W, double* r2, int N);
__global__ void setI(double* Id, int N);
__global__ void getB(double* W, double beta_, int N);
__global__ void update_Fake_G (float* g, int i);
__global__ void init_Fake_G (float* g);
__global__ void setval(float* f, float val, int N);
__global__ void setval(double* f, double val, int N);
__global__ void setval(cuComplex* f, cuComplex val, int N);
__global__ void WinG(double* res, double* Win, double* G, int Q, int M);
__global__ void update_state(double* res, double* A, double* x, int K, int N);
__global__ void myPrep(double* x, double* r, int* col, int NK);
__global__ void mySpMV(double* x2, double* xy, double* y2,
		       double* y, double* x, double* A, double* r, int K, int N);
__global__ void est_eval(double e, double* fLf, double * f2);
__global__ void setA(double* A, double fac, int N);
__global__ void inv_scale_kernel(double* res, const double* f, const double* scalar, int N);
__global__ void eig_residual(double* y, double* A, double* x, double* R,
			     double* r2, double eval, int K, int N);
  
__global__ void scale_kernel(cuComplex* res, double s);
__global__ void scale_kernel(cuComplex* res, cuComplex s);
__global__ void scale_singlemom_kernel(cuComplex* res, cuComplex* m, cuComplex s);
__global__ void scale_singlemom_kernel(cuComplex* res, cuComplex* m, float scalar);
// Should we have some eqfix options in the singlemom_kernels? 

__global__ void add_section(cuComplex *res, const cuComplex *tmp, int ntot);

__global__ void add_scaled_singlemom_kernel(cuComplex* res,
					    double c1, const cuComplex* m1,
					    double c2, const cuComplex* m2);
__global__ void add_scaled_singlemom_kernel(cuComplex* res, 
					    double c1, const cuComplex* m1,
					    double c2, const cuComplex* m2,
					    double c3, const cuComplex* m3);
__global__ void add_scaled_singlemom_kernel(cuComplex* res,
					    cuComplex c1, cuComplex* m1,
					    cuComplex c2, cuComplex* m2);

__global__ void reality_kernel(cuComplex* g, int N);

__device__ int get_ikx(int idx);
__device__ bool unmasked(int idx, int idy);
__device__ bool   masked(int idx, int idy);

__global__ void Tbar(cuComplex* t_bar, const cuComplex* g, const cuComplex* phi, const float *kperp2);
		     
__global__ void growthRates(const cuComplex *phi, const cuComplex *phiOld, double dt, cuComplex *omega);

__global__ void J0phiToGrid(cuComplex* J0phi, const cuComplex* phi, const float* b,
			    const float* muB, float rho2_s);

__global__ void acc(float *a, const float *b);

__global__ void bracket(float* g_res,
			const float* dg_dx, const float* dJ0phi_dy,
			const float* dg_dy, const float* dJ0Phi_dx, float kxfac);

__global__ void  d2x (cuComplex *res, cuComplex *f, float *kx);
__global__ void  ddx (cuComplex *res, cuComplex *f, float *kx);
__global__ void  ddy (cuComplex *res, cuComplex *f, float *ky);
__global__ void mddy (cuComplex *res, cuComplex *f, float *ky);

__global__ void castDoubleToFloat (const cuDoubleComplex *array_d, cuComplex *array_f, int size);

__global__ void beer_toroidal_closures(const cuComplex* g, cuComplex* gRhs,
				       const float* omegad, const cuComplex* nu, const float* tz);

__global__ void smith_perp_toroidal_closures(const cuComplex* g, cuComplex* gRhs,
					     const float* omegad, const cuComplex* Aclos, int q, const float* tz);

__global__ void stirring_kernel(const cuComplex force, cuComplex *moments, int forcing_index);

__global__ void xytranspose(float *in, float *out);
__global__ void yzavg(float *vE, float *vEavg, float *vol_fac);
__global__ void fieldlineaverage(cuComplex *favg, cuComplex *df, const cuComplex *f, const float *volJac);

__global__ void W_summand(float *G2, const cuComplex* g, const float* volJac, const float* nt);

__global__ void vol_summand(float *rmom, const cuComplex* f, const cuComplex* g, const float* volJac);

__global__ void get_pzt (float* primary, float* secondary, float* tertiary, const cuComplex* phi, const cuComplex* tbar);
__global__ void rescale_kernel(cuComplex* f, float* phi_max, int N);
__global__ void maxPhi(float* phi_max, const cuComplex *phi);

__global__ void Wphi_scale(float* p2, float alpha);

__global__ void Wphi2_summand(float *p2, const cuComplex *phi, const float *volJac);
  
__global__ void Wphi_summand(float* p2, const cuComplex* phi, const float* volJac, const float* kperp2, float rho2_s);
__global__ void Wphi_summand_krehm(float* p2, const cuComplex* phi, const float* volJac, const float* kx, const float* ky, float rho_i);
  
__global__ void heat_flux_summand(float* qflux, const cuComplex* phi, const cuComplex* g, const float* ky, 
				  const float* flxJac, const float *kperp2, float rho2_s, float p_s);

__global__ void part_flux_summand(float* pflux, const cuComplex* phi, const cuComplex* g, const float* ky, 
				  const float* flxJac, const float *kperp2, float rho2_s, float n_s);

__global__ void init_kperp2(float* kperp2, const float* kx, const float* ky,
			    const float* gds2, const float* gds21, const float* gds22,
			    const float* bmagInv, float shat);

__global__ void init_omegad(float* omegad, float* cv_d, float* gb_d, const float* kx, const float* ky,
			    const float* cv, const float* gb, const float* cv0, const float* gb0, float shat);

__global__ void calc_bgrad(float* bgrad, const float* bgrad_temp, const float* bmag, float scale);
__global__ void init_kxs(float* kxs, float* kx, float* th0);
__global__ void update_kxs(float* kxs, float* dth0);
__global__ void update_theta0(float* th0, double dt);
__global__ void update_geo(float* kxs, float* ky, float* cv_d, float* gb_d, float* kperp2,
			   float* cv, float* cv0, float* gb, float* gb0, float* omegad,  			      
			   float* gds2, float* gds21, float* gds22, float* bmagInv, float shat);
  
__device__ cuComplex i_kxs(void *dataIn, size_t offset, void *kxsData, void *sharedPtr);

__device__ cuComplex i_kx(void *dataIn, size_t offset, void *kxData, void *sharedPtr);
__device__ cuComplex i_ky(void *dataIn, size_t offset, void *kyData, void *sharedPtr);
__device__ void mask_and_scale(void *dataOut, size_t offset, cufftComplex element, void *data, void * sharedPtr);

extern __managed__ cufftCallbackLoadC i_kxs_callbackPtr;
extern __managed__ cufftCallbackLoadC i_kx_callbackPtr;
extern __managed__ cufftCallbackLoadC i_ky_callbackPtr;
extern __managed__ cufftCallbackStoreC mask_and_scale_callbackPtr;
  
__device__ void zfts(void *dataOut, size_t offset, cufftComplex element, void *data, void *sharedPtr);
__device__ void i_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void abs_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void i_kz_1d(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);

extern __managed__ cufftCallbackStoreC zfts_callbackPtr;
extern __managed__ cufftCallbackStoreC i_kz_callbackPtr;
extern __managed__ cufftCallbackStoreC i_kz_1d_callbackPtr;
extern __managed__ cufftCallbackStoreC abs_kz_callbackPtr;

__global__ void kInit(float* kx, float* ky, float* kz, int* kzm, float* kzp, const float X0, const float Y0, const int Zp, bool dealias_kz);  
__global__ void qneut(cuComplex* Phi, const cuComplex* g, const float* kperp2, const float* rho2s,
		      const float* qn, const float* nzs);

__global__ void ampere(cuComplex* Apar, const cuComplex* gu, const float* kperp2, const float* rho2s,
		       const float* as, const float* amps, const float beta);

__global__ void rhs_linear_krehm(const cuComplex* g, const cuComplex* phi, const cuComplex* apar, 
			  const float nu_ei, const float rhos, const float de, cuComplex* rhs_par);
__global__ void phiSolve_krehm (cuComplex *phi, cuComplex *G0, float* kx, float* ky, float rho_i);
__global__ void aparSolve_krehm (cuComplex *apar, cuComplex *G1, float* kx, float* ky, float rho_s, float d_e);

__global__ void real_space_density(cuComplex* nbar, const cuComplex* g, const float *kperp2,
				   const float *rho2s, const float *nzs);

//__global__ void qneut_fieldlineaveraged(cuComplex *Phi, const cuComplex *nbar, const float *PhiAvgDenom, 
//					const float *kperp2, const float *jacobian,
//					const specie *species, const float ti_ov_te, float *work);

__global__ void qneutAdiab_part1(cuComplex* PhiAvgNum_tmp, const cuComplex* nbar,
				 const float* kperp2, const float* jacobian,
				 const float* rho2s, const float* qns, float tau_fac);

__global__ void qneutAdiab_part2(cuComplex* Phi, const cuComplex* PhiAvgNum_tmp, const cuComplex* nbar,
				 const float* PhiAvgDenom, const float* kperp2, 
				 const float* rho2s, const float* qns, float tau_fac);

__global__ void calc_phiavgdenom(float* PhiAvgDenom, const float* kperp2, const float* jacobian,
				 const float* rho2s, const float* qns, float tau_fac);

__global__ void add_source(cuComplex* f, const float source);

__global__ void qneutAdiab(cuComplex* Phi, const cuComplex* nbar,
			   const float* kperp2, const float* rho2s, const float* qns, float tau_fac);

__global__ void linkedCopy(const cuComplex* G, cuComplex* G_linked, int nLinks, int nChains,
			   const int* ikx, const int* iky, int nMoms);

__global__ void linkedCopyBack(const cuComplex* G_linked, cuComplex* G, int nLinks, int nChains,
			       const int* ikx, const int* iky, int nMoms);

__device__ void zfts_Linked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void i_kzLinked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void abs_kzLinked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__global__ void init_kzLinked(float* kz, int nLinks);

extern __managed__ cufftCallbackStoreC zfts_Linked_callbackPtr;
extern __managed__ cufftCallbackStoreC i_kzLinked_callbackPtr;
extern __managed__ cufftCallbackStoreC abs_kzLinked_callbackPtr;

__global__ void getPhi (cuComplex *phi, cuComplex *G, float* ky);
__global__ void rhs_lin_vp(const cuComplex *G, const cuComplex* phi, cuComplex* GRhs, float* ky,
			   bool closure, float nu, float nuh, int alpha, int alpha_h);
__global__ void kz_dealias (cuComplex *G, int *kzm, int LMS);
__global__ void nlvp(float *res, const float *Gy, const float *dphi);
__global__ void nlks(float *res, const float *Gy, const float *dG);
__global__ void nlks1(float *res, const float *Gy);
__global__ void nlks2(cuComplex *res, const float *ky);
__global__ void rhs_ks (const cuComplex *G, cuComplex *GRhs, float *ky, float eps_ks);
__global__ void streaming_rhs (const cuComplex* g, const cuComplex* phi, const cuComplex* apar, const float* kperp2, const float* rho2s, 
			       const float gradpar, const float* vt, const float* zt, cuComplex* rhs_par);

__global__ void rhs_linear(const cuComplex *g, const cuComplex* phi, const cuComplex* apar,
			   const cuComplex* upar_bar, const cuComplex* uperp_bar, const cuComplex* t_bar,
			   const float* b, const float* cv_d, const float* gb_d, const float* bgrad,
			   const float* ky, const float* vt, const float* zt, const float* tz, const float* nz, const float* as,
			   const float* nu_ss, const float* tprim, const float* uprim, const float* fprim, 
			   const float* rho2s, const int* typs, cuComplex* rhs, bool hegna);  // bb6126 - hegna test

__global__ void get_s1 (float* s10, float* s11, const float* kx, const float* ky, const cuComplex* df, float w_osc);
__global__ void get_s01 (float* s01, const cuComplex* favg, const float* kx, const float w_osc);
__global__ void HB_hyper (const cuComplex* G, const float* s01, const float* s10, const float* s11,
			  const float* kx, const float* ky, const float D_HB, const int p_HB, cuComplex* RHS);

__global__ void conservation_terms(cuComplex* upar_bar, cuComplex* uperp_bar,
				   cuComplex* t_bar, const cuComplex* G, const cuComplex* phi, const cuComplex* apar,
				   const float *b, const float *zt, const float *rho2s, const float *vt);

__global__ void hyperdiff(const cuComplex* g, const float* kx, const float* ky,
			  float nu_hyper, float D_hyper, cuComplex* rhs);

__global__ void hypercollisions(const cuComplex* g, const float nu_hyper_l, const float nu_hyper_m,
				const int p_hyper_l, const int p_hyper_m, const float* vt, cuComplex* rhs);



