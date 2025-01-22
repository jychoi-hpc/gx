#pragma once
#include "cufft.h"
#include "cufftXt.h"
#include "precision_types.h"
#include "species.h"

#define idXYZ unsigned int idy = get_id1(); unsigned int idx = get_id2(); unsigned int idz = get_id3();


__device__ unsigned int get_id1(void);
__device__ unsigned int get_id2(void);
__device__ unsigned int get_id3(void);
__device__ unsigned int get_idxyz(unsigned int idx, unsigned int idy, unsigned int idz);

__host__ __device__ float factorial(int m);
__device__ float Jflr(int l, float b, bool enforce_JL_0=true);
__device__ float JflrB(int l, float b, bool enforce_JL_0=true);
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

void setdev_constants(int Nx, int Ny, int Nyc, int Nz, int Nspecies, int Nm, int Nl, int Nj, int Zp, int ikxf, int ikyf,
		      int is_lo_in, int is_up_in, int m_lo_in, int m_up_in, int m_ghost_in, int Nm_glob);

__global__ void acc_scaled_kernel(cuComplex* res,
				  double c1, const cuComplex* m1);

__global__ void abs(float *f, int N);

__global__ void add_scaled_singlemom_kernel(float* res,
					    double c1, const float* m1,
					    double c2, const float* m2);

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
__device__ int get_idx(int ikx);
__device__ bool unmasked(int idx, int idy);
__device__ bool   masked(int idx, int idy);

__global__ void Tbar(cuComplex* t_bar, const cuComplex* g, const cuComplex* phi, const float *kperp2);
		     
__global__ void growthRates(const cuComplex *phi, const cuComplex *phiOld, double dt, cuComplex *omega);

__global__ void J0fToGrid(cuComplex* J0f, const cuComplex* f, const float* kperp2,
			    const float* muB, const float rho2_s, const float fac);

__global__ void J0phiAndBparToGrid(cuComplex* J0phiB, const cuComplex* phi, const cuComplex* bpar, const float* kperp2,
			    const float* muB, const float rho2_s, const float tz, const float fphi, const float fbpar);

__global__ void iKxJ0ftoGrid(cuComplex* iKxf, const cuComplex* f, const cuComplex* iKx, bool exb_flag);
__global__ void iKxgtoGrid(cuComplex* iKxg, const cuComplex* g, const cuComplex* iKx, bool exb_flag);
__global__ void iKxgsingletoGrid(cuComplex* iKxg_single, const cuComplex* g_single, const cuComplex* iKx, bool exb_flag);
__global__ void iKxphitoGrid(cuComplex* iKxphi, const cuComplex* phi, const cuComplex* iKx, bool exb_flag);
__global__ void acc(float *a, const float *b);

__global__ void bracket(float* __restrict__ g_res,
			const float* __restrict__ dg_dx, const float* __restrict__ dJ0phi_dy,
			const float* __restrict__ dg_dy, const float* __restrict__ dJ0Phi_dx, float kxfac);

__global__ void bracket_cetg(float* __restrict__ g_res,
			const float* __restrict__ dg_dx, const float* __restrict__ dphi_dy,
			const float* __restrict__ dg_dy, const float* __restrict__ dPhi_dx, float kxfac);

__global__ void  d2x (cuComplex *res, cuComplex *f, float *kx);
__global__ void  ddx (cuComplex *res, cuComplex *f, float *kx);
__global__ void  ddy (cuComplex *res, cuComplex *f, float *ky);
__global__ void mddy (cuComplex *res, cuComplex *f, float *ky);

__global__ void castDoubleToFloat (const cuDoubleComplex *array_d, cuComplex *array_f, int size);

__global__ void beer_toroidal_closures(const cuComplex* g, cuComplex* gRhs,
				       const float* omegad, const cuComplex* nu, const float tz);

__global__ void smith_perp_toroidal_closures(const cuComplex* g, cuComplex* gRhs,
					     const float* omegad, const cuComplex* Aclos, int q, const float tz);

__global__ void stirring_kernel(const cuComplex force, cuComplex *moments, int forcing_index);
__global__ void kz_stirring_kernel(const cuComplex force, cuComplex *moments, int kx, int ky, int kz);

__global__ void xytranspose(float *in, float *out);
__global__ void yzavg(float *vE, float *vEavg, float *vol_fac);
__global__ void fieldlineaverage(cuComplex *favg, cuComplex *df, const cuComplex *f, const float *volJac);

__global__ void Wg_summand(float *G2, const cuComplex* g, const float* volJac, const float nt);

__global__ void vol_summand(float *rmom, const cuComplex* f, const cuComplex* g, const float* volJac);

__global__ void rescale_kernel(cuComplex* f, float* phi_max, int N);
__global__ void maxPhi(float* phi_max, const cuComplex *phi);

__global__ void Wphi_scale(float* p2, float alpha);

__global__ void Phi2_summand(float *p2, const cuComplex *phi, const float *volJac);
__global__ void Phi2_zonal_summand(float *p2, const cuComplex *phi, const float *volJac);
  
__global__ void Wphi_summand(float* p2, const cuComplex* phi, const float* volJac, const float* kperp2, float rho2_s);
__global__ void Wphi_summand_krehm(float* p2, const cuComplex* phi, const float* volJac, const float* kx, const float* ky, float rho_i);
__global__ void Wapar_summand(float* p2, const cuComplex* apar, const float* volJac, const float* kperp2, const float* bmag);
__global__ void Wapar_summand_krehm(float* p2, const cuComplex* apar, const cuComplex* apar_ext, const float* volJac, const float* kx, const float* ky, float rho_i);
  
__global__ void heat_flux_summand(float* qflux, const cuComplex* phi, const cuComplex* apar, const cuComplex* bpar, const cuComplex* g, const float* ky, 
				  const float* flxJac, const float *kperp2, float rho2_s, float p_s, float vts, float tzs);
__global__ void Wphi_summand_cetg(float* p2, const cuComplex* phi, const float* volJac);

__global__ void heat_flux_summand_cetg(float* qflux, const cuComplex* phi, const cuComplex* g, const float* ky, 
				       const float* flxJac, float p_s);
__global__ void heat_flux_ES_summand(float* qflux, const cuComplex* phi, const cuComplex* g, const float* ky, 
				  const float* flxJac, const float *kperp2, float rho2_s, float p_s, float vts);
__global__ void heat_flux_Apar_summand(float* qflux, const cuComplex* apar, const cuComplex* g, const float* ky, 
				  const float* flxJac, const float *kperp2, float rho2_s, float pres, float vts);
__global__ void heat_flux_Bpar_summand(float* qflux, const cuComplex* bpar, const cuComplex* g, const float* ky, 
				  const float* flxJac, const float *kperp2, float rho2_s, float pres, float tzs);

__global__ void particle_flux_summand(float* pflux, const cuComplex* phi, const cuComplex* apar, const cuComplex* bpar, const cuComplex* g, const float* ky, 

				  const float* flxJac, const float *kperp2, float rho2_s, float n_s, float vts, float tzs);

__global__ void particle_flux_ES_summand(float* pflux, const cuComplex* phi, const cuComplex* g, const float* ky, 
				  const float* flxJac, const float *kperp2, float rho2_s, float n_s, float vts, float tzs);
__global__ void particle_flux_Apar_summand(float* pflux, const cuComplex* apar, const cuComplex* g, const float* ky, 
				  const float* flxJac, const float *kperp2, float rho2_s, float n_s, float vts, float tzs);
__global__ void particle_flux_Bpar_summand(float* pflux, const cuComplex* bpar, const cuComplex* g, const float* ky, 
				  const float* flxJac, const float *kperp2, float rho2_s, float n_s, float vts, float tzs);

__global__ void init_ftwist(float* ftwist, const float* gds21, const float* gds22, float shat);
__global__ void init_m0(int* m0, const float x0, const float* ky, const float* ftwist, float shat, const float kxfac);
__global__ void init_deltaKx(float* deltaKx, const int* m0, const float x0, const float* ky, const float* ftwist);

__global__ void init_kperp2_ntft(float* kperp2, const float* kx, const float* ky, 
				 const float* gds2, const float* gds21, const float* gds22, 
				 const float* ftwist, const float* bmagInv, float shat, const float* deltaKx);
__global__ void init_omegad_ntft(float* omegad, float* cv_d, float* gb_d, const float* kx, const float* ky,
			         const float* cv, const float* gb, const float* cv0, const float* gb0,
				 float shat, const int* m0, const float x0);
__global__ void init_iKx(cuComplex* iKx, const float* kx, const float* deltaKx);
__global__ void init_phasefac_ntft(cuComplex* phasefac, const float* x, const float* deltaKx, bool sign);
__global__ void calc_n_bar(cuComplex* n_bar, cuComplex* g, const cuComplex* phi, const cuComplex* bpar, const float *kperp2, const specie sp);
__global__ void calc_upar_bar(cuComplex* upar_bar, cuComplex* g, const cuComplex* apar, const float *kperp2, const specie sp);
__global__ void calc_uperp_bar(cuComplex* uperp_bar, cuComplex* g, const cuComplex* phi, const cuComplex* bpar, const float *kperp2, const specie sp);
__global__ void calc_T_bar(cuComplex* T_bar, cuComplex* g, const cuComplex* phi, const cuComplex* bpar, const float *kperp2, const specie sp);

__global__ void turbulent_heating_summand(float* heat, const cuComplex* phi, const cuComplex* apar, const cuComplex* bpar, 
                                          const cuComplex* phi_old, const cuComplex* apar_old, const cuComplex* bpar_old, 
                                          const cuComplex* g, const cuComplex* g_old, const float* volJac, const float* kperp2, const specie sp, float dt);

__global__ void init_kperp2(float* kperp2, const float* kx, const float* ky,
			    const float* gds2, const float* gds21, const float* gds22,
			    const float* bmagInv, const float shat);

__global__ void init_omegad(float* omegad, float* cv_d, float* gb_d, const float* kx, const float* ky,
			    const float* cv, const float* gb, const float* cv0, const float* gb0, const float shat);

__global__ void calc_bgrad(float* bgrad, const float* bgrad_temp, const float* bmag, float scale);
  
__device__ cuComplex i_kxstar(void *dataIn, size_t offset, void *kxstarData, void *sharedPtr);

__device__ cuComplex i_kx(void *dataIn, size_t offset, void *kxData, void *sharedPtr);
__device__ cuComplex i_ky(void *dataIn, size_t offset, void *kyData, void *sharedPtr);
__device__ void mask_and_scale(void *dataOut, size_t offset, cufftComplex element, void *data, void * sharedPtr);

extern __device__ cufftCallbackLoadC i_kxstar_callbackPtr;
extern __device__ cufftCallbackLoadC i_kx_callbackPtr;
extern __device__ cufftCallbackLoadC i_ky_callbackPtr;
//extern __device__ cufftCallbackLoadC phasefac_exb_callbackPtr;
//extern __device__ cufftCallbackLoadC phasefacminus_exb_callbackPtr;
extern __device__ cufftCallbackStoreC mask_and_scale_callbackPtr;
extern __device__ cufftCallbackStoreC scale_ky_callbackPtr;
  
__device__ void    zfts(void *dataOut, size_t offset, cufftComplex element, void *data,   void *sharedPtr);
__device__ void    i_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void    mkz2(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void  abs_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void i_kz_1d(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);

extern __device__ cufftCallbackStoreC zfts_callbackPtr;
extern __device__ cufftCallbackStoreC i_kz_callbackPtr;
extern __device__ cufftCallbackStoreC mkz2_callbackPtr;
extern __device__ cufftCallbackStoreC i_kz_1d_callbackPtr;
extern __device__ cufftCallbackStoreC mkz2_1d_callbackPtr;
extern __device__ cufftCallbackStoreC abs_kz_callbackPtr;

__global__ void kInit(float* kx, float* ky, float* kz, int* kzm, float* kzp, const float X0, const float Y0, const int Zp, bool dealias_kz);  

__global__ void rhs_linear_krehm(const cuComplex* g, const cuComplex* phi, const cuComplex* apar, const cuComplex* apar_ext,
			  const float nu_ei, const float rhos, const float de, const float gradpar, cuComplex* rhs);
__global__ void krehm_collisions(const cuComplex* g, const cuComplex* apar, const cuComplex* apar_ext, const float* kx, const float* ky,
			  const float nu_ei, const float rhos, const float de, cuComplex* rhs);
__global__ void phiSolve_krehm (cuComplex *phi, cuComplex *G0, float* kx, float* ky, float rho_i);
__global__ void aparSolve_krehm (cuComplex *apar, cuComplex *G1, float* kx, float* ky, float rho_s, float d_e);
__global__ void equilibrium_current_krehm (cuComplex *G1, float* kx, float* ky, float rho_s, float d_e, cuComplex* apar_ext);

__global__ void phiSolve_cetg (cuComplex *phi, cuComplex *G0, float tau_bar);
__global__ void rhs_diff_cetg(const cuComplex* density, const cuComplex* temperature, const cuComplex* phi,
			      const float gpar, const float c1, const float C21, const float C23, cuComplex* rhs_diff);
__global__ void rhs_lin_cetg(const cuComplex* phi, const float* ky, cuComplex* rhs);
__global__ void hyper_cetg(const cuComplex* g, const float* kx, const float* ky,
			   const float nu_hyper, const float D_hyper, cuComplex* rhs);

__global__ void real_space_density(cuComplex* nbar, const cuComplex* g, const float *kperp2, const specie sp);
__global__ void real_space_par_current(cuComplex* jbar, const cuComplex* g, const float *kperp2, const specie sp);
__global__ void real_space_perp_current(cuComplex* jbar, const cuComplex* g, const float *kperp2, const float *bmagInv, const specie sp);

__global__ void sum_solverFacs(float* qneutFacPhi, float* qneutFacBpar, float* ampereParFac, float* amperePerpFacPhi, float* amperePerpFacBpar,
                               const float* kperp2, const float* bmag, const float* bmagInv, const specie sp,
			       const float beta, const bool first, const float fapar, const float fbpar, bool long_wavelength_GK);

__global__ void qneut(cuComplex* Phi, const cuComplex* nbar, const float* denom, float fphi);
__global__ void ampere_apar(cuComplex* apar, cuComplex* jbar, float* denom, float fapar);
__global__ void qneut_and_ampere_perp(cuComplex* Phi, cuComplex* Bpar, const cuComplex* SQ, const cuComplex* SA, 
		      const float* QPhi, const float* QB, const float* APhi, const float* AB, const float fphi, const float fbpar);

//__global__ void qneut_fieldlineaveraged(cuComplex *Phi, const cuComplex *nbar, const float *PhiAvgDenom, 
//					const float *kperp2, const float *jacobian,
//					const specie *species, const float ti_ov_te, float *work);

__global__ void qneutAdiab_part1(cuComplex* PhiAvgNum_tmp, const cuComplex* nbar,
				 const float* jacobian, const float* qneutDenom, float tau_fac);

__global__ void qneutAdiab_part2(cuComplex* Phi, const cuComplex* PhiAvgNum_tmp, const cuComplex* nbar,
				 const float* PhiAvgDenom, const float* qneutDenom, float tau_fac, float fphi);

__global__ void calc_phiavgdenom(float* PhiAvgDenom, const float* jacobian,
				 const float* qneutDenom, float tau_fac);

__global__ void add_source(cuComplex* f, const float source);

__global__ void qneutAdiab(cuComplex* Phi, const cuComplex* nbar, const float* qneutDenom, float tau_fac, float fphi);

__global__ void dampEnds_linked(cuComplex* G, cuComplex* phi, cuComplex* apar, cuComplex* bpar, float* kperp2, specie sp,
			       int nLinks, int nChains, const int* ikx, const int* iky, int nMoms,
			       cuComplex* GRhs, float widthfrac, float amp);

__global__ void dampEnds_linkedNTFT(cuComplex* G, cuComplex* phi, cuComplex* apar, cuComplex* bpar, float* kperp2, specie sp,
			       int nLinks, int nChains, const int* ikx, const int* iky, int nMoms,
			       cuComplex* GRhs, float widthfrac, float amp);

__global__ void zeroEnds_linked(cuComplex* G, cuComplex* phi, cuComplex* apar, float* kperp2, specie sp,
			       int nLinks, int nChains, const int* ikx, const int* iky, int nMoms);

__global__ void linkedFilterEnds(cuComplex* G, int ifilter,
			       int nLinks, int nChains, const int* ikx, const int* iky, int nMoms);

__global__ void linkedCopy(const cuComplex* __restrict__ G, cuComplex* __restrict__ G_linked, int nLinks, int nChains,
			   const int* __restrict__ ikx, const int* __restrict__ iky, int nMoms);

__global__ void linkedCopyBack(const cuComplex* __restrict__ G_linked, cuComplex* __restrict__ G, int nLinks, int nChains,
			       const int* __restrict__ ikx, const int* __restrict__ iky, int nMoms);

__global__ void linkedAccumulateBack(const cuComplex* __restrict__ G_linked, cuComplex* __restrict__ G, int nLinks, int nChains,
			       const int* __restrict__ ikx, const int* __restrict__ iky, int nMoms, float scale);

__global__ void linkedCopyNTFT(const cuComplex* __restrict__ G, cuComplex* __restrict__ G_linked, int nLinks, int nChains,
			   const int* __restrict__ ikx, const int* __restrict__ iky, int nMoms);

__global__ void linkedCopyBackNTFT(const cuComplex* __restrict__ G_linked, cuComplex* __restrict__ G, int nLinks, int nChains,
			       const int* __restrict__ ikx, const int* __restrict__ iky, int nMoms);

__global__ void linkedAccumulateBackNTFT(const cuComplex* __restrict__ G_linked, cuComplex* __restrict__ G, int nLinks, int nChains,
			       const int* __restrict__ ikx, const int* __restrict__ iky, int nMoms, float scale);

__device__ void   zfts_Linked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void zfts_LinkedNTFT(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void    i_kzLinked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void i_kzLinkedNTFT(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void   mkz2_Linked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void   mkz2_LinkedNTFT(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void  abs_kzLinked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void  abs_kzLinkedNTFT(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__global__ void init_kzLinked(float* kz, int nLinks, bool dealias);
__global__ void init_kzLinkedNTFT(float* kz, int nLinks, bool dealias);

extern __device__ cufftCallbackStoreC zfts_Linked_callbackPtr;
extern __device__ cufftCallbackStoreC zfts_LinkedNTFT_callbackPtr;
extern __device__ cufftCallbackStoreC i_kzLinked_callbackPtr;
extern __device__ cufftCallbackStoreC i_kzLinkedNTFT_callbackPtr;
extern __device__ cufftCallbackStoreC hyperkzLinked_callbackPtr;
extern __device__ cufftCallbackStoreC hyperkzLinkedNTFT_callbackPtr;
extern __device__ cufftCallbackStoreC mkz2_Linked_callbackPtr;
extern __device__ cufftCallbackStoreC mkz2_LinkedNTFT_callbackPtr;
extern __device__ cufftCallbackStoreC abs_kzLinked_callbackPtr;
extern __device__ cufftCallbackStoreC abs_kzLinkedNTFT_callbackPtr;

__global__ void getPhi (cuComplex *phi, cuComplex *G, float* ky);
__global__ void rhs_lin_vp(const cuComplex *G, const cuComplex* phi, cuComplex* GRhs, float* ky,
			   bool closure, float nu, float nuh, int alpha, int alpha_h);
__global__ void kz_dealias (cuComplex *G, int *kzm, int LMS);
__global__ void nlvp(float *res, const float *Gy, const float *dphi);
__global__ void nlks(float *res, const float *Gy, const float *dG);
__global__ void nlks1(float *res, const float *Gy);
__global__ void nlks2(cuComplex *res, const float *ky);
__global__ void rhs_ks (const cuComplex *G, cuComplex *GRhs, float *ky, float eps_ks);
__global__ void streaming_rhs(const cuComplex* __restrict__ g,
			      const cuComplex* __restrict__ phi,
			      const cuComplex* __restrict__ apar,
			      const cuComplex* __restrict__ bpar,
			      const float* __restrict__ kperp2, 
			      const float gradpar,
			      const specie sp,
			      cuComplex* __restrict__ rhs_par);

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
			   bool ei_colls,
                           float rhoc,
                           float g_exb,
                           float RBzeta,
                           float qsf); 

__global__ void get_s1 (float* s10, float* s11, const float* kx, const float* ky, const cuComplex* df, float w_osc);
__global__ void get_s01 (float* s01, const cuComplex* favg, const float* kx, const float w_osc);
__global__ void HB_hyper (const cuComplex* G, const float* s01, const float* s10, const float* s11,
			  const float* kx, const float* ky, const float D_HB, const int p_HB, cuComplex* RHS);

__global__ void conservation_terms(cuComplex* upar_bar,
				   cuComplex* uperp_bar,
				   cuComplex* t_bar,
				   const cuComplex* g,
				   const cuComplex* phi,
				   const cuComplex* apar,
				   const cuComplex* bpar,
				   const float *kperp2,
				   const specie sp);

__global__ void hyperdiff(const cuComplex* g, const float* kx, const float* ky,
			  float nu_hyper, float D_hyper, cuComplex* rhs);

__global__ void hypercollisions(const cuComplex* g, const float nu_hyper_l, const float nu_hyper_m, const float nu_hyper_lm,
				const int p_hyper_l, const int p_hyper_m, const int p_hyper_lm, cuComplex* rhs, const float vt);
__global__ void hypercollisions_kz(const cuComplex* g, const float nu, const int p, cuComplex* res);

__global__ void init_kxstar_kxbar_phasefac(double* kxstar, int* kxbar_ikx_new, int* kxbar_ikx_old, cuComplex* phasefac, cuComplex* phasefac_minus, const float* kx);
__global__ void geo_shift(const double* kxstar, const float* ky, float* cv_d, float* gb_d, float* kperp2,
                           const float* cv, const float* cv0, const float* gb, const float* gb0, float* omegad,
                           const float* gds2, const float* gds21, const float* gds22, const float* bmagInv, const float shat);
__global__ void geo_shift_ntft(const double* kxstar, const float* ky, float* cv_d, float* gb_d, float* kperp2,
                               const float* cv, const float* cv0, const float* gb, const float* gb0, float* omegad,
                               const float* gds2, const float* gds21, const float* gds22, const float* bmagInv, const float shat,
			       const float* ftwist, float* deltaKx, const int* m0, const float x0);
__global__ void iKx_shift_ntft(cuComplex* iKx, const float g_exb, const double dt, const float* ky);

__global__ void kxstar_phase_shift(double* kxstar, int* kxbar_ikx_new, int* kxbar_ikx_old, const float* ky, const float* x, cuComplex* phasefac, cuComplex* phasefac_minus, const float g_exb, const double dt, const float x0, const bool ExBshear_phase);
__global__ void field_shift(cuComplex* field_new, const cuComplex* field_old, const int* kxbar_ikx_new, const int* kxbar_ikx_old, const float g_exb);
__global__ void g_shift(cuComplex* g_new, const cuComplex* g_old, const int* kxbar_ikx_new, const int* kxbar_ikx_old, const float g_exb);
