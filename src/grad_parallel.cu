#include "grad_parallel.h"
#define GGPF <<< dGf, dBf >>> 
#define GGP2 <<< dGd, dBd >>> 
#define GGP <<< dG, dB >>>

GradParallelPeriodic::GradParallelPeriodic(Grids* grids) :
  grids_(grids)
{
  // (ky, kx, theta) <-> (ky, kx, kpar)
  cufftCreate(&zft_plan_forward);
  cufftCreate(&zft_plan_inverse);
  cufftCreate(&dz_plan_forward);
  cufftCreate(&dz_plan_inverse);
  cufftCreate(&dz2_plan_forward);
  cufftCreate(&abs_dz_plan_forward);

  int n = grids_->Nz; 			// size of FFT
  int isize = grids_->NxNycNz;		// size of input data
  int osize = grids_->NxNycNz;		// size of output data
  int dim = 1;				// 1 dimensional
  int istride = grids_->NxNyc;		// distance between two elements in a batch 
					// = distance between (ky,kx,z=1) and (ky,kx,z=2) = Nx*(Ny/2+1)
  int idist = 1;			// idist = distance between first element of consecutive batches 
					// = distance between (ky=1,kx=1,z=1) and (ky=2,kx=1,z=1) = 1
  int ostride = grids_->NxNyc;
  int odist = 1;
  int batchsize = grids_->NxNyc;	// number of consecutive transforms
  size_t workSize;

  cufftMakePlanMany(zft_plan_forward, dim, &n, &isize, istride, idist, &osize, ostride, odist, CUFFT_C2C, batchsize, &workSize);
  cufftMakePlanMany(zft_plan_inverse, dim, &n, &isize, istride, idist, &osize, ostride, odist, CUFFT_C2C, batchsize, &workSize);
  cufftMakePlanMany( dz_plan_forward, dim, &n, &isize, istride, idist, &osize, ostride, odist, CUFFT_C2C, batchsize, &workSize);
  cufftMakePlanMany( dz_plan_inverse, dim, &n, &isize, istride, idist, &osize, ostride, odist, CUFFT_C2C, batchsize, &workSize);
  cufftMakePlanMany(dz2_plan_forward, dim, &n, &isize, istride, idist, &osize, ostride, odist, CUFFT_C2C, batchsize, &workSize);
  cufftMakePlanMany(abs_dz_plan_forward,
 		                      dim, &n, &isize, istride, idist, &osize, ostride, odist, CUFFT_C2C, batchsize, &workSize);

  // set up callback functions
  cudaDeviceSynchronize();

  cufftCallbackStoreC zfts_callbackPtr_h;
  cufftCallbackStoreC i_kz_callbackPtr_h;
  cufftCallbackStoreC mkz2_callbackPtr_h;
  cufftCallbackStoreC abs_kz_callbackPtr_h;
  checkCuda(cudaMemcpyFromSymbol(&zfts_callbackPtr_h,   zfts_callbackPtr,   sizeof(zfts_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&i_kz_callbackPtr_h,   i_kz_callbackPtr,   sizeof(i_kz_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&mkz2_callbackPtr_h,   mkz2_callbackPtr,   sizeof(mkz2_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&abs_kz_callbackPtr_h, abs_kz_callbackPtr, sizeof(abs_kz_callbackPtr_h)));

  checkCuda(cufftXtSetCallback(   zft_plan_forward, (void**)   &zfts_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kz));
  checkCuda(cufftXtSetCallback(    dz_plan_forward, (void**)   &i_kz_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kzp));
  checkCuda(cufftXtSetCallback(abs_dz_plan_forward, (void**) &abs_kz_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kzp));
  checkCuda(cufftXtSetCallback(   dz2_plan_forward, (void**)   &mkz2_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kzp));
  cudaDeviceSynchronize();

  int nn1, nt1, nb1, nn2, nt2, nb2, nn3, nt3, nb3;

  nn1 = grids_->NxNyc;                          nt1 = min(nn1, 32);         nb1 = 1 + (nn1-1)/nt1;
  nn2 = grids_->Nz;                             nt2 = min(nn2, 32);         nb2 = 1 + (nn2-1)/nt2;
  nn3 = grids_->Nmoms;                          nt3 = min(nn3, 1);          nb3 = 1 + (nn3-1)/nt3;

  dBd = dim3(nt1, nt2, nt3);
  dGd = dim3(nb1, nb2, nb3);
  
  nn1 = grids_->NxNyc;                          nt1 = min(nn1, 32);         nb1 = 1 + (nn1-1)/nt1;
  nn2 = grids_->Nz;                             nt2 = min(nn2, 32);         nb2 = 1 + (nn2-1)/nt2;

  dBf = dim3(nt1, nt2, 1);
  dGf = dim3(nb1, nb2, 1);
  
}

GradParallelPeriodic::~GradParallelPeriodic() {
  cufftDestroy(zft_plan_forward);
  cufftDestroy(zft_plan_inverse);
  cufftDestroy( dz_plan_forward);
  cufftDestroy( dz_plan_inverse);
  cufftDestroy( dz2_plan_forward);
  cufftDestroy(abs_dz_plan_forward);
}

// Dealias in kz
void GradParallelPeriodic::dealias(MomentsG* G)
{
  for (int i = 0; i < grids_->Nmoms; i++) cufftExecC2C(zft_plan_forward, G->G(i), G->G(i), CUFFT_FORWARD);
  kz_dealias GGP2 (G->G(), grids_->kzm, grids_->Nmoms);
  for (int i = 0; i < grids_->Nmoms; i++) cufftExecC2C(zft_plan_inverse, G->G(i), G->G(i), CUFFT_INVERSE);  
}

// Dealias in kz
void GradParallelPeriodic::dealias(cuComplex* f)
{
  cufftExecC2C(zft_plan_forward, f, f, CUFFT_FORWARD);
  int one  = 1;
  kz_dealias GGPF (f, grids_->kzm, one);
  cufftExecC2C(zft_plan_inverse, f, f, CUFFT_INVERSE);  
}

// Fourier transform all moments 
void GradParallelPeriodic::zft(MomentsG* G)
{
  // for now, loop over all l and m because cannot batch 
  for(int i = 0; i < grids_->Nmoms; i++) cufftExecC2C(zft_plan_forward, G->G(i), G->G(i), CUFFT_FORWARD);
}

void GradParallelPeriodic::zft_inverse(MomentsG* G)
{
  // for now, loop over all l and m because cannot batch 
  for(int i = 0; i < grids_->Nmoms; i++) cufftExecC2C(zft_plan_inverse, G->G(i), G->G(i), CUFFT_INVERSE);
}

// Fourier transform for a single moment
void GradParallelPeriodic::zft(cuComplex* mom, cuComplex* res)
{
  cufftExecC2C(zft_plan_forward, mom, res, CUFFT_FORWARD);
}
/*
// inverse Fourier transform for a single moment
void GradParallelPeriodic::zft_inverse(cuComplex* mom, cuComplex* res)
{
  cufftExecC2C(zft_plan_inverse, mom, res, CUFFT_INVERSE);
}
*/

// FFT and derivative for all moments
void GradParallelPeriodic::dz(MomentsG* G, MomentsG* res, bool accumulate)
{
  // FFT and derivative on parallel term
  // i*kz*G calculated via callback, defined as part of dz_plan_forward
  // for now, loop over all l and m because cannot batch 
  // eventually will optimize by first transposing so that z is fastest index

  for(int i = 0; i < grids_->Nmoms; i++) {
    // forward FFT (z -> kz) & multiply by i kz (via callback)
    cufftExecC2C(dz_plan_forward, G->G(i), G->G(i), CUFFT_FORWARD);

    // backward FFT (kz -> z)
    cufftExecC2C(dz_plan_inverse, G->G(i), G->G(i), CUFFT_INVERSE);
  }
}

// FFT and two derivatives for all moments
void GradParallelPeriodic::dz2(MomentsG* G)
{
  // FFT and second derivative on parallel term
  // -kz*kz*G calculated via callback, defined as part of dz2_plan_forward
  // for now, loop over all l and m because cannot batch 
  // eventually will optimize by first transposing so that z is fastest index

  for(int i = 0; i < grids_->Nmoms; i++) {
    // forward FFT (z -> kz) & multiply by -kz**2 (via callback)
    cufftExecC2C(dz2_plan_forward, G->G(i), G->G(i), CUFFT_FORWARD);

    // backward FFT (kz -> z)
    cufftExecC2C(dz_plan_inverse, G->G(i), G->G(i), CUFFT_INVERSE);
  }
}

// FFT and two derivatives for a single moment
void GradParallelPeriodic::dz2(cuComplex* mom, cuComplex* res)
{
  cufftExecC2C(dz2_plan_forward, mom, res, CUFFT_FORWARD);
  cufftExecC2C(dz_plan_inverse, res, res, CUFFT_INVERSE);
}

// FFT and derivative for a single moment
void GradParallelPeriodic::dz(cuComplex* mom, cuComplex* res, bool accumulate)
{
  cufftExecC2C(dz_plan_forward, mom, res, CUFFT_FORWARD);
  cufftExecC2C(dz_plan_inverse, res, res, CUFFT_INVERSE);
}

// FFT and |kz| operator for a single moment
void GradParallelPeriodic::abs_dz(cuComplex* mom, cuComplex* res, bool accumulate)
{
  cufftExecC2C(abs_dz_plan_forward, mom, res, CUFFT_FORWARD);
  cufftExecC2C(dz_plan_inverse, res, res, CUFFT_INVERSE);
}

// FFT only for a single moment -- deprecated. Should change to zft, dropping dir parameter
void GradParallelPeriodic::fft_only(cuComplex* mom, cuComplex* res, int dir)
{
  // use dz_plan_inverse since it does not multiply by i kz via callback 
  cufftExecC2C(dz_plan_inverse, mom, res, dir);
}

GradParallelLocal::GradParallelLocal(Grids* grids) :
  grids_(grids)
{
  dB = 512;
  dG = 1 + (grids_->NxNycNz-1)/dB.x;
  kpar = 1./((float) grids_->Zp);
  mkpar2 = -kpar/((float) grids_->Zp);
}

void GradParallelLocal::dz(MomentsG *G, MomentsG *res, bool accumulate)
{
  G->scale(make_cuComplex(0.,kpar));
}

void GradParallelLocal::dz2(MomentsG *G)
{
  G->scale(mkpar2);
}

void GradParallelLocal::zft(MomentsG *G) {return;}
void GradParallelLocal::zft(cuComplex* mom, cuComplex* res) {
  scale_singlemom_kernel GGP (res, mom, make_cuComplex(1.,0.));
}
void GradParallelLocal::zft_inverse(MomentsG *G) {return;}
//void GradParallelLocal::zft_inverse(MomentsG *G, cuComplex* res) {return;}

// single moment
void GradParallelLocal::dz(cuComplex* mom, cuComplex* res, bool accumulate) {
  scale_singlemom_kernel GGP (res, mom, make_cuComplex(0.,kpar));
}
// single moment
void GradParallelLocal::abs_dz(cuComplex* mom, cuComplex* res, bool accumulate) {
  scale_singlemom_kernel GGP (res, mom, make_cuComplex(kpar,0.));
}
void GradParallelLocal::dz2(cuComplex* mom, cuComplex* res) {
  scale_singlemom_kernel GGP (res, mom, mkpar2);
}

GradParallel1D::GradParallel1D(Grids* grids) :
  grids_(grids)
{
  // (theta) <-> (kpar)
  cufftCreate(&dz_plan_forward);
  cufftCreate(&dz_plan_inverse);

  cufftCreate(&dz2_plan_forward);
  
  // MFM: Plan for 1d FFT
  cufftPlan1d(&dz_plan_forward, grids_->Nz, CUFFT_R2C, 1);
  cufftPlan1d(&dz_plan_inverse, grids_->Nz, CUFFT_C2R, 1);

  cufftPlan1d(&dz2_plan_forward, grids_->Nz, CUFFT_R2C, 1);

  cudaDeviceSynchronize();
  cufftCallbackStoreC i_kz_1d_callbackPtr_h;
  checkCuda(cudaMemcpyFromSymbol(&i_kz_1d_callbackPtr_h, i_kz_1d_callbackPtr, sizeof(i_kz_1d_callbackPtr_h)));
  checkCuda(cufftXtSetCallback(dz_plan_forward, (void**) &i_kz_1d_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kz));
  cudaDeviceSynchronize();

  cufftCallbackStoreC mkz2_1d_callbackPtr_h;
  checkCuda(cudaMemcpyFromSymbol(&mkz2_1d_callbackPtr_h, mkz2_1d_callbackPtr, sizeof(mkz2_1d_callbackPtr_h)));
  checkCuda(cufftXtSetCallback(dz2_plan_forward, (void**) &mkz2_1d_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kz));
  cudaDeviceSynchronize();

  cudaMalloc((void**) &b_complex, sizeof(cuComplex)*(grids_->Nz/2+1));
}

GradParallel1D::~GradParallel1D() {
  cufftDestroy(dz_plan_forward);
  cufftDestroy(dz_plan_inverse);
  cufftDestroy(dz2_plan_forward);
  cudaFree(b_complex);
}

void GradParallel1D::dz1D(float* b)  // even tho cuda 11+ overwrites inputs, this is ok
{
  checkCuda(cufftExecR2C(dz_plan_forward, b, b_complex)); 
  checkCuda(cufftExecC2R(dz_plan_inverse, b_complex, b));
}

