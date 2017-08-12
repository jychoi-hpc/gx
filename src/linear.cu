#include "linear.h"
#include "device_funcs.h"
#include "cufft.h"
#include "get_error.h"
#include "species.h"
#include "cuda_constants.h"

__global__ void rhs_linear(cuComplex *g, cuComplex* phi, cuComplex* upar_bar, cuComplex* uperp_bar, cuComplex* t_bar,
			   float* b, float* iomegad, float* bgrad, float* ky, specie* s, cuComplex* rhs_par, cuComplex* rhs);
__global__ void conservation_terms(cuComplex* upar_bar, cuComplex* uperp_bar, cuComplex* t_bar, cuComplex* ghl, cuComplex* phi, float *b, specie* species);
__global__ void hypercollisions(cuComplex* g, float nu_hyper_l, float nu_hyper_m, int p_hyper_l, int p_hyper_m, cuComplex* rhs);


Linear::Linear(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo)
{
  mRhs_par = new Moments(grids_);

  // set up parallel ffts
  if(pars_->local_limit) {
    printf("Using local limit for grad parallel.\n");
    grad_par = new GradParallelLocal(grids_);
  }
//  else if(pars_->boundary_option_linked) {
//    grad_par = new GradParallelLinked(grids_);
//  }
  else {
    grad_par = new GradParallel(grids_);
  }
 
  if(pars_->closure_model==BEER42) {
    printf("Initializing Beer 4+2 closures\n");
    closures = new Beer42(grids_, geo_, pars_->local_limit);
  } else if (pars_->closure_model==SMITHPERP) {
    printf("Initializing Smith perpendicular toroidal closures\n");
    closures = new SmithPerp(grids_, geo_, pars_->smith_perp_q, pars_->smith_perp_w0);
  } else if (pars_->closure_model == SMITHPAR) {
    printf("Initializing Smith parallel closures\n");
    closures = new SmithPar(grids_, geo_, pars_->smith_par_q);
  }


  // allocate conservation terms for collision operator
  int size = sizeof(cuComplex)*grids_->NxNycNz*grids_->Nspecies;
  cudaMalloc((void**) &upar_bar, size);
  cudaMalloc((void**) &uperp_bar, size);
  cudaMalloc((void**) &t_bar, size);
  cudaMemset(upar_bar, 0., size);
  cudaMemset(uperp_bar, 0., size);
  cudaMemset(t_bar, 0., size);

  // set up CUDA grids for main linear kernel
  // NOTE: dimBlock.x = sharedSize.x = 32 gives best performance, but using 8 is only 5% worse.
  // this allows use of 4x more HL resolution without changing shared memory layouts.
  dimBlock = dim3(8, min(4, grids_->Nlaguerre), min(4, grids_->Nhermite));
  dimGrid = dim3(grids_->NxNycNz/dimBlock.x+1, 1, 1);
  sharedSize = dimBlock.x*(grids_->Nlaguerre+2)*(grids_->Nhermite+4)*sizeof(cuComplex);
  printf("For linear RHS: size of shared memory block = %f KB\n", sharedSize/1024.);
  if(sharedSize/1024.>48.) {
    printf("Error: currently cannot support this velocity resolution due to shared memory constraints.\n");
    printf("size of shared memory block must be less than 48 KB, so make sure (nhermite+4)*(nlaaguerre+2)<%d.\n", 48*1024/8/dimBlock.x);
    exit(1);
  }
}

Linear::~Linear()
{
  delete grad_par;
  delete mRhs_par;
  if(pars_->closure_model>0) delete closures;
}

int Linear::rhs(Moments* m, Fields* f, Moments* mRhs) {

  // calculate RHS
  rhs_linear<<<dimGrid, dimBlock, sharedSize>>>
      	(m->ghl, f->phi,
	geo_->kperp2, geo_->omegad, geo_->bgrad, 
       	grids_->ky, pars_->species, mRhs_par->ghl, mRhs->ghl);
  // mRhs_par = sqrt(l+1) G_l+1,m + sqrt(l) G_l-1,m
  // mRhs = ...

  // parallel gradient term
  grad_par->eval(mRhs_par);

  // combine
  mRhs->add_scaled(1., mRhs, (float) geo_->gradpar, mRhs_par);

  // closures
  if(pars_->closure_model>0) {
    closures->apply_closures(m, mRhs);
  }

  return 0;
}

// handle dissipative terms implicitly
int Linear::dissipation(Moments* m, Fields* f, double dt) {

  // calculate conservation terms for collision operator
  conservation_terms<<<grids_->NxNycNz/pars_->maxThreadsPerBlock+1, pars_->maxThreadsPerBlock>>>
	(upar_bar, uperp_bar, t_bar, m->ghl, f->phi, geo_->kperp2, pars_->species);

  if(pars_->nu_kpar!=0.) {
    for(int i = 0; i < grids_->Nmoms*grids_->Nspecies; i++) {
      grad_par->fft_only(&m->ghl[grids_->NxNycNz*i], &m->ghl[grids_->NxNycNz*i], CUFFT_FORWARD);
    }
  }

  dissipation_kernel<<<dimGrid,dimBlock>>>(m->ghl, f->phi, geo_->kperp2, grids_->kz, pars_->species, pars_->nu_kpar, upar_bar, uperp_bar, t_bar,
                                           pars_->nu_hyper_l, pars_->nu_hyper_m, pars_->p_hyper_l, pars_->p_hyper_m, dt);

  if(pars_->nu_kpar!=0.) {
    for(int i = 0; i < grids_->Nmoms*grids_->Nspecies; i++) {
      grad_par->fft_only(&m->ghl[grids_->NxNycNz*i], &m->ghl[grids_->NxNycNz*i], CUFFT_INVERSE);
    }
  }
}

// main kernel function for calculating RHS
# define S_G(L, M) s_g[sidxyz + (sDimx)*(M) + (sDimx)*(sDimy)*(L)]
__global__ void rhs_linear(cuComplex *g, cuComplex* phi, 
	float* b, float* omegad, float* bgrad, float* ky, specie* species,
	cuComplex* rhs_par, cuComplex* rhs)
{
  extern __shared__ cuComplex s_g[]; // aliased below by macro S_G, defined above

  unsigned int idxyz = threadIdx.x + blockIdx.x*blockDim.x;
  if(idxyz<nx*nyc*nz) {
    const unsigned int sidxyz = threadIdx.x;
    // these modulo operations are expensive... better way to get these indices?
    const unsigned int idy = idxyz % (nx*nyc) % nyc; 
    const unsigned int idz = idxyz / (nx*nyc);
  
    // shared memory blocks of size blockDim.x * (nlaguerre+2) * (nhermite+4)
    const int sDimx = blockDim.x;
    const int sDimy = nlaguerre+2;
  
    // read these values into (hopefully) register memory. 
    // local to each thread (i.e. each idxyz).
    // since idxyz is linear, these accesses are coalesced.
    const cuComplex phi_ = phi[idxyz];
    const float b_ = b[idxyz];
    const cuComplex iomegad_ = make_cuComplex(0., omegad[idxyz]);
  
    // all threads in a block will likely have same value of idz, so they will be reading same value of bgrad[idz].
    // if bgrad was in shared memory, would have bank conflicts.
    // no bank conflicts for reading from global memory though. 
    const float bgrad_ = bgrad[idz];  
  
    // this is coalesced?
    const cuComplex iomegastar_ = make_cuComplex(0., ky[idy]); 
  
   //#pragma unroll
   for(int is=0; is<nspecies; is++) { // might be a better way to handle species loop here...
    specie s = species[is];
  
    // species-specific constants
    const float nu_ = s.nu_ss; 
    const float tprim_ = s.tprim;
    const float fprim_ = s.fprim;
    //const float rho_ = s.rho;

    // conservation terms (species-specific)
    cuComplex upar_bar_ = upar_bar[idxyz + is*nx*nyc*nz];
    cuComplex uperp_bar_ = uperp_bar[idxyz + is*nx*nyc*nz];
    cuComplex t_bar_ = t_bar[idxyz + is*nx*nyc*nz];
  
    // read tile of g into shared mem
    // each thread in the block reads in multiple values of l and m
    for (int l = threadIdx.z; l < nhermite; l += blockDim.z) {
     for (int m = threadIdx.y; m < nlaguerre; m += blockDim.y) {
      int globalIdx = idxyz + nx*nyc*nz*m + nx*nyc*nz*nlaguerre*l + nx*nyc*nz*nlaguerre*nhermite*is; 
      int sl = l + 2;
      int sm = m + 1;
      S_G(sl, sm) = g[globalIdx];
     }
    }
  
    // this syncthreads isnt necessary unless ghosts require information from interior cells
    //__syncthreads();
  
    // set up ghost cells in l (for all m's)
    for (int m = threadIdx.y; m < nlaguerre; m += blockDim.y) {
      int sm = m + 1;
      int sl = threadIdx.z + 2;
      if(sl < 4) {
        // set ghost to zero at low l
        S_G(sl-2, sm) = make_cuComplex(0., 0.);
  
        // set ghost with closures at high l
        S_G(sl+nhermite, sm) = make_cuComplex(0., 0.);
      }
    }
  
    // set up ghost cells in m (for all l's)
    for (int l = threadIdx.z; l < nhermite+2; l += blockDim.z) {
      int sl = l + 1; // this takes care of corners...
      int sm = threadIdx.y + 1;
      if(sm < 2) {
        // set ghost to zero at low m
        S_G(sl, sm-1) = make_cuComplex(0., 0.);
  
        // set ghost with closures at high m
        S_G(sl, sm+nlaguerre) = make_cuComplex(0., 0.);
      }
    }
  
    __syncthreads();
  
    // stencil (on non-ghost cells)
    for (int l = threadIdx.z; l < nhermite; l += blockDim.z) {
     for (int m = threadIdx.y; m < nlaguerre; m += blockDim.y) {
      int globalIdx = idxyz + nx*nyc*nz*m + nx*nyc*nz*nlaguerre*l + nx*nyc*nz*nlaguerre*nhermite*is; 
      int sl = l + 2; // offset to get past ghosts
      int sm = m + 1; // offset to get past ghosts
  
      // need to calculate parallel terms separately because need to take derivative via fft 
      rhs_par[globalIdx] = -( sqrtf(l+1)*S_G(sl+1,sm) + sqrtf(l)*S_G(sl-1,sm) );
  
      // remaining terms
      rhs[globalIdx] = 
       - bgrad_ * ( -sqrtf(l+1)*(m+1)*S_G(sl+1,sm) - sqrtf(l+1)*m*S_G(sl+1,sm-1)
  
                    + sqrtf(l)*m*S_G(sl-1,sm) + sqrtf(l)*(m+1)*S_G(sl-1,sm+1) )
  
          - iomegad_ * ( sqrtf((l+1)*(l+2))*S_G(sl+2,sm) + sqrtf(l*(l-1))*S_G(sl-2,sm)
  
                        + (m+1)*S_G(sl,sm+1) + m*S_G(sl,sm-1) + 2.*(l+m+1)*S_G(sl,sm) )
  
           - nu_ * ( b_ + l + 2*m ) * ( S_G(sl,sm) + Jflr(m, b_)*phi_ );
  
      // add drive and conservation terms in low hermite moments
      if(l==0) {
        rhs[globalIdx] = rhs[globalIdx] + phi_*(
                  Jflr(m-1,b_)*( -m*iomegad_ + m*tprim_*iomegastar_ ) 
                + Jflr(m,  b_)*( -2*(m+1)*iomegad_ + (fprim_ + tprim_*2*m)*iomegastar_ )
                + Jflr(m+1,b_)*( -(m+1)*iomegad_ + (m+1)*tprim_*iomegastar_ ) )
		+ nu_ * sqrtf(b_) * ( Jflr(m, b_) + Jflr(m-1, b_) ) * uperp_bar_
		+ nu_ * 2. * ( m*Jflr(m-1,b_) + 2.*m*Jflr(m,b_) + (m+1)*Jflr(m+1,b_) ) * t_bar_;
      } 
      if(l==1) {
        rhs_par[globalIdx] = rhs_par[globalIdx] - Jflr(m,b_)*phi_;
        rhs[globalIdx] = rhs[globalIdx] - phi_ * ( m*Jflr(m,b_) + (m+1)*Jflr(m+1,b_) ) * bgrad_ 
      		+ nu_ * Jflr(m,b_) * upar_bar_;
      }
      if(l==2) {
        rhs[globalIdx] = rhs[globalIdx] + phi_ * Jflr(m,b_) * (-2*iomegad_ + tprim_*iomegastar_)/sqrtf(2) 
		+ nu_ * sqrtf(2) * Jflr(m,b_) * t_bar_;
      }  
     } // m loop
    } // l loop
  
   } // species loop
  } // idxyz < NxNycNz
}

# define H(XYZ, L, M, S) ghl[(XYZ) + nx*nyc*nz*(M) + nx*nyc*nz*nlaguerre*(L) + nx*nyc*nz*nlaguerre*nhermite*(S)] + Jflr(M,b_)*phi_
__global__ void conservation_terms(cuComplex* upar_bar, cuComplex* uperp_bar, cuComplex* t_bar, cuComplex* ghl, cuComplex* phi, float *b, specie* species)
{
  unsigned int idxyz = get_id1();

  if(idxyz<nx*nyc*nz) {
    float b_ = b[idxyz];
    cuComplex phi_ = phi[idxyz];
    for(int is=0; is<nspecies; is++) {
      int index = idxyz + nx*nyc*nz*is;
      upar_bar[index] = make_cuComplex(0., 0.);
      uperp_bar[index] = make_cuComplex(0., 0.);
      t_bar[index] = make_cuComplex(0., 0.);
      // sum over m
      for(int m=0; m<nlaguerre; m++) {
        // H(...) is defined by macro above
        upar_bar[index] = upar_bar[index] + Jflr(m,b_)*H(idxyz, 1, m, is);
        uperp_bar[index] = uperp_bar[index] + (Jflr(m,b_) + Jflr(m-1,b_))*H(idxyz, 0, m, is);
        t_bar[index] = t_bar[index] + sqrtf(2)*Jflr(m,b_)*H(idxyz, 2, m, is)
		+ ( m*Jflr(m-1,b_) + 2.*m*Jflr(m,b_) + (m+1)*Jflr(m+1,b_) )*H(idxyz, 0, m, is);
      }
      uperp_bar[index] = uperp_bar[index]*sqrtf(b_);
    }
  }
}

__global__ void dissipation_kernel(cuComplex* g, cuComplex* phi, float* b, float* kz, specie* species, float nu_kpar,
				   cuComplex* upar_bar, cuComplex* uperp_bar, cuComplex* t_bar,
  				   float nu_hyper_l, float nu_hyper_m, int p_hyper_l, int p_hyper_m, float dt)
{

  const unsigned int idxyz = get_id1();
  const unsigned int idz = idxyz / (nx*nyc);
  if(idxyz<nx*nyc*nz) {
   for(int is=0; is<nspecies; is++) { 
    for (int l = threadIdx.z; l < nhermite; l += blockDim.z) {
     for (int m = threadIdx.y; m < nlaguerre; m += blockDim.y) {
      int globalIdx = idxyz + nx*nyc*nz*m + nx*nyc*nz*nlaguerre*l + nx*nyc*nz*nlaguerre*nhermite*is; 
      int c = 1;
      if(l<=2) c=0; // to switch off hypercollisions for l=0,1,2

      float damp = species[is].nu_ss*( b[idxyz] + l + 2*m ) 
		+ c*nu_hyper_l*pow((float) l/nhermite, (float) p_hyper_l)
		+ c*nu_hyper_m*pow((float)m/nlaguerre, p_hyper_m)
		+ nu_kpar*pow(kz[idz],2);

      g[globalIdx] = (g[globalIdx] - dt*damp*phi[idxyz])/(1+dt*damp);
 
      // need conservation terms

     }
    }
   }
  }
}
				   

__global__ void hypercollisions(cuComplex* g, float nu_hyper_l, float nu_hyper_m, int p_hyper_l, int p_hyper_m, cuComplex* rhs) {
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
   for(int is=0; is<nspecies; is++) { 
    for (int l = threadIdx.z; l < nhermite; l += blockDim.z) {
     for (int m = threadIdx.y; m < nlaguerre; m += blockDim.y) {
      int globalIdx = idxyz + nx*nyc*nz*m + nx*nyc*nz*nlaguerre*l + nx*nyc*nz*nlaguerre*nhermite*is; 
      if(l>2) {
        rhs[globalIdx] = rhs[globalIdx] - (nu_hyper_l*pow((float) l/nhermite, (float) p_hyper_l)+nu_hyper_m*pow((float)m/nlaguerre, p_hyper_m))*g[globalIdx];
      }
     }
    }
   }
  }
}
