#include "solver.h"
#define GQN <<< dG, dB >>>

//=======================================
// Solver_GK
// object for handling field solve in GK
//=======================================
Solver_GK::Solver_GK(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo),
  tmp(nullptr), nbar(nullptr), jparbar(nullptr), jperpbar(nullptr), phiavgdenom(nullptr), 
  qneutFacPhi(nullptr), ampereParFac(nullptr),
  qneutFacBpar(nullptr), amperePerpFacPhi(nullptr), amperePerpFacBpar(nullptr)
{

  if (pars_->ks) return;
  
  count = grids_->NxNycNz;
  if(pars_->fapar > 0.) count = 2*grids_->NxNycNz;
  if(pars_->fbpar > 0.) count = 3*grids_->NxNycNz;

  // the "nbar" array contains nbar, jparbar, and jperpbar
  size_t cgrid = sizeof(cuComplex)*count;
  if (pars_->use_NCCL) {
    checkCuda(cudaMalloc((void**) &nbar, cgrid)); zero(nbar);
    checkCuda(cudaMalloc((void**) &nbar_tmp, cgrid)); zero(nbar_tmp);
  } else {
    cudaMallocHost((void**) &nbar, cgrid); zero(nbar);
    cudaMallocHost((void**) &nbar_tmp, cgrid); zero(nbar_tmp);
  }
  // set offset pointers to jparbar and jperpbar
  if(pars_->fapar > 0.) jparbar = nbar+grids_->NxNycNz;
  if(pars_->fbpar > 0.) jperpbar = nbar+2*grids_->NxNycNz;

  int nn1, nn2, nn3, nt1, nt2, nt3, nb1, nb2, nb3;
  
  nn1 = grids_->Nyc;        nt1 = min(nn1, 32 );   nb1 = 1 + (nn1-1)/nt1;
  nn2 = grids_->Nx;         nt2 = min(nn2,  4 );   nb2 = 1 + (nn2-1)/nt2;
  nn3 = grids_->Nz;         nt3 = min(nn3,  4 );   nb3 = 1 + (nn3-1)/nt3;
  
  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3);
  
  nn1 = grids_->Nyc;        nt1 = min(nn1, 32 );   nb1 = 1 + (nn1-1)/nt1;
  nn2 = grids_->Nx;         nt2 = min(nn2, 32 );   nb2 = 1 + (nn2-1)/nt2;
  nn3 = 1;                  nt3 = min(nn3,  1 );   nb3 = 1 + (nn3-1)/nt3;
  
  db = dim3(nt1, nt2, nt3);
  dg = dim3(nb1, nb2, nb3);
  
  if(!pars_->all_kinetic && (pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS)) {cudaMalloc((void**) &tmp,  cgrid); zero(tmp);}
  
  if(pars_->fphi > 0.) {
    cudaMalloc(&qneutFacPhi,    sizeof(float)*grids_->NxNycNz);
    cudaMemset(qneutFacPhi, 0., sizeof(float)*grids_->NxNycNz);    
  }

  if(pars_->fapar > 0.) {
    cudaMalloc(&ampereParFac,    sizeof(float)*grids_->NxNycNz);
    cudaMemset(ampereParFac, 0., sizeof(float)*grids_->NxNycNz);    
  }

  if(pars_->fbpar > 0.) {
    cudaMalloc(&qneutFacBpar,    sizeof(float)*grids_->NxNycNz);
    cudaMemset(qneutFacBpar, 0., sizeof(float)*grids_->NxNycNz);    
    cudaMalloc(&amperePerpFacPhi,    sizeof(float)*grids_->NxNycNz);
    cudaMemset(amperePerpFacPhi, 0., sizeof(float)*grids_->NxNycNz);    
    cudaMalloc(&amperePerpFacBpar,    sizeof(float)*grids_->NxNycNz);
    cudaMemset(amperePerpFacBpar, 0., sizeof(float)*grids_->NxNycNz);    
  }
  
  // int threads, blocks;
  // threads = min(grids_->NxNycNz, 128);
  // blocks = 1 + (grids_->NxNycNz-1)/threads;
  
  // compute qneutFacPhi  = sum_s z_s^2*n_s/tau_s*(1- sum_l J_l^2)
  //         qneutFacBpar = -sum_s z_s*n_s*sum_l J_l*(J_l + J_{l-1})
  //         ampereParFac = kperp2 + beta/2*sum_s z_s^2*n_s/m_s*sum_l J_l^2
  //         amperePerpFacPhi  = beta/2*sum_s z_s*n_s*sum_l J_l*(J_l + J_{l-1})
  //         amperePerpFacBpar = 1 + beta/2*sum_s n_s*t_s*sum_l (J_l + J_{l-1})^2
  for(int is_glob=0; is_glob<grids_->Nspecies_glob; is_glob++) {
    sum_solverFacs GQN (qneutFacPhi, qneutFacBpar, ampereParFac, amperePerpFacPhi, amperePerpFacBpar, geo_->kperp2, geo_->bmag, geo_->bmagInv,
                        pars_->species_h[is_glob], pars_->beta, is_glob==0, pars_->fapar, pars_->fbpar, pars_->long_wavelength_GK);
  }

  // set up phiavgdenom, which is stored for quasineutrality calculation as appropriate
  if (!pars_->all_kinetic && (pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS) && !pars_->no_fields) {     
    cudaMalloc(&phiavgdenom,    sizeof(float)*grids_->Nx);
    cudaMemset(phiavgdenom, 0., sizeof(float)*grids_->Nx);    
    
    int threads, blocks;
    threads = min(grids_->Nx, 128);
    blocks = 1 + (grids_->Nx-1)/threads;
    
    calc_phiavgdenom <<<blocks, threads>>> (phiavgdenom, geo_->jacobian, qneutFacPhi, pars_->tau_fac);
  }
}


Solver_GK::~Solver_GK() 
{
  if (nbar)        cudaFree(nbar);
  if (tmp)         cudaFree(tmp);
  if (qneutFacPhi)  cudaFree(qneutFacPhi);
  if (qneutFacBpar)  cudaFree(qneutFacBpar);
  if (amperePerpFacPhi)  cudaFree(amperePerpFacPhi);
  if (amperePerpFacBpar)  cudaFree(amperePerpFacBpar);
  if (ampereParFac) cudaFree(ampereParFac);
  if (phiavgdenom) cudaFree(phiavgdenom);

}

void Solver_GK::fieldSolve(MomentsG** G, Fields* fields)
// Calculates all the fields, i.e., phi, apar, bpar
{

  if (pars_->ks) return;
  if (pars_->vp) {
    getPhi GQN (fields->phi, G[0]->G(), grids_->ky);
    return;
  }
  
  if (pars_->no_fields) { zero(fields->phi); return; }
  
  zero(nbar);

  for(int is=0; is<grids_->Nspecies; is++) {
    if(grids_->m_lo == 0) { // only compute density on procs with m=0
      real_space_density GQN (nbar, G[is]->G(), geo_->kperp2, *G[is]->species);
      if(pars_->fbpar>0.0) {
        // jperpbar is an offset pointer to a location in nbar
        real_space_perp_current GQN (jperpbar, G[is]->G(), geo_->kperp2, geo_->bmagInv, *G[is]->species);
      }
    }
    if(grids_->m_lo <= 1 && grids_->m_up > 1) { // only compute current on procs with m=1
      if(pars_->fapar>0.0) {
        // jparbar is an offset pointer to a location in nbar
        real_space_par_current GQN (jparbar, G[is]->G(), geo_->kperp2, *G[is]->species);
      }
    }
  }

  if(grids_->nprocs>1) {
    // factor of 2 in count*2 is from cuComplex -> float conversion
    // here, "nbar" actually packages nbar, jparbar, and jperpbar
    if(pars_->use_NCCL) {
      // do AllReduce only across procs with m=0
      if(grids_->iproc_m==0) {
        checkCuda(ncclAllReduce((void*) nbar, (void*) nbar_tmp, count*2, ncclFloat, ncclSum, grids_->ncclComm_m0, 0));
      }
      // broadcast result to all procs
      checkCuda(ncclBroadcast((void*) nbar_tmp, (void*) nbar, count*2, ncclFloat, 0, grids_->ncclComm_s, 0));
      cudaStreamSynchronize(0);
    } else {
      MPI_Allreduce((void*) nbar_tmp, (void*) nbar, count*2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      CP_ON_GPU(nbar, nbar_tmp, sizeof(cuComplex)*count);
    }
  }

  if(pars_->all_kinetic) {
    if (pars_->fbpar>0.0) {
      qneut_and_ampere_perp GQN (fields->phi, fields->bpar, nbar, jperpbar, qneutFacPhi, qneutFacBpar, amperePerpFacPhi, amperePerpFacBpar, pars_->fphi, pars_->fbpar);
    } else {
      qneut GQN (fields->phi, nbar, qneutFacPhi, pars_->fphi);
    }
    if (pars_->fapar>0.0) ampere_apar  GQN (fields->apar, jparbar, ampereParFac, pars_->fapar);
  } else if (pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS) {
    qneutAdiab_part1 GQN (             tmp, nbar, geo_->jacobian, qneutFacPhi, pars_->tau_fac);
    qneutAdiab_part2 GQN (fields->phi, tmp, nbar, phiavgdenom,    qneutFacPhi, pars_->tau_fac, pars_->fphi);
  } else if (pars_->Boltzmann_opt == BOLTZMANN_IONS) {
    qneutAdiab GQN (fields->phi, nbar, qneutFacPhi, pars_->tau_fac, pars_->fphi);
    // can still have finite apar with Boltzmann ions
    if (pars_->fapar>0.0) ampere_apar  GQN (fields->apar, jparbar, ampereParFac, pars_->fapar);
  }
  
  if(pars_->source_option==PHIEXT) add_source GQN (fields->phi, pars_->phi_ext);
}

void Solver_GK::svar (cuComplex* f, int N)
{
  cuComplex* f_h = (cuComplex*) malloc(sizeof(cuComplex)*N);

  for (int i=0; i<N; i++) { f_h[i].x=0.; f_h[i].y=0.; }

  CP_TO_CPU (f_h, f, N*sizeof(cuComplex));

  for (int i=0; i<N; i++) printf("solver: var(%d) = (%e, %e) \n", i, f_h[i].x, f_h[i].y);
  printf("\n");

  free (f_h);
}

void Solver_GK::svar (float* f, int N)
{
  float* f_h = (float*) malloc(sizeof(float)*N);

  CP_TO_CPU (f_h, f, N*sizeof(float));

  for (int i=0; i<N; i++) printf("solver: var(%d) = %e \n", i, f_h[i]);
  printf("\n");
  
  free (f_h);
}

void Solver_GK::zero (cuComplex* f)
{
  cudaMemset(f, 0., sizeof(cuComplex)*count);
}

//==========================================
// Solver_KREHM
// object for handling field solve in KREHM
//==========================================
Solver_KREHM::Solver_KREHM(Parameters* pars, Grids* grids) :
  pars_(pars), grids_(grids)
{
  int nn1, nn2, nn3, nt1, nt2, nt3, nb1, nb2, nb3;
  
  nn1 = grids_->Nyc;        nt1 = min(nn1, 32 );   nb1 = 1 + (nn1-1)/nt1;
  nn2 = grids_->Nx;         nt2 = min(nn2,  4 );   nb2 = 1 + (nn2-1)/nt2;
  nn3 = grids_->Nz;         nt3 = min(nn3,  4 );   nb3 = 1 + (nn3-1)/nt3;
  
  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3);

  count = grids_->NxNycNz*2; // 2 moments, density and current
  size_t cgrid = sizeof(cuComplex)*count;
  checkCuda(cudaMalloc((void**) &moms, cgrid)); 
  // set offset pointers
  density = moms;
  current = moms + grids_->NxNycNz;
}

Solver_KREHM::~Solver_KREHM() 
{
  cudaFree(moms);
}

void Solver_KREHM::fieldSolve(MomentsG** G, Fields* fields)
{
  if(grids_->iproc_m==0) {
    CP_ON_GPU(density, G[0]->Gm(0), sizeof(cuComplex)*grids_->NxNycNz);
    CP_ON_GPU(current, G[0]->Gm(1), sizeof(cuComplex)*grids_->NxNycNz);
  }
  if(grids_->nprocs>1) {
    // broadcast moments to all procs
    // factor of 2 in count*2 is from cuComplex -> float conversion
    // moms includes both density and current
    checkCuda(ncclBroadcast((void*) moms, (void*) moms, count*2, ncclFloat, 0, grids_->ncclComm, 0));
    cudaStreamSynchronize(0);
  } 
  phiSolve_krehm<<<dG, dB>>>(fields->phi, density, grids_->kx, grids_->ky, pars_->rho_i);
  aparSolve_krehm<<<dG, dB>>>(fields->apar, current, grids_->kx, grids_->ky, pars_->rho_s, pars_->d_e);
}

void Solver_KREHM::set_equilibrium_current(MomentsG* G, Fields* fields)
{
  if(grids_->m_lo <= 1 && grids_->m_up > 1) { // only compute current on procs with m=1
    int m = 1;
    int m_local = m - grids_->m_lo;
    equilibrium_current_krehm<<<dG, dB>>>(G->Gm(m_local), grids_->kx, grids_->ky, pars_->rho_s, pars_->d_e, fields->apar_ext);
  }
}

//==========================================
// Solver_cetg
// object for handling field solve in the Adkins collisional ETG model
//==========================================
Solver_cetg::Solver_cetg(Parameters* pars, Grids* grids) :
  pars_(pars), grids_(grids)
{
  int nn1, nn2, nn3, nt1, nt2, nt3, nb1, nb2, nb3;
  
  nn1 = grids_->Nyc;        nt1 = min(nn1, 32 );   nb1 = 1 + (nn1-1)/nt1;
  nn2 = grids_->Nx;         nt2 = min(nn2,  4 );   nb2 = 1 + (nn2-1)/nt2;
  nn3 = grids_->Nz;         nt3 = min(nn3,  4 );   nb3 = 1 + (nn3-1)/nt3;
  
  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3);

  count = grids_->NxNycNz; 
  size_t cgrid = sizeof(cuComplex)*count;
  checkCuda(cudaMalloc((void**) &moms, cgrid)); 

  // set pointer for convenience
  density = moms;
}

Solver_cetg::~Solver_cetg() 
{
  cudaFree(moms);
}

void Solver_cetg::fieldSolve(MomentsG** G, Fields* fields)
{
  if(grids_->iproc_m==0) {
    CP_ON_GPU(density, G[0]->Gm(0), sizeof(cuComplex)*grids_->NxNycNz);
  }
  if(grids_->nprocs>1) {
    assert(false && "Cannot use multiple GPUs for the Adkins collisional ETG model. \n");
  }
  
  phiSolve_cetg<<<dG, dB>>>(fields->phi, density, pars_->tau_fac);
}

//=======================================
// Solver_VP
// object for handling field solve in VP
//=======================================
Solver_VP::Solver_VP(Parameters* pars, Grids* grids) :
  pars_(pars), grids_(grids)
{

  int nn1 = grids_->Nyc;        int nt1 = min(nn1, 512);     int nb1 = 1 + (nn1-1)/nt1;
  
  dB = dim3(nt1, 1, 1);
  dG = dim3(nb1, 1, 1);
  
}

Solver_VP::~Solver_VP() 
{
  // nothing
}

void Solver_VP::fieldSolve(MomentsG** G, Fields* fields)
{
  if (pars_->ks) return;

  getPhi GQN (fields->phi, G[0]->G(), grids_->ky);
}

void Solver_VP::svar (cuComplex* f, int N)
{
  cuComplex* f_h = (cuComplex*) malloc(sizeof(cuComplex)*N);

  for (int i=0; i<N; i++) { f_h[i].x=0.; f_h[i].y=0.; }

  CP_TO_CPU (f_h, f, N*sizeof(cuComplex));

  for (int i=0; i<N; i++) printf("solver: var(%d) = (%e, %e) \n", i, f_h[i].x, f_h[i].y);
  printf("\n");

  free (f_h);
}

void Solver_VP::svar (float* f, int N)
{
  float* f_h = (float*) malloc(sizeof(float)*N);

  CP_TO_CPU (f_h, f, N*sizeof(float));

  for (int i=0; i<N; i++) printf("solver: var(%d) = %e \n", i, f_h[i]);
  printf("\n");
  
  free (f_h);
}

void Solver_VP::zero (cuComplex* f)
{
  cudaMemset(f, 0., sizeof(cuComplex)*grids_->Nyc);
}

