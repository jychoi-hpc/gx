#include "solver.h"
#define GQN <<< dG, dB >>>

//=======================================
// Solver_GK
// object for handling field solve in GK
//=======================================
Solver_GK::Solver_GK(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo),
  tmp(nullptr), nbar(nullptr), phiavgdenom(nullptr), 
  qneutDenom(nullptr), ampereDenom(nullptr)
{

  if (pars_->ks) return;
  
  size_t cgrid = sizeof(cuComplex)*grids_->NxNycNz;
  cudaMalloc((void**) &nbar, cgrid); zero(nbar);

  if(pars_->beta > 0.) {
    cudaMalloc((void**) &jbar, cgrid); zero(jbar);
  }

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
  
  cudaMalloc(&qneutDenom,    sizeof(float)*grids_->NxNycNz);
  cudaMemset(qneutDenom, 0., sizeof(float)*grids_->NxNycNz);    

  if(pars_->beta > 0.) {
    cudaMalloc(&ampereDenom,    sizeof(float)*grids_->NxNycNz);
    cudaMemset(ampereDenom, 0., sizeof(float)*grids_->NxNycNz);    
  }
  
  int threads, blocks;
  threads = min(grids_->NxNycNz, 128);
  blocks = 1 + (grids_->NxNycNz-1)/threads;
  
  // compute qneutDenom = sum_s z_s^2*n_s/tau_s*(1- sum_l J_l^2)
  // and ampereDenom = kperp2 + beta/2*sum_s z_s^2*n_s/m_s*sum_l J_l^2
  for(int is=0; is<grids_->Nspecies; is++) {
    sum_qneutDenom GQN (qneutDenom, geo_->kperp2, pars_->species_h[is]);
    if(pars_->beta > 0.) sum_ampereDenom GQN (ampereDenom, geo_->kperp2, pars_->species_h[is], is==0);
  }

  // set up phiavgdenom, which is stored for quasineutrality calculation as appropriate
  if (!pars_->all_kinetic && (pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS) && !pars_->no_fields) {     
    cudaMalloc(&phiavgdenom,    sizeof(float)*grids_->Nx);
    cudaMemset(phiavgdenom, 0., sizeof(float)*grids_->Nx);    
    
    int threads, blocks;
    threads = min(grids_->Nx, 128);
    blocks = 1 + (grids_->Nx-1)/threads;
    
    calc_phiavgdenom <<<blocks, threads>>> (phiavgdenom, geo_->jacobian, qneutDenom, pars_->tau_fac);
  }
    
}

Solver_GK::~Solver_GK() 
{
  if (nbar)        cudaFree(nbar);
  if (jbar)        cudaFree(jbar);
  if (tmp)         cudaFree(tmp);
  if (qneutDenom)  cudaFree(qneutDenom);
  if (ampereDenom) cudaFree(ampereDenom);
  if (phiavgdenom) cudaFree(phiavgdenom);
}

void Solver_GK::fieldSolve(MomentsG** G, Fields* fields)
{
  if (pars_->ks) return;
  if (pars_->vp) {
    getPhi GQN (fields->phi, G[0]->G(), grids_->ky);
    return;
  }
  
  if (pars_->no_fields) { zero(fields->phi); return; }
  
  bool em = pars_->beta > 0. ? true : false;
  
  if (pars_->all_kinetic) {
    zero(nbar);
    if(em) zero(jbar);

    for(int is=0; is<grids_->Nspecies; is++) {
      real_space_density GQN (nbar, G[is]->G(), geo_->kperp2, *G[is]->species);
      if(em) real_space_current GQN (jbar, G[is]->G(), geo_->kperp2, *G[is]->species);
    }
    
    qneut GQN (fields->phi, nbar, qneutDenom);
    if (em) ampere GQN (fields->apar, jbar, ampereDenom);

  } else {

    zero(nbar);

    for(int is=0; is<grids_->Nspecies; is++) {
      real_space_density GQN (nbar, G[is]->G(), geo_->kperp2, *G[is]->species);
    }

    // In these routines there is inefficiency because multiple threads
    // calculate the same field line averages. It is correct but inefficient.

    if(pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS) {
      zero(fields->phi);
      qneutAdiab_part1 GQN (             tmp, nbar, geo_->jacobian, qneutDenom, pars_->tau_fac);
      qneutAdiab_part2 GQN (fields->phi, tmp, nbar, phiavgdenom,    qneutDenom, pars_->tau_fac);
    } 
    
    if(pars_->Boltzmann_opt == BOLTZMANN_IONS) qneutAdiab GQN (fields->phi, nbar, qneutDenom, pars_->tau_fac);
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
  cudaMemset(f, 0., sizeof(cuComplex)*grids_->NxNycNz);
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
}

Solver_KREHM::~Solver_KREHM() 
{
  // nothing
}

void Solver_KREHM::fieldSolve(MomentsG** G, Fields* fields)
{
  phiSolve_krehm<<<dG, dB>>>(fields->phi, G[0]->G(0), grids_->kx, grids_->ky, pars_->rho_i);
  aparSolve_krehm<<<dG, dB>>>(fields->apar, G[0]->G(1), grids_->kx, grids_->ky, pars_->rho_s, pars_->d_e);
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

