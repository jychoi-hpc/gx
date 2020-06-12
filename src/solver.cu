#include "solver.h"
#include "qneut_kernel.cu"
#include "get_error.h"
#include "cuda_constants.h"

Solver::Solver(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo)
{
  size_t cgrid = sizeof(cuComplex)*grids_->NxNycNz;
  cudaMalloc((void**) &nbar, cgrid); cudaMemset(nbar, 0., cgrid);
  cudaMalloc((void**) &tmp,  cgrid); cudaMemset(tmp,  0., cgrid);

  // set up phiavgdenom, which is stored for quasineutrality calculation
  cudaMalloc((void**) &phiavgdenom, sizeof(float)*grids_->Nx);
  cudaMemset(phiavgdenom, 0., sizeof(float)*grids_->Nx);
  
  cudaDeviceProp prop;
  int dev;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  maxThreadsPerBlock_ = prop.maxThreadsPerBlock;

  int threads, blocks;
  threads = maxThreadsPerBlock_;
  blocks = grids_->Nx/threads+1;
  
  calc_phiavgdenom <<<blocks, threads>>>
    (phiavgdenom, geo_->kperp2, geo_->jacobian, pars_->species, pars_->ti_ov_te); 
  
  // cuda dims for qneut calculation
  dimBlock_qneut = dim3(32, 4, 4); // MM ?? are these grid/block dimensions based on shared memory limitations?
  dimGrid_qneut = dim3(grids_->Nyc/dimBlock_qneut.x+1, grids_->Nx/dimBlock_qneut.y+1, grids_->Nz/dimBlock_qneut.z+1);
}

Solver::~Solver() 
{
  cudaFree(nbar);
  cudaFree(tmp);
  cudaFree(phiavgdenom);
}

void Solver::svar (cuComplex* f, int N)
{
  cuComplex* f_h;
  cudaMallocHost((void**) &f_h, sizeof(cuComplex)*N);
  for (int i=0; i<N; i++) {f_h[i].x=0.; f_h[i].y=0.;}
  CP_TO_CPU (f_h, f, N*sizeof(cuComplex));
  for (int i=0; i<N; i++) {
    printf("solver: var(%d) = (%e, %e) \n", i, f_h[i].x, f_h[i].y);
  }
  printf("\n");

  cudaFreeHost (f_h);
}

void Solver::svar (float* f, int N)
{
  float* f_h;
  cudaMallocHost((void**) &f_h, sizeof(float)*N);

  CP_TO_CPU (f_h, f, N*sizeof(float));

  for (int i=0; i<N; i++) printf("solver: var(%d) = %e \n", i, f_h[i]);
  printf("\n");
  
  cudaFreeHost (f_h);
}

int Solver::fieldSolve(MomentsG* G, Fields* fields)
{
  if(pars_->adiabatic_electrons) {
    real_space_density <<<grids_->NxNycNz/maxThreadsPerBlock_+1, maxThreadsPerBlock_>>>
      (nbar, G->G(), geo_->kperp2, pars_->species);

    if(pars_->iphi00==2) {
      qneutAdiab_part1 <<<dimGrid_qneut, dimBlock_qneut>>>
	(tmp, nbar, geo_->kperp2, geo_->jacobian, pars_->species, pars_->ti_ov_te);

      cudaMemset(fields->phi, 0., sizeof(cuComplex)*grids_->NxNycNz);
      
      qneutAdiab_part2 <<<dimGrid_qneut, dimBlock_qneut>>>
	(fields->phi, tmp, nbar, phiavgdenom, geo_->kperp2,
      	 geo_->jacobian, pars_->species, pars_->ti_ov_te);
    } 

    if(pars_->iphi00==1) {
      
      qneutAdiab <<<dimGrid_qneut, dimBlock_qneut>>>
	(fields->phi, nbar, geo_->kperp2, geo_->jacobian, pars_->species, pars_->ti_ov_te);
    }
  }
  if(pars_->source_option==PHIEXT) {
    add_source <<<dimGrid_qneut, dimBlock_qneut>>> (fields->phi, pars_->phi_ext);
  }
  return 0;
}

