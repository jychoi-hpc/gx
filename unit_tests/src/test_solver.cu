#include "gtest/gtest.h"
#include "parameters.h"
#include "grids.h"
#include "moments.h"
#include "fields.h"
#include "geometry.h"
#include "solver.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  RUN_ALL_TESTS();
  MPI_Finalize();
}

class TestSolver : public ::testing::Test {

protected:
  virtual void SetUp() {
    char** argv;
    int argc = 0;
    MPI_Comm mpcom = MPI_COMM_WORLD;
    MPI_Comm_rank(mpcom, &iproc);
    MPI_Comm_size(mpcom, &nprocs);

    int devid = 0; // This should be determined (optionally) on the command line
    int nGPUs = 0;
    cudaGetDeviceCount(&nGPUs);
    checkCuda(cudaSetDevice(iproc%nGPUs));
    cudaDeviceSynchronize();
    pars = new Parameters(iproc, nprocs, mpcom);
    pars->get_nml_vars("inputs/cyc_nl_ke");

    grids = new Grids(pars);
    geo = new S_alpha_geo(pars, grids);
  }

  virtual void TearDown() {
    delete pars;
    delete grids;
    delete geo;
  }

  Parameters* pars;
  Grids *grids;
  Geometry* geo;
  int iproc, nprocs;
};

TEST_F(TestSolver, fieldSolve_phi_only) {
  Solver* solver;
  pars->fapar = 0.0;
  pars->fbpar = 0.0;
  solver = new Solver_GK(pars, grids, geo);

  Fields* fields;
  MomentsG  ** G = (MomentsG**) malloc(sizeof(void*)*grids->Nspecies);
  for(int is=0; is<grids->Nspecies; is++) {
    G[is] = nullptr;
  }
  for(int is=0; is<grids->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G[is] = new MomentsG (pars, grids, is_glob);
  }
  fields = new Fields(pars, grids);

  for(int is=0; is<grids->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G[is] -> set_zero();
    G[is] -> initialConditions();
    G[is] -> sync(true);
  }

  int NLOOP = 100;
  int counter = 0;           float timer = 0;          cudaEvent_t start, stop;
  cudaEventCreate(&start);   cudaEventCreate(&stop);   cudaEventRecord(start,0);
  for(int i=0; i<NLOOP; i++) {
    solver -> fieldSolve(G, fields);                
  }
  cudaEventRecord(stop,0);    cudaEventSynchronize(stop);    cudaEventElapsedTime(&timer,start,stop);
  printf("Avg time for fieldSolve (phi only) = %e s\n", timer/1000./NLOOP);

  for(int is=0; is<grids->Nspecies; is++) {
    if (G[is])         delete G[is];
  }
  free(G);
  delete fields;
  delete solver;
}

TEST_F(TestSolver, fieldSolve_phi_and_apar) {
  Solver* solver;
  pars->fapar = 1.0;
  pars->fbpar = 0.0;
  solver = new Solver_GK(pars, grids, geo);

  Fields* fields;
  MomentsG  ** G = (MomentsG**) malloc(sizeof(void*)*grids->Nspecies);
  for(int is=0; is<grids->Nspecies; is++) {
    G[is] = nullptr;
  }
  for(int is=0; is<grids->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G[is] = new MomentsG (pars, grids, is_glob);
  }
  fields = new Fields(pars, grids);

  for(int is=0; is<grids->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G[is] -> set_zero();
    G[is] -> initialConditions();
    G[is] -> sync(true);
  }

  int NLOOP = 100;
  int counter = 0;           float timer = 0;          cudaEvent_t start, stop;
  cudaEventCreate(&start);   cudaEventCreate(&stop);   cudaEventRecord(start,0);
  for(int i=0; i<NLOOP; i++) {
    solver -> fieldSolve(G, fields);                
  }
  cudaEventRecord(stop,0);    cudaEventSynchronize(stop);    cudaEventElapsedTime(&timer,start,stop);
  printf("Avg time for fieldSolve (phi, Apar) = %e s\n", timer/1000./NLOOP);

  for(int is=0; is<grids->Nspecies; is++) {
    if (G[is])         delete G[is];
  }
  free(G);
  delete fields;
  delete solver;
}

TEST_F(TestSolver, fieldSolve_phi_apar_bpar) {
  Solver* solver;
  pars->fapar = 1.0;
  pars->fbpar = 1.0;
  solver = new Solver_GK(pars, grids, geo);

  Fields* fields;
  MomentsG  ** G = (MomentsG**) malloc(sizeof(void*)*grids->Nspecies);
  for(int is=0; is<grids->Nspecies; is++) {
    G[is] = nullptr;
  }
  for(int is=0; is<grids->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G[is] = new MomentsG (pars, grids, is_glob);
  }
  fields = new Fields(pars, grids);

  for(int is=0; is<grids->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G[is] -> set_zero();
    G[is] -> initialConditions();
    G[is] -> sync(true);
  }

  int NLOOP = 100;
  int counter = 0;           float timer = 0;          cudaEvent_t start, stop;
  cudaEventCreate(&start);   cudaEventCreate(&stop);   cudaEventRecord(start,0);
  for(int i=0; i<NLOOP; i++) {
    solver -> fieldSolve(G, fields);                
  }
  cudaEventRecord(stop,0);    cudaEventSynchronize(stop);    cudaEventElapsedTime(&timer,start,stop);
  printf("Avg time for fieldSolve (phi,Apar,Bpar) = %e s\n", timer/1000./NLOOP);

  for(int is=0; is<grids->Nspecies; is++) {
    if (G[is])         delete G[is];
  }
  free(G);
  delete fields;
  delete solver;
}
