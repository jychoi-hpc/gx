#include "gtest/gtest.h"
#include "parameters.h"
#include "grids.h"
#include "moments.h"
#include "fields.h"
#include "geometry.h"
#include "nonlinear.h"

#define NLOOP 1000

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  RUN_ALL_TESTS();
  MPI_Finalize();
}

class TestNonlinear : public ::testing::Test {

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
    pars->get_nml_vars("inputs/cyc_nl");

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

TEST_F(TestNonlinear, warmup_nlps_ES) {
  Nonlinear *nonlinear;
  pars->beta = 0.0; // electrostatic limit
  pars->fapar = 0.0;
  pars->fbpar = 0.0;
  nonlinear = new Nonlinear_GK(pars, grids, geo);

  MomentsG *G, *GRes;
  Fields* fields;
  G = new MomentsG(pars, grids, 0);
  GRes = new MomentsG(pars, grids, 0);
  fields = new Fields(pars, grids);

  // set initial conditions
  G->initialConditions();

  // set phi = n
  CP_ON_GPU(fields->phi, G->Gm(0), sizeof(cuComplex)*grids->NxNycNz);

  // evaluate nonlinear term
  GRes->set_zero();

  int counter = 0;           float timer = 0;          cudaEvent_t start, stop;
  cudaDeviceSynchronize();
  cudaEventCreate(&start);   cudaEventCreate(&stop);   cudaEventRecord(start,0);
  for(int i=0; i<NLOOP; i++) {
    nonlinear->nlps(G, fields, GRes);
  }
  cudaEventRecord(stop,0);    cudaEventSynchronize(stop);    cudaEventElapsedTime(&timer,start,stop);
  printf("Avg time for warmup (nlps ES) = %e s\n", timer/1000./NLOOP);

  delete G;
  delete GRes;
  delete fields;
  delete nonlinear;
}

TEST_F(TestNonlinear, nlps_ES) {
  Nonlinear *nonlinear;
  pars->beta = 0.0; // electrostatic limit
  pars->fapar = 0.0;
  pars->fbpar = 0.0;
  nonlinear = new Nonlinear_GK(pars, grids, geo);

  MomentsG *G, *GRes;
  Fields* fields;
  G = new MomentsG(pars, grids, 0);
  GRes = new MomentsG(pars, grids, 0);
  fields = new Fields(pars, grids);

  // set initial conditions
  G->initialConditions();

  // set phi = n
  CP_ON_GPU(fields->phi, G->Gm(0), sizeof(cuComplex)*grids->NxNycNz);

  // evaluate nonlinear term
  GRes->set_zero();

  int counter = 0;           float timer = 0;          cudaEvent_t start, stop;
  cudaDeviceSynchronize();
  cudaEventCreate(&start);   cudaEventCreate(&stop);   cudaEventRecord(start,0);
  for(int i=0; i<NLOOP; i++) {
    nonlinear->nlps(G, fields, GRes);
  }
  cudaEventRecord(stop,0);    cudaEventSynchronize(stop);    cudaEventElapsedTime(&timer,start,stop);
  printf("Avg time for nlps (ES) = %e s\n", timer/1000./NLOOP);

  delete G;
  delete GRes;
  delete fields;
  delete nonlinear;
}

TEST_F(TestNonlinear, nlps_EM) {
  Nonlinear *nonlinear;
  nonlinear = new Nonlinear_GK(pars, grids, geo);

  MomentsG *G, *GRes;
  Fields* fields;
  G = new MomentsG(pars, grids, 0);
  GRes = new MomentsG(pars, grids, 0);
  fields = new Fields(pars, grids);

  // set initial conditions
  G->initialConditions();

  // set phi = n
  CP_ON_GPU(fields->phi, G->Gm(0), sizeof(cuComplex)*grids->NxNycNz);

  // evaluate nonlinear term
  GRes->set_zero();

  int counter = 0;           float timer = 0;          cudaEvent_t start, stop;
  cudaEventCreate(&start);   cudaEventCreate(&stop);   cudaEventRecord(start,0);
  for(int i=0; i<NLOOP; i++) {
    nonlinear->nlps(G, fields, GRes);
  }
  cudaEventRecord(stop,0);    cudaEventSynchronize(stop);    cudaEventElapsedTime(&timer,start,stop);
  printf("Avg time for nlps (EM) = %e s\n", timer/1000./NLOOP);

  delete G;
  delete GRes;
  delete fields;
  delete nonlinear;
}

TEST_F(TestNonlinear, get_max_frequency) {
  Nonlinear *nonlinear;
  nonlinear = new Nonlinear_GK(pars, grids, geo);

  Fields* fields;
  fields = new Fields(pars, grids);

  double omega_max[3];
  int counter = 0;           float timer = 0;          cudaEvent_t start, stop;
  cudaEventCreate(&start);   cudaEventCreate(&stop);   cudaEventRecord(start,0);
  for(int i=0; i<NLOOP; i++) {
    nonlinear->get_max_frequency(fields, omega_max);
  }
  cudaEventRecord(stop,0);    cudaEventSynchronize(stop);    cudaEventElapsedTime(&timer,start,stop);
  printf("Avg time for get_max_frequency = %e s\n", timer/1000./NLOOP);
  delete fields;
  delete nonlinear;
}

TEST_F(TestNonlinear, sync_nl_overlap_ES) {
  Nonlinear *nonlinear;
  pars->beta = 0.0; // electrostatic limit
  pars->fapar = 0.0;
  pars->fbpar = 0.0;
  nonlinear = new Nonlinear_GK(pars, grids, geo);

  MomentsG *G, *GRes;
  Fields* fields;
  G = new MomentsG(pars, grids, 0);
  GRes = new MomentsG(pars, grids, 0);
  fields = new Fields(pars, grids);

  // set initial conditions
  G->initialConditions();

  // set phi = n
  CP_ON_GPU(fields->phi, G->Gm(0), sizeof(cuComplex)*grids->NxNycNz);
  
  GRes->set_zero();
  double omega_max[3];

  int counter = 0;           float timer = 0;          cudaEvent_t start, stop;
  cudaDeviceSynchronize();
  cudaEventCreate(&start);   cudaEventCreate(&stop);   cudaEventRecord(start,0);
  for(int i=0; i<NLOOP; i++) {
    G->sync();

    // compute timestep
    nonlinear->get_max_frequency(fields, omega_max);

    GRes->set_zero();
    nonlinear->nlps (G, fields, GRes);

    cudaStreamSynchronize(G->syncStream);
  }
  cudaEventRecord(stop,0);    cudaEventSynchronize(stop);    cudaEventElapsedTime(&timer,start,stop);
  printf("Avg time for sync+NL (ES) = %e s\n", timer/1000./NLOOP);

  delete G;
  delete GRes;
  delete fields;
  delete nonlinear;
}

TEST_F(TestNonlinear, sync_nl_overlap_EM) {
  Nonlinear *nonlinear;
  nonlinear = new Nonlinear_GK(pars, grids, geo);

  MomentsG *G, *GRes;
  Fields* fields;
  G = new MomentsG(pars, grids, 0);
  GRes = new MomentsG(pars, grids, 0);
  fields = new Fields(pars, grids);

  // set initial conditions
  G->initialConditions();

  // set phi = n
  CP_ON_GPU(fields->phi, G->Gm(0), sizeof(cuComplex)*grids->NxNycNz);
  
  GRes->set_zero();
  double omega_max[3];

  int counter = 0;           float timer = 0;          cudaEvent_t start, stop;
  cudaEventCreate(&start);   cudaEventCreate(&stop);   cudaEventRecord(start,0);
  for(int i=0; i<NLOOP; i++) {
    G->sync();

    // compute timestep
    nonlinear->get_max_frequency(fields, omega_max);

    GRes->set_zero();
    nonlinear->nlps (G, fields, GRes);

    cudaStreamSynchronize(G->syncStream);
  }
  cudaEventRecord(stop,0);    cudaEventSynchronize(stop);    cudaEventElapsedTime(&timer,start,stop);
  printf("Avg time for sync+NL (EM) = %e s\n", timer/1000./NLOOP);

  delete G;
  delete GRes;
  delete fields;
  delete nonlinear;
}
