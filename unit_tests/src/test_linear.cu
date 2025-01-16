#include "gtest/gtest.h"
#include "parameters.h"
#include "grids.h"
#include "moments.h"
#include "fields.h"
#include "geometry.h"
#include "linear.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  RUN_ALL_TESTS();
  MPI_Finalize();
}

class TestLinear : public ::testing::Test {

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
    linear = new Linear_GK(pars, grids, geo);
  }

  virtual void TearDown() {
    delete pars;
    delete grids;
    delete geo;
    delete linear;
  }

  Parameters* pars;
  Grids *grids;
  Geometry* geo;
  Linear* linear;
  int iproc, nprocs;
};


TEST_F(TestLinear, rhs) {
  MomentsG *G, *GRes;
  Fields* fields;
  G = new MomentsG(pars, grids, 0);
  GRes = new MomentsG(pars, grids, 0);
  fields = new Fields(pars, grids);

  // set initial conditions
  G->initialConditions();

  // set phi = n
  CP_ON_GPU(fields->phi, G->Gm(0), sizeof(cuComplex)*grids->NxNycNz);

  // evaluate linear term
  GRes->set_zero();

  int NLOOP = 100;
  int counter = 0;           float timer = 0;          cudaEvent_t start, stop;
  cudaEventCreate(&start);   cudaEventCreate(&stop);   cudaEventRecord(start,0);
  for(int i=0; i<NLOOP; i++) {
    linear->rhs(G, fields, GRes, 0.1);
  }
  cudaEventRecord(stop,0);    cudaEventSynchronize(stop);    cudaEventElapsedTime(&timer,start,stop);
  printf("Avg time for linear rhs = %e s\n", timer/1000./NLOOP);

  delete G;
  delete GRes;
  delete fields;
}

