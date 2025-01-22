#include "gtest/gtest.h"
#include "device_test.h"
#include "parameters.h"
#include "grids.h"
#include "device_funcs.h"
#include "mpi.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  RUN_ALL_TESTS();
  MPI_Finalize();
}

class TestGrids : public ::testing::Test {
protected:
  virtual void SetUp() {
    char** argv;
    int argc = 0;
    MPI_Comm mpcom = MPI_COMM_WORLD;
    MPI_Comm_rank(mpcom, &iproc);
    MPI_Comm_size(mpcom, &nprocs);

    int devid = 0; // This should be determined (optionally) on the command line
    checkCuda(cudaSetDevice(devid));
    cudaDeviceSynchronize();
    pars = new Parameters(iproc, nprocs, mpcom);
    pars->get_nml_vars("inputs/cyc_nl");
    checkCuda(cudaGetLastError());
//    pars->nx_in = 20;
//    pars->ny_in = 48;
//    pars->nz_in = 32;
//    pars->nspec_in = 2;
//    pars->nm_in = 4;
//    pars->nl_in = 2;
    pars->Zp = 1.;
    pars->x0 = 10.;
    pars->y0 = 10.;

    grids = new Grids(pars);
    grids->init_ks_and_coords();
    checkCuda(cudaGetLastError());
  }

  virtual void TearDown() {
    delete grids;
    delete pars;
    cudaDeviceReset();
  }

  Grids *grids;
  Parameters* pars;
  int iproc, nprocs;
};

TEST_F(TestGrids, Dimensions) {

  EXPECT_EQ(grids->Nx, 256);
  EXPECT_EQ(grids->Ny, 256);
  EXPECT_EQ(grids->Nyc, 129);
  EXPECT_EQ(grids->Nz, 64);
  EXPECT_EQ(grids->NxNycNz, 256*129*64);
  
}

TEST_F(TestGrids, Ks_on_host) {
 for(int i=0; i<grids->Nyc; i++) {
   EXPECT_FLOAT_EQ(grids->ky_h[i], i/10.);
 } 
 for(int i=0; i<grids->Nx; i++) {
   if(i<grids->Nx/2+1) {
     EXPECT_FLOAT_EQ(grids->kx_h[i], i/10.);
   } else {
     EXPECT_FLOAT_EQ(grids->kx_h[i], (i-grids->Nx)/10.);
   }
 } 
}


TEST_F(TestGrids, Ks_on_device) {
 for(int i=0; i<grids->Nyc; i++) {
   EXPECT_FLOAT_EQ_D(&grids->ky[i], i/10.);
 } 
 for(int i=0; i<grids->Nx; i++) {
   if(i<grids->Nx/2+1) {
     EXPECT_FLOAT_EQ_D(&grids->kx[i], i/10.);
   } else {
     EXPECT_FLOAT_EQ_D(&grids->kx[i], (i-grids->Nx)/10.);
   }
 } 
}

TEST_F(TestGrids, ParTest) {
  if(nprocs<=pars->nspec_in) {
    EXPECT_EQ(grids->Nspecies, pars->nspec_in/nprocs);
    EXPECT_EQ(grids->Nm, pars->nm_in);
  } else {
    EXPECT_EQ(grids->Nspecies, 1);
    int nprocs_m = nprocs/pars->nspec_in;
    EXPECT_EQ(grids->m_ghost, 2);
    EXPECT_EQ(grids->Nm, pars->nm_in/nprocs_m);
  } 
}

int get_ikx(int idx, int nx) {
  if (idx < nx/2+1)
    return idx;
  else
    return idx-nx;
}

bool unmasked(int idx, int idy, int nx, int ny) {
  int ikx = get_ikx(idx, nx);
  if ( !(idx==0 && idy==0)
       && idy <  (ny-1)/3 + 1
       && idx <   nx                 // both indices must be in range 
       && ikx <  (nx-1)/3 + 1
       && ikx > -(nx-1)/3 - 1)
    return true;
  else
    return false;
}

TEST_F(TestGrids, MaskTest) {
  int count = 1; // account for masked (0,0) mode
  int nshift = grids->Nx - grids->Nakx;
  for(int i=0; i<grids->Nx; i++) {
    for(int j=0; j<grids->Nyc; j++) {
      if (unmasked(i, j, grids->Nx, grids->Ny)) {
	count++;

	int akx = i;
	if (i >= (grids->Nakx+1)/2) {
	  akx = i - nshift;
	}
	int index = j + grids->Naky*akx;
	EXPECT_TRUE(index < grids->Naky*grids->Nakx);
      }
    }
  }
  EXPECT_EQ(grids->Naky*grids->Nakx, count);
}

//TEST_F(TestGrids, DeviceConstants) {
//  int Nx, Nyc, Nz;
//  cudaMemcpyFromSymbol(&Nx, GPU_SYMBOL(nx), sizeof(int), 0, cudaMemcpyDeviceToHost);
//  cudaMemcpyFromSymbol(&Nyc, GPU_SYMBOL(nyc), sizeof(int), 0, cudaMemcpyDeviceToHost);
//  cudaMemcpyFromSymbol(&Nz, GPU_SYMBOL(nz), sizeof(int), 0, cudaMemcpyDeviceToHost);
//
//  EXPECT_EQ(Nx, 20);
//  EXPECT_EQ(Nyc, 25);
//  EXPECT_EQ(Nz, 32);
//}

