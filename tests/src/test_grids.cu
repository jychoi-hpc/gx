#include "gtest/gtest.h"
#include "device_test.h"
#include "parameters.h"
#include "grids.h"
#include "device_funcs.h"

class TestGrids : public ::testing::Test {
protected:
  virtual void SetUp() {
    Parameters* pars = new Parameters;
    checkCuda(cudaGetLastError());
    pars->nx_in = 20;
    pars->ny_in = 48;
    pars->nz_in = 32;
    pars->nspec_in = 1;
    pars->nm_in = 4;
    pars->nl_in = 2;
    pars->Zp = 1.;
    pars->x0 = 10.;
    pars->y0 = 10.;

    grids = new Grids(pars);
    checkCuda(cudaGetLastError());
  }

  virtual void TearDown() {
    delete grids;
  }

  Grids *grids;
};

TEST_F(TestGrids, Dimensions) {

  EXPECT_EQ(grids->Nx, 20);
  EXPECT_EQ(grids->Ny, 48);
  EXPECT_EQ(grids->Nyc, 25);
  EXPECT_EQ(grids->Nz, 32);
  EXPECT_EQ(grids->NxNycNz, 20*25*32);
  
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

//TEST_F(TestGrids, DeviceConstants) {
//  int Nx, Nyc, Nz;
//  cudaMemcpyFromSymbol(&Nx, nx, sizeof(int), 0, cudaMemcpyDeviceToHost);
//  cudaMemcpyFromSymbol(&Nyc, nyc, sizeof(int), 0, cudaMemcpyDeviceToHost);
//  cudaMemcpyFromSymbol(&Nz, nz, sizeof(int), 0, cudaMemcpyDeviceToHost);
//
//  EXPECT_EQ(Nx, 20);
//  EXPECT_EQ(Nyc, 25);
//  EXPECT_EQ(Nz, 32);
//}
