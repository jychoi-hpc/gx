#include "gtest/gtest.h"
#include "device_test.h"
#include "parameters.h"
#include "grids.h"
#include "grad_perp.h"
#include "moments.h"
#include "diagnostics.h"
#include "device_funcs.h"
#include "diagnostics.h"

class TestGradPerp : public ::testing::Test {
protected:
  virtual void SetUp() {
    pars = new Parameters;
    pars->nx_in = 32;
    pars->ny_in = 32;
    pars->nz_in = 1;
    pars->nperiod = 1;
    pars->nspec_in = 1;
    pars->nm_in = 1;
    pars->nl_in = 1;
    pars->Zp = 1.;
    pars->x0 = 10.;
    pars->y0 = 10.;

    grids = new Grids(pars);
    grad_perp = new GradPerp(grids, grids->Nz*grids->Nl, grids->NxNycNz*grids->Nl);
  }

  virtual void TearDown() {
    delete grids;
    delete grad_perp;
    delete pars;
  }

  Parameters* pars;
  Grids *grids;
  GradPerp *grad_perp;
};


TEST_F(TestGradPerp, EvaluateDerivative) {

  float* init_h = (float*) malloc(sizeof(float)*grids->NxNyNz*grids->Nl);
  float* dxcheck = (float*) malloc(sizeof(float)*grids->NxNyNz*grids->Nl);
  float* dycheck = (float*) malloc(sizeof(float)*grids->NxNyNz*grids->Nl);

  float *init, *dx, *dy;
  cuComplex* comp; 
  cudaMalloc((void**) &init, sizeof(float)*grids->NxNyNz*grids->Nl);
  cudaMalloc((void**) &dx, sizeof(float)*grids->NxNyNz*grids->Nl);
  cudaMalloc((void**) &dy, sizeof(float)*grids->NxNyNz*grids->Nl);
  cudaMalloc((void**) &comp, sizeof(cuComplex)*grids->NxNycNz*grids->Nl);

  float kx = 0.4;
  float ky = 0.2;

  srand(22);
  for(int idz=0; idz<grids->Nz; idz++) {
    for(int idl=0; idl<grids->Nl; idl++) {
      float ra = (float) (rand()-RAND_MAX/2)/RAND_MAX;
      for(int idx=0; idx<grids->Nx; idx++) {
        for(int idy=0; idy<grids->Ny; idy++) {
          int globalIdx = idy + grids->Ny*idx + grids->Nx*grids->Ny*idz + grids->NxNyNz*idl;
          float x = pars->x0*2.*M_PI*(float)(idx-grids->Nx/2)/grids->Nx;
          float y = pars->y0*2.*M_PI*(float)(idy-grids->Ny/2)/grids->Ny;
          init_h[globalIdx] = ra*sin(kx*x + ky*y);
          dxcheck[globalIdx] = kx*ra*cos(kx*x + ky*y);
          dycheck[globalIdx] = ky*ra*cos(kx*x + ky*y);
        }
      }
    }
  }
  cudaMemcpy(init, init_h, sizeof(float)*grids->NxNyNz*grids->Nl, cudaMemcpyHostToDevice);
          
  grad_perp->R2C(init, comp, false);
  grad_perp->dxC2R(comp, dx);
  grad_perp->dyC2R(comp, dy);

  printf("Checking...\n");

  for(int idz=0; idz<grids->Nz; idz++) {
    for(int idl=0; idl<grids->Nl; idl++) {
      for(int idx=0; idx<grids->Nx; idx++) {
        for(int idy=0; idy<grids->Ny; idy++) {
          int globalIdx = idy + grids->Ny*idx + grids->Nx*grids->Ny*idz + grids->NxNyNz*idl;
          EXPECT_FLOAT_EQ_D(&dx[globalIdx], dxcheck[globalIdx], 1.e-6);
          EXPECT_FLOAT_EQ_D(&dy[globalIdx], dycheck[globalIdx], 1.e-6);
        }
      }
    }
  }

  free(init_h);
  free(dxcheck);
  free(dycheck);
  cudaFree(init);
  cudaFree(dx);
  cudaFree(dy);
  cudaFree(comp);
}
