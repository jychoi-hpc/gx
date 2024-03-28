#include "gtest/gtest.h"
#include "device_test.h"
#include "parameters.h"
#include "grids.h"
#include "grad_parallel.h"
#include "moments.h"
#include "diagnostics.h"
#include "device_funcs.h"
#include "diagnostics.h"

class TestGradParallelPeriodic1D : public ::testing::Test {
protected:
  virtual void SetUp() {
    pars = new Parameters;
    pars->nx_in = 1;
    pars->ny_in = 1;
    pars->nz_in = 32;
    pars->nperiod = 2;
    pars->nspec_in = 1;
    pars->nm_in = 4;
    pars->nl_in = 2;
    pars->Zp = 1.;
    pars->x0 = 10.;
    pars->y0 = 10.;

    grids = new Grids(pars);
    grad_par = new GradParallelPeriodic(grids);
  }

  virtual void TearDown() {
    delete grids;
    delete grad_par;
    delete pars;
  }

  Parameters* pars;
  Grids *grids;
  GradParallel *grad_par;
};

class TestGradParallelPeriodic3D : public ::testing::Test {
protected:
  virtual void SetUp() {
    pars = new Parameters;
    pars->nx_in = 16;
    pars->ny_in = 16;
    pars->nz_in = 16;
    pars->nperiod = 1;
    pars->nspec_in = 1;
    pars->nm_in = 4;
    pars->nl_in = 2;
    pars->Zp = 1.;
    pars->x0 = 10.;
    pars->y0 = 10.;

    grids = new Grids(pars);
    grad_par = new GradParallelPeriodic(grids);
  }

  virtual void TearDown() {
    delete grids;
    delete grad_par;
    delete pars;
  }

  Parameters* pars;
  Grids *grids;
  GradParallel *grad_par;
};

TEST_F(TestGradParallelPeriodic1D, EvaluateDerivative) {
  Geometry* geo;
  geo = new S_alpha_geo(pars, grids);
  
  MomentsG* GInit, *GRes;
  GInit = new MomentsG(grids);
  GRes = new MomentsG(grids);
  pars->init_single = false;
  pars->init = DENS;
  pars->init_amp = .01;
  pars->kpar_init = 2.;
  GInit->initialConditions(pars, geo);

  float* init_check = (float*) malloc(sizeof(float)*grids->Nz);
  float* deriv_check = (float*) malloc(sizeof(float)*grids->Nz);

  srand(22);
  float ra = (float) (pars->init_amp *(rand()-RAND_MAX/2)/RAND_MAX);
  for(int i=0; i<grids->Nz; i++) {
    init_check[i] = ra*cos(2.*geo->z_h[i]);
    deriv_check[i] = -2.*ra*sin(2.*geo->z_h[i]);
  }

  // check initial condition
  for(int i=0; i<grids->Nz; i++) {
    EXPECT_FLOAT_EQ_D(&GInit->dens_ptr[0][i].x, init_check[i]);
  }

  // out-of-place, with only a single moment
  grad_par->dz(GInit->dens_ptr[0], GRes->dens_ptr[0]);
  
  for(int i=0; i<grids->Nz; i++) {
    EXPECT_FLOAT_EQ_D(&GRes->dens_ptr[0][i].x, deriv_check[i], 1.e-7);
  }

  // in place, with entire HL moms array
  grad_par->dz(GInit);
  for(int i=0; i<grids->Nz; i++) {
    EXPECT_FLOAT_EQ_D(&GInit->dens_ptr[0][i].x, deriv_check[i], 1.e-7);
  }

  free(init_check);
  free(deriv_check);
  delete GInit;
  delete GRes;
  delete geo;
}

TEST_F(TestGradParallelPeriodic3D, EvaluateDerivative) {
  Geometry* geo;
  geo = new S_alpha_geo(pars, grids);
  //strncpy(pars->run_name, "test", strlen("test")-1);
  //Diagnostics* diagnostics;
  //diagnostics = new Diagnostics(pars, grids, geo);

  MomentsG *GInit, *GRes;
  GInit = new MomentsG(grids);
  GRes = new MomentsG(grids);
  pars->init_single = false;
  pars->init = UPAR;
  pars->init_amp = .01;
  pars->kpar_init = 2.;
  pars->linear = true;
  strncpy(pars->run_name, "gpar_test", strlen("gpar_test"));
  GInit->initialConditions(pars, geo);

  float* init_check = (float*) malloc(sizeof(float)*grids->NxNycNz);
  float* deriv_check = (float*) malloc(sizeof(float)*grids->NxNycNz);

  srand(22);
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      float ra = (float) (pars->init_amp * (rand()-RAND_MAX/2) / RAND_MAX);
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        init_check[index] = ra*cos(2.*geo->z_h[k]);
        deriv_check[index] = -2.*ra*sin(2.*geo->z_h[k]);
      }
    }
  }
  // reality condition
  for(int j=0; j<grids->Nx/2; j++) {
    for(int k=0; k<grids->Nz; k++) {
      int index = 0 + (grids->Ny/2+1)*j + grids->Nx*(grids->Ny/2+1)*k;
      int index2 = 0 + (grids->Ny/2+1)*(grids->Nx-j) + grids->Nx*(grids->Ny/2+1)*k;
      if(j!=0) init_check[index2] = init_check[index];
      if(j!=0) deriv_check[index2] = deriv_check[index];
    }
  }

  // check initial condition
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&GInit->upar_ptr[0][index].x, init_check[index]);
      }
    }
  }

  // out-of-place, with only a single moment
  grad_par->dz(GInit->upar_ptr[0], GRes->upar_ptr[0]);

  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&GRes->upar_ptr[0][index].x, deriv_check[index], 1.e-7);
      }
    }
  }

  // in place, with entire LH G array
  grad_par->dz(GInit);
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&GInit->upar_ptr[0][index].x, deriv_check[index], 1.e-7);
      }
    }
  }
  free(init_check);
  free(deriv_check);
  delete GInit;
  delete GRes;
  delete geo;
}
