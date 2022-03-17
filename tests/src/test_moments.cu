#include "gtest/gtest.h"
#include "device_test.h"
#include "parameters.h"
#include "grids.h"
#include "moments.h"
#include "device_funcs.h"
#include "diagnostics.h"

class TestMomentsG : public ::testing::Test {

protected:
  virtual void SetUp() {
    pars = new Parameters;
    pars->nx_in = 32;
    pars->ny_in = 32;
    pars->nz_in = 32;
    pars->nperiod = 1;
    pars->nspec_in = 1;
    pars->nm_in = 4;
    pars->nl_in = 2;
    pars->Zp = 1.;
    pars->x0 = 10.;
    pars->y0 = 10.;

    grids = new Grids(pars);
    G = new MomentsG(grids);
    geo = new S_alpha_geo(pars,grids);
  }

  virtual void TearDown() {
    delete grids;
    delete G;
    delete pars;
    delete geo;
  }

  Parameters* pars;
  Grids *grids;
  MomentsG *G;
  Geometry* geo;
};

TEST_F(TestMomentsG, InitConditions) {

  pars->init_single = false;
  pars->init_amp = .01;
  pars->kpar_init = 2.;

  float* init_check = (float*) malloc(sizeof(float)*grids->NxNycNz);

  srand(22);
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      float ra = (float) (pars->init_amp * (rand()-RAND_MAX/2) / RAND_MAX);
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        init_check[index] = ra*cos(2.*geo->z_h[k]);
      }
    }
  }
  // reality condition
  for(int j=0; j<grids->Nx/2; j++) {
    for(int k=0; k<grids->Nz; k++) {
      int index = 0 + (grids->Ny/2+1)*j + grids->Nx*(grids->Ny/2+1)*k;
      int index2 = 0 + (grids->Ny/2+1)*(grids->Nx-j) + grids->Nx*(grids->Ny/2+1)*k;
      if(j!=0) init_check[index2] = init_check[index];
    }
  }

  // initialize upar
  pars->init = UPAR;
  G->initialConditions(pars, geo);

  // check initial condition
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&G->upar_ptr[0][index].x, init_check[index]);
      }
    }
  }

  // initialize dens
  pars->init = DENS;
  G->initialConditions(pars, geo);

  // check initial condition
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&G->dens_ptr[0][index].x, init_check[index]);
        EXPECT_FLOAT_EQ_D(&G->upar_ptr[0][index].x, init_check[index]);
        EXPECT_FLOAT_EQ_D(&G->tpar_ptr[0][index].x, 0.);
        EXPECT_FLOAT_EQ_D(&G->G(0,3)[index].x, 0.);
      }
    }
  }
 
  free(init_check);
}

TEST_F(TestMomentsG, AddMomentsG) 
{
  pars->init_single = false;
  pars->init_amp = .01;
  pars->kpar_init = 2.;
  float* init_check = (float*) malloc(sizeof(float)*grids->NxNycNz);
  srand(22);
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      float ra = (float) (pars->init_amp * (rand()-RAND_MAX/2) / RAND_MAX);
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        init_check[index] = ra*cos(2.*geo->z_h[k]);
      }
    }
  }
  // reality condition
  for(int j=0; j<grids->Nx/2; j++) {
    for(int k=0; k<grids->Nz; k++) {
      int index = 0 + (grids->Ny/2+1)*j + grids->Nx*(grids->Ny/2+1)*k;
      int index2 = 0 + (grids->Ny/2+1)*(grids->Nx-j) + grids->Nx*(grids->Ny/2+1)*k;
      if(j!=0) init_check[index2] = init_check[index];
    }
  }
  // initialize upar
  pars->init = UPAR;
  G->initialConditions(pars, geo);
  // G = init(z) * upar

  MomentsG* G2;
  G2 = new MomentsG(grids);
  pars->init = DENS;
  G2->initialConditions(pars, geo);
  // G2 = init(z) * dens

  G->add_scaled(1., G, 2., G2);
  // G = init(z) * ( 2*dens + upar )

  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&G->dens_ptr[0][index].x, 2.*init_check[index]);
        EXPECT_FLOAT_EQ_D(&G->upar_ptr[0][index].x, init_check[index]);
        EXPECT_FLOAT_EQ_D(&G->tpar_ptr[0][index].x, 0.);
        EXPECT_FLOAT_EQ_D(&G->G(0,3)[index].x, 0.);
      }
    }
  }
  
  pars->init = UPAR;
  G2->initialConditions(pars, geo);
  // G2 = init(z) * ( dens + upar )
  G->add_scaled(1., G, 2., G2);
  // G = init(z) * ( 4*dens + 3*upar )

  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&G->dens_ptr[0][index].x, 4.*init_check[index]);
        EXPECT_FLOAT_EQ_D(&G->upar_ptr[0][index].x, 3.*init_check[index]);
        EXPECT_FLOAT_EQ_D(&G->tpar_ptr[0][index].x, 0.);
        EXPECT_FLOAT_EQ_D(&G->G(1,3)[index].x, 0.);
      }
    }
  }

  /////////  single moment addition /////////

  // 1d thread blocks over xyz
  dim3 dimBlock = 512;
  dim3 dimGrid = grids->NxNycNz/dimBlock.x+1;
  add_scaled_singlemom_kernel<<<dimGrid,dimBlock>>>
      (G->G(0,3), 1., G->G(0,1), 1., G->G(0,0));
  // qpar = 7*init(z)
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&G->G(0,3)[index].x, 7.*init_check[index]);
      }
    }
  }
}
