#include "gtest/gtest.h"
#include "device_test.h"
#include "parameters.h"
#include "grids.h"
#include "grad_parallel.h"
#include "moments.h"
#include "diagnostics.h"
#include "device_funcs.h"
#include "diagnostics.h"

class TestGradParallelLinked3D_nLink1 : public ::testing::Test {
protected:
  virtual void SetUp() {
    pars = new Parameters;
    pars->nx_in = 1;
    pars->ny_in = 16;
    pars->nz_in = 16;
    pars->nperiod = 1;
    pars->nspec_in = 1;
    pars->nm_in = 4;
    pars->nl_in = 2;
    pars->Zp = 1.;
    pars->x0 = 10.;
    pars->y0 = 10.;
    int jtwist = 5;

    grids = new Grids(pars);
    grad_par = new GradParallelLinked(grids, jtwist);
  }

  virtual void TearDown() {
    delete grids;
    delete grad_par;
    delete pars;
  }

  Parameters* pars;
  Grids *grids;
  GradParallelLinked *grad_par;
};

class TestGradParallelLinked3D : public ::testing::Test {
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
    int jtwist = 5;

    grids = new Grids(pars);
    grad_par = new GradParallelLinked(grids, jtwist);
  }

  virtual void TearDown() {
    delete grids;
    delete grad_par;
    delete pars;
  }

  Parameters* pars;
  Grids *grids;
  GradParallelLinked *grad_par;
};

TEST_F(TestGradParallelLinked3D_nLink1, linkPrint) {
  grad_par->linkPrint();
}
TEST_F(TestGradParallelLinked3D, linkPrint) {
  grad_par->linkPrint();
}

TEST_F(TestGradParallelLinked3D_nLink1, identity) {
  Geometry* geo;
  geo = new S_alpha_geo(pars, grids);

  MomentsG *G, *G2;
  G = new MomentsG(grids);
  G2 = new MomentsG(grids);
  pars->init_single = false;
  pars->init = UPAR;
  pars->init_amp = .01;
  pars->kpar_init = 2.;
  pars->linear = true;
  G->initialConditions(pars, geo);
  G->reality();
  G2->copyFrom(G);

  grad_par->identity(G);
  for(int index=0; index<grids->Nx*grids->Nyc*grids->Nz*grids->Nmoms; index++) {
    EXPECT_FLOAT_EQ_D(&G->G()[index].x, &G2->G()[index].x, 1.e-7);
    EXPECT_FLOAT_EQ_D(&G->G()[index].y, &G2->G()[index].y, 1.e-7);
  }
  
}
TEST_F(TestGradParallelLinked3D, identity) {
  Geometry* geo;
  geo = new S_alpha_geo(pars, grids);

  MomentsG *G, *G2;
  G = new MomentsG(grids);
  G2 = new MomentsG(grids);
  pars->init_single = false;
  pars->init = UPAR;
  pars->init_amp = .01;
  pars->kpar_init = 2.;
  pars->linear = true;
  G->initialConditions(pars, geo);
  G->reality();
  G2->copyFrom(G);

  grad_par->identity(G);
  for(int index=0; index<grids->Nx*grids->Nyc*grids->Nz*grids->Nmoms; index++) {
    EXPECT_FLOAT_EQ_D(&G->G()[index].x, &G2->G()[index].x, 1.e-7);
    EXPECT_FLOAT_EQ_D(&G->G()[index].y, &G2->G()[index].y, 1.e-7);
  }
  
}

TEST_F(TestGradParallelLinked3D_nLink1, EvaluateDerivative) {
  Geometry* geo;
  geo = new S_alpha_geo(pars, grids);
  strncpy(pars->run_name, "gpar_test", strlen("gpar_test"));

  MomentsG *GInit, *GRes;
  GInit = new MomentsG(grids);
  GRes = new MomentsG(grids);
  pars->init_single = false;
  pars->init = UPAR;
  pars->init_amp = .01;
  pars->kpar_init = 2.;
  pars->linear = true;
  GInit->initialConditions(pars, geo);

  Diagnostics* diagnostics;
  diagnostics = new Diagnostics(pars, grids, geo);

  //diagnostics->writeMomOrField(GInit->upar_ptr[0],"upar0_nlink1");

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
  //for(int i=0; i<((Nx-1)/3+1); i++) {
  //  for(int j=0; j<grids->Naky; j++) {
  //    for(int k=0; k<grids->Nz; k++) {
  //      int index = j + grids->Nyc*i + grids->NxNyc*k;
  //      EXPECT_FLOAT_EQ_D(&GInit->upar_ptr[0][index].x, deriv_check[index], 1.e-7);
  //    }
  //  }
  //}
  for(int i=0; i<grids->Nx; i++) {
    for(int j=0; j<grids->Nyc; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = j + grids->Nyc*i + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&GInit->upar_ptr[0][index].x, deriv_check[index], 1.e-7);
      }
    }
  }

  //diagnostics->writeMomOrField(GInit->upar_ptr[0], "upar_nlink1");

  free(init_check);
  free(deriv_check);
  delete GInit;
  delete GRes;
  delete geo;
}

//TEST_F(TestGradParallelLinked3D, EvaluateDerivative) {
//  Geometry* geo;
//  geo = new S_alpha_geo(pars, grids);
//  strncpy(pars->run_name, "gpar_test", strlen("gpar_test"));
//
//  MomentsG *GInit, *GRes;
//  GInit = new MomentsG(grids);
//  GRes = new MomentsG(grids);
//  pars->init_single = false;
//  pars->init = UPAR;
//  pars->init_amp = .01;
//  pars->kpar_init = 2.;
//  pars->linear = false;
//  GInit->initialConditions(pars, geo);
//
//  Diagnostics* diagnostics;
//  diagnostics = new Diagnostics(pars, grids, geo);
//
//  diagnostics->writeMomOrField(GInit->upar_ptr[0],"upar0");
//
//  float* init_check = (float*) malloc(sizeof(float)*grids->NxNycNz);
//  float* deriv_check = (float*) malloc(sizeof(float)*grids->NxNycNz);
//
//  srand(22);
//  for(int i=0; i<grids->Nyc; i++) {
//    for(int j=0; j<grids->Nx; j++) {
//      float ra = (float) (pars->init_amp * (rand()-RAND_MAX/2) / RAND_MAX);
//      for(int k=0; k<grids->Nz; k++) {
//        int index = i + grids->Nyc*j + grids->NxNyc*k;
//        init_check[index] = ra*cos(2.*geo->z_h[k]);
//        deriv_check[index] = -2.*ra*sin(2.*geo->z_h[k]);
//      }
//    }
//  }
//  // reality condition
//  for(int j=0; j<grids->Nx/2; j++) {
//    for(int k=0; k<grids->Nz; k++) {
//      int index = 0 + (grids->Ny/2+1)*j + grids->Nx*(grids->Ny/2+1)*k;
//      int index2 = 0 + (grids->Ny/2+1)*(grids->Nx-j) + grids->Nx*(grids->Ny/2+1)*k;
//      if(j!=0) init_check[index2] = init_check[index];
//      if(j!=0) deriv_check[index2] = deriv_check[index];
//    }
//  }
//
//  // check initial condition
//  for(int i=0; i<grids->Nyc; i++) {
//    for(int j=0; j<grids->Nx; j++) {
//      for(int k=0; k<grids->Nz; k++) {
//        int index = i + grids->Nyc*j + grids->NxNyc*k;
//        EXPECT_FLOAT_EQ_D(&GInit->upar_ptr[0][index].x, init_check[index]);
//      }
//    }
//  }
//
//  // out-of-place, with only a single moment
//  grad_par->dz(GInit->upar_ptr[0], GRes->upar_ptr[0]);
//
//  for(int i=0; i<((grids->Nx-1)/3+1); i++) {
//    for(int j=0; j<grids->Naky; j++) {
//      for(int k=0; k<grids->Nz; k++) {
//        int index = j + grids->Nyc*i + grids->NxNyc*k;
//        EXPECT_FLOAT_EQ_D(&GInit->upar_ptr[0][index].x, deriv_check[index], 1.e-7);
//      }
//    }
//  }
//  for(int i=2*grids->Nx/3+1; i<grids->Nx; i++) {
//    for(int j=0; j<grids->Naky; j++) {
//      for(int k=0; k<grids->Nz; k++) {
//        int index = j + grids->Nyc*i + grids->NxNyc*k;
//        EXPECT_FLOAT_EQ_D(&GInit->upar_ptr[0][index].x, deriv_check[index], 1.e-7);
//      }
//    }
//  }
//
//  // in place, with entire LH G array
//  grad_par->dz(GInit);
//  for(int i=0; i<((grids->Nx-1)/3+1); i++) {
//    for(int j=0; j<grids->Naky; j++) {
//      for(int k=0; k<grids->Nz; k++) {
//        int index = j + grids->Nyc*i + grids->NxNyc*k;
//        EXPECT_FLOAT_EQ_D(&GInit->upar_ptr[0][index].x, deriv_check[index], 1.e-7);
//      }
//    }
//  }
//  for(int i=2*grids->Nx/3+1; i<grids->Nx; i++) {
//    for(int j=0; j<grids->Naky; j++) {
//      for(int k=0; k<grids->Nz; k++) {
//        int index = j + grids->Nyc*i + grids->NxNyc*k;
//        EXPECT_FLOAT_EQ_D(&GInit->upar_ptr[0][index].x, deriv_check[index], 1.e-7);
//      }
//    }
//  }
//
//  diagnostics->writeMomOrField(GInit->upar_ptr[0], "upar");
//
//  free(init_check);
//  free(deriv_check);
//  delete GInit;
//  delete GRes;
//  delete geo;
//}
