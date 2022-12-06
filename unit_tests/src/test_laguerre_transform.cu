#include "laguerre_transform.h"
#include "gtest/gtest.h"
#include "device_test.h"
#include "parameters.h"
#include "grids.h"
#include "get_error.h"

class TestLaguerreTransform : public ::testing::Test {
protected:
  virtual void SetUp() {
    pars = new Parameters;
    pars->nx_in = 4;
    pars->ny_in = 4;
    pars->nz_in = 4;
    pars->nspec_in = 1;
    pars->nm_in = 4;
    pars->nl_in = 9; // leave this value
    pars->Zp = 1.;
    pars->x0 = 10.;
    pars->y0 = 10.;

    grids = new Grids(pars);
    laguerre = new LaguerreTransform(grids,1);
    L = grids->Nl - 1;
    J = (3*L-1)/2;
  }

  virtual void TearDown() {
    delete grids;
    delete laguerre;
    delete pars;
  }

  Parameters *pars;
  Grids *grids;
  LaguerreTransform *laguerre;
  int L, J;
};

TEST_F(TestLaguerreTransform, toGrid) {
  float toGrid_correct[108] = {
   1.00000000e+00,  -8.84277880e-01,   7.75251567e-01,  -6.72662795e-01,
   5.76260686e-01,  -4.85801816e-01,   4.01049733e-01,  -3.21775049e-01,
   2.47755125e-01,   1.00000000e+00,  -3.88242513e-01,  -3.63913588e-02,
   3.12059700e-01,  -4.71084774e-01,   5.40666878e-01,  -5.43525815e-01,
   4.98475462e-01,  -4.20938134e-01,   1.00000000e+00,   5.12610257e-01,
  -8.81225646e-01,   6.82652235e-01,  -2.75575489e-01,  -1.33453578e-01,
   4.40667242e-01,  -6.08770728e-01,   6.40757143e-01,   1.00000000e+00,
   1.83375132e+00,  -6.52429342e-01,  -7.51392841e-01,   1.27194440e+00,
  -9.67510760e-01,   2.56868631e-01,   4.56239164e-01,  -9.18599963e-01,
   1.00000000e+00,   3.59922767e+00,   2.37799215e+00,  -2.71716285e+00,
  -1.52671695e-01,   2.30810499e+00,  -2.33504939e+00,   8.23941171e-01,
   9.71965134e-01,   1.00000000e+00,   5.84452534e+00,   1.07347136e+01,
   2.70380044e+00,  -8.15612793e+00,   1.35302496e+00,   5.85969639e+00,
  -6.31248045e+00,   1.30792463e+00,   1.00000000e+00,   8.62131691e+00,
   2.80422344e+01,   3.74498062e+01,   3.51027584e+00,  -2.95236473e+01,
   3.85872912e+00,   2.34434929e+01,  -1.91382790e+01,   1.00000000e+00,
   1.20060549e+01,   5.95666237e+01,   1.50960510e+02,   1.81994324e+02,
   2.50474415e+01,  -1.43287506e+02,  -2.15931778e+01,   1.30758530e+02,
   1.00000000e+00,   1.61168556e+01,   1.13259659e+02,   4.46705719e+02,
   1.04486951e+03,   1.33884631e+03,   4.94196930e+02,  -8.56934387e+02,
  -6.59173035e+02,   1.00000000e+00,   2.11510906e+01,   2.02033218e+02,
   1.14092932e+03,   4.17005566e+03,   1.00554131e+04,   1.52130898e+04,
   1.12691260e+04,  -3.23813623e+03,   1.00000000e+00,   2.74879665e+01,
   3.49806213e+02,   2.72042017e+03,   1.43517207e+04,   5.37608359e+04,
   1.44734859e+05,   2.74154812e+05,   3.35580906e+05,   1.00000000e+00,
   3.60991211e+01,   6.14974121e+02,   6.55597705e+03,   4.88710586e+04,
   2.69401969e+05,   1.13113325e+06,   3.66327200e+06,   9.12964500e+06
  };

  for(int j = 0; j<(J+1); j++) {
    for(int l = 0; l<(L+1); l++) {
      EXPECT_FLOAT_EQ_REL_D(&laguerre->get_toGrid()[l+(L+1)*j], toGrid_correct[l+(L+1)*j], 1.e-4);
    }
  }
}

TEST_F(TestLaguerreTransform, toSpectral) {
  float toSpectral_correct[108] = {
2.6473e-01,	3.7776e-01,	2.4408e-01,	9.0449e-02,	2.0102e-02,	2.6640e-03,	2.0323e-04,	8.3651e-06,	1.6685e-07,	1.3424e-09,	3.0616e-12,	8.1481e-16,	
-2.3410e-01,	-1.4666e-01,	1.2512e-01,	1.6586e-01,	7.2353e-02,	1.5570e-02,	1.7521e-03,	1.0043e-04,	2.6891e-06,	2.8393e-08,	8.4157e-11,	2.9414e-14,	
2.0523e-01,	-1.3747e-02,	-2.1509e-01,	-5.9012e-02,	4.7803e-02,	2.8597e-02,	5.6991e-03,	4.9828e-04,	1.8897e-05,	2.7121e-07,	1.0710e-09,	5.0109e-13,	
-1.7807e-01,	1.1788e-01,	1.6662e-01,	-6.7963e-02,	-5.4621e-02,	7.2029e-03,	7.6110e-03,	1.2628e-03,	7.4533e-05,	1.5316e-06,	8.3288e-09,	5.3419e-12,	
1.5255e-01,	-1.7796e-01,	-6.7263e-02,	1.1505e-01,	-3.0691e-03,	-2.1728e-02,	7.1340e-04,	1.5224e-03,	1.7434e-04,	5.5978e-06,	4.3939e-08,	3.9821e-11,	
-1.2861e-01,	2.0424e-01,	-3.2574e-02,	-8.7511e-02,	4.6398e-02,	3.6044e-03,	-6.0001e-03,	2.0952e-04,	2.2339e-04,	1.3498e-05,	1.6459e-07,	2.1951e-10,	
1.0617e-01,	-2.0532e-01,	1.0756e-01,	2.3234e-02,	-4.6940e-02,	1.5610e-02,	7.8422e-04,	-1.1986e-03,	8.2456e-05,	2.0422e-05,	4.4312e-07,	9.2166e-10,	
-8.5184e-02,	1.8830e-01,	-1.4859e-01,	4.1266e-02,	1.6563e-02,	-1.6816e-02,	4.7645e-03,	-1.8063e-04,	-1.4298e-04,	1.5128e-05,	8.3935e-07,	2.9849e-09,	
6.5589e-02,	-1.5901e-01,	1.5640e-01,	-8.3087e-02,	1.9539e-02,	3.4843e-03,	-3.8895e-03,	1.0938e-03,	-1.0998e-04,	-4.3468e-06,	1.0274e-06,	7.4389e-09 };

  for(int j = 0; j<(J+1); j++) {
    for(int l = 0; l<(L+1); l++) {
      EXPECT_FLOAT_EQ_REL_D(&laguerre->get_toSpectral()[j+(J+1)*l], toSpectral_correct[j+(J+1)*l], 1.e-4);
    }
  }
}

TEST_F(TestLaguerreTransform, roots) {
  float roots_correct[12] = {
  1.1572e-01,	6.1176e-01,	1.5126e+00,	2.8338e+00,	4.5992e+00,	6.8445e+00,	
  9.6213e+00,	1.3006e+01,	1.7117e+01,	2.2151e+01,	2.8488e+01,	3.7099e+01
  };

  for(int j = 0; j<(J+1); j++) {
    EXPECT_FLOAT_EQ_REL_D(&laguerre->get_roots()[j], roots_correct[j], 1.e-4);
  }
}

TEST_F(TestLaguerreTransform, identity) {
  
  float *G, *g, *Gres, *init_h;
  Geometry *geo = new S_alpha_geo(pars,grids);

  cudaMallocHost((void**) &init_h, sizeof(float)*grids->NxNyNz*grids->Nl);
  cudaMalloc((void**) &G, sizeof(float)*grids->NxNyNz*grids->Nl);
  cudaMalloc((void**) &Gres, sizeof(float)*grids->NxNyNz*grids->Nl);
  cudaMalloc((void**) &g, sizeof(float)*grids->NxNyNz*(laguerre->J+1));

  srand(22);
  float samp = 1.;
  for(int l=0; l<grids->Nl; l++) {
    for(int i=0; i<grids->Ny; i++) {
      for(int j=0; j<grids->Nx; j++) {
          float ra = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
          for(int k=0; k<grids->Nz; k++) {
            int index = i + grids->Ny*j + grids->NxNy*k + grids->NxNyNz*l;
    	    init_h[index] = ra*cos(1.*geo->z_h[k]/pars->Zp);
          }
      }
    }
  }
  cudaMemcpy(G, init_h, sizeof(float)*grids->NxNyNz*grids->Nl, cudaMemcpyHostToDevice);

  checkCuda(laguerre->transformToGrid(G, g));
  checkCuda(laguerre->transformToSpectral(g, Gres));
  
  for(int i=0; i<grids->NxNyNz*grids->Nl; i++) {
    EXPECT_FLOAT_EQ_D(&Gres[i], init_h[i], 1.e-7);
  }

  delete geo;

}

__global__ void scale(float* res, float* g, float scaler, int size) {
  for(int i=0; i<size; i++) {
    res[i] = g[i]*scaler;
  }
}

TEST_F(TestLaguerreTransform, identity2) {
  
  float *G, *g, *Gres, *init_h;
  Geometry *geo = new S_alpha_geo(pars,grids);

  cudaMallocHost((void**) &init_h, sizeof(float)*grids->NxNyNz*grids->Nl);
  cudaMalloc((void**) &G, sizeof(float)*grids->NxNyNz*grids->Nl);
  cudaMalloc((void**) &Gres, sizeof(float)*grids->NxNyNz*grids->Nl);
  cudaMalloc((void**) &g, sizeof(float)*grids->NxNyNz*(laguerre->J+1));

  srand(22);
  float samp = 1.;
  for(int l=0; l<grids->Nl; l++) {
    for(int i=0; i<grids->Ny; i++) {
      for(int j=0; j<grids->Nx; j++) {
          float ra = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
          for(int k=0; k<grids->Nz; k++) {
            int index = i + grids->Ny*j + grids->NxNy*k + grids->NxNyNz*l;
    	    init_h[index] = ra*cos(1.*geo->z_h[k]/pars->Zp);
          }
      }
    }
  }
  cudaMemcpy(G, init_h, sizeof(float)*grids->NxNyNz*grids->Nl, cudaMemcpyHostToDevice);

  checkCuda(laguerre->transformToGrid(G, g));
  scale<<<1,1>>>(g,g,2.,grids->NxNyNz*(laguerre->J+1));
  checkCuda(laguerre->transformToSpectral(g, Gres));
  
  for(int i=0; i<grids->NxNyNz*grids->Nl; i++) {
    EXPECT_FLOAT_EQ_D(&Gres[i], 2.*init_h[i], 1.e-6);
  }

  delete geo;

}
