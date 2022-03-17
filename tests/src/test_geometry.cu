#include "gtest/gtest.h"
#include "device_test.h"
#include "parameters.h"
#include "geometry.h"
#include "device_funcs.h"

class TestGeometry : public ::testing::Test {
protected:
  virtual void SetUp() {
    pars = new Parameters;
    checkCuda(cudaGetLastError());
    char f[200];
    sprintf(f, "inputs/z");
    pars->get_nml_vars(f);
    grids = new Grids(pars);
    checkCuda(cudaGetLastError());
  }

  virtual void TearDown() {
    checkCuda(cudaGetLastError());
    delete pars;
    checkCuda(cudaGetLastError());
    delete grids;
    checkCuda(cudaGetLastError());
  }

  Parameters* pars;
  Grids* grids;
};


TEST_F(TestGeometry, SlabGeoCoefficentArrays) {
  pars->slab = true;
  Geometry *geo_slab;
  geo_slab = new S_alpha_geo(pars,grids);

  for(int k=0; k<pars->nz_in; k++) {
    EXPECT_FLOAT_EQ_D(&geo_slab->z[k], 2.*M_PI*pars->Zp*(k-pars->nz_in/2)/pars->nz_in);
    EXPECT_FLOAT_EQ_D(&geo_slab->bmag[k], 1.);
    EXPECT_FLOAT_EQ_D(&geo_slab->bmagInv[k], 1./geo_slab->bmag_h[k]);
    EXPECT_FLOAT_EQ_D(&geo_slab->bgrad[k], 0.);           
    EXPECT_FLOAT_EQ_D(&geo_slab->gds2[k], 1. + pow((pars->shat*geo_slab->z_h[k]-pars->shift*sin(geo_slab->z_h[k])),2));
    EXPECT_FLOAT_EQ_D(&geo_slab->gds21[k], -pars->shat*(pars->shat*geo_slab->z_h[k]-pars->shift*sin(geo_slab->z_h[k])));
    EXPECT_FLOAT_EQ_D(&geo_slab->gds22[k], pow(pars->shat,2));
    EXPECT_FLOAT_EQ_D(&geo_slab->gbdrift[k], 0.);
    EXPECT_FLOAT_EQ_D(&geo_slab->cvdrift[k], 0.);
    EXPECT_FLOAT_EQ_D(&geo_slab->gbdrift0[k], 0.);
    EXPECT_FLOAT_EQ_D(&geo_slab->cvdrift0[k], 0.);
    EXPECT_FLOAT_EQ_D(&geo_slab->grho[k], 1);
    EXPECT_FLOAT_EQ_D(&geo_slab->jacobian[k], 1. / abs(pars->drhodpsi*geo_slab->gradpar*geo_slab->bmag_h[k]));
  }
  delete geo_slab;
}


TEST_F(TestGeometry, GeoCoefficentArrays) {

  Geometry *geo;
  geo = new S_alpha_geo(pars,grids);

  for(int k=0; k<pars->nz_in; k++) {
    EXPECT_FLOAT_EQ_D(&geo->z[k], 2.*M_PI*pars->Zp*(k-pars->nz_in/2)/pars->nz_in);
    EXPECT_FLOAT_EQ_D(&geo->bmag[k], 1./(1.+pars->eps*cos(geo->z_h[k])));
    EXPECT_FLOAT_EQ_D(&geo->bmagInv[k], 1./geo->bmag_h[k]);
    EXPECT_FLOAT_EQ_D(&geo->bgrad[k], geo->gradpar*pars->eps*sin(geo->z_h[k])*geo->bmag_h[k]);           
    EXPECT_FLOAT_EQ_D(&geo->gds2[k], 1. + pow((pars->shat*geo->z_h[k]-pars->shift*sin(geo->z_h[k])),2));
    EXPECT_FLOAT_EQ_D(&geo->gds21[k], -pars->shat*(pars->shat*geo->z_h[k]-pars->shift*sin(geo->z_h[k])));
    EXPECT_FLOAT_EQ_D(&geo->gds22[k], pow(pars->shat,2));
    EXPECT_FLOAT_EQ_D(&geo->gbdrift[k], 1./(2.*pars->rmaj)*( cos(geo->z_h[k]) + (pars->shat*geo->z_h[k]-pars->shift*sin(geo->z_h[k]))*sin(geo->z_h[k]) ));
    EXPECT_FLOAT_EQ_D(&geo->cvdrift[k], geo->gbdrift_h[k]);
    EXPECT_FLOAT_EQ_D(&geo->gbdrift0[k], -1./(2.*pars->rmaj)*pars->shat*sin(geo->z_h[k]));
    EXPECT_FLOAT_EQ_D(&geo->cvdrift0[k], geo->gbdrift0_h[k]);
    EXPECT_FLOAT_EQ_D(&geo->grho[k], 1);
    EXPECT_FLOAT_EQ_D(&geo->jacobian[k], 1. / abs(pars->drhodpsi*geo->gradpar*geo->bmag_h[k]));
  }
  checkCuda(cudaGetLastError());
  delete geo;
  checkCuda(cudaGetLastError());
  
}
