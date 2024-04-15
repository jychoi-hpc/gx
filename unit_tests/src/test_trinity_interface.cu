#include "gtest/gtest.h"
#include "device_test.h"
#include "parameters.h"
#include "grids.h"
#include "geometry.h"
#include "device_funcs.h"
#include "trinity_interface.h"

class TestTrinityInterface : public ::testing::Test {
protected:
  virtual void SetUp() {
    pars = new Parameters;
    char f[200];
    sprintf(f, "inputs/cyc_nl");
    pars->get_nml_vars(f);
    grids = new Grids(pars);
    geo = new S_alpha_geo(pars,grids);
    checkCuda(cudaGetLastError());
  }

  virtual void TearDown() {
    checkCuda(cudaGetLastError());
    delete geo;
    delete grids;
    delete pars;
  }

  Parameters *pars;
  Grids *grids;
  Geometry *geo;
};

TEST_F(TestTrinityInterface, SetFromTrin) {

  trin_parameters_struct tpars;
  tpars.ntspec = 1;
  tpars.rgeo_local = 13.;
  tpars.dens[0] = 2.;
  tpars.temp[0] = 2.;
  tpars.fprim[0] = 2.;
  tpars.tprim[0] = 2.;
  tpars.nu[0] = 2.;
  tpars.nstep = 10;

  EXPECT_FLOAT_EQ(1., pars->rmaj);

  set_from_trinity(pars, &tpars);

  EXPECT_FLOAT_EQ(tpars.rgeo_local, pars->rmaj);
  EXPECT_FLOAT_EQ(tpars.dens[0], pars->species_h[0].dens);
}

TEST_F(TestTrinityInterface, CopyFluxesToTrin) {
  trin_fluxes_struct tfluxes;
  copy_fluxes_to_trinity(pars, geo, &tfluxes);

  EXPECT_FLOAT_EQ(49.65884399414062, tfluxes.qflux[0]);
}
