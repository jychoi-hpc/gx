#include "gtest/gtest.h"
#include "device_test.h"

#include "parameters.h"

TEST(TestParameters, ReadNamelist) {
  Parameters* pars = new Parameters;

  char f[200];
  sprintf(f, "inputs/test_parameters.in");
  pars->read_namelist(f);

  // kt_grids_box_parameters
  EXPECT_EQ(16, pars->nx_in);
  EXPECT_EQ(16, pars->ny_in);
  EXPECT_FLOAT_EQ(10., pars->y0);
  EXPECT_EQ( (int) round(2*M_PI*pars->shat*pars->Zp), pars->jtwist);

  // theta_grid_parameters
  EXPECT_EQ(96, pars->nz_in);
  EXPECT_EQ(2, pars->nperiod);
  EXPECT_FLOAT_EQ(0.18, pars->eps);
  EXPECT_FLOAT_EQ(0.8, pars->shat);
  EXPECT_FLOAT_EQ(1.4, pars->qsf);

  // species_parameters_1
  EXPECT_FLOAT_EQ(6.9, pars->species_h[0].tprim);
  EXPECT_FLOAT_EQ_D(&pars->species[0].tprim, 6.9);

  // these parameters are commented out in
  // inputs/test_parameters.in so that
  // values are taken from the default namelist
  EXPECT_FLOAT_EQ(0.05, pars->dt);
}
