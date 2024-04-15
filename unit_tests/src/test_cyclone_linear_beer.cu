#include "gtest/gtest.h"
#include "device_test.h"
#include "parameters.h"
#include "grids.h"
#include "diagnostics.h"
#include "run_gx.h"
#include "get_error.h"


TEST(TestParameters, CycloneLinearBeer) {

  Parameters* pars = new Parameters;
  char f[200];
  sprintf(f, "inputs/test_cyclone_linear_beer.in");
  pars->read_namelist(f);
  pars->iproc = 0;
  checkCuda(cudaGetLastError());
  printf("Initializing grids...\n");
  Grids* grids = new Grids(pars);
  printf("Grid dimensions: Nx=%d, Ny=%d, Nz=%d, Nl=%d, Nm=%d, Nspecies=%d\n", 
     grids->Nx, grids->Ny, grids->Nz, grids->Nl, grids->Nm, grids->Nspecies);
  checkCuda(cudaGetLastError());
  printf("Initializing geometry...\n");
  Geometry* geo = new S_alpha_geo(pars,grids);
  checkCuda(cudaGetLastError());
  printf("Initializing diagnostics...\n");
  Diagnostics* diagnostics = new Diagnostics(pars, grids, geo);
  checkCuda(cudaGetLastError());
  
  run_gx(pars, grids, geo, diagnostics);

  cuDoubleComplex growth_rates_correct[11] = {
	0,		0,
	0.052153,	0.038351,
	0.154689,	0.100977,
	0.278426,	0.159365,
	0.437399,	0.191831,
	0.604324,	0.224568,
	0.775701,	0.222670,
	0.943826,	0.206715,
	1.105725,	0.162845,
	1.258766,	0.102780,
	1.398853,	0.025864
  };

  for(int i=1; i<11; i++) {
    EXPECT_NEAR(growth_rates_correct[i].x, diagnostics->growth_rates_h[i].x, 1e-4);
    EXPECT_NEAR(growth_rates_correct[i].y, diagnostics->growth_rates_h[i].y, 1e-4);
  }

  delete grids;
  delete geo;
  delete diagnostics;
  delete pars;
}
