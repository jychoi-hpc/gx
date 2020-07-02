#include <iostream>
#include "catch.h"
#include "parameters.h"
#include "grids.h"
#include "reductions.h"
#include "device_funcs.h"
#include <fstream>
#include <vector>


TEST_CASE_METHOD(Parameters, "check if able to get maxThreadsPerBlock from Parameters constructor", "[parameters]") {
  Parameters *pars = new Parameters;
  char *in_file;
  in_file = "tests/unit_tests/generic_input";
  pars->get_nml_vars(in_file);
  CHECK(maxThreadsPerBlock == 1024);
  CHECK(pars->nz_in == 32);
  CHECK(pars->ny_in == 20);
  pars->ny_in = 10;
  Grids* grids = new Grids(pars);
  CHECK(grids->Nz == 32);
  CHECK(grids->Ny == 20);
  CHECK(grids->Nx == 1);
  CHECK(grids->Nyc == 11);
  CHECK(pars->x0 == 10);
  CHECK(pars->y0 == 20);
  CHECK(pars->Zp == 1);
  int Nmax = std::max(std::max(grids->Nx, grids->Nyc),grids->Nz);
  CHECK(Nmax == 32);
  std::vector<float> ky_grid{0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5};
  
  for (int i=0; i<grids->Nyc; i++) {
    CHECK(grids->ky_h[i] == ky_grid[i]);
  }
  
}

/*
TEST_CASE_METHOD(Kernels, "Kernal Testing", "[kernels]") {
  SECTION("Test kInit Kernel", "[kInit]") {
    int gridsize{1};
    int nthreads{32};
    int Nx{1};
    int Nyc{11};
    int Nz{32};
    float X0{10};
    float Y0{20};
    float Zp{1};
    Kernels::kInit(gridsize, nthreads, Nx, Nyc, Nz, kx_h, ky_h, kz_h, X0, Y0, Zp);
    std::vector<float> ky_grid{0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,0.5};
    for (int i=0; i<Nyc; i++) {
      CHECK(ky_h[i] == ky_grid[i]);
    }
  }
}
*/

TEST_CASE_METHOD(Parameters, "device_functions", "[device]") {
  // Need to create a Parameters and Grids object to get/set variables that are found in some of the device functions
  Parameters *pars = new Parameters;
  char *in_file;
  in_file = "tests/unit_tests/generic_input";
  pars->get_nml_vars(in_file);
  Grids* grids = new Grids(pars);

  // -------------------------------------------------------
  // __host__ __device__ factorial(m)

  CHECK( factorial(0) == 1 );
  CHECK( factorial(1) == 1 );
  CHECK( factorial(2) == 2 );
  CHECK( factorial(3) == 6 );
  CHECK( factorial(4) == 24 );
  CHECK( factorial(5) == 120 );
  CHECK( factorial(6) == 720 );
  // algorithm to calculate factorials >6 are not exact
  CHECK( factorial(7) == Approx(5040).epsilon(0.01) );
  CHECK( factorial(10) == Approx(3628800).epsilon(0.01) );
  
  // -------------------------------------------------------
  // __host__ __device__ Jflr(l, b, enforce_JL_0)

  // *** Nl = 8
  CHECK( Jflr(-1, 0.2, 0) == 0 ); // negative l should always give zero
  CHECK( Jflr(-10, 0, 1) == 0 ); // negative l should always give zero
  CHECK( Jflr(31, 0, 1) == 0 ); // l > 30 should always give zero

  // if l >= Nl and enforce_JL_0 = 1 should give zero
  CHECK( Jflr(8, 0.2, 1) == 0 );
  
  CHECK( Jflr(7, 0.2, 1) == Approx(0).margin(0.01) );
  CHECK( Jflr(7, 0.2, 1) == -Approx(1.79531e-11).margin(1.e-11) );
  CHECK( Jflr(7, -0.2, 1) == Approx(1.79531e-11).margin(1.e-11) );
  
  CHECK( Jflr(1, 0.2, 0) == Approx(-0.0904837).epsilon(0.01) );
  CHECK( Jflr(1, 0.2, 1) == Approx(-0.0904837).epsilon(0.01) );

  // -------------------------------------------------------
  // __host__ __device__ get_ikx(idx)
  // changing the value of nx_in in pars, and then creating a new object to reset the grid parameters based on this new Nx value
  // This larger Nx allows "get_ikx" to be tested non-trivially
  pars->nx_in = 10;
  Grids* grids2 = new Grids(pars);

  // Nx = 10
  CHECK( get_ikx(0) == 0 );
  CHECK( get_ikx(1) == 1 );
  CHECK( get_ikx(5) == 5 );
  CHECK( get_ikx(6) == -4 );
  CHECK( get_ikx(7) == -3 );
  CHECK( get_ikx(9) == -1 );

  delete grids2;
  delete grids;
  delete pars;
}

TEST_CASE_METHOD(Parameters, "cub library", "[cub]") {
  Parameters *pars = new Parameters;
  char *in_file;
  in_file = "tests/unit_tests/generic_input";
  pars->get_nml_vars(in_file);
  Grids *grids = new Grids(pars);
  Red *red = new Red(grids);
  int i = 1;
  CHECK(i == 1);
  float arr_1[10] {0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  float *ptr_1{arr_1};
  float out1[1] {0.0};
  float *out_ptr1{out1};
  red->Sum(ptr_1, out_ptr1);
  CHECK(out1[0] == 2);

  float arr_2[10] {0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  float *ptr_2{arr_2};
  float out2[1] {0.0};
  float *out_ptr2{out2};
  red->Sum(ptr_2, out_ptr2);
  CHECK(out2[0] == 3);

  float arr_3[10] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  float *ptr_3{arr_3};
  float out3[1] {0.0};
  float *out_ptr3{out3};
  red->Sum(ptr_3, out_ptr3);
  CHECK(out3[0] == 55);
}
