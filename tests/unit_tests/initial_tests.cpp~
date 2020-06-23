#include "catch.h"
#include "parameters.h"
#include "grids.h"
#include "reductions.h"
#include "kernel_class.h"
#include <fstream>
#include <vector>


TEST_CASE_METHOD(Parameters, "check if able to get maxThreadsPerBlock from Parameters constructor", "[parameters]") {
  Parameters *pars = new Parameters;
  char *in_file;
  in_file = "catch2_tests/generic_input";
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

TEST_CASE_METHOD(Parameters, "cub library", "[cub]") {
  Parameters *pars = new Parameters;
  char *in_file;
  in_file = "catch2_tests/generic_input";
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
