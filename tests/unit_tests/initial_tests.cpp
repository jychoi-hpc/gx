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
  // __host__ __device__ int get_ikx(int idx)
  // changing the value of nx_in in pars, and then creating a new object to reset the grid parameters based on this new Nx value
  // This larger Nx allows "get_ikx" to be tested non-trivially
  pars->nx_in = 10;
  pars->ny_in = 10;
  Grids* grids2 = new Grids(pars);

  // Nx = 10
  CHECK( get_ikx(0) == 0 );
  CHECK( get_ikx(1) == 1 );
  CHECK( get_ikx(5) == 5 );
  CHECK( get_ikx(6) == -4 );
  CHECK( get_ikx(7) == -3 );
  CHECK( get_ikx(9) == -1 );


  // -------------------------------------------------------
  // __host__ __device__ float g0(float b)
  //
  // g0 is equivalent to \Gamma_0(b) = I_0(b)*exp(-b)
  // where I_0 is the modified Bessel function of the first kind

  CHECK( g0(1.e-8) == 1 ); // should give 1 for b < tol = 1.e-7
  CHECK( g0(-0.1) == 1 ); // negative arguments should return 1
  CHECK( g0(0.001) == Approx(0.999001).epsilon(0.01) );
  CHECK( g0(0.01) == Approx(0.990075).epsilon(0.01) );
  CHECK( g0(0.1) == Approx(0.907101).epsilon(0.01) );
  CHECK( g0(0.5) == Approx(0.645035).epsilon(0.01) );
  CHECK( g0(1.0) == Approx(0.465760).epsilon(0.01) );
  CHECK( g0(10.0) == Approx(0.127833).epsilon(0.01) );

  // -------------------------------------------------------
  // __host__ __device__ float g1(float b)
  //
  // g1 is equivalent to \Gamma_1(b) = I_1(b)*exp(-b)
  // ** Note that this differs from the value defined in Howes 2006
  // ** where \Gamma_1(b) = (I_0(b) - I_1(b))*exp(-b)

  CHECK( g1(1.e-8) == 0 ); // for b < tol = 1.e-7, should return 0
  CHECK( g1(-0.1) == 0 ); // should return 0 for negative arguments as well
  CHECK( g1(0.001) == Approx(4.995e-4).epsilon(0.01) );
  CHECK( g1(0.01) == Approx(4.95031e-3).epsilon(0.01) );
  CHECK( g1(0.1) == Approx(0.0452984).epsilon(0.01) );
  CHECK( g1(0.5) == Approx(0.156421).epsilon(0.01) );
  CHECK( g1(1.0) == Approx(0.20791).epsilon(0.01) );
  CHECK( g1(10.0) == Approx(0.121263).epsilon(0.01) );
  
  // -------------------------------------------------------
  // __host__ __device__ float sgam0(float b)
  //
  // This take sqrt( \Gamma_0(b) ), which is defined above
  CHECK( sgam0(1.e-8) == 1 );
  CHECK( sgam0(-0.1) == 1 );
  CHECK( sgam0(0.001) == Approx(0.9995).epsilon(0.01) );
  CHECK( sgam0(0.01) == Approx(0.995025).epsilon(0.01) );
  CHECK( sgam0(0.1) == Approx(0.952418).epsilon(0.01) );
  CHECK( sgam0(0.5) == Approx(0.803141).epsilon(0.01) );
  CHECK( sgam0(1.0) == Approx(0.682466).epsilon(0.01) );
  CHECK( sgam0(10.) == Approx(0.357537).epsilon(0.01) );

  // -------------------------------------------------------
  // __host__ __device__ bool operator>(cuComplex f, cuComplex g)
  // __host__ __device__ bool operator<(cuComplex f, cuComplex g)
  // If complex arguments are on either side of ">" or "<"
  // the complex numbers will be compared via (f.real)^2 + (f.imag)^2
  cuComplex f, g;

  // Testing ">"
  f.x = 0.0; f.y = 0.0; g.x = 0.0; g.y = 0.0;
  CHECK( (f > g) == false );
  f.x = 2.0; f.y = 2.0; g.x = 2.0; g.y = 2.0;
  CHECK( (f > g) == false );  
  f.x = 0.0; f.y = 0.0; g.x = 1.0; g.y = 0.0;
  CHECK( (f > g) == false );
  f.x = 1.0; f.y = 0.0; g.x = 0.0; g.y = 0.0;
  CHECK( (f > g) == true );
  f.x = 1.0; f.y = 2.0; g.x = 3.0; g.y = 1.0;
  CHECK( (f > g) == false );
  f.x = -1.0; f.y = -2.0; g.x = -3.0; g.y = -1.0;
  CHECK( (f > g) == false );
  f.x = -1.0; f.y = 2.0; g.x = 3.0; g.y = -1.0;
  CHECK( (f > g) == false );  
  // Testing "<"
  f.x = 0.0; f.y = 0.0; g.x = 0.0; g.y = 0.0;
  CHECK( (f < g) == false );
  f.x = 2.0; f.y = 2.0; g.x = 2.0; g.y = 2.0;
  CHECK( (f < g) == false );  
  f.x = 0.0; f.y = 0.0; g.x = 1.0; g.y = 0.0;
  CHECK( (f < g) == true );
  f.x = 1.0; f.y = 0.0; g.x = 0.0; g.y = 0.0;
  CHECK( (f < g) == false );
  f.x = 1.0; f.y = 2.0; g.x = 3.0; g.y = 1.0;
  CHECK( (f < g) == true );
  f.x = -1.0; f.y = -2.0; g.x = -3.0; g.y = -1.0;
  CHECK( (f < g) == true );
  f.x = -1.0; f.y = 2.0; g.x = 3.0; g.y = -1.0;
  CHECK( (f < g) == true );  

  // -------------------------------------------------------
  // __host__ __device__ cuComplex operator+(cuComplex f, cuComplex g)
  // __host__ __device__ cuComplex operator-(cuComplex f, cuComplex g)
  //
  // Adds and subtracts the respective real/imag components of each argument
  cuComplex h;
  // Testing "+"
  f.x = 0.0; f.y = 0.0; g.x = 0.0; g.y = 0.0;
  h = f + g; // This is where the overloaded operator is actually used. catch cannot handle cuComplex
  CHECK( h.x == 0 ); CHECK( h.y == 0 ); // check real/imag components
  f.x = 2.0; f.y = 2.0; g.x = 2.0; g.y = 2.0; h = f + g;
  CHECK( h.x == 4 ); CHECK( h.y == 4 );
  f.x = 1.0; f.y = 2.0; g.x = 3.0; g.y = 4.0; h = f + g;  
  CHECK( h.x == 4 ); CHECK( h.y == 6 );
  f.x = 1.0; f.y = -2.0; g.x = -3.0; g.y = 4.0; h = f + g;  
  CHECK( h.x == -2 ); CHECK( h.y == 2 );
  f.x = -1.0; f.y = -2.0; g.x = -3.0; g.y = -4.0; h = f + g;  
  CHECK( h.x == -4 ); CHECK( h.y == -6 );
  // Testing "-"
  f.x = 0.0; f.y = 0.0; g.x = 0.0; g.y = 0.0;
  h = f - g; // This is where the overloaded operator is actually used. catch cannot handle cuComplex
  CHECK( h.x == 0 ); CHECK( h.y == 0 ); // check real/imag components
  f.x = 2.0; f.y = 2.0; g.x = 2.0; g.y = 2.0; h = f - g;
  CHECK( h.x == 0 ); CHECK( h.y == 0 );
  f.x = 1.0; f.y = 2.0; g.x = 3.0; g.y = 4.0; h = f - g;  
  CHECK( h.x == -2 ); CHECK( h.y == -2 );
  f.x = 1.0; f.y = -2.0; g.x = -3.0; g.y = 4.0; h = f - g;  
  CHECK( h.x == 4 ); CHECK( h.y == -6 );
  f.x = -1.0; f.y = -2.0; g.x = -3.0; g.y = -4.0; h = f - g;  
  CHECK( h.x == 2 ); CHECK( h.y == 2 );

  // -------------------------------------------------------
  // __host__ __device__ cuComplex operator-(cuComplex f)
  //
  // Compared to the overloaded "-" above, this assumes that the first argument is zero
  // -f = 0 - f
  f.x = 0.0; f.y = 0.0; h = -f;
  CHECK( h.x == 0 ); CHECK( h.y == 0 );
  f.x = 2.0; f.y = 0.0; h = -f;
  CHECK( h.x == -2 ); CHECK( h.y == 0 );
  f.x = 0.0; f.y = 2.0; h = -f;
  CHECK( h.x == 0 ); CHECK( h.y == -2 );
  f.x = 2.0; f.y = 2.0; h = -f;
  CHECK( h.x == -2 ); CHECK( h.y == -2 );
  f.x = -2.0; f.y = -2.0; h = -f;
  CHECK( h.x == 2 ); CHECK( h.y == 2 );
  f.x = -5.0; f.y = 3.0; h = -f;
  CHECK( h.x == 5 ); CHECK( h.y == -3 );

  // -------------------------------------------------------
  // __host__ __device__ cuComplex operator*(float scalar, cuComplex f)
  // __host__ __device__ cuComplex operator*(cuComplex f, float scalar)
  //
  // This overloaded operator multiplies both the real/imag component of f
  // by the scalar value. The operator with f on either the left or right
  // side of "*" should produce the same result
  float scalar;
  cuComplex hl, hr;
  f.x = 0.0; f.y = 0.0; scalar = 1.0;
  hr = scalar*f; CHECK( hr.x == 0 ); CHECK( hr.y == 0 );
  hl = f*scalar; CHECK( hl.x == 0 ); CHECK( hl.y == 0 );
  f.x = 1.0; f.y = 1.0; scalar = 0.0;
  hr = scalar*f; CHECK( hr.x == 0 ); CHECK( hr.y == 0 );
  hl = f*scalar; CHECK( hl.x == 0 ); CHECK( hl.y == 0 );
  f.x = 0.0; f.y = 0.0; scalar = -1.0;
  hr = scalar*f; CHECK( hr.x == 0 ); CHECK( hr.y == 0 );
  hl = f*scalar; CHECK( hl.x == 0 ); CHECK( hl.y == 0 );
  f.x = -1.0; f.y = -1.0; scalar = 0.0;
  hr = scalar*f; CHECK( hr.x == 0 ); CHECK( hr.y == 0 );
  hl = f*scalar; CHECK( hl.x == 0 ); CHECK( hl.y == 0 );
  f.x = 1.0; f.y = 1.0; scalar = 1.0;
  hr = scalar*f; CHECK( hr.x == 1 ); CHECK( hr.y == 1 );
  hl = f*scalar; CHECK( hl.x == 1 ); CHECK( hl.y == 1 );
  f.x = 1.0; f.y = 1.0; scalar = -1.0;
  hr = scalar*f; CHECK( hr.x == -1 ); CHECK( hr.y == -1 );
  hl = f*scalar; CHECK( hl.x == -1 ); CHECK( hl.y == -1 );
  f.x = 4.0; f.y = 3.0; scalar = -2.0;
  hr = scalar*f; CHECK( hr.x == -8 ); CHECK( hr.y == -6 );
  hl = f*scalar; CHECK( hl.x == -8 ); CHECK( hl.y == -6 );
  f.x = 2.0; f.y = -5.0; scalar = 2.0;
  hr = scalar*f; CHECK( hr.x == 4 ); CHECK( hr.y == -10 );
  hl = f*scalar; CHECK( hl.x == 4 ); CHECK( hl.y == -10 );

  // -------------------------------------------------------
  // __host__ __device__ cuComplex operator*(cuComplex f, cuComplex g)
  // __host__ __device__ cuDoubleComplex operator*(cuDoubleComplex f, cuDoubleComplex g)
  f.x = 0.0; f.y = 0.0; g.x = 0.0; g.y = 0.0; h = f*g;
  CHECK( h.x == 0 ); CHECK( h.y == 0 );
  f.x = 1.0; f.y = 0.0; g.x = 0.0; g.y = 0.0; h = f*g;
  CHECK( h.x == 0 ); CHECK( h.y == 0 );
  f.x = 0.0; f.y = 1.0; g.x = 0.0; g.y = 0.0; h = f*g;
  CHECK( h.x == 0 ); CHECK( h.y == 0 );
  f.x = 0.0; f.y = 0.0; g.x = 1.0; g.y = 0.0; h = f*g;
  CHECK( h.x == 0 ); CHECK( h.y == 0 );
  f.x = 0.0; f.y = 0.0; g.x = 0.0; g.y = 1.0; h = f*g;
  CHECK( h.x == 0 ); CHECK( h.y == 0 );
  f.x = 1.0; f.y = 0.0; g.x = 0.0; g.y = 1.0; h = f*g;
  CHECK( h.x == 0 ); CHECK( h.y == 1 );
  f.x = 1.0; f.y = 0.0; g.x = 0.0; g.y = -1.0; h = f*g;
  CHECK( h.x == 0 ); CHECK( h.y == -1 );
  f.x = 1.0; f.y = 0.0; g.x = 1.0; g.y = 0.0; h = f*g;
  CHECK( h.x == 1 ); CHECK( h.y == 0 );
  f.x = 1.0; f.y = 1.0; g.x = 0.0; g.y = 0.0; h = f*g;
  CHECK( h.x == 0 ); CHECK( h.y == 0 );
  f.x = 0.0; f.y = 1.0; g.x = 1.0; g.y = 0.0; h = f*g;
  CHECK( h.x == 0 ); CHECK( h.y == 1 );
  f.x = 1.0; f.y = -1.0; g.x = -1.0; g.y = 1.0; h = f*g;
  CHECK( h.x == 0 ); CHECK( h.y == 2 );
  f.x = -1.0; f.y = 1.0; g.x = 1.0; g.y = -1.0; h = f*g;
  CHECK( h.x == 0 ); CHECK( h.y == 2 );
  f.x = 2.0; f.y = -1.0; g.x = 5.0; g.y = -4.0; h = f*g;
  CHECK( h.x == 6 ); CHECK( h.y == -13 );
  f.x = -2.0; f.y = -5.0; g.x = 1.0; g.y = -6.0; h = f*g;
  CHECK( h.x == -32 ); CHECK( h.y == 7 );

  cuDoubleComplex fd, gd, hd;
  fd.x = 0.0; fd.y = 0.0; gd.x = 0.0; gd.y = 0.0; hd = fd*gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == 0 );
  fd.x = 1.0; fd.y = 0.0; gd.x = 0.0; gd.y = 0.0; hd = fd*gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == 0 );
  fd.x = 0.0; fd.y = 1.0; gd.x = 0.0; gd.y = 0.0; hd = fd*gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == 0 );
  fd.x = 0.0; fd.y = 0.0; gd.x = 1.0; gd.y = 0.0; hd = fd*gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == 0 );
  fd.x = 0.0; fd.y = 0.0; gd.x = 0.0; gd.y = 1.0; hd = fd*gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == 0 );
  fd.x = 1.0; fd.y = 0.0; gd.x = 0.0; gd.y = 1.0; hd = fd*gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == 1 );
  fd.x = 1.0; fd.y = 0.0; gd.x = 0.0; gd.y = -1.0; hd = fd*gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == -1 );
  fd.x = 1.0; fd.y = 0.0; gd.x = 1.0; gd.y = 0.0; hd = fd*gd;
  CHECK( hd.x == 1 ); CHECK( hd.y == 0 );
  fd.x = 1.0; fd.y = 1.0; gd.x = 0.0; gd.y = 0.0; hd = fd*gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == 0 );
  fd.x = 0.0; fd.y = 1.0; gd.x = 1.0; gd.y = 0.0; hd = fd*gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == 1 );
  fd.x = 1.0; fd.y = -1.0; gd.x = -1.0; gd.y = 1.0; hd = fd*gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == 2 );
  fd.x = -1.0; fd.y = 1.0; gd.x = 1.0; gd.y = -1.0; hd = fd*gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == 2 );
  fd.x = 2.0; fd.y = -1.0; gd.x = 5.0; gd.y = -4.0; hd = fd*gd;
  CHECK( hd.x == 6 ); CHECK( hd.y == -13 );
  fd.x = -2.0; fd.y = -5.0; gd.x = 1.0; gd.y = -6.0; hd = fd*gd;
  CHECK( hd.x == -32 ); CHECK( hd.y == 7 );
  
  // -------------------------------------------------------
  // __host__ __device__ cuComplex operator/(cuComplex f, cuComplex g)
  // __host__ __device__ cuDoubleComplex operator/(cuDoubleComplex f, cuDoubleComplex g)
  f.x = 0.0; f.y = 0.0; g.x = 1.0; g.y = 0.0; h = f/g;
  CHECK( h.x == 0 ); CHECK( h.y == 0 );
  f.x = 0.0; f.y = 0.0; g.x = 0.0; g.y = 1.0; h = f/g;
  CHECK( h.x == 0 ); CHECK( h.y == 0 );
  f.x = 1.0; f.y = 0.0; g.x = 0.0; g.y = 1.0; h = f/g;
  CHECK( h.x == 0 ); CHECK( h.y == -1 );
  f.x = 0.0; f.y = 1.0; g.x = 0.0; g.y = 1.0; h = f/g;
  CHECK( h.x == 1 ); CHECK( h.y == 0 );
  f.x = -1.0; f.y = 0.0; g.x = 0.0; g.y = 1.0; h = f/g;
  CHECK( h.x == 0 ); CHECK( h.y == 1 );
  f.x = 0.0; f.y = -1.0; g.x = 0.0; g.y = 1.0; h = f/g;
  CHECK( h.x == -1 ); CHECK( h.y == 0 );
  f.x = 1.0; f.y = 0.0; g.x = 1.0; g.y = 0.0; h = f/g;
  CHECK( h.x == 1 ); CHECK( h.y == 0 );
  f.x = 0.0; f.y = 1.0; g.x = 1.0; g.y = 0.0; h = f/g;
  CHECK( h.x == 0 ); CHECK( h.y == 1 );
  f.x = -1.0; f.y = 0.0; g.x = 1.0; g.y = 0.0; h = f/g;
  CHECK( h.x == -1 ); CHECK( h.y == 0 );
  f.x = 0.0; f.y = -1.0; g.x = 1.0; g.y = 0.0; h = f/g;
  CHECK( h.x == 0 ); CHECK( h.y == -1 );
  f.x = 1.0; f.y = 0.0; g.x = 0.0; g.y = -1.0; h = f/g;
  CHECK( h.x == 0 ); CHECK( h.y == 1 );
  f.x = 0.0; f.y = 1.0; g.x = 0.0; g.y = -1.0; h = f/g;
  CHECK( h.x == -1 ); CHECK( h.y == 0 );
  f.x = -1.0; f.y = 1.0; g.x = 1.0; g.y = -1.0; h = f/g;
  CHECK( h.x == -1 ); CHECK( h.y == 0 );
  f.x = -2.0; f.y = -5.0; g.x = 1.0; g.y = -6.0; h = f/g;
  CHECK( h.x == Approx(28./37.).epsilon(0.01) ); CHECK( h.y == Approx(-17./37.).epsilon(0.01) );
  f.x = 2.0; f.y = 5.0; g.x = 1.0; g.y = 6.0; h = f/g;
  CHECK( h.x == Approx(32./37.).epsilon(0.01) ); CHECK( h.y == Approx(-7./37.).epsilon(0.01) );
  f.x = 10.0; f.y = -10.0; g.x = 9.0; g.y = 5.0; h = f/g;
  CHECK( h.x == Approx(20./53.).epsilon(0.01) ); CHECK( h.y == Approx(-70./53.).epsilon(0.01) );

  // cuDoubleComplex
  fd.x = 0.0; fd.y = 0.0; gd.x = 1.0; gd.y = 0.0; hd = fd/gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == 0 );
  fd.x = 0.0; fd.y = 0.0; gd.x = 0.0; gd.y = 1.0; hd = fd/gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == 0 );
  fd.x = 1.0; fd.y = 0.0; gd.x = 0.0; gd.y = 1.0; hd = fd/gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == -1 );
  fd.x = 0.0; fd.y = 1.0; gd.x = 0.0; gd.y = 1.0; hd = fd/gd;
  CHECK( hd.x == 1 ); CHECK( hd.y == 0 );
  fd.x = -1.0; fd.y = 0.0; gd.x = 0.0; gd.y = 1.0; hd = fd/gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == 1 );
  fd.x = 0.0; fd.y = -1.0; gd.x = 0.0; gd.y = 1.0; hd = fd/gd;
  CHECK( hd.x == -1 ); CHECK( hd.y == 0 );
  fd.x = 1.0; fd.y = 0.0; gd.x = 1.0; gd.y = 0.0; hd = fd/gd;
  CHECK( hd.x == 1 ); CHECK( hd.y == 0 );
  fd.x = 0.0; fd.y = 1.0; gd.x = 1.0; gd.y = 0.0; hd = fd/gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == 1 );
  fd.x = -1.0; fd.y = 0.0; gd.x = 1.0; gd.y = 0.0; hd = fd/gd;
  CHECK( hd.x == -1 ); CHECK( hd.y == 0 );
  fd.x = 0.0; fd.y = -1.0; gd.x = 1.0; gd.y = 0.0; hd = fd/gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == -1 );
  fd.x = 1.0; fd.y = 0.0; gd.x = 0.0; gd.y = -1.0; hd = fd/gd;
  CHECK( hd.x == 0 ); CHECK( hd.y == 1 );
  fd.x = 0.0; fd.y = 1.0; gd.x = 0.0; gd.y = -1.0; hd = fd/gd;
  CHECK( hd.x == -1 ); CHECK( hd.y == 0 );
  fd.x = -1.0; fd.y = 1.0; gd.x = 1.0; gd.y = -1.0; hd = fd/gd;
  CHECK( hd.x == -1 ); CHECK( hd.y == 0 );
  fd.x = -2.0; fd.y = -5.0; gd.x = 1.0; gd.y = -6.0; hd = fd/gd;
  CHECK( hd.x == Approx(28./37.).epsilon(0.01) ); CHECK( hd.y == Approx(-17./37.).epsilon(0.01) );
  fd.x = 2.0; fd.y = 5.0; gd.x = 1.0; gd.y = 6.0; hd = fd/gd;
  CHECK( hd.x == Approx(32./37.).epsilon(0.01) ); CHECK( hd.y == Approx(-7./37.).epsilon(0.01) );
  fd.x = 10.0; fd.y = -10.0; gd.x = 9.0; gd.y = 5.0; hd = fd/gd;
  CHECK( hd.x == Approx(20./53.).epsilon(0.01) ); CHECK( hd.y == Approx(-70./53.).epsilon(0.01) );

  // -------------------------------------------------------
  // __host__ __device__ cuComplex operator/(cuComplex f, cuComplex g)
  f.x = 0.0; f.y = 0.0; scalar = 1.0; h = f/scalar;
  CHECK( h.x == 0 ); CHECK (h.y == 0 );
  f.x = 0.0; f.y = 0.0; scalar = -1.0; h = f/scalar;
  CHECK( h.x == 0 ); CHECK (h.y == 0 );
  f.x = 0.0; f.y = 1.0; scalar = 1.0; h = f/scalar;
  CHECK( h.x == 0 ); CHECK (h.y == 1 );
  f.x = 0.0; f.y = 1.0; scalar = -1.0; h = f/scalar;
  CHECK( h.x == 0 ); CHECK (h.y == -1 );
  f.x = 1.0; f.y = 0.0; scalar = 1.0; h = f/scalar;
  CHECK( h.x == 1 ); CHECK (h.y == 0 );
  f.x = 1.0; f.y = 0.0; scalar = -1.0; h = f/scalar;
  CHECK( h.x == -1 ); CHECK (h.y == 0 );
  f.x = 1.0; f.y = 1.0; scalar = 1.0; h = f/scalar;
  CHECK( h.x == 1 ); CHECK (h.y == 1 );
  f.x = 5.0; f.y = -4.0; scalar = 1.0; h = f/scalar;
  CHECK( h.x == 5 ); CHECK (h.y == -4 );
  f.x = 4.0; f.y = -3.0; scalar = 2.0; h = f/scalar;
  CHECK( h.x == 2 ); CHECK (h.y == -1.5 );
  f.x = 4.33; f.y = -3.40; scalar = 1.095; h = f/scalar;
  CHECK( h.x == Approx(3.95434).epsilon(0.01) ); CHECK (h.y == Approx(-3.10502).epsilon(0.01) );
  f.x = -14.33; f.y = 2.01; scalar = -12.67; h = f/scalar;
  CHECK( h.x == Approx(1.13102).epsilon(0.01) ); CHECK (h.y == Approx(-0.15864).epsilon(0.01) );

  // -------------------------------------------------------
  // __host__ __device__ bool unmasked(int idx, int idy)
  //
  // Returns true if the (kx,ky) mode is not removed after dealiasing; false otherwise
  // indices will always be positive
  // For this test nx=ny=10
  // Here, running get_ikx on idx=[1,5] gives positive values, [6,9] negative, and get_ikx(0)=0
  CHECK( unmasked(0,0) == false ); // (0,0) mode not evolved
  CHECK( unmasked(0,4) == false ); // idy > (ny-1)/3 + 1
  CHECK( unmasked(10,0) == false ); // idx > nx
  CHECK( unmasked(5,0) == false ); // ikx > (nx-1)/3 + 1
  CHECK( unmasked(6,0) == false ); // ikx < -(nx-1)/3 + 1
  CHECK( unmasked(0,3) == true );
  CHECK( unmasked(7,0) == true );
  CHECK( unmasked(4,3) == false );
  CHECK( unmasked(3,3) == true );
  
  // -------------------------------------------------------
  // __host__ __device__ bool masked(int idx, int idy)
  //
  // Returns true if (kx,ky) mode is removed during dealiasing AND if mode is
  // actually within idx < nx and idy < ny
  CHECK( masked(0,0) == true );
  CHECK( masked(0,4) == true );
  CHECK( masked(10,0) == false ); // idx > nx
  CHECK( masked(0,10) == false ); // idy > ny
  CHECK( masked(5,0) == true );
  CHECK( masked(6,0) == true );
  CHECK( masked(0,3) == false );
  CHECK( masked(7,0) == false );
  CHECK( masked(4,3) == true );
  CHECK( masked(3,3) == false );
  
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
