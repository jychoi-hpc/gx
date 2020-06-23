#include "catch.h"
#include "regression.h"
#include <vector>

TEST_CASE_METHOD(Regression, "regression tests", "[regression]") {
  Regression *reg = new Regression;
  reg->kh01();
  // ny = 4 ---> naky = 2
  // nx = 4 ---> nakx = 3

  // gamma[0] --> (ky=0, kx=0),  gamma[1] --> (ky=0.1, kx=0)
  // gamma[2] --> (ky=0, kx=0.5),  gamma[3] --> (ky=0.1, kx=0.5)
  // gamma[4] --> (ky=0, kx=-0.5),  gamma[5] --> (ky=0.1, kx=-0.5)
  
  // omega should be zero for kx=ky=0 mode
  CHECK(reg->omega[0] == 0);
  CHECK(reg->gamma[0] == 0);

  // real frequency should be equal and opposite for omega with same ky and opposite signs of kx
  CHECK(reg->omega[2] == -reg->omega[4]);
  CHECK(reg->omega[3] == -reg->omega[5]);

  // check values of growth rates with tolerance
  Approx target = Approx(4.90198).epsilon(0.01);
  CHECK(reg->gamma[2] == target);
  CHECK(reg->gamma[3] == target);
  CHECK(reg->gamma[4] == target);
  CHECK(reg->gamma[5] == target);

  // check values of real frequency
  CHECK(reg->omega[2] == Approx(-2.04917e-05).epsilon(1.e-5));
  CHECK(reg->omega[3] == Approx(-1.11399e-05).epsilon(1.e-5));
}
