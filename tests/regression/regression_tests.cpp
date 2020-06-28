#include "catch.h"
#include "regression.h"
#include <vector>
#include <iostream>

TEST_CASE_METHOD(Regression, "KH Regression Test", "[regression]") {
  Regression *reg = new Regression;
  reg->kh01();
  // ny = 4 ---> naky = 2
  // nx = 4 ---> nakx = 3
  CHECK(reg->Naky_ == 2);
  CHECK(reg->Nakx_ == 3);
  
  // gamma_[0] --> (ky=0, kx=0),  gamma_[1] --> (ky=0.1, kx=0)
  // gamma_[2] --> (ky=0, kx=0.5),  gamma_[3] --> (ky=0.1, kx=0.5)
  // gamma_[4] --> (ky=0, kx=-0.5),  gamma_[5] --> (ky=0.1, kx=-0.5)
  
  // omega should be zero for kx=ky=0 mode
  CHECK(reg->omega_[0] == 0);
  CHECK(reg->gamma_[0] == 0);
  
  // real frequency should be equal and opposite for omega_ with same ky and opposite signs of kx
  CHECK(reg->omega_[2] == -reg->omega_[4]);
  CHECK(reg->omega_[3] == -reg->omega_[5]);
  
  // check values of growth rates with tolerance
  Approx target = Approx(4.90198).epsilon(0.01);
  CHECK(reg->gamma_[2] == target);
  CHECK(reg->gamma_[3] == target);
  CHECK(reg->gamma_[4] == target);
  CHECK(reg->gamma_[5] == target);
  
  // check values of real frequency
  CHECK(reg->omega_[2] == Approx(-2.04917e-05).epsilon(1.e-5));
  CHECK(reg->omega_[3] == Approx(-1.11399e-05).epsilon(1.e-5));
  
  delete reg;
}

TEST_CASE_METHOD(Regression, "Slab Regression Test", "[regression]") {
  
  Regression *reg = new Regression;
  reg->slab();
  // ny = 32 ---> naky = 11
  // nx = 1 ---> nakx = 1
  CHECK(reg->Naky_ == 6);
  CHECK(reg->Nakx_ == 1);
  
  // gamma_[0] --> (ky=0, kx=0)
  // gamma_[1] --> (ky=0.5, kx=0)
  // gamma_[2] --> (ky=1.0, kx=0)
  // ...
  // gamma_[5] -> (ky=2.5, kx=0)

  // omega should be zero for kx=ky=0 mode
  CHECK(reg->omega_[0] == 0);
  CHECK(reg->gamma_[0] == 0);
  
  // check values of growth rates with tolerance
  CHECK(reg->gamma_[2] == Approx(0.082298).epsilon(0.01));
  CHECK(reg->gamma_[3] == Approx(0.074535).epsilon(0.01));
  CHECK(reg->gamma_[4] == Approx(0.060325).epsilon(0.01));
  CHECK(reg->gamma_[5] == Approx(0.039144).epsilon(0.01));

  // check values of real frequency within larger range
  CHECK(reg->omega_[2] == Approx(0.1658728).margin(0.01));
  CHECK(reg->omega_[3] == Approx(0.1680077).margin(0.01));
  CHECK(reg->omega_[4] == Approx(0.1559375).margin(0.01));
  CHECK(reg->omega_[5] == Approx(0.1436503).margin(0.01));

  delete reg;

}
