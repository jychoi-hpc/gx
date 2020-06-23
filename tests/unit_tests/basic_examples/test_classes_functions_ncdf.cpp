#include <iostream>
#include <fstream>
#include <iomanip>
#include "catch.h"
#include "example_class.h"
#include <vector>

//using namespace std;
TEST_CASE_METHOD(Grids, "check if this connects to grids.h", "[grids]") {
  REQUIRE(something == 3);
  CHECK(something == 2);
  REQUIRE(other == 3);
  
}

TEST_CASE_METHOD(Grids, "check if member function works", "[grids_func]") {
  REQUIRE(simple_function(2) == 3);
  REQUIRE(simple_function(2) == 2);
}

TEST_CASE_METHOD(Grids, "reading file", "[read_grid]") {
  std::vector<int> vector;
  get_filename(vector);
  // This case reads in a simple "file.txt" with values 3, 4, and 6
  REQUIRE(vector[0] == 3);
  REQUIRE(vector[1] == 4);
  REQUIRE(vector[2] == 1); // should fail
}

TEST_CASE_METHOD(Grids, "netcdf read", "[netcdf]") {
  int ret_val;
  ret_val = get_netcdf_varname();
  // reads in the netcdf file cyc01.nc, and get the value of ntheta = 32
  CHECK (ret_val == 1); // should fail
  REQUIRE (ret_val == 32);
}
