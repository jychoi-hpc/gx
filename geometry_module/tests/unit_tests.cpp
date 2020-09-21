#include <iostream>
#include "catch.h"
#include "geometric_coefficients.h"
#include "vmec_variables.h"
#include "solver.h"
#include "parameters.h"

TEST_CASE_METHOD(VMEC_variables, "W7-X stdconf test", "[w7x]") {
  
  VMEC_variables *vmec = new VMEC_variables();
