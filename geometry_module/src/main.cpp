#include "vmec_variables.h"
#include "geometric_coefficients.h"
#include "solver.h"

int main(int argc, char* argv[]) {

  VMEC_variables *vmec = new VMEC_variables(argv[1]);
  Geometric_coefficients *geo = new Geometric_coefficients(vmec);

  delete vmec;
  delete geo;
  
  return 0;
}
