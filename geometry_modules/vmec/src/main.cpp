#include "vmec_variables.h"
#include "geometric_coefficients.h"
#include "solver.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]) {

  char nml_file[512];
  strcpy (nml_file, argv[1]);
  strcat (nml_file, ".ing");

  // check for an additional input (by looking flag or whatever) and force the output file name
  // to be something the users wishes

  VMEC_variables *vmec = new VMEC_variables(nml_file);
  Geometric_coefficients *geo = new Geometric_coefficients(nml_file, vmec);

  delete vmec;
  delete geo;
  
  return 0;
}
