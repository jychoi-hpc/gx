#pragma once
#include "geometric_coefficients.h"
#include "vmec_variables.h"

struct g_params {
  double theta_pest_target_f;
  double zeta0_f;
  int vmec_radial_index_half_f[2];
  double vmec_radial_weight_half_f[2];
  VMEC_variables *vmec_f;
};

double fzero_residual(double, void*);
void solver_vmec_theta(double*, double*, int, double, double, VMEC_variables*, int*, double*);
void interp_to_new_grid(double*, double*, double*, int);
