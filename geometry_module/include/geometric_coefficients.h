#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "parameters.h"
#include "vmec_variables.h"
#define _USE_MATH_DEFINES
#include <math.h>

class Geometric_coefficients {

 public:
  Geometric_coefficients(VMEC_variables*);
  ~Geometric_coefficients();
  void test_arrays(double*, double*, int, double, const std::string&);

  void get_GX_geo_arrays(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);

  void write_geo_arrays_to_file(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);

  void get_cut_indices_custom(std::vector<double>&, int&, int&, int&);
  void get_revised_theta_custom(std::vector<double>&, std::vector<double>&, std::vector<double>&);

  void get_cut_indices_zeros(std::vector<double>&, int&, int&, int&, int&, int&);
  void get_revised_theta_zeros(std::vector<double>&, std::vector<double>&, std::vector<double>&, std::vector<double>&);

  friend void solver_vmec_theta(double*, double*, int, double, double, VMEC_variables*, int*, double*);

  std::vector<double> slice(std::vector<double> const &, int, int);

  VMEC_variables *vmec;

  template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
  }
  
  // output geometric quantities
  double *theta_grid;
  double *bmag;
  double *gradpar;
  double *grho;
  double *gds2;
  double *gds21;
  double *gds22;
  double *gbdrift;
  double *gbdrift0;
  double *cvdrift;
  double *cvdrift0;
  
 private:
  // local variables for computing geometric coefficients
  double L_reference;  
  double edge_toroidal_flux_over_2pi;
  double B_reference;
  double *normalized_toroidal_flux_full_grid;
  double *normalized_toroidal_flux_half_grid;
  double normalized_toroidal_flux_used;
  double *dr2_half;
  double *dr2_full;
  int vmec_radial_index_full[2];
  int vmec_radial_index_half[2];
  double vmec_radial_weight_full[2];
  double vmec_radial_weight_half[2];
  double iota;
  double safety_factor_q;
  double *d_iota_ds_on_half_grid;
  double *d_pressure_ds_on_half_grid;
  double ds;
  double d_iota_ds;
  double d_pressure_ds;
  double shat;
  int number_field_periods_to_include_final;
  double *zeta;
  double *theta;
  double *theta_vmec;

  // geometry variables
  double *B;
  double *temp2D;
  double *sqrt_g;
  double *R;
  double *dB_dtheta_vmec;
  double *dB_dzeta;
  double *dB_ds;
  double *dR_dtheta_vmec;
  double *dR_dzeta;
  double *dR_ds;
  double *dZ_dtheta_vmec;
  double *dZ_dzeta;
  double *dZ_ds;
  double *dLambda_dtheta_vmec;
  double *dLambda_dzeta;
  double *dLambda_ds;
  double *B_sub_s;
  double *B_sub_theta_vmec;
  double *B_sub_zeta;
  double *B_sup_theta_vmec;
  double *B_sup_zeta;

  double *dB_ds_mnc;
  double *dB_ds_mns;
  double *dR_ds_mnc;
  double *dR_ds_mns;
  double *dZ_ds_mnc;
  double *dZ_ds_mns;
  double *dLambda_ds_mnc;
  double *dLambda_ds_mns;

  double *dX_ds;
  double *dX_dtheta_vmec;
  double *dX_dzeta;
  double *dY_ds;
  double *dY_dtheta_vmec;
  double *dY_dzeta;

  double *grad_s_X;
  double *grad_s_Y;
  double *grad_s_Z;
  double *grad_theta_vmec_X;
  double *grad_theta_vmec_Y;
  double *grad_theta_vmec_Z;
  double *grad_zeta_X;
  double *grad_zeta_Y;
  double *grad_zeta_Z;
  double *grad_psi_X;
  double *grad_psi_Y;
  double *grad_psi_Z;
  double *grad_alpha_X;
  double *grad_alpha_Y;
  double *grad_alpha_Z;

  double *B_X;
  double *B_Y;
  double *B_Z;
  double *grad_B_X;
  double *grad_B_Y;
  double *grad_B_Z;
  double *B_cross_grad_B_dot_grad_alpha;
  double *B_cross_grad_B_dot_grad_alpha_alternate;
  double *B_cross_grad_s_dot_grad_alpha;
  double *B_cross_grad_s_dot_grad_alpha_alternate;

  double *B_dot_grad_theta_pest_over_B_dot_grad_zeta;
  double *diff_arr;
  double *sum_arr;
  double *array1_temp;

  double *theta_grid_temp;
  double *bmag_temp;
  double *gradpar_temp;
  double *grho_temp;
  double *gds2_temp;
  double *gds21_temp;
  double *gds22_temp;
  double *gbdrift_temp;
  double *gbdrift0_temp;
  double *cvdrift_temp;
  double *cvdrift0_temp;

  double dtheta_custom;
  std::vector<double> revised_theta;
  double *revised_theta_grid;
  std::vector<double> theta_grid_cut;
  std::vector<double> theta_cut_temp;
  double *theta_cut;
  std::vector<double> bmag_cut;
  std::vector<double> gradpar_cut;
  std::vector<double> grho_cut;
  std::vector<double> gds2_cut;
  std::vector<double> gds21_cut;
  std::vector<double> gds22_cut;
  std::vector<double> gbdrift_cut;
  std::vector<double> gbdrift0_cut;
  std::vector<double> cvdrift_cut;
  std::vector<double> cvdrift0_cut;
  
  bool non_Nyquist_mode_available;
  bool found_imn;

  double angle, cos_angle, sin_angle;
  double temp;
  double sqrt_s;
  double mu_0 = 4.0*M_PI*1.0e-7;

  // variables for the get_GX_geo_arrays member function
  double *gradpar_half_grid;
  double *temp_grid;
  double *z_on_theta_grid;
  double *uniform_zgrid;
  double dtheta;
  double dtheta_pi;
  double desired_gradpar;
  int index_of_middle;

  // variables for geometric coeff rootfinder
  double *temp_tgrid;
  
  // misc variables
  int i, index, m, n, imn, imn_ind;
  double min_dr2, scale_factor;
  int ileft, iright;
  int root_idx_left, root_idx_right;
  double domain_scaling_factor;

  // input variables for the interface
  double alpha = 0.0;
  int nzgrid = 16;
  int npol = 1;
  int sign_psi;
  double desired_normalized_toroidal_flux = 0.25;
  int vmec_surface_option = 2;
  int verbose = 1;
  std::string flux_tube_cut = "none";
  double custom_length = M_PI;
  int which_crossing = 1;

  // strings for test_arrays function
  const std::string iota_name = "iota";
  const std::string grad_zeta_X_name = "grad_zeta_X";
  const std::string grad_zeta_Y_name = "grad_zeta_Y";
  const std::string grad_zeta_Z_name = "grad_zeta_Z";
  const std::string sqrt_g_name = "sqrt_g";
  const std::string inv_sqrt_g_name = "1/sqrt_g";
  const std::string B_sub_theta_vmec_name = "B_sub_theta_vmec";
  const std::string B_sub_zeta_name = "B_sub_zeta";
  const std::string B_sub_s_name = "B_sub_s";
  const std::string B_sup_theta_vmec_name = "B_sup_theta_vmec";
  const std::string B_sup_zeta_name = "B_sup_zeta";
  const std::string B_sup_s_name = "B_sup_s";
  const std::string B_cross_grad_s_dot_grad_alpha_name = "B_cross_grad_s_dot_grad_alpha";
  const std::string B_cross_grad_B_dot_grad_alpha_name = "B_cross_grad_B_dot_grad_alpha";
  
};
  
