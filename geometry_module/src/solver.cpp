#include "solver.h"
#include "vmec_variables.h"
#include "geometric_coefficients.h"
#include "gsl_roots.h"
#include "gsl_math.h"
#include "gsl_errno.h"
#include "gsl_spline.h"

void solver_vmec_theta(double *theta_vmec, double *zeta, int nzgrid, double alpha, double iota, VMEC_variables *vmec, int *radial_index, double *radial_weight) {

  int status;
  double r;
  //  double r_exp=0.0;
  int iter, max_iter;
  double theta_pest_target, theta_vmec_min, theta_vmec_max;
  double zeta0;
  double root_solve_absolute_tol = 1.0e-5;
  double root_solve_relative_tol = 1.0e-5;

  // Root solve for theta_vmec at each point on the uniform grid
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    
    iter=0;
    max_iter=100;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    gsl_function F;

    zeta0 = zeta[izeta];
    if (!zeta0 == 0) {
      theta_pest_target = alpha + iota * zeta0;
      theta_vmec_min = theta_pest_target - 0.5;
      theta_vmec_max = theta_pest_target + 0.5;
      r = 0.1;
      g_params g_parameters = {theta_pest_target, zeta0, {radial_index[0], radial_index[1]} , {radial_weight[0], radial_weight[1]}, vmec};
      F.function = &fzero_residual;
      F.params = &g_parameters;
    
      T = gsl_root_fsolver_brent;
      s = gsl_root_fsolver_alloc (T);
      gsl_root_fsolver_set (s, &F, theta_vmec_min, theta_vmec_max);

      /*printf ("using %s method\n",
	      gsl_root_fsolver_name (s));

      printf ("%5s [%9s, %9s] %9s %10s %9s\n",
	      "iter", "lower", "upper", "root",
	      "err", "err(est)");
      */
      do {
	iter++;
	status = gsl_root_fsolver_iterate(s);
	r = gsl_root_fsolver_root(s);
	theta_vmec_min = gsl_root_fsolver_x_lower(s);
	theta_vmec_max = gsl_root_fsolver_x_upper(s);
	status = gsl_root_test_interval(theta_vmec_min, theta_vmec_max,
					root_solve_absolute_tol, root_solve_relative_tol);
	
	//if (status == GSL_SUCCESS) { printf("Converged:\n"); }
	/*	printf ("%5d [%.7f, %.7f] %.7f %+.7f %.7f\n",
		iter, theta_vmec_min, theta_vmec_max,
		r, r - r_exp,
		theta_vmec_max - theta_vmec_min);*/
      }
      while (status == GSL_CONTINUE && iter < max_iter);
      
      gsl_root_fsolver_free (s);
      theta_vmec[izeta] = r;
    }
    else {
      theta_vmec[izeta] = 0;
    }
  }
}

double fzero_residual(double theta_vmec_try, void *p) {
  struct g_params *params = (struct g_params *)p;
  double theta_pest_target = (params->theta_pest_target_f);
  double zeta0 = (params->zeta0_f);
  int *vmec_radial_index_half = (params->vmec_radial_index_half_f);
  double *vmec_radial_weight_half = (params->vmec_radial_weight_half_f);
  VMEC_variables *vmec = (params->vmec_f);
  
  double fzero_residual, angle, sinangle, cosangle; 
  fzero_residual = theta_vmec_try - theta_pest_target;
  for (int imn=0; imn < vmec->mnmax; imn++) {
    angle = vmec->xm[imn]*theta_vmec_try - vmec->xn[imn]*zeta0;
    sinangle = sin(angle);
    cosangle = cos(angle);
    for (int which_surface=0; which_surface<2; which_surface++) {
      fzero_residual = fzero_residual + vmec_radial_weight_half[which_surface]*vmec->lmns[imn+vmec_radial_index_half[which_surface]*vmec->mnmax]*sinangle;
      if (vmec->lasym) {
	fzero_residual = fzero_residual + vmec_radial_weight_half[which_surface]*vmec->lmnc[imn+vmec_radial_index_half[which_surface]*vmec->mnmax]*cosangle;
      }
    }
  }
  return fzero_residual;
}

void interp_to_new_grid(double *geo_array, double *z_on_theta_grid, double *uniform_grid, int nzgrid, bool include_final_grid_point) {

  gsl_interp_accel *acc = gsl_interp_accel_alloc ();
  gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, 2*nzgrid+1);

  gsl_spline_init (spline, z_on_theta_grid, geo_array, 2*nzgrid+1);

  if (include_final_grid_point) {
    for (int j=0; j < 2*nzgrid+1; j++) {
      geo_array[j] = gsl_spline_eval(spline, uniform_grid[j], acc);
    }
  }
  
  else {
    for (int j=0; j < 2*nzgrid; j++) {
      geo_array[j] = gsl_spline_eval(spline, uniform_grid[j], acc);
    }
  }

  gsl_spline_free (spline);
  gsl_interp_accel_free (acc);

}

double find_zero_crossing(double* geo_array, double* theta_grid, int npoints) {
  double zero_loc = 0;
  //  if (geo_array.size() == theta_grid.size()) {
  //    std::cout << "length of arrays are equal\n";
  //  }
  /* gsl_interp_accel *acc = gsl_interp_accel_alloc ();
  gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, 2*nzgrid+1);

  gsl_spline_init (spline, theta_grid, geo_array, npoints);
  gsl_function F;
  F.function = &spline;*/
  return zero_loc;
}
