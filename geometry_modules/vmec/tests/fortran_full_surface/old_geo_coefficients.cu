// Adapted from Fortran module written by:
// Matt Landreman, University of Maryland

#include <iostream>
#include "geometric_coefficients.h"
#include <netcdf.h>
#include "parameters.h"
#include <cmath>
#include "vmec_variables.h"
#include "solver.h"
#include <algorithm>

Geometric_coefficients::Geometric_coefficients(VMEC_variables *vmec_vars) : vmec(vmec_vars) {

  // the following are literals for now, but should be input file parameters
  alpha = 0.0;
  nzgrid = 16;
  desired_normalized_toroidal_flux = 0.27;
  zeta_center = 0.0;
  number_field_periods_to_include = 0;
  vmec_surface_option = 2;
  verbose = 1;

  // Reference length and magnetic field are chosen to be the GIST values
  // MM * may have to adapt this for GX normalizations
  std::cout << "Phi_vmec at LCFS = " << vmec->phi[vmec->ns-1] << "\n";
  //  std::cout << "signgs_vmec = " << vmec->signgs << "\n";
  // Note the first index difference of Fortran and c++ arrays

  // MM * removed the factor of signgs = -1 (not sure why that is here?)
  edge_toroidal_flux_over_2pi = vmec->phi[vmec->ns-1] / (2*M_PI);
  //  edge_toroidal_flux_over_2pi = vmec->phi[vmec->ns-1] / ( (2*M_PI)*vmec->signgs );
  L_reference = vmec->Aminor_p;

  // The following differs from the fortran version to allow for the toroidal flux to be either sign
  B_reference = (2*edge_toroidal_flux_over_2pi) / (L_reference*L_reference);
  //  B_reference = 2*abs(edge_toroidal_flux_over_2pi) / (L_reference*L_reference); // old value
  std::cout << "Reference Length = " << L_reference << "\n";
  std::cout << "Reference B Field = " << B_reference << "\n";

  // ------------------------------------------------------------
  // Allocate full and half grids of normalized toroidal flux

  normalized_toroidal_flux_full_grid = new double[vmec->ns];
  normalized_toroidal_flux_half_grid = new double[vmec->ns-1];
  for (int i=0; i<vmec->ns; i++) {
    normalized_toroidal_flux_full_grid[i] = static_cast<double>(i)/(vmec->ns-1);
    //    std::cout << normalized_toroidal_flux_full_grid[i] << "\n";
  }

  for (int i=0; i<vmec->ns-1; i++) {
    normalized_toroidal_flux_half_grid[i] = 0.5*(normalized_toroidal_flux_full_grid[i] + normalized_toroidal_flux_full_grid[i+1]);
    //    std::cout << normalized_toroidal_flux_half_grid[i] << "\n";
  }

  /* ------------------------------------------------------------
   Determine flux surface to use based on user inputs:
   -- desired_normalized_toroidal_flux
   -- vmec_surface_option

   Possible values of vmec_surface_option:
   0 = use exact radius requested
   1 = use nearest value of the VMEC half grid
   2 = use nearest value of the VMEC full grid
  */
  dr2_half = new double[vmec->ns-1];
  dr2_full = new double[vmec->ns];
  
  switch (vmec_surface_option) {
    
  case 0:
    // use exact radius requested
    normalized_toroidal_flux_used = desired_normalized_toroidal_flux;
    break;
      
  case 1:
    // use nearest value of the VMEC half grid
    for (int i=0; i<vmec->ns-1; i++) {
      dr2_half[i] = std::pow(normalized_toroidal_flux_half_grid[i] - desired_normalized_toroidal_flux,2);
      //      std::cout << "dr2["<< i << "] = " << dr2_half[i] << "\n";
    }

    index = 0;
    min_dr2 = dr2_half[0];
    // Find the index of minimum error (?)
    for (int i=1; i<vmec->ns-1; i++) {
      if (dr2_half[i] < min_dr2) {
	index = i;
	min_dr2 = dr2_half[i];
      }
    }
    normalized_toroidal_flux_used = normalized_toroidal_flux_half_grid[index];
    std::cout << "normalized flux used = " << normalized_toroidal_flux_used << "\n";
    break;

  case 2:
    // use nearest value of VMEC full grid
    for (int i=0; i<vmec->ns; i++) {
      dr2_full[i] = std::pow(normalized_toroidal_flux_full_grid[i] - desired_normalized_toroidal_flux,2);
    }
    index = 0;
    min_dr2 = dr2_full[0];

    // Find the index of minimum error (?)
    for (int i=1; i<vmec->ns; i++) {
      if (dr2_full[i] < min_dr2) {
	index = i;
	min_dr2 = dr2_full[i];
      }
    }
    normalized_toroidal_flux_used = normalized_toroidal_flux_full_grid[index];
    std::cout << "normalized flux used = " << normalized_toroidal_flux_used << "\n";
    break;

  default:
    std::cout << "Error! vmec_surface_option must be 0, 1, or 2. It is instead " << vmec_surface_option << "\n";
    exit(1);
  }
  delete[] dr2_half;
  delete[] dr2_full;

  /* In general, we get quantities for gs2 by linear interpolation, taking a weighted average of the quantity from 2 surfaces in the VMEC file. Sometimes the weights are 0 and 1, i.e. no interpolation is needed.
  
     For any VMEC quantity Q on the full grid, the value used in GS2 will be Q_gs2 = Q(vmec_radial_index_full(1))*vmec_radial_weight_full(1) + Q(vmec_radial_index_full(2))*vmec_radial_weight_full(2)

     For any VMEC quantity Q on the half grid, the value used in GS2 will be Q_gs2 = Q(vmec_radial_index_half(1))*vmec_radial_weight_half(1) + Q(vmec_radial_index_half(2))*vmec_radial_weight_half(2)
  */

  // For quantities on the full grid
  if (normalized_toroidal_flux_used > 1) {
    std::cout << "Error! normalized_toroidal_flux_used cannot be > 1\n";
    exit(1);
  }
  else if (normalized_toroidal_flux_used < 0) {
    std::cout << "Error! normalized_toroidal_flux_used cannot be < 0\n";
    exit(1);
  }
  else if (normalized_toroidal_flux_used == 1) {
    vmec_radial_index_full[0] = vmec->ns - 2;
    vmec_radial_index_full[1] = vmec->ns - 1;
    vmec_radial_weight_full[0] = 0.0;
  }
  else {
    std::cout << "***********************************************\n";
    std::cout << "confusing issue with indexing compared to fortran code. Check for more cases\n";
    vmec_radial_index_full[0] = std::floor(normalized_toroidal_flux_used*(vmec->ns));
    vmec_radial_index_full[1] = vmec_radial_index_full[0] + 1;
    //    std::cout << "vmec_radial_index_full[0, 1] = " << vmec_radial_index_full[0] << ", " << vmec_radial_index_full[1] << "\n";
    vmec_radial_weight_full[0] = vmec_radial_index_full[0] - normalized_toroidal_flux_used*(vmec->ns - 1.0)+1;
  }
  vmec_radial_weight_full[1] = 1.0 - vmec_radial_weight_full[0];
  //  std::cout << "vmec_radial_weight_full[0, 1] = " << vmec_radial_weight_full[0] << ", " << vmec_radial_weight_full[1] << "\n";

  // Handle quantities for the half grid:

  if (normalized_toroidal_flux_used < normalized_toroidal_flux_half_grid[0]) {
    std::cout << "Warning: extrapolating beyond the end of VMEC's half grid\n";
    std::cout << "(Extrapolating towards the magnetic axis) Results are likely to be inaccurate\n";
    
  // starting at element 2 since element 1 is always 0 for quantities on the half grid
    vmec_radial_index_half[0] = 1;
    vmec_radial_index_half[1] = 2;
    vmec_radial_weight_half[0] = (normalized_toroidal_flux_half_grid[1] - normalized_toroidal_flux_used) / (normalized_toroidal_flux_half_grid[1] - normalized_toroidal_flux_half_grid[0]);
  }
  
  else if (normalized_toroidal_flux_used > normalized_toroidal_flux_half_grid[vmec->ns-2]) {
    std::cout << "warning: extrapolating beyond the end of VMEC's half grid\n";
    std::cout << "(extrapolating towards the last closed flux surface) Results may be inaccurate\n";
    // largest index of half grid is vmec->ns - 2
    vmec_radial_index_half[0] = vmec->ns - 2;
    vmec_radial_index_half[1] = vmec->ns - 1;
    vmec_radial_weight_half[0] = (normalized_toroidal_flux_half_grid[vmec->ns-2] - normalized_toroidal_flux_used) / (normalized_toroidal_flux_half_grid[vmec->ns-2] - normalized_toroidal_flux_half_grid[vmec->ns-3]);
  }

  else if (normalized_toroidal_flux_used == normalized_toroidal_flux_half_grid[vmec->ns-2]) {
    // we are exactly at the last point of the half grid
    vmec_radial_index_half[0] = vmec->ns - 2;
    vmec_radial_index_half[1] = vmec->ns - 1;
    vmec_radial_weight_half[0] = 0.0;
  }

  else {
    // normalized_toroidal_flux_used is inside the half grid
    // this is the most common case
    vmec_radial_index_half[0] = std::floor(normalized_toroidal_flux_used*(vmec->ns-1) + 0.5);
    if (vmec_radial_index_half[0] < 2) {
      // This can occur sometimes due to roundoff error
      vmec_radial_index_half[0] = 2;
    }
    vmec_radial_index_half[1] = vmec_radial_index_half[0] + 1;
    vmec_radial_weight_half[0] = vmec_radial_index_half[0] - normalized_toroidal_flux_used*(vmec->ns - 1.0) + 0.5;
  }
  vmec_radial_weight_half[1] = 1.0 - vmec_radial_weight_half[0];
  std::cout << "***********************************************\n";
  std::cout << "confusing issue with indexing compared to fortran code. Check for more cases\n";
  
  if (verbose) {
    if (abs(vmec_radial_weight_half[0] < 1.e-14)) {
      std::cout << "Using radial index " << vmec_radial_index_half[1] << "of " << vmec->ns-1 << "from VMEC's half mesh\n";
      }
    else if (abs(vmec_radial_weight_half[1]) < 1.e-14) {
      std::cout << "Using radial index " << vmec_radial_index_half[0] << "of " << vmec->ns-1 << "from VMEC's half mest\n";
    }
    else {
      std::cout << "Interpolating using radial indices " << vmec_radial_index_half[0] << " and " << vmec_radial_index_half[1] << " of " << vmec->ns-1 << " from VMEC's half mesh\n";
      std::cout << "Weight for half mesh = " << vmec_radial_weight_half[0] << " and " << vmec_radial_weight_half[1] << "\n";
      std::cout << "Interpolating using radial indicies " << vmec_radial_index_full[0] << " and " << vmec_radial_index_full[1] << " of " << vmec->ns << " from VMEC's full mesh\n";
      std::cout << "Weight for full mesh = " << vmec_radial_weight_full[0] << " and " << vmec_radial_weight_full[1] << "\n";
    }
  }

  // Evaluate several radial-profile functions at the flux surface we ended up choosing

  iota = vmec->iotas[vmec_radial_index_half[0]]*vmec_radial_weight_half[0] + vmec->iotas[vmec_radial_index_half[1]]*vmec_radial_weight_half[1];
  if (verbose) { std::cout << "iota = " << iota << "\n"; }
  safety_factor_q = 1./iota;

  d_iota_ds_on_half_grid = new double[vmec->ns];
  d_pressure_ds_on_half_grid = new double[vmec->ns];

  d_iota_ds_on_half_grid[0] = 0;
  d_pressure_ds_on_half_grid[0] = 0.0;
  ds = normalized_toroidal_flux_full_grid[1] - normalized_toroidal_flux_full_grid[0];
  if (verbose) { std::cout << "ds = " << ds << "\n"; }

  for (int i=1; i<vmec->ns; i++) {
    d_iota_ds_on_half_grid[i] = (vmec->iotaf[i] - vmec->iotaf[i-1]) / ds;
    d_pressure_ds_on_half_grid[i] = (vmec->presf[i] - vmec->presf[i-1]) / ds;
    //    std::cout << "iota[1] = " << iotaf_vmec[i] << ", iota[0] = " << iotaf_vmec[i-1] << "\n";
    //    std::cout << "diota/ds = " << d_iota_ds_on_half_grid[i] << "\n";
  }
  d_iota_ds = d_iota_ds_on_half_grid[vmec_radial_index_half[0]]*vmec_radial_weight_half[0] + d_iota_ds_on_half_grid[vmec_radial_index_half[1]]*vmec_radial_weight_half[1];
  d_pressure_ds = d_pressure_ds_on_half_grid[vmec_radial_index_half[0]]*vmec_radial_weight_half[0] + d_pressure_ds_on_half_grid[vmec_radial_index_half[1]]*vmec_radial_weight_half[1];

  // shat = (r/q)(dq/dr) where r = a sqrt(s)
  //      = -(r/iota)(d iota / dr) = -2 (s/iota) (d iota/ ds)
  shat = (-2 * normalized_toroidal_flux_used / iota) * d_iota_ds;
  
  delete[] d_iota_ds_on_half_grid;
  delete[] d_pressure_ds_on_half_grid;
  if (verbose) {
    std::cout << "d iota / ds = " << d_iota_ds << "\n";
    std::cout << "d pressure / ds = " << d_pressure_ds << "\n";
  }

  // ----------------------------------------------------------------------
  // Set up the coordinate grids
  //
  //
  number_field_periods_to_include_final = number_field_periods_to_include;
  if (number_field_periods_to_include <= 0) {
    number_field_periods_to_include_final = vmec->nfp;
    if (verbose) {
      std::cout << "Since number_field_periods_to_include was <= 0, it is being reset to nfp = " << vmec->nfp << "\n";
    }
  }


  zeta = new double[2*nzgrid+1];
  for (int i=0; i<2*nzgrid+1; i++) {
    zeta[i] = zeta_center + (M_PI*(i-nzgrid)*number_field_periods_to_include_final) / (vmec->nfp*nzgrid);
  }
  
  
  // theta_pest = alpha + iota*zeta
  // Need to determine ---> theta_vmec = theta_pest - Lambda
  double* theta_vmec_double = new double[2*nzgrid+1]; // this is not a VMEC input
  theta_vmec = new double[2*nzgrid+1];

  // calling GSL rootfinder
  std::cout << "\n---------------------------------------------\n";
  std::cout << "Beginning root solves to determine theta_vmec\n";
  solver_vmec_theta(theta_vmec_double, zeta, nzgrid, alpha, iota, vmec, vmec_radial_index_half, vmec_radial_weight_half);
  std::cout << "Values of theta_vmec from GSL solver:\n[";
  for(int i=0; i<2*nzgrid+1; i++) {
    theta_vmec[i] = static_cast<double>(theta_vmec_double[i]);
    std::cout << theta_vmec[i] << " ";
  }
  std::cout << "]\n";
  std::cout << "---------------------------------------------\n\n";
  delete[] theta_vmec_double;
  
  // ---------------------------------------------------------
  // Alllocating geometry arrays
  // ---------------------------------------------------------

  // geometry variables
  B = new double[2*nzgrid+1]{};
  temp2D = new double[2*nzgrid+1]{};
  sqrt_g = new double[2*nzgrid+1]{};
  R = new double[2*nzgrid+1]{};
  dB_dtheta_vmec = new double[2*nzgrid+1]{};
  dB_dzeta = new double[2*nzgrid+1]{};
  dB_ds = new double[2*nzgrid+1]{};
  dR_dtheta_vmec = new double[2*nzgrid+1]{};
  dR_dzeta = new double[2*nzgrid+1]{};
  dR_ds = new double[2*nzgrid+1]{};
  dZ_dtheta_vmec = new double[2*nzgrid+1]{};
  dZ_dzeta = new double[2*nzgrid+1]{};
  dZ_ds = new double[2*nzgrid+1]{};
  dLambda_dtheta_vmec = new double[2*nzgrid+1]{};
  dLambda_dzeta = new double[2*nzgrid+1]{};
  dLambda_ds = new double[2*nzgrid+1]{};
  B_sub_s = new double[2*nzgrid+1]{};
  B_sub_theta_vmec = new double[2*nzgrid+1]{};
  B_sub_zeta = new double[2*nzgrid+1]{};
  B_sup_theta_vmec = new double[2*nzgrid+1]{};
  B_sup_zeta = new double[2*nzgrid+1]{};

  dB_ds_mnc = new double[vmec->ns]{};
  dB_ds_mns = new double[vmec->ns]{};
  dR_ds_mnc = new double[vmec->ns]{};
  dR_ds_mns = new double[vmec->ns]{};
  dZ_ds_mnc = new double[vmec->ns]{};
  dZ_ds_mns = new double[vmec->ns]{};
  dLambda_ds_mnc = new double[vmec->ns]{};
  dLambda_ds_mns = new double[vmec->ns]{};

  dX_ds = new double[2*nzgrid+1]{};
  dX_dtheta_vmec = new double[2*nzgrid+1]{};
  dX_dzeta = new double[2*nzgrid+1]{};
  dY_ds = new double[2*nzgrid+1]{};
  dY_dtheta_vmec = new double[2*nzgrid+1]{};
  dY_dzeta = new double[2*nzgrid+1]{};

  grad_s_X = new double[2*nzgrid+1]{};
  grad_s_Y = new double[2*nzgrid+1]{};
  grad_s_Z = new double[2*nzgrid+1]{};
  grad_theta_vmec_X = new double[2*nzgrid+1]{};
  grad_theta_vmec_Y = new double[2*nzgrid+1]{};
  grad_theta_vmec_Z = new double[2*nzgrid+1]{};
  grad_zeta_X = new double[2*nzgrid+1]{};
  grad_zeta_Y = new double[2*nzgrid+1]{};
  grad_zeta_Z = new double[2*nzgrid+1]{};
  grad_psi_X = new double[2*nzgrid+1]{};
  grad_psi_Y = new double[2*nzgrid+1]{};
  grad_psi_Z = new double[2*nzgrid+1]{};
  grad_alpha_X = new double[2*nzgrid+1]{};
  grad_alpha_Y = new double[2*nzgrid+1]{};
  grad_alpha_Z = new double[2*nzgrid+1]{};

  B_X = new double[2*nzgrid+1]{};
  B_Y = new double[2*nzgrid+1]{};
  B_Z = new double[2*nzgrid+1]{};
  grad_B_X = new double[2*nzgrid+1]{};
  grad_B_Y = new double[2*nzgrid+1]{};
  grad_B_Z = new double[2*nzgrid+1]{};
  B_cross_grad_B_dot_grad_alpha = new double[2*nzgrid+1]{};
  B_cross_grad_B_dot_grad_alpha_alternate = new double[2*nzgrid+1]{};
  B_cross_grad_s_dot_grad_alpha = new double[2*nzgrid+1]{};
  B_cross_grad_s_dot_grad_alpha_alternate = new double[2*nzgrid+1]{};
  
  //--------------------------------------------------------------
  // Now that we know grid points in theta_vmec, we can evaluate
  // all the geometric quantities on the grid points
  //--------------------------------------------------------------

  // All quantities needed except for R, Z, and Lambda use the _nyq mode numbers
  for(int imn_nyq=0; imn_nyq<vmec->mnmax_nyq; imn_nyq++) {

    m = vmec->xm_nyq[imn_nyq];
    n = vmec->xn_nyq[imn_nyq]/vmec->nfp;
    
    if (abs(m) >= vmec->mpol or abs(n) > vmec->ntor) {
      non_Nyquist_mode_available = false;
    }
    else {
      non_Nyquist_mode_available = true;
      // Find the imn in the non-Nyquist arrays that corresponds to the same m and n
      found_imn = false;
      for (int imn_ind=0; imn_ind<vmec->mnmax; imn_ind++) {
	if (vmec->xm[imn_ind] == m and vmec->xn[imn_ind] == n*vmec->nfp) {
	  found_imn = true;
	  imn = imn_ind;
	  break;
	}
      }

      if ( (vmec->xm[imn] != m) or (vmec->xn[imn] != n*vmec->nfp) ) {
	std::cout << "Something went wrong!\n";
	exit(1);
      }
      if (!found_imn) {
	std::cout << "Error! imn could not be found matching the given imn_nyq\n";
	exit(1);
      }
    }

    // All quantities are multiplied by a variable scale_factor which can in principle depend on m and n
    // For now we just set scale_factor = 1. In the future, scale_factor could be used to lower the symmetry-breaking Fourier comonents, or filter out certain Fourier components in some way
    scale_factor = 1;

    // -----------------------------------------------------
    // First, consider just the stellarator-symmetric terms
    // -----------------------------------------------------

    // Evaluate the radial derivative we will need:

    // B and Lambda are on the half mesh, so their radial derivatives are on the full mesh
    // R and Z are on the full mesh, so their radial derivatives are on the half mesh

    //if (imn_nyq == 3 and imn ==3 ) { std::cout << "dB_ds_mnc:\n[ "; }
    for (int i=1; i<vmec->ns-1; i++) {
      dB_ds_mnc[i] = (vmec->bmnc[imn_nyq + (i+1)*vmec->mnmax_nyq] - vmec->bmnc[imn_nyq + i*vmec->mnmax_nyq]) / ds;
      /*if (imn_nyq == 3 and imn == 3) {
      	std::cout << dB_ds_mnc[i] << ", ";
	}*/
    }
    // Simplistic extrapolation at the endpoints:
    dB_ds_mnc[0] = dB_ds_mnc[1];
    dB_ds_mnc[vmec->ns-1] = dB_ds_mnc[vmec->ns-2];
    //if (imn_nyq == 3 and imn ==3 ) { std::cout << "]\n\n\n"; }
    
    if (non_Nyquist_mode_available) {
      
      // R is on the full mesh
      //if (imn_nyq == 3 and imn ==3 ) { std::cout << "dR_ds_mnc:\n[ "; }
      for (int i=1; i<vmec->ns; i++) {
	dR_ds_mnc[i] = (vmec->rmnc[imn + i*vmec->mnmax] - vmec->rmnc[imn + (i-1)*vmec->mnmax]) / ds;
	dR_ds_mnc[0] = 0;
	/*if (imn_nyq == 3 and imn == 3) {
	  std::cout << dR_ds_mnc[i] << ", ";
	  }*/
      }
      //if (imn_nyq == 3 and imn == 3) { std::cout << "]\n\n\n"; }

      // Z is on the full mesh
      //if (imn_nyq == 3 and imn ==3 ) { std::cout << "dZ_ds_mns:\n[ "; }
      for (int i=1; i<vmec->ns; i++) {
	dZ_ds_mns[i] = (vmec->zmns[imn + i*vmec->mnmax] - vmec->zmns[imn + (i-1)*vmec->mnmax]) / ds;
	dZ_ds_mns[0] = 0;
	/*if (imn_nyq == 3 and imn == 3) {
	  std::cout << dZ_ds_mns[i] << ", ";
	  }*/
      }
      //if (imn_nyq == 3 and imn ==3 ) { std::cout << "]\n\n\n"; }
      
      // Lambda is on the half mesh
      //if (imn_nyq == 3 and imn ==3 ) { std::cout << "dLambda_ds_mnc:\n[ "; }
      for (int i=1; i<vmec->ns-1; i++) {
	dLambda_ds_mns[i] = (vmec->lmns[imn + (i+1)*vmec->mnmax] - vmec->lmns[imn + i*vmec->mnmax]) / ds;
	/*if (imn_nyq == 3 and imn == 3) {
	  std::cout << dLambda_ds_mnc[i] << ", ";
	  }*/
      }
      // Simplistic extrapolation at the endpoints:
      dLambda_ds_mns[0] = dLambda_ds_mns[1];
      dLambda_ds_mns[vmec->ns-1] = dLambda_ds_mns[vmec->ns-2];
      //if (imn_nyq == 3 and imn ==3 ) { std::cout << "]\n\n\n"; }
    }

    
    // End of evaluating radial derivatives

    // This next part would normally involve a loop over alpha, which is ignored here since nalpha = 1
    for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
      
      angle = m * theta_vmec[izeta] - n * vmec->nfp * zeta[izeta];
      cos_angle = cos(angle);
      sin_angle = sin(angle);
      for (int isurf=0; isurf<2; isurf++) {

	// Handle |B|
	temp = vmec->bmnc[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	temp = temp * scale_factor;
	B[izeta] = B[izeta] + temp*cos_angle;
	dB_dtheta_vmec[izeta] = dB_dtheta_vmec[izeta] - m*temp*sin_angle;
	dB_dzeta[izeta] = dB_dzeta[izeta] + n*vmec->nfp*temp*sin_angle;
	
	// Handle Jacobian
	temp = vmec->gmnc[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	temp = temp * scale_factor;
	sqrt_g[izeta] = sqrt_g[izeta] + temp * cos_angle;

	// Handle B sup theta
	temp = vmec->bsupumnc[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	temp = temp * scale_factor;
	B_sup_theta_vmec[izeta] = B_sup_theta_vmec[izeta] + temp * cos_angle;

	// Handle B sup zeta
	temp = vmec->bsupvmnc[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	temp = temp * scale_factor;
	B_sup_zeta[izeta] = B_sup_zeta[izeta] + temp * cos_angle;

	// Handle B sub theta
	temp = vmec->bsubumnc[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	temp = temp * scale_factor;
	B_sub_theta_vmec[izeta] = B_sub_theta_vmec[izeta] + temp * cos_angle;

	// Handle B sub zeta
	temp = vmec->bsubvmnc[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	temp = temp * scale_factor;
	B_sub_zeta[izeta] = B_sub_zeta[izeta] + temp * cos_angle;

	// Handle B sub psi
	// Unlike the other components of B, this one is on the full mesh
	temp = vmec->bsubsmns[imn_nyq + vmec_radial_index_full[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_full[isurf];
	//	std::cout << "\n\n";
	temp = temp * scale_factor;
	B_sub_s[izeta] = B_sub_s[izeta] + temp * sin_angle;

	// Handle dB/ds
	// Since bmnc is on the half mesh, its radial derivative is on the full mesh
	temp = dB_ds_mnc[vmec_radial_index_full[isurf]] * vmec_radial_weight_full[isurf];
	temp = temp * scale_factor;
	dB_ds[izeta] = dB_ds[izeta] + temp * cos_angle;

	// Handle arrays that use xm and xn instead of xm_nyq and xn_nyq
	if (non_Nyquist_mode_available) {

	  // Handle R, which is on the full mesh
	  temp = vmec->rmnc[imn + vmec_radial_index_full[isurf] * vmec->mnmax] * vmec_radial_weight_full[isurf];
	  temp = temp * scale_factor;
	  R[izeta] = R[izeta] + temp * cos_angle;
	  dR_dtheta_vmec[izeta] = dR_dtheta_vmec[izeta] - temp * m * sin_angle;
	  dR_dzeta[izeta] = dR_dzeta[izeta] + temp * n * vmec->nfp * sin_angle;

	  // Handle Z, which is on the full mesh
	  temp = vmec->zmns[imn + vmec_radial_index_full[isurf] * vmec->mnmax] * vmec_radial_weight_full[isurf];
	  temp = temp * scale_factor;
	  // only need derivatives of Z
	  dZ_dtheta_vmec[izeta] = dZ_dtheta_vmec[izeta] + temp * m * cos_angle;
	  dZ_dzeta[izeta] = dZ_dzeta[izeta] - temp * n * vmec->nfp * cos_angle;

	  // Handle Lambda
	  temp = vmec->lmns[imn + vmec_radial_index_half[isurf] * vmec->mnmax] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  // only need derivative of Lambda
	  dLambda_dtheta_vmec[izeta] = dLambda_dtheta_vmec[izeta] + m * temp * cos_angle;
	  dLambda_dzeta[izeta] = dLambda_dzeta[izeta] - n * vmec->nfp * temp * cos_angle;

	  // Handle dR/ds
	  // Since R is on the full mesh, its radial derivative is on the half mesh
	  temp = dR_ds_mnc[vmec_radial_index_half[isurf]] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  dR_ds[izeta] = dR_ds[izeta] + temp * cos_angle;

	  // Handle dZ/ds
	  // Since Z is on the full mesh, its radial derivative is on the half mesh
	  temp = dZ_ds_mns[vmec_radial_index_half[isurf]] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  dZ_ds[izeta] = dZ_ds[izeta] + temp * sin_angle;

	  // Handle dLambda/ds
	  // Since Lambda is on the half mesh, its radial derivative is on the full mesh
	  temp = dLambda_ds_mns[vmec_radial_index_full[isurf]] * vmec_radial_weight_full[isurf];
	  temp = temp * scale_factor;
	  dLambda_ds[izeta] = dLambda_ds[izeta] + temp * sin_angle;

	}
      }
     
    }

    if (vmec->lasym) {

      // Evaluate the radial derivatives we will need
      // B and Lambda are on the half mesh, so their radial derivatives are on the full mesh
      // R and Z are on the full mesh, so their radial derivatives are on the half mesh
      
      for (int i=1; i<vmec->ns-1; i++) {
	dB_ds_mns[i] = (vmec->bmns[imn_nyq + (i+1)*vmec->mnmax_nyq] - vmec->bmns[imn_nyq + i*vmec->mnmax_nyq]) / ds;
      }
      // Simplistic extrapolation at the endpoints:
      dB_ds_mns[0] = dB_ds_mns[1];
      dB_ds_mns[vmec->ns-1] = dB_ds_mns[vmec->ns-2];
    
      if (non_Nyquist_mode_available) {
	
	// R is on the full mesh
	for (int i=1; i<vmec->ns; i++) {
	  dR_ds_mns[i] = (vmec->rmns[imn + i*vmec->mnmax] - vmec->rmns[imn + (i-1)*vmec->mnmax]) / ds;
	  dR_ds_mns[0] = 0;
	}
	
	// Z is on the full mesh
	for (int i=1; i<vmec->ns; i++) {
	  dZ_ds_mnc[i] = (vmec->zmnc[imn + i*vmec->mnmax] - vmec->zmnc[imn + (i-1)*vmec->mnmax]) / ds;
	  dZ_ds_mnc[0] = 0;
	}
	
	// Lambda is on the half mesh
	for (int i=1; i<vmec->ns-1; i++) {
	  dLambda_ds_mnc[i] = (vmec->lmnc[imn + (i+1)*vmec->mnmax] - vmec->lmnc[imn + i*vmec->mnmax]) / ds;
	}
	// Simplistic extrapolation at the endpoints:
	dLambda_ds_mnc[0] = dLambda_ds_mnc[1];
	dLambda_ds_mnc[vmec->ns-1] = dLambda_ds_mnc[vmec->ns-2];
      }
      
      // This next part would normally involve a loop over alpha, which is ignored here since nalpha = 1
      for (int izeta=0; izeta < 2*nzgrid+1; izeta++) {

	angle = m * theta_vmec[izeta] - n * vmec->nfp * zeta[izeta];
	cos_angle = cos(angle);
	sin_angle = sin(angle);
	for (int isurf=0; isurf<2; isurf++) {
	  
	  // Handle |B|
	  temp = vmec->bmns[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  B[izeta] = B[izeta] + temp*cos_angle;
	  dB_dtheta_vmec[izeta] = dB_dtheta_vmec[izeta] - m*temp*sin_angle;
	  dB_dzeta[izeta] = dB_dzeta[izeta] + n*vmec->nfp*temp*sin_angle;
	  
	  // Handle Jacobian
	  temp = vmec->gmns[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  sqrt_g[izeta] = sqrt_g[izeta] + temp * cos_angle;
	  
	  // Handle B sup theta
	  temp = vmec->bsupumns[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  B_sup_theta_vmec[izeta] = B_sup_theta_vmec[izeta] + temp * cos_angle;
	  
	  // Handle B sup zeta
	  temp = vmec->bsupvmns[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  B_sup_zeta[izeta] = B_sup_zeta[izeta] + temp * cos_angle;
	  
	  // Handle B sub theta
	  temp = vmec->bsubumns[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  B_sub_theta_vmec[izeta] = B_sub_theta_vmec[izeta] + temp * cos_angle;
	  
	  // Handle B sub zeta
	  temp = vmec->bsubvmns[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  B_sub_zeta[izeta] = B_sub_zeta[izeta] + temp * cos_angle;

	  // Handle B sub psi
	  // Unlike the other components of B, this one is on the full mesh
	  temp = vmec->bsubsmnc[imn_nyq + vmec_radial_index_full[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_full[isurf];
	  temp = temp * scale_factor;
	  B_sub_s[izeta] = B_sub_s[izeta] + temp * sin_angle;

	  // Handle dB/ds
	  // Since bmnc is on the half mesh, its radial derivative is on the full mesh
	  temp = dB_ds_mns[vmec_radial_index_full[isurf]] * vmec_radial_weight_full[isurf];
	  temp = temp * scale_factor;
	  dB_ds[izeta] = dB_ds[izeta] + temp * cos_angle;

	  // Handle arrays that use xm and xn instead of xm_nyq and xn_nyq
	  if (non_Nyquist_mode_available) {
	    
	    // Handle R, which is on the full mesh
	    temp = vmec->rmns[imn + vmec_radial_index_full[isurf] * vmec->mnmax] * vmec_radial_weight_full[isurf];
	    temp = temp * scale_factor;
	    R[izeta] = R[izeta] + temp * cos_angle;
	    dR_dtheta_vmec[izeta] = dR_dtheta_vmec[izeta] - temp * m * sin_angle;
	    dR_dzeta[izeta] = dR_dzeta[izeta] + temp * n * vmec->nfp * sin_angle;
	    
	    // Handle Z, which is on the full mesh
	    temp = vmec->zmnc[imn + vmec_radial_index_full[isurf] * vmec->mnmax] * vmec_radial_weight_full[isurf];
	    temp = temp * scale_factor;
	    // only need derivatives of Z
	    dZ_dtheta_vmec[izeta] = dZ_dtheta_vmec[izeta] + temp * m * cos_angle;
	    dZ_dzeta[izeta] = dZ_dzeta[izeta] - temp * n * vmec->nfp * cos_angle;
	    
	    // Handle Lambda
	    temp = vmec->lmnc[imn + vmec_radial_index_half[isurf] * vmec->mnmax] * vmec_radial_weight_half[isurf];
	    temp = temp * scale_factor;
	    // only need derivative of Lambda
	    dLambda_dtheta_vmec[izeta] = dLambda_dtheta_vmec[izeta] + m * temp * cos_angle;
	    dLambda_dzeta[izeta] = dLambda_dzeta[izeta] - n * vmec->nfp * temp * cos_angle;

	    // Handle dR/ds
	    // Since R is on the full mesh, its radial derivative is on the half mesh
	    temp = dR_ds_mns[vmec_radial_index_half[isurf]] * vmec_radial_weight_half[isurf];
	    temp = temp * scale_factor;
	    dR_ds[izeta] = dR_ds[izeta] + temp * cos_angle;

	    // Handle dZ/ds
	    // Since Z is on the full mesh, its radial derivative is on the half mesh
	    temp = dZ_ds_mnc[vmec_radial_index_half[isurf]] * vmec_radial_weight_half[isurf];
	    temp = temp * scale_factor;
	    dZ_ds[izeta] = dZ_ds[izeta] + temp * sin_angle;
	    
	    // Handle dLambda/ds
	    // Since Lambda is on the half mesh, its radial derivative is on the full mesh
	    temp = dLambda_ds_mnc[vmec_radial_index_full[isurf]] * vmec_radial_weight_full[isurf];
	    temp = temp * scale_factor;
	    dLambda_ds[izeta] = dLambda_ds[izeta] + temp * sin_angle;

	  }
	}	
      } // nzgrid loop
    } // stellarator-asymmetric if-statement 
  } // loop over imn_nyq

  
  //--------------------------------------------------------------------
  // Sanity check: If the conversion to theta_pest has been done correctly
  // we should find that
  // (B dot grad theta_pest) / (B dot grad zeta) = iota
  // Verifying that:
  //--------------------------------------------------------------------

  
  B_dot_grad_theta_pest_over_B_dot_grad_zeta = new double[2*nzgrid+1]{};
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    B_dot_grad_theta_pest_over_B_dot_grad_zeta[izeta] = (B_sup_theta_vmec[izeta] * (1.0 + dLambda_dtheta_vmec[izeta]) + B_sup_zeta[izeta] * dLambda_dzeta[izeta]) / B_sup_zeta[izeta];
    temp2D[izeta] = iota;
  }
  test_arrays(B_dot_grad_theta_pest_over_B_dot_grad_zeta, temp2D, false, 0.01, "iota");
  delete[] B_dot_grad_theta_pest_over_B_dot_grad_zeta;

  //--------------------------------------------------------------------
  // Using R(theta, zeta) and Z(theta, zeta), compute Cartesian
  // components of gradient basis vectors using the dual relations
  //--------------------------------------------------------------------

  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    cos_angle = cos(zeta[izeta]);
    sin_angle = sin(zeta[izeta]);

    // X = R*cos(zeta)
    dX_dtheta_vmec[izeta] = dR_dtheta_vmec[izeta]*cos_angle;
    dX_dzeta[izeta] = dR_dzeta[izeta]*cos_angle - R[izeta]*sin_angle;
    dX_ds[izeta] = dR_ds[izeta]*cos_angle;

    // Y = R*sin(zeta)
    dY_dtheta_vmec[izeta] = dR_dtheta_vmec[izeta]*sin_angle;
    dY_dzeta[izeta] = dR_dzeta[izeta]*sin_angle + R[izeta]*cos_angle;
    dY_ds[izeta] = dR_ds[izeta]*sin_angle;
  }

  // Use the dual relations to get the Cartesian components of grad s, grad theta, grad theta_vmec, and grad_zeta
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    grad_s_X[izeta] = (dY_dtheta_vmec[izeta] * dZ_dzeta[izeta] - dZ_dtheta_vmec[izeta] * dY_dzeta[izeta]) / sqrt_g[izeta];
    grad_s_Y[izeta] = (dZ_dtheta_vmec[izeta] * dX_dzeta[izeta] - dX_dtheta_vmec[izeta] * dZ_dzeta[izeta]) / sqrt_g[izeta];
    grad_s_Z[izeta] = (dX_dtheta_vmec[izeta] * dY_dzeta[izeta] - dY_dtheta_vmec[izeta] * dX_dzeta[izeta]) / sqrt_g[izeta];

    grad_theta_vmec_X[izeta] = (dY_dzeta[izeta] * dZ_ds[izeta] - dZ_dzeta[izeta] * dY_ds[izeta]) / sqrt_g[izeta];
    grad_theta_vmec_Y[izeta] = (dZ_dzeta[izeta] * dX_ds[izeta] - dX_dzeta[izeta] * dZ_ds[izeta]) / sqrt_g[izeta];
    grad_theta_vmec_Z[izeta] = (dX_dzeta[izeta] * dY_ds[izeta] - dY_dzeta[izeta] * dX_ds[izeta]) / sqrt_g[izeta];

    grad_zeta_X[izeta] = (dY_ds[izeta] * dZ_dtheta_vmec[izeta] - dZ_ds[izeta] * dY_dtheta_vmec[izeta]) / sqrt_g[izeta];
    grad_zeta_Y[izeta] = (dZ_ds[izeta] * dX_dtheta_vmec[izeta] - dX_ds[izeta] * dZ_dtheta_vmec[izeta]) / sqrt_g[izeta];
    grad_zeta_Z[izeta] = (dX_ds[izeta] * dY_dtheta_vmec[izeta] - dY_ds[izeta] * dX_dtheta_vmec[izeta]) / sqrt_g[izeta];
  }
  // End of the dual relations

  // Sanity check: grad_zeta_X should be -sin(zeta)/R
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    temp2D[izeta] = -sin(zeta[izeta]) / R[izeta];
  }
  test_arrays(grad_zeta_X, temp2D, false, 1.0e-2, "grad_zeta_X");
  // Might as well use the exact value, which is currently in temp2D
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    grad_zeta_X[izeta] = temp2D[izeta];
  }

  // Sanity check: grad_zeta_Y should be cos(zeta)/R
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    temp2D[izeta] = cos(zeta[izeta]) / R[izeta];
  }
  test_arrays(grad_zeta_Y, temp2D, false, 1.0e-2, "grad_zeta_X");
  // Might as well use the exact value, which is currently in temp2D
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    grad_zeta_Y[izeta] = temp2D[izeta];
  }

  // Sanity check: grad_zeta_Z should be 0
  test_arrays(grad_zeta_Z, temp2D, true, 1.0e-6, "grad_zeta_Z");
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    grad_zeta_Z[izeta] = 0.0;
  }

  //-------------------------------------------------------------
  // Compute Cartesian components of other quantities we need
  //-------------------------------------------------------------

  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    grad_psi_X[izeta] = grad_s_X[izeta] * abs(edge_toroidal_flux_over_2pi);
    grad_psi_Y[izeta] = grad_s_Y[izeta] * abs(edge_toroidal_flux_over_2pi);
    grad_psi_Z[izeta] = grad_s_Z[izeta] * abs(edge_toroidal_flux_over_2pi);

    /*    grad_psi_X[izeta] = grad_s_X[izeta] * edge_toroidal_flux_over_2pi;
    grad_psi_Y[izeta] = grad_s_Y[izeta] * edge_toroidal_flux_over_2pi;
    grad_psi_Z[izeta] = grad_s_Z[izeta] * edge_toroidal_flux_over_2pi;*/

    grad_alpha_X[izeta] = (dLambda_ds[izeta] - zeta[izeta]*d_iota_ds) * grad_s_X[izeta] + (1.0 + dLambda_dtheta_vmec[izeta]) * grad_theta_vmec_X[izeta] + (-iota + dLambda_dzeta[izeta]) * grad_zeta_X[izeta];
    grad_alpha_Y[izeta] = (dLambda_ds[izeta] - zeta[izeta]*d_iota_ds) * grad_s_Y[izeta] + (1.0 + dLambda_dtheta_vmec[izeta]) * grad_theta_vmec_Y[izeta] + (-iota + dLambda_dzeta[izeta]) * grad_zeta_Y[izeta];
    grad_alpha_Z[izeta] = (dLambda_ds[izeta] - zeta[izeta]*d_iota_ds) * grad_s_Z[izeta] + (1.0 + dLambda_dtheta_vmec[izeta]) * grad_theta_vmec_Z[izeta] + (-iota + dLambda_dzeta[izeta]) * grad_zeta_Z[izeta];

    grad_B_X[izeta] = dB_ds[izeta] * grad_s_X[izeta] + dB_dtheta_vmec[izeta] * grad_theta_vmec_X[izeta] + dB_dzeta[izeta] * grad_zeta_X[izeta];
    grad_B_Y[izeta] = dB_ds[izeta] * grad_s_Y[izeta] + dB_dtheta_vmec[izeta] * grad_theta_vmec_Y[izeta] + dB_dzeta[izeta] * grad_zeta_Y[izeta];
    grad_B_Z[izeta] = dB_ds[izeta] * grad_s_Z[izeta] + dB_dtheta_vmec[izeta] * grad_theta_vmec_Z[izeta] + dB_dzeta[izeta] * grad_zeta_Z[izeta];

    // Need to be careful with sign of toroidal flux here!
    /*B_X[izeta] = edge_toroidal_flux_over_2pi * ((1.0 + dLambda_dtheta_vmec[izeta]) * dX_dzeta[izeta] + (iota - dLambda_dzeta[izeta]) * dX_dtheta_vmec[izeta]) / sqrt_g[izeta];
    B_Y[izeta] = edge_toroidal_flux_over_2pi * ((1.0 + dLambda_dtheta_vmec[izeta]) * dY_dzeta[izeta] + (iota - dLambda_dzeta[izeta]) * dY_dtheta_vmec[izeta]) / sqrt_g[izeta];
    B_Z[izeta] = edge_toroidal_flux_over_2pi * ((1.0 + dLambda_dtheta_vmec[izeta]) * dZ_dzeta[izeta] + (iota - dLambda_dzeta[izeta]) * dZ_dtheta_vmec[izeta]) / sqrt_g[izeta];*/

    B_X[izeta] = abs(edge_toroidal_flux_over_2pi) * ((1.0 + dLambda_dtheta_vmec[izeta]) * dX_dzeta[izeta] + (iota - dLambda_dzeta[izeta]) * dX_dtheta_vmec[izeta]) / sqrt_g[izeta];
    B_Y[izeta] = abs(edge_toroidal_flux_over_2pi) * ((1.0 + dLambda_dtheta_vmec[izeta]) * dY_dzeta[izeta] + (iota - dLambda_dzeta[izeta]) * dY_dtheta_vmec[izeta]) / sqrt_g[izeta];
    B_Z[izeta] = abs(edge_toroidal_flux_over_2pi) * ((1.0 + dLambda_dtheta_vmec[izeta]) * dZ_dzeta[izeta] + (iota - dLambda_dzeta[izeta]) * dZ_dtheta_vmec[izeta]) / sqrt_g[izeta];

  }
    
  sqrt_s = sqrt(normalized_toroidal_flux_used);

  //---------------------------------------------------------------
  // Sanity Tests: Verify that the Jacobian equals the appropriate
  // cross product of the basis vectors
  //---------------------------------------------------------------

  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    temp2D[izeta] = 0.0
      + dX_ds[izeta] * dY_dtheta_vmec[izeta] * dZ_dzeta[izeta]
      + dY_ds[izeta] * dZ_dtheta_vmec[izeta] * dX_dzeta[izeta]
      + dZ_ds[izeta] * dX_dtheta_vmec[izeta] * dY_dzeta[izeta]
      - dZ_ds[izeta] * dY_dtheta_vmec[izeta] * dX_dzeta[izeta]
      - dX_ds[izeta] * dZ_dtheta_vmec[izeta] * dY_dzeta[izeta]
      - dY_ds[izeta] * dX_dtheta_vmec[izeta] * dZ_dzeta[izeta];
  }
  test_arrays(sqrt_g, temp2D, false, 3.0e-3, "sqrt_g");

  double *inv_sqrt_g = new double[2*nzgrid+1]{};
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    temp2D[izeta] = 0.0
      + grad_s_X[izeta] * grad_theta_vmec_Y[izeta] * grad_zeta_Z[izeta]
      + grad_s_Y[izeta] * grad_theta_vmec_Z[izeta] * grad_zeta_X[izeta]
      + grad_s_Z[izeta] * grad_theta_vmec_X[izeta] * grad_zeta_Y[izeta]
      - grad_s_Z[izeta] * grad_theta_vmec_Y[izeta] * grad_zeta_X[izeta]
      - grad_s_X[izeta] * grad_theta_vmec_Z[izeta] * grad_zeta_Y[izeta]
      - grad_s_Y[izeta] * grad_theta_vmec_X[izeta] * grad_zeta_Z[izeta];
    inv_sqrt_g[izeta] = 1./sqrt_g[izeta];
  }
  
  test_arrays(inv_sqrt_g, temp2D, false, 1.e-2, "1/sqrt_g");
  delete[] inv_sqrt_g;

  //---------------------------------------------------------------
  // Sanity Tests: Verify that 
  // \vec{B} dot (each of the covariant and contravariant basis vectors)
  // matches the corresponding term in VMEC
  //---------------------------------------------------------------

  double *B_sub_theta_calc = new double[2*nzgrid+1]{};
  double *B_sub_zeta_calc = new double[2*nzgrid+1]{};
  double *B_sub_s_calc = new double[2*nzgrid+1]{};
  double *B_sup_theta_calc = new double[2*nzgrid+1]{};
  double *B_sup_zeta_calc = new double[2*nzgrid+1]{};
  double *B_sup_s_calc = new double[2*nzgrid+1]{};
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    B_sub_theta_calc[izeta] = B_X[izeta] * dX_dtheta_vmec[izeta] + B_Y[izeta] * dY_dtheta_vmec[izeta] + B_Z[izeta] * dZ_dtheta_vmec[izeta];
    B_sub_zeta_calc[izeta] = B_X[izeta] * dX_dzeta[izeta] + B_Y[izeta] * dY_dzeta[izeta] + B_Z[izeta] * dZ_dzeta[izeta];
    B_sub_s_calc[izeta] = B_X[izeta] * dX_ds[izeta] + B_Y[izeta] * dY_ds[izeta] + B_Z[izeta] * dZ_ds[izeta];
    
    B_sup_theta_calc[izeta] = B_X[izeta] * grad_theta_vmec_X[izeta] + B_Y[izeta] * grad_theta_vmec_Y[izeta] + B_Z[izeta] * grad_theta_vmec_Z[izeta];
    B_sup_zeta_calc[izeta] = B_X[izeta] * grad_zeta_X[izeta] + B_Y[izeta] * grad_zeta_Y[izeta] + B_Z[izeta] * grad_zeta_Z[izeta];
    B_sup_s_calc[izeta] = B_X[izeta] * grad_s_X[izeta] + B_Y[izeta] * grad_s_Y[izeta] + B_Z[izeta] * grad_s_Z[izeta];
  }
  // The VMEC arrays B_sub_i do not appear to have a concept of the sign of the toroidal flux, so if edge_toroidal_flux_over_2pi is negative, this will cause our comparison to have opposite signs

  std::cout << "B_sub_s = [";
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    std::cout << B_sub_s[izeta] << ", ";
  }
  std::cout << "]\n\n";

  std::cout << "B_sub_s_calc = [";
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    std::cout << B_sub_s_calc[izeta] << ", ";
  }
  std::cout << "]\n\n";
  exit(1);
  /*
  test_arrays(B_sub_theta_calc, B_sub_theta_vmec, false, 1.0e-2, "B_sub_theta_vmec");
  test_arrays(B_sub_zeta_calc, B_sub_zeta, false, 1.0e-2, "B_sub_zeta");
  test_arrays(B_sub_s_calc, B_sub_s, false, 1.0e-2, "B_sub_s");

  test_arrays(B_sup_theta_calc, B_sup_theta_vmec, false, 1.0e-2, "B_sup_theta_vmec");
  test_arrays(B_sup_zeta_calc, B_sup_zeta, false, 1.0e-2, "B_sup_zeta");
  test_arrays(B_sup_s_calc, temp2D, true, 1.0e-2, "B_sup_s");
  */

  //---------------------------------------------------------------
  // For gbdrift, we need \vec{B} cross grad |B| dot grad alpha
  // For cvdrift, we also need \vec{B} cross grad s dot grad alpha
  // Let us compute both of these quantities 2 ways, and make sure the two
  // approaches give the same answer (within some tolerance)
  //---------------------------------------------------------------

  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    
    B_cross_grad_s_dot_grad_alpha[izeta] = ( B_sub_zeta[izeta] * (1.0 + dLambda_dtheta_vmec[izeta]) - B_sub_theta_vmec[izeta] * (dLambda_dzeta[izeta] - iota) ) / sqrt_g[izeta];
    
    B_cross_grad_s_dot_grad_alpha_alternate[izeta] = 0
      + B_X[izeta] * grad_s_Y[izeta] * grad_alpha_Z[izeta]
      + B_Y[izeta] * grad_s_Z[izeta] * grad_alpha_X[izeta]
      + B_Z[izeta] * grad_s_X[izeta] * grad_alpha_Y[izeta]
      - B_Z[izeta] * grad_s_Y[izeta] * grad_alpha_X[izeta]
      - B_X[izeta] * grad_s_Z[izeta] * grad_alpha_Y[izeta]
      - B_Y[izeta] * grad_s_X[izeta] * grad_alpha_Z[izeta];

    B_cross_grad_B_dot_grad_alpha[izeta] = 0
      + (B_sub_s[izeta] * dB_dtheta_vmec[izeta] * (dLambda_dzeta[izeta] - iota)
	 + B_sub_theta_vmec[izeta] * dB_dzeta[izeta] * (dLambda_ds[izeta] - zeta[izeta] * d_iota_ds)
	 + B_sub_zeta[izeta] * dB_ds[izeta] * (1.0 + dLambda_dtheta_vmec[izeta])
	 - B_sub_zeta[izeta] * dB_dtheta_vmec[izeta] * (dLambda_ds[izeta] - zeta[izeta] * d_iota_ds)
	 - B_sub_theta_vmec[izeta] * dB_ds[izeta] * (dLambda_dzeta[izeta] - iota)
	 - B_sub_s[izeta] * dB_dzeta[izeta] * (1.0 + dLambda_dtheta_vmec[izeta])) / sqrt_g[izeta];

    B_cross_grad_B_dot_grad_alpha_alternate[izeta] = 0
      + B_X[izeta] * grad_B_Y[izeta] * grad_alpha_Z[izeta]
      + B_Y[izeta] * grad_B_Z[izeta] * grad_alpha_X[izeta]
      + B_Z[izeta] * grad_B_X[izeta] * grad_alpha_Y[izeta]
      - B_Z[izeta] * grad_B_Y[izeta] * grad_alpha_X[izeta]
      - B_X[izeta] * grad_B_Z[izeta] * grad_alpha_Y[izeta]
      - B_Y[izeta] * grad_B_X[izeta] * grad_alpha_Z[izeta];
  }
  //  test_arrays(B_cross_grad_s_dot_grad_alpha, B_cross_grad_s_dot_grad_alpha_alternate, false, 1.0e-2, "B_cross_grad_s_dot_grad_alpha");
  //  test_arrays(B_cross_grad_B_dot_grad_alpha, B_cross_grad_B_dot_grad_alpha_alternate, false, 1.0e-2, "B_cross_grad_B_dot_grad_alpha");

  //--------------------------------------------------------------
  // Finally, assemble the quantities needed for GX
  //
  // For a derivation of the following formulae, see the .tex document
  // included in this directory

  bmag_temp = new double[2*nzgrid+1]{};
  gradpar_temp = new double[2*nzgrid+1]{};
  gds2_temp = new double[2*nzgrid+1]{};
  gds21_temp = new double[2*nzgrid+1]{};
  gds22_temp = new double[2*nzgrid+1]{};
  gbdrift_temp = new double[2*nzgrid+1]{};
  gbdrift0_temp = new double[2*nzgrid+1]{};
  cvdrift_temp = new double[2*nzgrid+1]{};
  cvdrift0_temp = new double[2*nzgrid+1]{};
  
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    
    bmag_temp[izeta] = abs( B[izeta] / B_reference );

    // need to check this. Using theta to set gradpar, as opposed to zeta
    // for the full surface version
    gradpar_temp[izeta] = ( L_reference * B_sup_theta_vmec[izeta] ) / B[izeta];

    gds2_temp[izeta] = (grad_alpha_X[izeta] * grad_alpha_X[izeta] + grad_alpha_Y[izeta] * grad_alpha_Y[izeta] + grad_alpha_Z[izeta] * grad_alpha_Z[izeta]) * L_reference * L_reference * normalized_toroidal_flux_used;

    gds21_temp[izeta] = (grad_alpha_X[izeta] * grad_psi_X[izeta] + grad_alpha_Y[izeta] * grad_psi_Y[izeta] + grad_alpha_Z[izeta] * grad_psi_Z[izeta]) * (shat / B_reference);
    
    gds22_temp[izeta] = (grad_psi_X[izeta] * grad_psi_X[izeta] + grad_psi_Y[izeta] * grad_psi_Y[izeta] + grad_psi_Z[izeta] * grad_psi_Z[izeta]) * ( (shat * shat) / (L_reference * L_reference * B_reference * B_reference * normalized_toroidal_flux_used) );

    gbdrift_temp[izeta] = 2 * B_reference * L_reference * L_reference * sqrt_s * B_cross_grad_B_dot_grad_alpha[izeta] / ( B[izeta] * B[izeta] * B[izeta] );

    gbdrift0_temp[izeta] = ( (B_sub_theta_vmec[izeta] * dB_dzeta[izeta] - B_sub_zeta[izeta] * dB_dtheta_vmec[izeta]) / sqrt_g[izeta] )
      * ( (edge_toroidal_flux_over_2pi * 2 * shat) / (B[izeta] * B[izeta] * B[izeta] * sqrt_s) );
    // In the above expression for gbdrift0, the first line and the edge_toroidal_flux_over_2pi is \vec{B} \times \nabla B \cdot \nabla \psi

    double mu_0 = 4*M_PI*(1.0e-7);
    cvdrift_temp[izeta] = gbdrift_temp[izeta] + 2 * B_reference * L_reference * L_reference * sqrt_s * mu_0 * d_pressure_ds * B_cross_grad_s_dot_grad_alpha[izeta] / (B[izeta] *B[izeta] * B[izeta] * B[izeta]);

    cvdrift0_temp[izeta] = gbdrift0_temp[izeta];

  }
  /*  
  std::cout << "bmag = [";
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    std::cout << bmag[izeta] << ", ";
  }
  std::cout << "\n\n";

  std::cout << "gradpar = [";
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    std::cout << gradpar[izeta] << ", ";
  }
  std::cout << "\n\n";

  std::cout << "gds2 = [";
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    std::cout << gds2[izeta] << ", ";
  }
  std::cout << "\n\n";

  std::cout << "gds21 = [";
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    std::cout << gds21[izeta] << ", ";
  }
  std::cout << "\n\n";

  std::cout << "gds22 = [";
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    std::cout << gds22[izeta] << ", ";
  }
  std::cout << "\n\n";

  std::cout << "gbdrift = [";
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    std::cout << gbdrift[izeta] << ", ";
  }
  std::cout << "\n\n";

  std::cout << "gbdrift0 = [";
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    std::cout << gbdrift0[izeta] << ", ";
  }
  std::cout << "\n\n";

  std::cout << "cvdrift = [";
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    std::cout << cvdrift[izeta] << ", ";
  }
  std::cout << "\n\n";

  std::cout << "cvdrift0 = [";
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    std::cout << cvdrift0[izeta] << ", ";
  }
  std::cout << "\n\n";
  */


  //-----------------------------------------------------------------
  // Due to the FFTs in GX, need to have a grid where gradpar=const
  //
  // So we must create such a grid, and interpolate the "gs2"
  // geometric arrays onto this uniform Z grid
  
  bmag = new double[2*nzgrid+1]{};
  gradpar = new double[2*nzgrid+1]{};
  gds2 = new double[2*nzgrid+1]{};
  gds21 = new double[2*nzgrid+1]{};
  gds22 = new double[2*nzgrid+1]{};
  gbdrift = new double[2*nzgrid+1]{};
  gbdrift0 = new double[2*nzgrid+1]{};
  cvdrift = new double[2*nzgrid+1]{};
  cvdrift0 = new double[2*nzgrid+1]{};
  
  gradpar_half_grid = new double[2*nzgrid]{};
  double *temp = new double[2*nzgrid+1]{};
  double *zOnThetaGrid = new double[2*nzgrid+1]{};
  
  //  double dtheta = theta_vmec[1] - theta_vmec[0];
  double dtheta = zeta[1] - zeta[0];
  
  std::cout << "gradpar_half_grid = [";
  for (int izeta=0; izeta<2*nzgrid; izeta++) {
    gradpar_half_grid[izeta] = 0.5 * (abs(gradpar_temp[izeta]) + abs(gradpar_temp[izeta+1]));
    std::cout << gradpar_half_grid[izeta] << ", ";
  }
  std::cout << "\n\n";

  for (int itheta=1; itheta<2*nzgrid; itheta++) {
    temp[itheta] = temp[itheta-1] + dtheta * (1. / abs(gradpar_half_grid[itheta]));
    std::cout << temp[itheta] << ", ";
  }
  std::cout << "\n\n";

  int index_of_middle = nzgrid;
  std::cout << "zOnThetaGrid = [";
  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    zOnThetaGrid[itheta] = temp[itheta] - temp[index_of_middle];
    std::cout << zOnThetaGrid[itheta] << ", ";
  }
  std::cout << "]\n\n";
  
  //write_geo_arrays_to_file(bmag, gradpar, gds2, gds21, gds22, gbdrift, gbdrift0, cvdrift, cvdrift0);
  

}
  
/*void Geometric_coefficients::write_geo_arrays_to_file(double* bmag, double* gradpar, double* gds2, double* gds21, double* gds22, double* gbdrift, double* gbdrift0, double* cvdrift, double* cvdrift0) {

 
  }*/

void Geometric_coefficients::test_arrays(double* array1, double* array2, int should_be_zero, double tolerance, char* name) {
  // This function is used for verifying the geometry arrays.
  // When should_be_zero = true, the subroutine verifies that |array1| = 0 to
  // within an absolute tolerance specified by 'tolerance'. array2 is ignored in this case
  // When should_be_zero = false, the function verifies that array1 = array2
  // to within a relative tolerance specified by 'tolerance'

  diff_arr = new double[2*nzgrid+1]{};
  sum_arr = new double[2*nzgrid+1]{};
  array1_temp = new double[2*nzgrid+1]{};
  double* max_value_sum;
  double* max_value_diff;
  double* max_value;
  double max_difference;
  for (int izeta=0; izeta<2*nzgrid+1; izeta++) {
    array1_temp[izeta] = abs(array1[izeta]);
    diff_arr[izeta] = abs(array1[izeta]-array2[izeta]);
    sum_arr[izeta] = abs(array1[izeta]) + abs(array2[izeta]);
  }
  max_value_sum = std::max_element(sum_arr, sum_arr+2*nzgrid);
  max_value_diff = std::max_element(diff_arr, diff_arr+2*nzgrid);
  max_difference = *max_value_diff / *max_value_sum;
  
  if (should_be_zero) {
    max_value = std::max_element(array1_temp, array1_temp+2*nzgrid);
    std::cout << "maxval(abs(" << name << ")) = " << *max_value << " (should be < 1.)\n";
    if (*max_value > tolerance) {
      std::cout << "Error! " << name << " should be 0\n";
      exit(1);
    }
  }
  else {
    std::cout << "Relative difference between two methods for computing " << name << " = " << max_difference << " (should be << 1)\n";
    if (max_difference > tolerance) {
      std::cout << "Error! Two methods for computing " << name << "disagree.\n";
      exit(1);
    }
  }
  delete[] sum_arr;
  delete[] diff_arr;
  delete[] array1_temp;
  
}

Geometric_coefficients::~Geometric_coefficients() {
  
  // Deallocate VMEC arrays (why can't these go in a destructor??)
  //  delete[] B;
}
