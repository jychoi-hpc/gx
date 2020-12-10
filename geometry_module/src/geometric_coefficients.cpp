// Adapted from Fortran full surface module written by:
// Matt Landreman, University of Maryland

#include <iostream>
#include "geometric_coefficients.h"
#include <netcdf.h>
#include "parameters.h"
#include <cmath>
#include "vmec_variables.h"
#include "solver.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>
#include "gsl_poly.h"

using namespace std;

Geometric_coefficients::Geometric_coefficients(VMEC_variables *vmec_vars) : vmec(vmec_vars) {

  // ------------------------------------------------------------------------
  // ------------------------------------------------------------------------
  // INPUT PARAMETERS
  // the following are literals for now, but should be input file parameters
  alpha = 0.0;
  nzgrid = 64;
  npol = 2;
  //  desired_normalized_toroidal_flux = 0.25;
  ////////////////////////
  // values for comparison with GIST files that are at surf=12
  //  desired_normalized_toroidal_flux = 0.12755; // W7-X
  desired_normalized_toroidal_flux = 0.19531; // Quasdex
  //  desired_normalized_toroidal_flux = 0.26042; // NCSX
  ////////////////////////
  vmec_surface_option = 2;
  flux_tube_cut = "gds21"; // default is "none"
  // the following are used or ignored based on choice of flux_tube_cut
  custom_length = 2.0; // default is [-pi, pi]
  which_crossing = 4;
  
  // ------------------------------------------------------------------------
  // ------------------------------------------------------------------------
  
  // Reference length and magnetic field are chosen to be the GIST values
  // MM * may have to adapt this for GX normalizations
  std::cout << "Phi_vmec at LCFS = " << vmec->phi[vmec->ns-1] << "\n";

  // Note the first index difference of Fortran and c++ arrays
  // factor of signgs must be there, or sign errors result in the test_arrays functions
  edge_toroidal_flux_over_2pi = vmec->signgs * ( vmec->phi[vmec->ns-1] / (2*M_PI) );
  sign_psi = vmec->phi[vmec->ns-1] / abs(vmec->phi[vmec->ns-1]);
  L_reference = vmec->Aminor_p;
  B_reference = abs((2*edge_toroidal_flux_over_2pi) / (L_reference*L_reference));
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
    //std::cout << "***********************************************\n";
    //std::cout << "confusing issue with indexing compared to fortran code. Check for more cases\n";
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
  //  std::cout << "***********************************************\n";
  //  std::cout << "confusing issue with indexing compared to fortran code. Check for more cases\n";
  
  if (verbose) {
    /*    if (abs(vmec_radial_weight_half[0] < 1.e-14)) {
      std::cout << "Using radial index " << vmec_radial_index_half[1] << "of " << vmec->ns-1 << "from VMEC's half mesh\n";
      }
    else if (abs(vmec_radial_weight_half[1]) < 1.e-14) {
      std::cout << "Using radial index " << vmec_radial_index_half[0] << "of " << vmec->ns-1 << "from VMEC's half mest\n";
    }
    else {*/
      std::cout << "Interpolating using radial indices " << vmec_radial_index_half[0] << " and " << vmec_radial_index_half[1] << " of " << vmec->ns-1 << " from VMEC's half mesh\n";
      std::cout << "Weight for half mesh = " << vmec_radial_weight_half[0] << " and " << vmec_radial_weight_half[1] << "\n";
      std::cout << "Interpolating using radial indicies " << vmec_radial_index_full[0] << " and " << vmec_radial_index_full[1] << " of " << vmec->ns << " from VMEC's full mesh\n";
      std::cout << "Weight for full mesh = " << vmec_radial_weight_full[0] << " and " << vmec_radial_weight_full[1] << "\n";
    
  }

  // Evaluate several radial-profile functions at the flux surface we ended up choosing

  iota = vmec->iotas[vmec_radial_index_half[0]]*vmec_radial_weight_half[0] + vmec->iotas[vmec_radial_index_half[1]]*vmec_radial_weight_half[1];
  //iota = vmec->iotaf[vmec_radial_index_full[0]]*vmec_radial_weight_full[0] + vmec->iotaf[vmec_radial_index_full[1]]*vmec_radial_weight_full[1];
  if (verbose) { std::cout << "iota = " << iota << "\n"; }
  safety_factor_q = 1./iota;

  d_iota_ds_on_half_grid = new double[vmec->ns]{};
  /*  double *d_iota_ds_on_full_grid = new double[vmec->ns-1]{};
  double *dq_ds_half_grid = new double[vmec->ns]{};
  double temp_d_iota;*/
  d_pressure_ds_on_half_grid = new double[vmec->ns]{};

  ds = normalized_toroidal_flux_full_grid[1] - normalized_toroidal_flux_full_grid[0];
  if (verbose) { std::cout << "ds = " << ds << "\n"; }

  for (int i=1; i<vmec->ns; i++) {
    //d_iota_ds_on_half_grid[i] = (vmec->iotas[i] - vmec->iotas[i-1]) / ds;
    //dq_ds_half_grid[i] = (1./(vmec->iotaf[i]) - 1./(vmec->iotaf[i-1])) / ds;
    d_iota_ds_on_half_grid[i] = (vmec->iotaf[i] - vmec->iotaf[i-1]) / ds;
    d_pressure_ds_on_half_grid[i] = (vmec->presf[i] - vmec->presf[i-1]) / ds;
    //    std::cout << "iota[1] = " << iotaf_vmec[i] << ", iota[0] = " << iotaf_vmec[i-1] << "\n";
    //    std::cout << "diota/ds = " << d_iota_ds_on_half_grid[i] << "\n";
  }
  /*  for (int i=1; i<vmec->ns-1; i++) {
    d_iota_ds_on_full_grid[i] = (vmec->iotas[i] - vmec->iotas[i-1]) / ds;
    }*/
  //  d_iota_ds = d_iota_ds_on_half_grid[vmec_radial_index_full[0]]*vmec_radial_weight_full[0] + d_iota_ds_on_half_grid[vmec_radial_index_full[1]]*vmec_radial_weight_full[1];
  //  temp_d_iota = d_iota_ds_on_full_grid[vmec_radial_index_full[0]+1]*vmec_radial_weight_full[0] + d_iota_ds_on_full_grid[vmec_radial_index_full[1]+1]*vmec_radial_weight_full[1];
  //  std::cout << "temp_d_iota = " << temp_d_iota << "\n";
  d_iota_ds = d_iota_ds_on_half_grid[vmec_radial_index_half[0]]*vmec_radial_weight_half[0] + d_iota_ds_on_half_grid[vmec_radial_index_half[1]]*vmec_radial_weight_half[1];
  d_pressure_ds = d_pressure_ds_on_half_grid[vmec_radial_index_half[0]]*vmec_radial_weight_half[0] + d_pressure_ds_on_half_grid[vmec_radial_index_half[1]]*vmec_radial_weight_half[1];

  // shat = (r/q)(dq/dr) where r = a sqrt(s)
  //      = -(r/iota)(d iota / dr) = -2 (s/iota) (d iota/ ds)
  shat = (-2 * normalized_toroidal_flux_used / iota) * d_iota_ds;
  //double shat2 = (-2 * normalized_toroidal_flux_used / iota) * temp_d_iota;
  
  delete[] d_iota_ds_on_half_grid;
  delete[] d_pressure_ds_on_half_grid;
  if (verbose) {
    std::cout << "shat = " << shat << "\n";
    //    std::cout << "shat2 = " << shat2 << "\n";
    std::cout << "d iota / ds = " << d_iota_ds << "\n";
    std::cout << "d pressure / ds = " << d_pressure_ds << "\n";
  }

  // ----------------------------------------------------------------------
  // Set up the coordinate grids
  //

  // Creating uniform theta grid to be [-npol*pi, npol*pi]
  theta = new double[2*nzgrid+1];
  std::vector<double> theta_std_copy (2*nzgrid+1, 0.0); // for potential use when cutting flux tube
  std::cout << "theta pest = [";
  for (int i=0; i<2*nzgrid+1; i++) {
    theta[i] = (npol*M_PI*(i-nzgrid))/nzgrid;
    theta_std_copy[i] = theta[i];
    std::cout << theta[i] << ", ";
  }
  std::cout << "]\n\n";

  // Creating zeta grid based on alpha = theta - iota*zeta
  zeta = new double[2*nzgrid+1]; 
  //std::cout << "zeta = [";
  //  for (int i=0; i<2*nzgrid+1; i++) {
  for (int i=0; i<2*nzgrid+1; i++) {
    //zeta[i] = (npol*M_PI*(i-nzgrid))/nzgrid;
    zeta[i] = (theta[i] - alpha) / iota;
    //std::cout << zeta[i] << ", ";
  }
  //  std::cout << "]\n\n";
  
  // theta_pest = alpha + iota*zeta
  // Need to determine ---> theta_vmec = theta_pest - Lambda
  theta_vmec = new double[2*nzgrid+1]; // this is not a VMEC input

  // calling GSL rootfinder
  std::cout << "\n---------------------------------------------\n";
  std::cout << "Beginning root solves to determine theta_vmec\n";

  solver_vmec_theta(theta_vmec, zeta, nzgrid, alpha, iota, vmec, vmec_radial_index_half, vmec_radial_weight_half);

  std::cout << "Values of theta_vmec from GSL solver:\n[";
  for(int i=0; i<2*nzgrid+1; i++) {
    std::cout << theta_vmec[i] << ", ";
  }
  std::cout << "]\n";
  std::cout << "---------------------------------------------\n\n";
  
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
    for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
      
      angle = m * theta_vmec[itheta] - n * vmec->nfp * zeta[itheta];
      cos_angle = cos(angle);
      sin_angle = sin(angle);
      for (int isurf=0; isurf<2; isurf++) {

	// Handle |B|
	temp = vmec->bmnc[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	temp = temp * scale_factor;
	B[itheta] = B[itheta] + temp*cos_angle;
	dB_dtheta_vmec[itheta] = dB_dtheta_vmec[itheta] - m*temp*sin_angle;
	dB_dzeta[itheta] = dB_dzeta[itheta] + n*vmec->nfp*temp*sin_angle;
	
	// Handle Jacobian
	temp = vmec->gmnc[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	temp = temp * scale_factor;
	sqrt_g[itheta] = sqrt_g[itheta] + temp * cos_angle;

	// Handle B sup theta
	temp = vmec->bsupumnc[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	temp = temp * scale_factor;
	B_sup_theta_vmec[itheta] = B_sup_theta_vmec[itheta] + temp * cos_angle;

	// Handle B sup zeta
	temp = vmec->bsupvmnc[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	temp = temp * scale_factor;
	B_sup_zeta[itheta] = B_sup_zeta[itheta] + temp * cos_angle;

	// Handle B sub theta
	temp = vmec->bsubumnc[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	temp = temp * scale_factor;
	B_sub_theta_vmec[itheta] = B_sub_theta_vmec[itheta] + temp * cos_angle;

	// Handle B sub zeta
	temp = vmec->bsubvmnc[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	temp = temp * scale_factor;
	B_sub_zeta[itheta] = B_sub_zeta[itheta] + temp * cos_angle;

	// Handle B sub psi
	// Unlike the other components of B, this one is on the full mesh
	temp = vmec->bsubsmns[imn_nyq + vmec_radial_index_full[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_full[isurf];
	temp = temp * scale_factor;
	B_sub_s[itheta] = B_sub_s[itheta] + temp * sin_angle;

	// Handle dB/ds
	// Since bmnc is on the half mesh, its radial derivative is on the full mesh
	temp = dB_ds_mnc[vmec_radial_index_full[isurf]] * vmec_radial_weight_full[isurf];
	temp = temp * scale_factor;
	dB_ds[itheta] = dB_ds[itheta] + temp * cos_angle;

	// Handle arrays that use xm and xn instead of xm_nyq and xn_nyq
	if (non_Nyquist_mode_available) {

	  // Handle R, which is on the full mesh
	  temp = vmec->rmnc[imn + vmec_radial_index_full[isurf] * vmec->mnmax] * vmec_radial_weight_full[isurf];
	  temp = temp * scale_factor;
	  R[itheta] = R[itheta] + temp * cos_angle;
	  dR_dtheta_vmec[itheta] = dR_dtheta_vmec[itheta] - temp * m * sin_angle;
	  dR_dzeta[itheta] = dR_dzeta[itheta] + temp * n * vmec->nfp * sin_angle;

	  // Handle Z, which is on the full mesh
	  temp = vmec->zmns[imn + vmec_radial_index_full[isurf] * vmec->mnmax] * vmec_radial_weight_full[isurf];
	  temp = temp * scale_factor;
	  // only need derivatives of Z
	  dZ_dtheta_vmec[itheta] = dZ_dtheta_vmec[itheta] + temp * m * cos_angle;
	  dZ_dzeta[itheta] = dZ_dzeta[itheta] - temp * n * vmec->nfp * cos_angle;

	  // Handle Lambda
	  temp = vmec->lmns[imn + vmec_radial_index_half[isurf] * vmec->mnmax] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  // only need derivative of Lambda
	  dLambda_dtheta_vmec[itheta] = dLambda_dtheta_vmec[itheta] + m * temp * cos_angle;
	  dLambda_dzeta[itheta] = dLambda_dzeta[itheta] - n * vmec->nfp * temp * cos_angle;

	  // Handle dR/ds
	  // Since R is on the full mesh, its radial derivative is on the half mesh
	  temp = dR_ds_mnc[vmec_radial_index_half[isurf]] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  dR_ds[itheta] = dR_ds[itheta] + temp * cos_angle;

	  // Handle dZ/ds
	  // Since Z is on the full mesh, its radial derivative is on the half mesh
	  temp = dZ_ds_mns[vmec_radial_index_half[isurf]] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  dZ_ds[itheta] = dZ_ds[itheta] + temp * sin_angle;

	  // Handle dLambda/ds
	  // Since Lambda is on the half mesh, its radial derivative is on the full mesh
	  temp = dLambda_ds_mns[vmec_radial_index_full[isurf]] * vmec_radial_weight_full[isurf];
	  temp = temp * scale_factor;
	  dLambda_ds[itheta] = dLambda_ds[itheta] + temp * sin_angle;

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
      for (int itheta=0; itheta < 2*nzgrid+1; itheta++) {

	angle = m * theta_vmec[itheta] - n * vmec->nfp * zeta[itheta];
	cos_angle = cos(angle);
	sin_angle = sin(angle);
	for (int isurf=0; isurf<2; isurf++) {
	  
	  // Handle |B|
	  temp = vmec->bmns[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  B[itheta] = B[itheta] + temp*cos_angle;
	  dB_dtheta_vmec[itheta] = dB_dtheta_vmec[itheta] - m*temp*sin_angle;
	  dB_dzeta[itheta] = dB_dzeta[itheta] + n*vmec->nfp*temp*sin_angle;
	  
	  // Handle Jacobian
	  temp = vmec->gmns[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  sqrt_g[itheta] = sqrt_g[itheta] + temp * cos_angle;
	  
	  // Handle B sup theta
	  temp = vmec->bsupumns[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  B_sup_theta_vmec[itheta] = B_sup_theta_vmec[itheta] + temp * cos_angle;
	  
	  // Handle B sup zeta
	  temp = vmec->bsupvmns[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  B_sup_zeta[itheta] = B_sup_zeta[itheta] + temp * cos_angle;
	  
	  // Handle B sub theta
	  temp = vmec->bsubumns[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  B_sub_theta_vmec[itheta] = B_sub_theta_vmec[itheta] + temp * cos_angle;
	  
	  // Handle B sub zeta
	  temp = vmec->bsubvmns[imn_nyq + vmec_radial_index_half[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_half[isurf];
	  temp = temp * scale_factor;
	  B_sub_zeta[itheta] = B_sub_zeta[itheta] + temp * cos_angle;

	  // Handle B sub psi
	  // Unlike the other components of B, this one is on the full mesh
	  temp = vmec->bsubsmnc[imn_nyq + vmec_radial_index_full[isurf]*vmec->mnmax_nyq] * vmec_radial_weight_full[isurf];
	  temp = temp * scale_factor;
	  B_sub_s[itheta] = B_sub_s[itheta] + temp * sin_angle;

	  // Handle dB/ds
	  // Since bmnc is on the half mesh, its radial derivative is on the full mesh
	  temp = dB_ds_mns[vmec_radial_index_full[isurf]] * vmec_radial_weight_full[isurf];
	  temp = temp * scale_factor;
	  dB_ds[itheta] = dB_ds[itheta] + temp * cos_angle;

	  // Handle arrays that use xm and xn instead of xm_nyq and xn_nyq
	  if (non_Nyquist_mode_available) {
	    
	    // Handle R, which is on the full mesh
	    temp = vmec->rmns[imn + vmec_radial_index_full[isurf] * vmec->mnmax] * vmec_radial_weight_full[isurf];
	    temp = temp * scale_factor;
	    R[itheta] = R[itheta] + temp * cos_angle;
	    dR_dtheta_vmec[itheta] = dR_dtheta_vmec[itheta] - temp * m * sin_angle;
	    dR_dzeta[itheta] = dR_dzeta[itheta] + temp * n * vmec->nfp * sin_angle;
	    
	    // Handle Z, which is on the full mesh
	    temp = vmec->zmnc[imn + vmec_radial_index_full[isurf] * vmec->mnmax] * vmec_radial_weight_full[isurf];
	    temp = temp * scale_factor;
	    // only need derivatives of Z
	    dZ_dtheta_vmec[itheta] = dZ_dtheta_vmec[itheta] + temp * m * cos_angle;
	    dZ_dzeta[itheta] = dZ_dzeta[itheta] - temp * n * vmec->nfp * cos_angle;
	    
	    // Handle Lambda
	    temp = vmec->lmnc[imn + vmec_radial_index_half[isurf] * vmec->mnmax] * vmec_radial_weight_half[isurf];
	    temp = temp * scale_factor;
	    // only need derivative of Lambda
	    dLambda_dtheta_vmec[itheta] = dLambda_dtheta_vmec[itheta] + m * temp * cos_angle;
	    dLambda_dzeta[itheta] = dLambda_dzeta[itheta] - n * vmec->nfp * temp * cos_angle;

	    // Handle dR/ds
	    // Since R is on the full mesh, its radial derivative is on the half mesh
	    temp = dR_ds_mns[vmec_radial_index_half[isurf]] * vmec_radial_weight_half[isurf];
	    temp = temp * scale_factor;
	    dR_ds[itheta] = dR_ds[itheta] + temp * cos_angle;

	    // Handle dZ/ds
	    // Since Z is on the full mesh, its radial derivative is on the half mesh
	    temp = dZ_ds_mnc[vmec_radial_index_half[isurf]] * vmec_radial_weight_half[isurf];
	    temp = temp * scale_factor;
	    dZ_ds[itheta] = dZ_ds[itheta] + temp * sin_angle;
	    
	    // Handle dLambda/ds
	    // Since Lambda is on the half mesh, its radial derivative is on the full mesh
	    temp = dLambda_ds_mnc[vmec_radial_index_full[isurf]] * vmec_radial_weight_full[isurf];
	    temp = temp * scale_factor;
	    dLambda_ds[itheta] = dLambda_ds[itheta] + temp * sin_angle;

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
  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    B_dot_grad_theta_pest_over_B_dot_grad_zeta[itheta] = (B_sup_theta_vmec[itheta] * (1.0 + dLambda_dtheta_vmec[itheta]) + B_sup_zeta[itheta] * dLambda_dzeta[itheta]) / B_sup_zeta[itheta];
    temp2D[itheta] = iota;
  }
  test_arrays(B_dot_grad_theta_pest_over_B_dot_grad_zeta, temp2D, false, 0.01, iota_name);
  delete[] B_dot_grad_theta_pest_over_B_dot_grad_zeta;

  //--------------------------------------------------------------------
  // Using R(theta, zeta) and Z(theta, zeta), compute Cartesian
  // components of gradient basis vectors using the dual relations
  //--------------------------------------------------------------------

  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    cos_angle = cos(zeta[itheta]);
    sin_angle = sin(zeta[itheta]);

    // X = R*cos(zeta)
    dX_dtheta_vmec[itheta] = dR_dtheta_vmec[itheta]*cos_angle;
    dX_dzeta[itheta] = dR_dzeta[itheta]*cos_angle - R[itheta]*sin_angle;
    dX_ds[itheta] = dR_ds[itheta]*cos_angle;

    // Y = R*sin(zeta)
    dY_dtheta_vmec[itheta] = dR_dtheta_vmec[itheta]*sin_angle;
    dY_dzeta[itheta] = dR_dzeta[itheta]*sin_angle + R[itheta]*cos_angle;
    dY_ds[itheta] = dR_ds[itheta]*sin_angle;
  }

  // Use the dual relations to get the Cartesian components of grad s, grad theta, grad theta_vmec, and grad_zeta
  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    grad_s_X[itheta] = (dY_dtheta_vmec[itheta] * dZ_dzeta[itheta] - dZ_dtheta_vmec[itheta] * dY_dzeta[itheta]) / sqrt_g[itheta];
    grad_s_Y[itheta] = (dZ_dtheta_vmec[itheta] * dX_dzeta[itheta] - dX_dtheta_vmec[itheta] * dZ_dzeta[itheta]) / sqrt_g[itheta];
    grad_s_Z[itheta] = (dX_dtheta_vmec[itheta] * dY_dzeta[itheta] - dY_dtheta_vmec[itheta] * dX_dzeta[itheta]) / sqrt_g[itheta];

    grad_theta_vmec_X[itheta] = (dY_dzeta[itheta] * dZ_ds[itheta] - dZ_dzeta[itheta] * dY_ds[itheta]) / sqrt_g[itheta];
    grad_theta_vmec_Y[itheta] = (dZ_dzeta[itheta] * dX_ds[itheta] - dX_dzeta[itheta] * dZ_ds[itheta]) / sqrt_g[itheta];
    grad_theta_vmec_Z[itheta] = (dX_dzeta[itheta] * dY_ds[itheta] - dY_dzeta[itheta] * dX_ds[itheta]) / sqrt_g[itheta];

    grad_zeta_X[itheta] = (dY_ds[itheta] * dZ_dtheta_vmec[itheta] - dZ_ds[itheta] * dY_dtheta_vmec[itheta]) / sqrt_g[itheta];
    grad_zeta_Y[itheta] = (dZ_ds[itheta] * dX_dtheta_vmec[itheta] - dX_ds[itheta] * dZ_dtheta_vmec[itheta]) / sqrt_g[itheta];
    grad_zeta_Z[itheta] = (dX_ds[itheta] * dY_dtheta_vmec[itheta] - dY_ds[itheta] * dX_dtheta_vmec[itheta]) / sqrt_g[itheta];
  }
  // End of the dual relations

  // Sanity check: grad_zeta_X should be -sin(zeta)/R
  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    temp2D[itheta] = -sin(zeta[itheta]) / R[itheta];
  }
  test_arrays(grad_zeta_X, temp2D, false, 1.0e-2, grad_zeta_X_name);
  // Might as well use the exact value, which is currently in temp2D
  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    grad_zeta_X[itheta] = temp2D[itheta];
  }

  // Sanity check: grad_zeta_Y should be cos(zeta)/R
  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    temp2D[itheta] = cos(zeta[itheta]) / R[itheta];
  }
  test_arrays(grad_zeta_Y, temp2D, false, 1.0e-2, grad_zeta_Y_name);
  // Might as well use the exact value, which is currently in temp2D
  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    grad_zeta_Y[itheta] = temp2D[itheta];
  }

  // Sanity check: grad_zeta_Z should be 0
  test_arrays(grad_zeta_Z, temp2D, true, 1.0e-6, grad_zeta_Z_name);
  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    grad_zeta_Z[itheta] = 0.0;
  }

  //-------------------------------------------------------------
  // Compute Cartesian components of other quantities we need
  //-------------------------------------------------------------

  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    grad_psi_X[itheta] = grad_s_X[itheta] * (edge_toroidal_flux_over_2pi);
    grad_psi_Y[itheta] = grad_s_Y[itheta] * (edge_toroidal_flux_over_2pi);
    grad_psi_Z[itheta] = grad_s_Z[itheta] * (edge_toroidal_flux_over_2pi);

    grad_alpha_X[itheta] = (dLambda_ds[itheta] - zeta[itheta]*d_iota_ds) * grad_s_X[itheta] + (1.0 + dLambda_dtheta_vmec[itheta]) * grad_theta_vmec_X[itheta] + (-iota + dLambda_dzeta[itheta]) * grad_zeta_X[itheta];
    grad_alpha_Y[itheta] = (dLambda_ds[itheta] - zeta[itheta]*d_iota_ds) * grad_s_Y[itheta] + (1.0 + dLambda_dtheta_vmec[itheta]) * grad_theta_vmec_Y[itheta] + (-iota + dLambda_dzeta[itheta]) * grad_zeta_Y[itheta];
    grad_alpha_Z[itheta] = (dLambda_ds[itheta] - zeta[itheta]*d_iota_ds) * grad_s_Z[itheta] + (1.0 + dLambda_dtheta_vmec[itheta]) * grad_theta_vmec_Z[itheta] + (-iota + dLambda_dzeta[itheta]) * grad_zeta_Z[itheta];

    grad_B_X[itheta] = dB_ds[itheta] * grad_s_X[itheta] + dB_dtheta_vmec[itheta] * grad_theta_vmec_X[itheta] + dB_dzeta[itheta] * grad_zeta_X[itheta];
    grad_B_Y[itheta] = dB_ds[itheta] * grad_s_Y[itheta] + dB_dtheta_vmec[itheta] * grad_theta_vmec_Y[itheta] + dB_dzeta[itheta] * grad_zeta_Y[itheta];
    grad_B_Z[itheta] = dB_ds[itheta] * grad_s_Z[itheta] + dB_dtheta_vmec[itheta] * grad_theta_vmec_Z[itheta] + dB_dzeta[itheta] * grad_zeta_Z[itheta];

    B_X[itheta] = (edge_toroidal_flux_over_2pi) * ((1.0 + dLambda_dtheta_vmec[itheta]) * dX_dzeta[itheta] + (iota - dLambda_dzeta[itheta]) * dX_dtheta_vmec[itheta]) / sqrt_g[itheta];
    B_Y[itheta] = (edge_toroidal_flux_over_2pi) * ((1.0 + dLambda_dtheta_vmec[itheta]) * dY_dzeta[itheta] + (iota - dLambda_dzeta[itheta]) * dY_dtheta_vmec[itheta]) / sqrt_g[itheta];
    B_Z[itheta] = (edge_toroidal_flux_over_2pi) * ((1.0 + dLambda_dtheta_vmec[itheta]) * dZ_dzeta[itheta] + (iota - dLambda_dzeta[itheta]) * dZ_dtheta_vmec[itheta]) / sqrt_g[itheta];
  }
    
  sqrt_s = sqrt(normalized_toroidal_flux_used);

  //---------------------------------------------------------------
  // Sanity Tests: Verify that the Jacobian equals the appropriate
  // cross product of the basis vectors
  //---------------------------------------------------------------

  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    temp2D[itheta] = 0.0
      + dX_ds[itheta] * dY_dtheta_vmec[itheta] * dZ_dzeta[itheta]
      + dY_ds[itheta] * dZ_dtheta_vmec[itheta] * dX_dzeta[itheta]
      + dZ_ds[itheta] * dX_dtheta_vmec[itheta] * dY_dzeta[itheta]
      - dZ_ds[itheta] * dY_dtheta_vmec[itheta] * dX_dzeta[itheta]
      - dX_ds[itheta] * dZ_dtheta_vmec[itheta] * dY_dzeta[itheta]
      - dY_ds[itheta] * dX_dtheta_vmec[itheta] * dZ_dzeta[itheta];
  }
  test_arrays(sqrt_g, temp2D, false, 3.0e-3, sqrt_g_name);

  double *inv_sqrt_g = new double[2*nzgrid+1]{};
  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    temp2D[itheta] = 0.0
      + grad_s_X[itheta] * grad_theta_vmec_Y[itheta] * grad_zeta_Z[itheta]
      + grad_s_Y[itheta] * grad_theta_vmec_Z[itheta] * grad_zeta_X[itheta]
      + grad_s_Z[itheta] * grad_theta_vmec_X[itheta] * grad_zeta_Y[itheta]
      - grad_s_Z[itheta] * grad_theta_vmec_Y[itheta] * grad_zeta_X[itheta]
      - grad_s_X[itheta] * grad_theta_vmec_Z[itheta] * grad_zeta_Y[itheta]
      - grad_s_Y[itheta] * grad_theta_vmec_X[itheta] * grad_zeta_Z[itheta];
    inv_sqrt_g[itheta] = 1./sqrt_g[itheta];
  }
  test_arrays(inv_sqrt_g, temp2D, false, 1.e-2, inv_sqrt_g_name);
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
  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    B_sub_theta_calc[itheta] = B_X[itheta] * dX_dtheta_vmec[itheta] + B_Y[itheta] * dY_dtheta_vmec[itheta] + B_Z[itheta] * dZ_dtheta_vmec[itheta];
    B_sub_zeta_calc[itheta] = B_X[itheta] * dX_dzeta[itheta] + B_Y[itheta] * dY_dzeta[itheta] + B_Z[itheta] * dZ_dzeta[itheta];
    B_sub_s_calc[itheta] = B_X[itheta] * dX_ds[itheta] + B_Y[itheta] * dY_ds[itheta] + B_Z[itheta] * dZ_ds[itheta];
    
    B_sup_theta_calc[itheta] = B_X[itheta] * grad_theta_vmec_X[itheta] + B_Y[itheta] * grad_theta_vmec_Y[itheta] + B_Z[itheta] * grad_theta_vmec_Z[itheta];
    B_sup_zeta_calc[itheta] = B_X[itheta] * grad_zeta_X[itheta] + B_Y[itheta] * grad_zeta_Y[itheta] + B_Z[itheta] * grad_zeta_Z[itheta];
    B_sup_s_calc[itheta] = B_X[itheta] * grad_s_X[itheta] + B_Y[itheta] * grad_s_Y[itheta] + B_Z[itheta] * grad_s_Z[itheta];
  }

  test_arrays(B_sub_theta_calc, B_sub_theta_vmec, false, 1.0e-2, B_sub_theta_vmec_name);
  test_arrays(B_sub_zeta_calc, B_sub_zeta, false, 1.0e-2, B_sub_zeta_name);
  test_arrays(B_sub_s_calc, B_sub_s, false, 1.0e-2, B_sub_s_name);

  test_arrays(B_sup_theta_calc, B_sup_theta_vmec, false, 1.0e-2, B_sup_theta_vmec_name);
  test_arrays(B_sup_zeta_calc, B_sup_zeta, false, 1.0e-2, B_sup_zeta_name);
  test_arrays(B_sup_s_calc, temp2D, true, 1.0e-2, B_sup_s_name);
  
  delete[] B_sub_theta_calc;
  delete[] B_sub_zeta_calc;
  delete[] B_sub_s_calc;
  delete[] B_sup_theta_calc;
  delete[] B_sup_zeta_calc;
  delete[] B_sup_s_calc;
  
  //---------------------------------------------------------------
  // For gbdrift, we need \vec{B} cross grad |B| dot grad alpha
  // For cvdrift, we also need \vec{B} cross grad s dot grad alpha
  // Let us compute both of these quantities 2 ways, and make sure the two
  // approaches give the same answer (within some tolerance)
  //---------------------------------------------------------------

  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    
    B_cross_grad_s_dot_grad_alpha[itheta] = ( B_sub_zeta[itheta] * (1.0 + dLambda_dtheta_vmec[itheta]) - B_sub_theta_vmec[itheta] * (dLambda_dzeta[itheta] - iota) ) / sqrt_g[itheta];
    
    B_cross_grad_s_dot_grad_alpha_alternate[itheta] = 0
      + B_X[itheta] * grad_s_Y[itheta] * grad_alpha_Z[itheta]
      + B_Y[itheta] * grad_s_Z[itheta] * grad_alpha_X[itheta]
      + B_Z[itheta] * grad_s_X[itheta] * grad_alpha_Y[itheta]
      - B_Z[itheta] * grad_s_Y[itheta] * grad_alpha_X[itheta]
      - B_X[itheta] * grad_s_Z[itheta] * grad_alpha_Y[itheta]
      - B_Y[itheta] * grad_s_X[itheta] * grad_alpha_Z[itheta];

    B_cross_grad_B_dot_grad_alpha[itheta] = 0
      + (B_sub_s[itheta] * dB_dtheta_vmec[itheta] * (dLambda_dzeta[itheta] - iota)
	 + B_sub_theta_vmec[itheta] * dB_dzeta[itheta] * (dLambda_ds[itheta] - zeta[itheta] * d_iota_ds)
	 + B_sub_zeta[itheta] * dB_ds[itheta] * (1.0 + dLambda_dtheta_vmec[itheta])
	 - B_sub_zeta[itheta] * dB_dtheta_vmec[itheta] * (dLambda_ds[itheta] - zeta[itheta] * d_iota_ds)
	 - B_sub_theta_vmec[itheta] * dB_ds[itheta] * (dLambda_dzeta[itheta] - iota)
	 - B_sub_s[itheta] * dB_dzeta[itheta] * (1.0 + dLambda_dtheta_vmec[itheta])) / sqrt_g[itheta];

    B_cross_grad_B_dot_grad_alpha_alternate[itheta] = 0
      + B_X[itheta] * grad_B_Y[itheta] * grad_alpha_Z[itheta]
      + B_Y[itheta] * grad_B_Z[itheta] * grad_alpha_X[itheta]
      + B_Z[itheta] * grad_B_X[itheta] * grad_alpha_Y[itheta]
      - B_Z[itheta] * grad_B_Y[itheta] * grad_alpha_X[itheta]
      - B_X[itheta] * grad_B_Z[itheta] * grad_alpha_Y[itheta]
      - B_Y[itheta] * grad_B_X[itheta] * grad_alpha_Z[itheta];
  }
  
  test_arrays(B_cross_grad_s_dot_grad_alpha, B_cross_grad_s_dot_grad_alpha_alternate, false, 1.0e-2, B_cross_grad_s_dot_grad_alpha_name);
  test_arrays(B_cross_grad_B_dot_grad_alpha, B_cross_grad_B_dot_grad_alpha_alternate, false, 1.0e-2, B_cross_grad_B_dot_grad_alpha_name);

  //--------------------------------------------------------------
  // Finally, assemble the quantities needed for GX
  //
  // For a derivation of the following formulae, see the .tex document
  // included in this directory

  std::vector<double> bmag_pest (2*nzgrid+1, 0.0);
  std::vector<double> gradpar_pest (2*nzgrid+1, 0.0);
  std::vector<double> grho_pest (2*nzgrid+1, 0.0);
  std::vector<double> gds2_pest (2*nzgrid+1, 0.0);
  std::vector<double> gds21_pest (2*nzgrid+1, 0.0);
  std::vector<double> gds22_pest (2*nzgrid+1, 0.0);
  std::vector<double> gbdrift_pest (2*nzgrid+1, 0.0);
  std::vector<double> gbdrift0_pest (2*nzgrid+1, 0.0);
  std::vector<double> cvdrift_pest (2*nzgrid+1, 0.0);
  std::vector<double> cvdrift0_pest (2*nzgrid+1, 0.0);

  // Except for bmag and gradpar, the following are related to dx/dpsi and/or dy/dalpha
  // Depending on the sign of the toroidal flux, the sign of dx/dpsi and dy/alpha will change to ensure that
  //
  // kxfac = B_ref * dx/dpsi * dy/dalpha = 1
  // Therefore, each instance of dx/dpsi or dy/alpha must also include a sign_psi factor
  
  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
   
    bmag_pest[itheta] = B[itheta] / B_reference;

    // Using theta to set gradpar, as opposed to zeta for the full surface version
    // gradpar should be calculated based on the PEST coordinates since the geometric quantities are calculated on a theta grid in PEST coordinates (although the values are found from the corresponding theta_vmec value)
    gradpar_pest[itheta] = ( L_reference * B_sup_zeta[itheta]*iota ) / B[itheta];
    //    std::cout << gradpar_pest[itheta] << ", ";

    grho_pest[itheta] = (1. / (L_reference * B_reference * sqrt_s) ) * sqrt( grad_psi_X[itheta]*grad_psi_X[itheta] + grad_psi_Y[itheta]*grad_psi_Y[itheta] + grad_psi_Z[itheta]*grad_psi_Z[itheta] );
    
    gds2_pest[itheta] = (grad_alpha_X[itheta] * grad_alpha_X[itheta] + grad_alpha_Y[itheta] * grad_alpha_Y[itheta] + grad_alpha_Z[itheta] * grad_alpha_Z[itheta]) * L_reference * L_reference * normalized_toroidal_flux_used;

    // Note that the gds21 value from GIST had the incorrect sign at some point. Thus, if comparing to GIST, it is possible that the signs will disagree
    // it seems like the sign needs to be flipped if the VMEC toroidal flux is positive, otherwise gds21 is incorrect (potentially related to signgs = -1?)
    gds21_pest[itheta] = (grad_alpha_X[itheta] * grad_psi_X[itheta] + grad_alpha_Y[itheta] * grad_psi_Y[itheta] + grad_alpha_Z[itheta] * grad_psi_Z[itheta]) * (shat / B_reference);
    //    gds21_pest[itheta] = (sign_psi * vmec->signgs) * (grad_alpha_X[itheta] * grad_psi_X[itheta] + grad_alpha_Y[itheta] * grad_psi_Y[itheta] + grad_alpha_Z[itheta] * grad_psi_Z[itheta]) * (shat / B_reference);
    
    gds22_pest[itheta] = (grad_psi_X[itheta] * grad_psi_X[itheta] + grad_psi_Y[itheta] * grad_psi_Y[itheta] + grad_psi_Z[itheta] * grad_psi_Z[itheta]) * ( (shat * shat) / (L_reference * L_reference * B_reference * B_reference * normalized_toroidal_flux_used) );


    gbdrift_pest[itheta] = sign_psi * 2 * B_reference * L_reference * L_reference * sqrt_s * B_cross_grad_B_dot_grad_alpha[itheta] / ( B[itheta] * B[itheta] * B[itheta] );

    gbdrift0_pest[itheta] = sign_psi * ( (B_sub_theta_vmec[itheta] * dB_dzeta[itheta] - B_sub_zeta[itheta] * dB_dtheta_vmec[itheta]) / sqrt_g[itheta] )
      * ( (abs(edge_toroidal_flux_over_2pi) * 2 * shat) / (B[itheta] * B[itheta] * B[itheta] * sqrt_s) );
    // In the above expression for gbdrift0, the first line and the edge_toroidal_flux_over_2pi is \vec{B} \times \nabla B \cdot \nabla \psi

    cvdrift_pest[itheta] = gbdrift_pest[itheta] + sign_psi * 2 * B_reference * L_reference * L_reference * sqrt_s * mu_0 * d_pressure_ds * B_cross_grad_s_dot_grad_alpha[itheta] / (B[itheta] *B[itheta] * B[itheta] * B[itheta]);

    cvdrift0_pest[itheta] = gbdrift0_pest[itheta];// + sign_psi * 2 * B_reference * L_reference * L_reference * sqrt_s * mu_0 * d_pressure_ds * B_cross_grad_s_dot_grad_alpha[itheta] / (B[itheta] *B[itheta] * B[itheta] * B[itheta]);;

  }
  std::cout << "\n";
  std::cout << "\n";
  std::cout << "\n";
  
  // ---------------------------------------------------------------------
  // Take subset of grid in theta for boundary condition considerations
  // ---------------------------------------------------------------------
  
  
  std::vector<double> theta_grid_cut;// (2*nzgrid+1, 0.0);
  std::vector<double> revised_theta_grid;// (2*nzgrid+1, 0.0);
  
  if (flux_tube_cut != "none") {

    if (flux_tube_cut == "custom") { // if user desires some subset of the arrays

      std::cout << "**************************************************\n";
      std::cout << "You have chosen to take a custom subset of the flux tube\n";
    
      if (custom_length > npol*M_PI) {
	std::cout << "ERROR! Custom flux tube [" << -custom_length << "," << custom_length << "] is longer than original length of [" << -npol*M_PI << "," << npol*M_PI << "]\n";
	std::cout << "Choose a smaller custom length. Exiting...\n";
	exit(1);
      }
      else if (custom_length < 0) {
	custom_length = abs(custom_length);
      }
      else if (custom_length < (theta[1]-theta[0])) {
	std::cout << "ERROR! Custom length must be greater than the spacing between grid points. Exiting...\n";
	exit(1);
      }
    
      std::cout << "The (unscaled) flux tube will go from [-" << custom_length << "," << custom_length << "]\n";
      std::cout << "**************************************************\n";
    
      // get_cut_indices will use input parameters to return a subset of the full theta grid
      // THIS WILL REDEFINE NZGRID!!!
      get_cut_indices_custom(theta_std_copy, ileft, iright, nzgrid);
          
      get_revised_theta_custom(theta_std_copy, theta_grid_cut, revised_theta_grid);
      
      //      exit(1);
    }
    
    else { // subset of flux tube where ends coincide with zeros of gds21 or gbdrift0
    
      if (flux_tube_cut == "gds21") {
	
	get_cut_indices_zeros(gds21_pest, ileft, iright, nzgrid, root_idx_left, root_idx_right);
	get_revised_theta_zeros(theta_std_copy, gds21_pest, theta_grid_cut, revised_theta_grid);
	
      }
      else if (flux_tube_cut == "gbdrift0") {
	
	get_cut_indices_zeros(gbdrift0_pest, ileft, iright, nzgrid, root_idx_left, root_idx_right);
	get_revised_theta_zeros(theta_std_copy, gbdrift0_pest, theta_grid_cut, revised_theta_grid);
      }
      else {
	std::cout << "The string " << flux_tube_cut << " is not valid.\n";
	std::cout << "This must be set to: none, custom, gds21, or gbdrift0. Exiting...\n";
	exit(1);
      }
                              
    }

    // take the subset of the geometric arrays corresponding to the theta value that is closest to the desired grid
    bmag_cut = slice(bmag_pest, ileft, iright);
    gradpar_cut = slice(gradpar_pest, ileft, iright);
    grho_cut = slice(grho_pest, ileft, iright);
    gds2_cut = slice(gds2_pest, ileft, iright);
    gds21_cut = slice(gds21_pest, ileft, iright);
    gds22_cut = slice(gds22_pest, ileft, iright);
    gbdrift_cut = slice(gbdrift_pest, ileft, iright);
    gbdrift0_cut = slice(gbdrift0_pest, ileft, iright);
    cvdrift_cut = slice(cvdrift_pest, ileft, iright);
    cvdrift0_cut = slice(cvdrift0_pest, ileft, iright);
        
    bmag_temp = &bmag_cut[0];
    gradpar_temp = &gradpar_cut[0];
    grho_temp = &grho_cut[0];
    gds2_temp = &gds2_cut[0];
    gds21_temp = &gds21_cut[0];
    gds22_temp = &gds22_cut[0];
    gbdrift_temp = &gbdrift_cut[0];
    gbdrift0_temp = &gbdrift0_cut[0];
    cvdrift_temp = &cvdrift_cut[0];
    cvdrift0_temp = &cvdrift0_cut[0];
    //theta_grid_temp = &theta_grid_cut[0];
 
    // Interpolate the cut grid onto the revised grid based on the type of cut
    interp_to_new_grid(bmag_temp, &theta_grid_cut[0], &revised_theta_grid[0], nzgrid, true);
    interp_to_new_grid(gradpar_temp, &theta_grid_cut[0], &revised_theta_grid[0], nzgrid, true);
    interp_to_new_grid(grho_temp, &theta_grid_cut[0], &revised_theta_grid[0], nzgrid, true);
    interp_to_new_grid(gds2_temp, &theta_grid_cut[0], &revised_theta_grid[0], nzgrid, true);
    interp_to_new_grid(gds21_temp, &theta_grid_cut[0], &revised_theta_grid[0], nzgrid, true);
    interp_to_new_grid(gds22_temp, &theta_grid_cut[0], &revised_theta_grid[0], nzgrid, true);
    interp_to_new_grid(gbdrift_temp, &theta_grid_cut[0], &revised_theta_grid[0], nzgrid, true);
    interp_to_new_grid(gbdrift0_temp, &theta_grid_cut[0], &revised_theta_grid[0], nzgrid, true);
    interp_to_new_grid(cvdrift_temp, &theta_grid_cut[0], &revised_theta_grid[0], nzgrid, true);
    interp_to_new_grid(cvdrift0_temp, &theta_grid_cut[0], &revised_theta_grid[0], nzgrid, true);

    std::cout << "Final gds21 = [";
    for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
      std::cout << gds21_temp[itheta] << ", ";
    }
    std::cout << "]\n\n";

    std::cout << "Final gbdrift0 = [";
    for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
      std::cout << gbdrift0_temp[itheta] << ", ";
    }
    std::cout << "]\n\n";

  }
  else {
    
    std::cout << "**************************************************\n";
    std::cout << "You have chosen not to take a subset of the flux tube. The (unscaled) flux tube will go from [-" << M_PI*npol << "," << M_PI*npol << "]\n";
    std::cout << "**************************************************\n";

    bmag_temp = &bmag_pest[0];
    gradpar_temp = &gradpar_pest[0];
    grho_temp = &grho_pest[0];
    gds2_temp = &gds2_pest[0];
    gds21_temp = &gds21_pest[0];
    gds22_temp = &gds22_pest[0];
    gbdrift_temp = &gbdrift_pest[0];
    gbdrift0_temp = &gbdrift0_pest[0];
    cvdrift_temp = &cvdrift_pest[0];
    cvdrift0_temp = &cvdrift0_pest[0];
    theta_grid_temp = &theta_std_copy[0];
    
  }

  // ---------------------------------------------------------------------
  // Interpolate the above geometric quantities onto a uniform grid for GX
  // ---------------------------------------------------------------------
  theta_grid_temp = new double[2*nzgrid+1]{};
  theta_grid = new double[2*nzgrid+1]{};
  bmag = new double[2*nzgrid+1]{};
  gradpar = new double[2*nzgrid+1]{};
  grho = new double[2*nzgrid+1]{};
  gds2 = new double[2*nzgrid+1]{};
  gds21 = new double[2*nzgrid+1]{};
  gds22 = new double[2*nzgrid+1]{};
  gbdrift = new double[2*nzgrid+1]{};
  gbdrift0 = new double[2*nzgrid+1]{};
  cvdrift = new double[2*nzgrid+1]{};
  cvdrift0 = new double[2*nzgrid+1]{};

  if (flux_tube_cut == "none") {
    
    get_GX_geo_arrays(bmag_temp, gradpar_temp, grho_temp, gds2_temp, gds21_temp, gds22_temp, gbdrift_temp, gbdrift0_temp, cvdrift_temp, cvdrift0_temp, theta_grid_temp, theta);

    // Flux tube length is always scaled to [-pi,pi] for GX grid files. The variable domain_scaling_factor accounts for that scaling for plotting results on the actual flux tube domain
    domain_scaling_factor = abs(theta[0]/theta_grid_temp[0]);
  }
  else {

    get_GX_geo_arrays(bmag_temp, gradpar_temp, grho_temp, gds2_temp, gds21_temp, gds22_temp, gbdrift_temp, gbdrift0_temp, cvdrift_temp, cvdrift0_temp, theta_grid_temp, &revised_theta_grid[0]);
    std::cout << "nzgrid = " << nzgrid << "\n";
    // see above
    domain_scaling_factor = abs(revised_theta_grid[0]/theta_grid_temp[0]);

  }

  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    theta_grid[itheta] = theta_grid_temp[itheta];
    bmag[itheta] = bmag_temp[itheta];
    gradpar[itheta] = gradpar_temp[itheta];
    grho[itheta] = grho_temp[itheta];
    gds2[itheta] = gds2_temp[itheta];
    gds21[itheta] = gds21_temp[itheta];
    gds22[itheta] = gds22_temp[itheta];
    gbdrift[itheta] = gbdrift_temp[itheta];
    gbdrift0[itheta] = gbdrift0_temp[itheta];
    cvdrift[itheta] = cvdrift_temp[itheta];
    cvdrift0[itheta] = cvdrift0_temp[itheta];
    }

  write_geo_arrays_to_file(theta_grid, bmag, gradpar, grho, gds2, gds21, gds22, gbdrift, gbdrift0, cvdrift, cvdrift0);


  std::cout << "Finished creating grid file\n";
}

void Geometric_coefficients::get_cut_indices_custom(std::vector<double>& theta, int& ileft_, int& iright_, int& nzgrid_) {
  //void Geometric_coefficients::get_cut_indices_custom(double* theta, double* bmag_pest, double* gradpar_pest, double* grho_pest, double* gds2_pest, double* gds21_pest, double* gds22_pest, double* gbdrift_pest, double* gbdrift0_pest, double* cvdrift_pest, double* cvdrift0_pest, int &nzgrid_) {

  //std::vector<double> data_std (2*nzgrid+1, 0.0);
  std::vector<double> short_theta;
  int theta_index_1, theta_index_2;
  
  // Converting to theta to std::vector type to enable some features
  //for (int i=0; i<2*nzgrid_+1; i++) {
  //  theta_std[i] = data[i];
  //}
    
  std::vector<double> theta_minus (2*nzgrid_+1, 0.0);
  std::vector<double> theta_plus (2*nzgrid_+1, 0.0);
    
  // Find index of full theta grid that is closest to the desired custom grid
  for (int i=0; i<2*nzgrid_+1; i++) {
    theta_minus[i] = abs(theta[i] - custom_length);
    theta_plus[i] = abs(theta[i] + custom_length);
  }
  
  auto min_val_1 = std::min_element(theta_minus.begin(), theta_minus.end());
  theta_index_1 = std::distance(theta_minus.begin(), min_val_1);
  auto min_val_2 = std::min_element(theta_plus.begin(), theta_plus.end());
  theta_index_2 = std::distance(theta_plus.begin(), min_val_2);
  
  if (theta[theta_index_1] < theta[theta_index_2]) {
    ileft_ = theta_index_1;
    iright_ = theta_index_2;
  }
  else {
    ileft_ = theta_index_2;
    iright_ = theta_index_1;
  }
  //  std::cout << "Desired custom length = [" << -custom_length << "," << custom_length << "]\n";
  //  std::cout << "Theta value at closest indices in full array = " << theta[ileft_] << "," << theta[iright_] << "\n";
    
  short_theta = slice(theta, ileft_, iright_);
  
  // nzgrid_ is now redefined for the subset of the input theta grid
  nzgrid_ = (short_theta.size() - 1) / 2;
}

void Geometric_coefficients::get_revised_theta_custom(std::vector<double>& theta, std::vector<double>& theta_cut_, std::vector<double>& theta_revised_) {

  theta_cut_ = slice(theta, ileft, iright);

  // Create a new theta grid from [-custom_theta,custom_theta] with the revised nzgrid
  //  std::cout << "temp_tgrid = [";
  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    theta_revised_.push_back( (custom_length*(itheta-nzgrid))/nzgrid );
    //std::cout << theta_revised_[itheta] << ", ";
  }
  //  std::cout << "]\n\n";

}

void Geometric_coefficients::get_cut_indices_zeros(std::vector<double>& data, int &ileft_, int &iright_, int &nzgrid_, int& root_idx_left_, int& root_idx_right_) {

  std::vector<double> data_cut;
  
  std::vector<int> isign;
  for (int itheta=0; itheta<2*nzgrid; itheta++) {
    if (sgn(data[itheta]) != sgn(data[itheta+1])) {
      if ((sgn(data[itheta]) != 0) and (sgn(data[itheta+1]) != 0)) {
	isign.push_back(itheta);
      }
    }
  }
  
  // Number of zeros on EACH SIDE of theta=0 (total zeros is 2*nzeros)
  int nzeros = isign.size()/2;
  // isign gives an array of indices for the element just prior to a sign change
  // includes all zeros except for the one at theta=0
  
  std::cout << "Indices of sign crossing = [";
  for (int i=0; i<isign.size(); i++) {
    std::cout << isign[i] << ", ";
  }
  std::cout << "]\n\n";
  
  if (which_crossing > nzeros) {
    std::cout << "There are not " << which_crossing << " zero crossings for a grid of this size.\n";
    std::cout << "You must select which_cross <= " << nzeros << "\n";
    std::cout << "Exiting...\n";
    exit(1);
  }
  else if (which_crossing <= 0) {
    std::cout << "which_crossing must be >0. Exiting...\n";
    exit(1);
  }

  ileft_ = isign[nzeros - which_crossing];
  iright_ = isign[which_crossing + (nzeros-1)] + 1;

  int region = 2; // 2*region is number of interpolating points for spline in the function "get_revised_theta_zeros"
  root_idx_left = isign[which_crossing + (nzeros-1)] - (region-1);
  root_idx_right = isign[which_crossing + (nzeros-1)] + region;
  
  data_cut = slice(data,ileft,iright);
  nzgrid_ = (data_cut.size() - 1) / 2;

}

void Geometric_coefficients::get_revised_theta_zeros(std::vector<double>& theta, std::vector<double>& data, std::vector<double>& theta_cut_, std::vector<double>& theta_revised_) {

  std::vector<double> data_cut;
  std::vector<double> zero_slice, theta_slice; // small slices of 2*region points on which to fit a spline
  //  int root_idx_left, root_idx_right;
  double r1, r2; // quadratic roots
  double theta_zero_loc; // theta value at location of zero crossing


  data_cut = slice(data,ileft,iright);
  theta_cut_ = slice(theta,ileft,iright);

  std::cout << "data_cut = [";
  for (int i=0; i<data_cut.size(); i++) {
    std::cout << data_cut[i] << ", ";
  }
  std::cout << "]\n\n";
  
  // use which_crossing value to determine which zero to find and take a 2*region slice of the geometric array
  
  // Location of zeros is symmetric about theta=0.
  // The zeros for positive theta values will be calculated, so skipping over the indices of the zeros for theta < 0 

  zero_slice = slice(data,root_idx_left,root_idx_right);
  theta_slice = slice(theta,root_idx_left,root_idx_right);

  std::cout << "zero_slice = [";
  for (int i=0; i<zero_slice.size(); i++) {
    std::cout << zero_slice[i] << ", ";
  }
  std::cout << "]\n\n";
  
  std::cout << "gds21 = [";
  for (int i=0; i<6; i++) {
    std::cout << zero_slice[i] << " ";
  }
  std::cout << "]\n\n";

  std::cout << "theta = [";
  for (int i=0; i<6; i++) {
    std::cout << theta_slice[i] << " ";
  }
  std::cout << "]\n\n";

  int region = 2;
  int ncoeff = 3;
  double coeff[ncoeff];
  // returns coefficients of P = coeff[0] + coeff[1]*x + coeff[2]*x^2 + ...
  PolyFit( &theta_slice[0], &zero_slice[0], 2*region, ncoeff, &coeff[0] );
  std::cout << "coeff[0] = " << coeff[0] << "\n\n";
  std::cout << "coeff[1] = " << coeff[1] << "\n\n";
  std::cout << "coeff[2] = " << coeff[2] << "\n\n";
  
  // solve a*x^2 + b*x + c = 0
  gsl_poly_solve_quadratic( coeff[2], coeff[1], coeff[0], &r1, &r2 );
  
  // Ensure that the root is within the interp region
  if ( (r1 > theta_slice[0]) and (r1 < theta_slice[2*region-1]) ) {
    theta_zero_loc = r1;
  }
  else if ( (r2 > theta_slice[0]) and (r2 < theta_slice[2*region-1]) ) {
    theta_zero_loc = r2;
  }
  else {
    std::cout << "Neither root is in the interpolation region around the zero. Something went wrong. Exiting...";
    exit(1);
  }

  // Create a new theta grid from [-theta_zero_loc,theta_zero_loc] with the revised nzgrid
  std::cout << "temp_tgrid = [";
  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    theta_revised_.push_back( (theta_zero_loc*(itheta-nzgrid))/nzgrid );
    std::cout << theta_revised_[itheta] << ", ";
  }
  std::cout << "]\n\n";

}

void Geometric_coefficients::get_GX_geo_arrays(double *bmag_temp, double *gradpar_temp, double* grho_temp, double *gds2_temp, double* gds21_temp, double *gds22_temp, double *gbdrift_temp, double *gbdrift0_temp, double *cvdrift_temp, double *cvdrift0_temp, double *final_theta_grid, double *theta) {

  //-----------------------------------------------------------------
  // Due to the FFTs in GX, need to have a uniform grid
  //
  // To create such a grid, we interpolate the "gs2"
  // geometric arrays onto this uniform grid, which we call Z
  //
  // This requires the parallel gradient of the coordinate Z to be
  // constant, so we must first find the Z values on the current theta
  // grid corresponding to gradpar = const.
  //
  // z(theta) = \int_0^theta (\hat{b}\cdot\nabla z) / (\hat{b}\cdot\nabla\theta) d\theta'
  // Taking \hat{b}\cdot\nabla z = 1, and noting that gradpar(d\theta') = \hat{b}\cdot\nabla\theta
  // do a simple trapezoidal integration of 1/gradpar by finding gradpar on the half grid

  gradpar_half_grid = new double[2*nzgrid]{};
  temp_grid = new double[2*nzgrid+1]{};
  z_on_theta_grid = new double[2*nzgrid+1]{};
  uniform_zgrid = new double[2*nzgrid+1]{};

  dtheta = theta[1] - theta[0]; // dtheta in pest coordinates
  dtheta_pi = M_PI/nzgrid; // dtheta on the SCALED uniform -pi,pi grid with 2*nzgrid+1 points
  index_of_middle = nzgrid;

  // Note: gradpar_half_grid has 1 less grid point than gradpar_temp
  for (int itheta=0; itheta<2*nzgrid-1; itheta++) {
    gradpar_half_grid[itheta] = 0.5 * (abs(gradpar_temp[itheta]) + abs(gradpar_temp[itheta+1]));
  }
  gradpar_half_grid[2*nzgrid-1] = gradpar_half_grid[0];

  //  std::cout << "temp grid = ";
  for (int itheta=1; itheta<2*nzgrid+1; itheta++) {
    temp_grid[itheta] = temp_grid[itheta-1] + dtheta * (1. / abs(gradpar_half_grid[itheta-1]));
    //std::cout << temp_grid[itheta] << ", ";
  }
  //  std::cout << "\n";

  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    z_on_theta_grid[itheta] = temp_grid[itheta] - temp_grid[index_of_middle];
  }
  desired_gradpar = M_PI/abs(z_on_theta_grid[0]);

  //  std::cout << "z_on_theta_grid = [";
  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    z_on_theta_grid[itheta] = z_on_theta_grid[itheta] * desired_gradpar;
    gradpar_temp[itheta] = desired_gradpar; // setting entire gradpar array to the constant value "desired_gradpar"
    //std::cout <<  z_on_theta_grid[itheta] << " ";
  }
  //  std::cout << "]\n\n";
  
  for (int itheta=0; itheta<2*nzgrid+1; itheta++) {
    uniform_zgrid[itheta] = z_on_theta_grid[0] + itheta*dtheta_pi;
    final_theta_grid[itheta] = uniform_zgrid[itheta];
  }

  // Interpolating each geometric array from the non-uniform theta grid, onto to the
  // uniform z grid where gradpar=const
  interp_to_new_grid(bmag_temp, z_on_theta_grid, uniform_zgrid, nzgrid, false);
  interp_to_new_grid(grho_temp, z_on_theta_grid, uniform_zgrid, nzgrid, false);
  interp_to_new_grid(gds2_temp, z_on_theta_grid, uniform_zgrid, nzgrid, false);
  interp_to_new_grid(gds21_temp, z_on_theta_grid, uniform_zgrid, nzgrid, false);
  interp_to_new_grid(gds22_temp, z_on_theta_grid, uniform_zgrid, nzgrid, false);
  interp_to_new_grid(gbdrift_temp, z_on_theta_grid, uniform_zgrid, nzgrid, false);
  interp_to_new_grid(gbdrift0_temp, z_on_theta_grid, uniform_zgrid, nzgrid, false);
  interp_to_new_grid(cvdrift_temp, z_on_theta_grid, uniform_zgrid, nzgrid, false);
  interp_to_new_grid(cvdrift0_temp, z_on_theta_grid, uniform_zgrid, nzgrid, false);
}
  
void Geometric_coefficients::write_geo_arrays_to_file(double *theta_grid, double* bmag, double* gradpar, double* grho, double* gds2, double* gds21, double* gds22, double* gbdrift, double* gbdrift0, double* cvdrift, double* cvdrift0) {

  std::string out_name;
  std::string tor_flux = std::to_string(normalized_toroidal_flux_used);
  std::string custom_info = std::to_string(custom_length);
  custom_info = custom_info.substr(0,5);
  std::string theta_grid_points = std::to_string(2*nzgrid);
  tor_flux = tor_flux.substr(0,5);
  std::string vmec_name = vmec->vmec_file;
  vmec_name = vmec_name.substr(0,vmec_name.size()-3);
  if (flux_tube_cut == "custom") {
    out_name = "grid.gx_" + vmec_name + "_psiN_" + tor_flux + "_custom_[-" + custom_info + "," + custom_info + "]_nt_" + theta_grid_points;
  }
  else if (flux_tube_cut == "gds21") {
    out_name = "grid.gx_" + vmec_name + "_psiN_" + tor_flux + "_gds21" + "_nt_" + theta_grid_points;
  }
  else if (flux_tube_cut == "gbdrift0") {
    out_name = "grid.gx_" + vmec_name + "_psiN_" + tor_flux + "_gbdrift0" + "_nt_" + theta_grid_points;
  }
  else {
    out_name = "grid.gx_" + vmec_name + "_psiN_" + tor_flux + "_nt_" + theta_grid_points;
  }
  
  std::ofstream out_file(out_name);
  //  std::ofstream out_file(".\\name.ext");
  if (out_file.is_open()) {
    out_file << "ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q scale\n";
    out_file << nzgrid << " 1.0 " << 2*nzgrid << " 1.0 1.0 " << shat << " 1.0 1.0 " << domain_scaling_factor << " \n";
    out_file << "gbdrift\t gradpar\t grho\t tgrid\n";
    for (int i=0; i<2*nzgrid+1; i++) {
      out_file << std::right << setprecision(10) << std::setw(20) << gbdrift[i] << "\t" << std::setw(20) << gradpar[i] << "\t" << std::setw(20) << grho[i] << "\t" << std::setw(20) << theta_grid[i] << "\n";
    }

    out_file << "cvdrift\t gds2\t bmag\t tgrid\n";
    for (int i=0; i<2*nzgrid+1; i++) {
      out_file << std::right << setprecision(10) << std::setw(20) << cvdrift[i] << "\t" << std::setw(20) << gds2[i] << "\t" << std::setw(20) << bmag[i] << "\t" << std::setw(20) << theta_grid[i] << "\n";
    }

    out_file << "gds21\t gds22\t tgrid\n";
    for (int i=0; i<2*nzgrid+1; i++) {
      out_file << std::right << setprecision(10) << std::setw(20) << gds21[i] << "\t" << std::setw(20) << gds22[i] << "\t" << std::setw(20) << theta_grid[i] << "\n";
    }

    out_file << "cvdrift0\t gbdrift0\t tgrid\n";
    for (int i=0; i<2*nzgrid+1; i++) {
      out_file << std::right << setprecision(10) << std::setw(20) << cvdrift0[i] << "\t" << std::setw(20) << gbdrift0[i] << "\t" << std::setw(20) << theta_grid[i] << "\n";
    }
  }
  
  out_file.close();
  
 
}

void Geometric_coefficients::test_arrays(double* array1, double* array2, int should_be_zero, double tolerance, const std::string &name) {
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
    //    diff_arr[izeta] = abs(array1[izeta])-abs(array2[izeta]);
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
      std::cout << "Error! Two methods for computing " << name << " disagree.\n";
      exit(1);
    }
  }
  delete[] sum_arr;
  delete[] diff_arr;
  delete[] array1_temp;
  
}

Geometric_coefficients::~Geometric_coefficients() {
  
  delete[] B;
  delete[] temp2D;
  delete[] sqrt_g;
  delete[] R;
  delete[] dB_dtheta_vmec;
  delete[] dB_dzeta;
  delete[] dB_ds;
  delete[] dR_dtheta_vmec;
  delete[] dR_dzeta;
  delete[] dR_ds;
  delete[] dZ_dtheta_vmec;
  delete[] dZ_dzeta;
  delete[] dZ_ds;
  delete[] dLambda_dtheta_vmec;
  delete[] dLambda_dzeta;
  delete[] dLambda_ds;
  delete[] B_sub_s;
  delete[] B_sub_theta_vmec;
  delete[] B_sub_zeta;
  delete[] B_sup_theta_vmec;
  delete[] B_sup_zeta;

  delete[] dB_ds_mnc;
  delete[] dB_ds_mns;
  delete[] dR_ds_mnc;
  delete[] dR_ds_mns;
  delete[] dZ_ds_mnc;
  delete[] dZ_ds_mns;
  delete[] dLambda_ds_mnc;
  delete[] dLambda_ds_mns;

  delete[] dX_ds;
  delete[] dX_dtheta_vmec;
  delete[] dX_dzeta;
  delete[] dY_ds;
  delete[] dY_dtheta_vmec;
  delete[] dY_dzeta;

  delete[] grad_s_X;
  delete[] grad_s_Y;
  delete[] grad_s_Z;
  delete[] grad_theta_vmec_X;
  delete[] grad_theta_vmec_Y;
  delete[] grad_theta_vmec_Z;
  delete[] grad_zeta_X;
  delete[] grad_zeta_Y;
  delete[] grad_zeta_Z;
  delete[] grad_psi_X;
  delete[] grad_psi_Y;
  delete[] grad_psi_Z;
  delete[] grad_alpha_X;
  delete[] grad_alpha_Y;
  delete[] grad_alpha_Z;

  delete[] B_X;
  delete[] B_Y;
  delete[] B_Z;
  delete[] grad_B_X;
  delete[] grad_B_Y;
  delete[] grad_B_Z;
  delete[] B_cross_grad_B_dot_grad_alpha;
  delete[] B_cross_grad_B_dot_grad_alpha_alternate;
  delete[] B_cross_grad_s_dot_grad_alpha;
  delete[] B_cross_grad_s_dot_grad_alpha_alternate;

}

std::vector<double> Geometric_coefficients::slice(std::vector<double> const &v, int m, int n) {
  auto first = v.cbegin() + m;
  auto last = v.cbegin() + n + 1;
  std::vector<double> vec(first, last);
  return vec;
}
