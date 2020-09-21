#include <iostream>
#include "vmec_variables.h"
#include <netcdf.h>
#include "parameters.h"
#include <cmath>

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

VMEC_variables::VMEC_variables(char *vmec) : vmec_file(vmec) {

  // Read in VMEC data and allocate some arrays
  // check if .nc file exists........
  int ncid;
  int varid;
  int retval;
  // variables to read from VMEC:
  // ns, nfp, iotas, iotaf, presf, xm_nyq, xn_nyq, mnmax, mnmax_nyquist, Aminor_p, xm, xn, signgs, phi
  if ((retval = nc_open(vmec_file, NC_NOWRITE, &ncid))) ERR(retval);

  // Stellarator-asymmetric: logical
  if ((retval = nc_inq_varid(ncid, "lasym__logical__", &varid)))  ERR(retval);
  if ((retval = nc_get_var_int(ncid, varid, &lasym)))  ERR(retval);
  
  // Number of surfaces
  if ((retval = nc_inq_varid(ncid, "ns", &varid)))  ERR(retval);
  if ((retval = nc_get_var_int(ncid, varid, &ns)))  ERR(retval);

  // Number of field periods
  if ((retval = nc_inq_varid(ncid, "nfp", &varid)))  ERR(retval);
  if ((retval = nc_get_var_int(ncid, varid, &nfp)))  ERR(retval);

  // Number of poloidal*toroidal*2 modes
  if ((retval = nc_inq_varid(ncid, "mnmax", &varid)))  ERR(retval);
  if ((retval = nc_get_var_int(ncid, varid, &mnmax)))  ERR(retval);

  // Twice the value of mnmax
  if ((retval = nc_inq_varid(ncid, "mnmax_nyq", &varid)))  ERR(retval);
  if ((retval = nc_get_var_int(ncid, varid, &mnmax_nyq)))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "signgs", &varid)))  ERR(retval);
  if ((retval = nc_get_var_int(ncid, varid, &signgs)))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "mpol", &varid)))  ERR(retval);
  if ((retval = nc_get_var_int(ncid, varid, &mpol)))  ERR(retval);
  
  if ((retval = nc_inq_varid(ncid, "ntor", &varid)))  ERR(retval);
  if ((retval = nc_get_var_int(ncid, varid, &ntor)))  ERR(retval);

  // Effective minor radius
  if ((retval = nc_inq_varid(ncid, "Aminor_p", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &Aminor_p)))  ERR(retval);
  
  // Allocating the size of other VMEC arrays based on the above variables
  iotas = new double[ns];
  iotaf = new double[ns];
  presf = new double[ns];
  phips = new double[ns];
  phi = new double[ns];

  xn = new int[mnmax];
  xm = new int[mnmax];
  xn_nyq = new int[mnmax_nyq];
  xm_nyq = new int[mnmax_nyq];
  // 2d VMEC arrays
  lmns = new double[ns*mnmax];
  rmnc = new double[ns*mnmax];
  zmns = new double[ns*mnmax];
  gmnc = new double[ns*mnmax_nyq];
  bmnc = new double[ns*mnmax_nyq];
  bsupumnc = new double[ns*mnmax_nyq];
  bsupvmnc = new double[ns*mnmax_nyq];
  bsubumnc = new double[ns*mnmax_nyq];
  bsubvmnc = new double[ns*mnmax_nyq];
  bsubsmns = new double[ns*mnmax_nyq];

  if ((retval = nc_inq_varid(ncid, "iotas", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &iotas[0])))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "iotaf", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &iotaf[0])))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "presf", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &presf[0])))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "phips", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &phips[0])))  ERR(retval);
  
  if ((retval = nc_inq_varid(ncid, "phi", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &phi[0])))  ERR(retval);
  
  if ((retval = nc_inq_varid(ncid, "xm", &varid)))  ERR(retval);
  if ((retval = nc_get_var_int(ncid, varid, &xm[0])))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "xn", &varid)))  ERR(retval);
  if ((retval = nc_get_var_int(ncid, varid, &xn[0])))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "xm_nyq", &varid)))  ERR(retval);
  if ((retval = nc_get_var_int(ncid, varid, &xm_nyq[0])))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "xn_nyq", &varid)))  ERR(retval);
  if ((retval = nc_get_var_int(ncid, varid, &xn_nyq[0])))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "lmns", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &lmns[0])))  ERR(retval);
  
  if ((retval = nc_inq_varid(ncid, "rmnc", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &rmnc[0])))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "zmns", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &zmns[0])))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "gmnc", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &gmnc[0])))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "bmnc", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &bmnc[0])))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "bsupumnc", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &bsupumnc[0])))  ERR(retval);
  
  if ((retval = nc_inq_varid(ncid, "bsupvmnc", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &bsupvmnc[0])))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "bsubumnc", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &bsubumnc[0])))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "bsubvmnc", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &bsubvmnc[0])))  ERR(retval);

  if ((retval = nc_inq_varid(ncid, "bsubsmns", &varid)))  ERR(retval);
  if ((retval = nc_get_var_double(ncid, varid, &bsubsmns[0])))  ERR(retval);

  sanity_check(phi, phips, xn, xm, xn_nyq, xm_nyq, lmns);
  
  if (lasym) {
    
    // 2d VMEC arrays
    lmnc = new double[ns*mnmax];
    if ((retval = nc_inq_varid(ncid, "lmnc", &varid)))  ERR(retval);
    if ((retval = nc_get_var_double(ncid, varid, &lmnc[0])))  ERR(retval);

    bmns = new double[ns*mnmax];
    if ((retval = nc_inq_varid(ncid, "bmns", &varid)))  ERR(retval);
    if ((retval = nc_get_var_double(ncid, varid, &bmns[0])))  ERR(retval);

    rmns = new double[ns*mnmax];
    if ((retval = nc_inq_varid(ncid, "rmns", &varid)))  ERR(retval);
    if ((retval = nc_get_var_double(ncid, varid, &rmns[0])))  ERR(retval);

    zmnc = new double[ns*mnmax];
    if ((retval = nc_inq_varid(ncid, "zmnc", &varid)))  ERR(retval);
    if ((retval = nc_get_var_double(ncid, varid, &zmnc[0])))  ERR(retval);

    gmns = new double[ns*mnmax];
    if ((retval = nc_inq_varid(ncid, "gmns", &varid)))  ERR(retval);
    if ((retval = nc_get_var_double(ncid, varid, &gmns[0])))  ERR(retval);

    bsupumns = new double[ns*mnmax];
    if ((retval = nc_inq_varid(ncid, "bsupumns", &varid)))  ERR(retval);
    if ((retval = nc_get_var_double(ncid, varid, &bsupumns[0])))  ERR(retval);

    bsupvmns = new double[ns*mnmax];
    if ((retval = nc_inq_varid(ncid, "bsupvmns", &varid)))  ERR(retval);
    if ((retval = nc_get_var_double(ncid, varid, &bsupvmns[0])))  ERR(retval);

    bsubumns = new double[ns*mnmax];
    if ((retval = nc_inq_varid(ncid, "bsubumns", &varid)))  ERR(retval);
    if ((retval = nc_get_var_double(ncid, varid, &bsubumns[0])))  ERR(retval);

    bsubvmns = new double[ns*mnmax];
    if ((retval = nc_inq_varid(ncid, "bsubvmns", &varid)))  ERR(retval);
    if ((retval = nc_get_var_double(ncid, varid, &bsubvmns[0])))  ERR(retval);

    bsubsmnc = new double[ns*mnmax];
    if ((retval = nc_inq_varid(ncid, "bsubsmnc", &varid)))  ERR(retval);
    if ((retval = nc_get_var_double(ncid, varid, &bsubsmnc[0])))  ERR(retval);

    for (int i=0; i<mnmax; i++) {
      if (abs(lmnc[i]) > 0) {
	std::cout << "Error! Expected lmnc to be on the half mesh, but its value at radial index 1 is nonzero\n";
	exit(1);
      }
    }
  
  }
  nc_close(ncid);
}

void VMEC_variables::sanity_check(double* phi_v, double* phips_v, int* xn_v, int* xm_v, int* xn_nyq_v, int* xm_nyq_v, double* lmns_v) {
  // Sanity checks on VMEC arrays
  if (abs(phi_v[0]) > 1.e-14) {
    std::cout << "Error! VMEC phi array does not begin with 0\n";
    std::cout << "phi_vmec[0] = " << phi_v[0] << "\n";
    exit(1);
  }

  double dphi = phi_v[1] - phi_v[0];
  for (int j=2; j<ns; j++) {
    //    std::cout << phi[j] << "\n";
    if (abs(phi_v[j]-phi_v[j-1]-dphi) > 1.e-5) { // double precision
	std::cout << "Error! VMEC phi array is not uniformly spaced\n";
	exit(1);
    }
  }
  //  std::cout << "Final value of phi = " << phi[ns-1] << "\n";

  // phips is on the half-mesh, so skip first point
  for (int j=1; j<ns; j++) {
    if (abs(phips_v[j]+phi_v[ns-1]/(2*M_PI)) > 1.e-5) {
      std::cout << "Error! VMEC phips array is not constant and equal to -phi(ns)/(2*pi)\n";
      exit(1);
    }
  }

  // The first mode in the m and n arrays should be m=n=0
  if (!xm_v[0] == 0) {
    std::cout << "First element of xm in the wout file should be 0\n";
    exit(1);
  }
  if (!xn_v[0] == 0) {
    std::cout << "First element of xn in the wout file should be 0\n";
    exit(1);
  }
  if (!xm_nyq_v[0] == 0) {
    std::cout << "First element of xm_nyq in the wout file should be 0\n";
    std::cout << xm_nyq_v[0] << "\n";
    exit(1);
    }
  if (!xn_nyq_v[0] == 0) {
    std::cout << "First element of xn_nyq in the wout file should be 0\n";
    exit(1);
  }

  // Lambda should be on the half mesh, so its value at radial index 1 should be 0 for all (m,n)
  for (int i=0; i<mnmax; i++) {
    if (abs(lmns_v[i]) > 0) {
      std::cout << "Error! Expected lmns to be on the half mesh, but its value at radial index 1 is nonzero\n";
      exit(1);
    }
  }

}  

VMEC_variables::~VMEC_variables() {

  delete[] iotas;
  delete[] iotaf;
  delete[] presf;
  delete[] phips;
  delete[] phi;
  delete[] xn;
  delete[] xm;
  delete[] xn_nyq;
  delete[] xm_nyq;
  delete[] lmns;
  delete[] rmnc;
  delete[] zmns;
  delete[] gmnc;
  delete[] bmnc;
  delete[] bsupumnc;
  delete[] bsupvmnc;
  delete[] bsubumnc;
  delete[] bsubvmnc;
  delete[] bsubsmns;
}
