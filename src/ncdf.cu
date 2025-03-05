#include "ncdf.h"

NetCDF::NetCDF(Parameters* pars, string suffix) 
{
  // create netcdf file
  int retval;
  char strb[1263];
  strcpy(strb, pars->run_name); 
  strcat(strb, suffix.c_str()); // suffix = ".out.nc" by default
  append_ = false;
  if (pars->restart && access(strb, F_OK) == 0 && pars->append_on_restart) {
    // if restarting and output file already exists, open it so that we can append
    if (retval = nc_open_par(strb, NC_WRITE, pars->mpcom, MPI_INFO_NULL, &fileid)) ERR(retval);
    append_ = true;
  } else { // no restart or restarting but no existing output file
    if (retval = nc_create_par(strb, NC_CLOBBER | NC_NETCDF4, pars->mpcom, MPI_INFO_NULL, &fileid)) ERR(retval);
  }

  nc_inputs = new NcInputs(fileid, append_);

}

void NetCDF::setup(Parameters* pars, Grids* grids, Geometry* geo)
{
  // get netcdf handles for the dimensions
  nc_dims = new NcDims(pars, grids, fileid, append_);

  // set-up and write grid variables (e.g. ky, kx, etc) to netcdf
  nc_grids = new NcGrids(grids, nc_dims, fileid, append_);

  // set-up and write species parameters to netcdf
  if (append_) {
    nc_species = nullptr;
  } else {
    nc_species = new NcSpecies(pars, nc_dims, fileid);
  }

  // set-up and write geometry variables to netcdf
  if (append_) {
    nc_geo = nullptr;
  } else {
    nc_geo = new NcGeo(grids, geo, nc_dims, fileid);
  }

  nc_diagnostics = new NcDiagnostics(fileid, append_);
}

NetCDF::~NetCDF()
{
  delete nc_dims;
  delete nc_grids;
  if (nc_geo) delete nc_geo;
  if (nc_species) delete nc_species;
  delete nc_diagnostics;

  // close netcdf file
  close_nc_file();  
  fflush(NULL);
}

void NetCDF::close_nc_file() {
  int retval;
  if (retval = nc_close(fileid)) ERR(retval);
}

void NetCDF::sync() {
  int retval;
  if (retval = nc_sync(fileid)) ERR(retval);
}
