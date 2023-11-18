#include "ncdf.h"

NetCDF::NetCDF(Parameters* pars, Grids* grids, Geometry* geo, string suffix) :
  pars_(pars), grids_(grids), geo_(geo)
{
  // create netcdf file
  int retval;
  char strb[263];
  strcpy(strb, pars_->run_name); 
  strcat(strb, suffix.c_str()); // suffix = ".out.nc" by default
  if (retval = nc_create_par(strb, NC_CLOBBER | NC_NETCDF4, pars_->mpcom, MPI_INFO_NULL, &fileid)) ERR(retval);

  // get netcdf handles for the dimensions
  nc_dims = new NcDims(pars_, grids_, fileid);

  // set-up and write grid variables (e.g. ky, kx, etc) to netcdf
  nc_grids = new NcGrids(grids_, nc_dims, fileid);

  // set-up and write geometry variables to netcdf
  nc_geo = new NcGeo(grids_, geo_, nc_dims, fileid);

  nc_diagnostics = new NcDiagnostics(fileid);
}

NetCDF::~NetCDF()
{
  delete nc_dims;
  delete nc_grids;
  delete nc_geo;
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
