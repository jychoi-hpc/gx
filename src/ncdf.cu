#include "ncdf.h"

NetCDF::NetCDF(Parameters* pars, Grids* grids, Geometry* geo, string suffix) :
  pars_(pars), grids_(grids), geo_(geo)
{
  // create netcdf file
  int retval;
  char strb[1263];
  strcpy(strb, pars_->run_name); 
  strcat(strb, suffix.c_str()); // suffix = ".out.nc" by default
  if (pars_->restart && access(strb, F_OK) == 0 && pars_->append_on_restart) {
    // if restarting and output file already exists, open it so that we can append
    if (retval = nc_open_par(strb, NC_WRITE, pars_->mpcom, MPI_INFO_NULL, &fileid)) ERR(retval);
  } else { // no restart or restarting but no existing output file
    if (retval = nc_create_par(strb, NC_CLOBBER | NC_NETCDF4, pars_->mpcom, MPI_INFO_NULL, &fileid)) ERR(retval);
  }

  // get netcdf handles for the dimensions
  nc_dims = new NcDims(pars_, grids_, fileid, pars_->restart && pars_->append_on_restart);

  // set-up and write grid variables (e.g. ky, kx, etc) to netcdf
  nc_grids = new NcGrids(grids_, nc_dims, fileid, pars_->restart && pars_->append_on_restart);

  // set-up and write geometry variables to netcdf
  if (pars_->restart && pars_->append_on_restart) {
    nc_geo = nullptr;
  } else {
    nc_geo = new NcGeo(grids_, geo_, nc_dims, fileid);
  }

  nc_diagnostics = new NcDiagnostics(fileid, pars_->restart && pars_->append_on_restart);
}

NetCDF::~NetCDF()
{
  delete nc_dims;
  delete nc_grids;
  if (nc_geo) delete nc_geo;
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
