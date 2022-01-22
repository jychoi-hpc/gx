#include "run_gx.h"
#include "helper_cuda.h"
#include "trinity_interface.h"

void gx_get_fluxes_(trin_parameters_struct* tpars, trin_fluxes_struct* tfluxes, char* run_name, int mpcom_f)
{
  Parameters* pars = nullptr;
  MPI_Comm mpcom = MPI_Comm_f2c(mpcom_f);
  int iproc;
  MPI_Comm_rank(mpcom, &iproc);
  pars = new Parameters(mpcom);
  // get default values from namelist
  pars->get_nml_vars(run_name);

  // overwrite parameters based on values from trinity
  pars->set_from_trinity(tpars);

  // initialize grids, geometry, and diagnostics
  Grids       * grids       = nullptr;
  
  DEBUGPRINT("Initializing grids...\n");
  grids = new Grids(pars);
  CUDA_DEBUG("Initializing grids: %s \n");

  DEBUGPRINT("Grid dimensions: Nx=%d, Ny=%d, Nz=%d, Nl=%d, Nm=%d, Nspecies=%d\n",
	     grids->Nx, grids->Ny, grids->Nz, grids->Nl, grids->Nm, grids->Nspecies);

  Geometry    * geo         = nullptr;
  Diagnostics_GK * diagnostics = nullptr;

  int igeo = pars->igeo;
  DEBUGPRINT("Initializing geometry...\n");
  if(igeo==0) {
    geo = new S_alpha_geo(pars, grids);
    CUDA_DEBUG("Initializing geometry s_alpha: %s \n");
  }
  else if(igeo==1) {
    geo = new File_geo(pars, grids);
    printf("************************* \n \n \n");
    printf("Warning: may have assumed grho = 1 \n \n \n");
    printf("************************* \n");
    CUDA_DEBUG("Initializing geometry from file: %s \n");
  } 
  else if(igeo==2) {
    geo = new geo_nc(pars, grids);
    CUDA_DEBUG("Initializing geometry from NetCDF file: %s \n");
  } 
  else if(igeo==3) {
    DEBUGPRINT("igeo = 3 not yet implemented!\n");
    exit(1);
    //geo = new Gs2_geo();
  }

  DEBUGPRINT("Initializing diagnostics...\n");
  diagnostics = new Diagnostics_GK(pars, grids, geo);
  CUDA_DEBUG("Initializing diagnostics: %s \n");    
  
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // run gx calculation using (updated) parameters
  run_gx(pars, grids, geo, diagnostics);

  // copy time-averaged fluxes to trinity
  diagnostics->copy_fluxes_to_trinity(tfluxes);

  delete pars;
  delete grids;
  delete geo;
  delete diagnostics;
}
