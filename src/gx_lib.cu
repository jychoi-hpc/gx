#include "gx_lib.h"
#ifdef USE_MPI
#include "mpi.h"
#else
typedef int MPI_Comm;
#endif
#include "cufft.h"
#include "parameters.h"
#include "run_gx.h"
#include "geometry.h"
#include "get_error.h"
#include "ncdf.h"
#include <assert.h>


__global__ void test(float* z) {
  printf("Device: %f\n", z[0]);
}

void gx_get_default_parameters_(struct external_parameters_struct * externalpars,
				char *run_name, MPI_Comm mpcom, int devid) {
  
  // MM // should the MPI functions be used only in the CPU code (or not at all)? Or is there some use for them in the GPU version?
  int iproc;

  //  printf("Communicator is %d\n", mpcom);
  
#ifdef USE_MPI
  MPI_Comm_rank(mpcom, &iproc);
#else
  iproc = 0;
#endif  
  // MM // GPU for next 4 lines, maybe some similar setup with #cores/#nodes for CPU
  int numdev;

  cudaGetDeviceCount(&numdev);
  cudaSetDevice(devid);

  cudaGetDevice(&externalpars->mpirank);
  if(iproc==0 && false) printf("Initializing gx ...\t runname is %s\n", run_name);

  
  /*  // MM // At least 2 variables in parametrs.h are GPU-specific
      - Parameters class constructor/destructor do all GPU stuff
      - Overloaded constructor for CPU/GPU?
      - Destructor with if-statements?
  */
  // read input parameters from namelist
  Parameters *pars = new Parameters; // MM // work to be done with the class 
  //  pars->read_namelist(run_name);
  pars->get_nml_vars(run_name);

  // copy elements of input_parameters_struct into external_parameters_struct externalpars
  if (iproc==0) pars->set_externalpars(externalpars); // MM // I think set_externalpars is processor-independent

  int nprocs;

#ifdef USE_MPI
  MPI_Comm_size(mpcom, &nprocs);
#else
  nprocs = 1;
#endif
  
  // MM ?? what are these serial strings?
  char serial_full[100];
  char serial[100];
  FILE *fp;

  fp = popen("nvidia-smi -q | grep Serial", "r"); // MM // probably GPU only
  //  while(fgets(serial_full, sizeof(serial_full)-1,fp) != NULL) {
  //    printf("%s\n", serial_full);
  //  }
  pclose(fp);
  
  for(int i=0; i<8; i++) {
    serial[i] = serial_full[strlen(serial_full) - (9-i)];
  }
  serial[8] = NULL;
  externalpars->job_id = atoi(serial);
  if (false) printf("SN: %d\n", externalpars->job_id);
  
  if (false) printf("About to broadcast externalpars %d %d\n", nprocs, iproc);

  // BD: We could communicate the externalpars through NetCDF files instead
  //  
  // EGH: this is a nasty way to broadcast externalpars... we should
  // really define a custom MPI datatype. However it should continue
  // to work as long as all MPI processes are running on the same
  // architecture. 
  int ret;
#ifdef USE_MPI
  ret = MPI_Bcast(&*externalpars, sizeof(external_parameters_struct), MPI_BYTE, 0, mpcom);
  if (false) printf("Broadcasted externalpars (%d) %d %d\n", ret, nprocs, iproc);
  // This has to be set after the broadcast
#endif  
  externalpars->pars_address = (void *)pars; 
  if (false) printf("Finished gx_get_default_parameters_\n");

}
  
void gx_get_fluxes_(struct external_parameters_struct *  externalpars, 
		    struct gx_outputs_struct * gxouts, MPI_Comm mpcom)
{
  int iproc;
  // iproc doesn't necessarily have to be the same as it was in 
  // gx_get_default_parameters_
#ifdef USE_MPI
  MPI_Comm_rank(mpcom, &iproc);
#else
  iproc = 0;
#endif
  Parameters* pars = (Parameters *)externalpars->pars_address;
  
  pars->iproc = iproc;
  
  //  int gpuID = externalpars->job_id;
  
  // Only proc0 needs to import paramters to gx
  // copy elements of external_parameters_struct externalpars into pars
  // this is done because externalpars may have been changed externally (i.e. by Trinity) 
  // between calls to gx_get_default_parameters and gx_get_fluxes.
  // pars then needs to be updated since pars is what is used in run_gx.
  if(iproc==0) { pars->import_externalpars(externalpars); }

  // MM ?? It seems the constructors of these 3 classes could be completely CPU or GPU specific, so the interface class would work well here
  Geometry* geo;  // geometry coefficient arrays // MM // Bunch of GPU allocation and a few kernels in this class
  Grids* grids;   // grids (e.g. kx, ky, z) // MM // Bunch of allocations and copies, probably could just use macros. A single kernel, kInit
  Diagnostics* diagnostics;
  //  HermiteTransform* herm;
  
  DEBUGPRINT("Initializing grids...\n");
  grids = new Grids(pars);
  CUDA_DEBUG("Initializing grids: %s \n");
  
  DEBUGPRINT("Grid dimensions: Nx=%d, Ny=%d, Nz=%d, Nl=%d, Nm=%d, Nspecies=%d\n",
	     grids->Nx, grids->Ny, grids->Nz, grids->Nl, grids->Nm, grids->Nspecies);

  if(iproc==0) {
    int igeo = pars->igeo;
    DEBUGPRINT("Initializing geometry...\n");
    if(igeo==0) {
      geo = new S_alpha_geo(pars, grids);
      CUDA_DEBUG("Initializing geometry s_alpha: %s \n");
    }
    else if(igeo==1) {
      geo = new File_geo(pars, grids);
      CUDA_DEBUG("Initializing geometry from file: %s \n");
    } 
    else if(igeo==2) {
      DEBUGPRINT("igeo = 2 not yet implemented!\n");
      exit(1);
      //geo = new Eik_geo();
    } 
    else if(igeo==3) {
      DEBUGPRINT("igeo = 3 not yet implemented!\n");
      exit(1);
      //geo = new Gs2_geo();
    }

    DEBUGPRINT("Initializing diagnostics...\n");
    diagnostics = new Diagnostics(pars, grids, geo);
    CUDA_DEBUG("Initializing diagnostics: %s \n");    

    DEBUGPRINT("Initializing Hermite transforms...\n");
    //    herm = new HermiteTransform(grids, 1); // batch size could ultimately be nspec
    CUDA_DEBUG("Initializing Hermite transforms: %s \n");    
  }

  cudaDeviceSynchronize(); // MM // GPU

  run_gx(pars, grids, geo, diagnostics);

  delete pars;
  delete grids;
  delete geo;
  delete diagnostics;
}  	

void gx_main(int argc, char* argv[], MPI_Comm mpcom) {
  struct external_parameters_struct externalpars;
  struct gx_outputs_struct gxouts;

  // MM // GPU
  int devid = 0; // This should be determined (optionally) on the command line
  
  char *run_name;
  if ( argc < 1) {
    fprintf(stderr, "The correct usage is:\n gx <runname>\n");
    exit(1);
  } else {    
    run_name = argv[1];
  }

  gx_get_default_parameters_(&externalpars, run_name, mpcom, devid);
  gx_get_fluxes_(&externalpars, &gxouts, mpcom);
}
