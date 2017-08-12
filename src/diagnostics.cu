#include "diagnostics.h"
#include "device_funcs.h"
#include "cuda_constants.h"
#include <sys/stat.h>

__global__ void growthRates(cuComplex *phi, cuComplex *phiOld, float dt, cuDoubleComplex *omega)
{
  unsigned int idxy = get_id1();
  
  //i_dt = i/dt
  cuDoubleComplex i_dt = make_cuDoubleComplex(0., (double) 1./dt);
  
  if(idxy<nx*nyc)
  {
    cuDoubleComplex ratio = cuComplexFloatToDouble(phi[idxy+nx*nyc*((int)(.5*nz))]) / cuComplexFloatToDouble(phiOld[idxy+nx*nyc*((int)(.5*nz))]);
    
    cuDoubleComplex logr;
    logr.x = log(cuCabs(ratio));
    logr.y = atan2(ratio.y,ratio.x);
    omega[idxy] = logr*i_dt;
  }
}


Diagnostics::Diagnostics(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo)
{
  fields_old = new Fields(grids_);
  grad_parallel = new GradParallel(grids_);

  cudaMalloc((void**) &growth_rates, sizeof(cuDoubleComplex)*grids_->NxNyc);
  cudaMallocManaged((void**) &hlspectrum, sizeof(float)*grids_->Nmoms);
  
  cudaMallocHost((void**) &growth_rates_h, sizeof(cuDoubleComplex)*grids_->NxNyc);

  cudaMallocHost((void**) &m_h, sizeof(cuComplex)*grids_->NxNycNz);
  cudaMalloc((void**) &res, sizeof(cuComplex)*grids_->NxNycNz);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  maxThreadsPerBlock_ = prop.maxThreadsPerBlock;

  // for volume averaging
  cudaDeviceSynchronize();
  fluxDenom = 0.;
  for(int i=0; i<grids_->Nz; i++) {
    fluxDenom += geo_->jacobian_h[i]*geo_->grho_h[i];
  }
  
  // set up stop file
  sprintf(stopfilename_, "%s.stop", pars_->run_name);

  if(pars_->source_option == PHIEXT) {
    char ofilename[2000];
    sprintf(ofilename, "%s.%s.time", pars_->run_name, "phi");
    // open new file (i.e. overwrite an existing one)
    timefile = fopen(ofilename,"w");
  }

  // skip over masked elements in x and y?
  mask = !pars_->linear;
}

Diagnostics::~Diagnostics()
{
  delete fields_old;
  cudaFree(growth_rates);
  cudaFree(hlspectrum);
  cudaFreeHost(growth_rates_h);
  if(pars_->source_option == PHIEXT) {
    fclose(timefile);
  }
}

// NOTE: needs to be called every step when calculating growth rates
// does not write to file every step
bool Diagnostics::loop_diagnostics(Moments* moms, Fields* fields, float dt, int counter, float time) 
{
  bool stop = false;
  // write instantaneous growth rates
  if(pars_->write_omega) {
    growthRates<<<grids_->NxNyc/maxThreadsPerBlock_+1, maxThreadsPerBlock_>>>
             (fields->phi, fields_old->phi, dt, growth_rates);
    // save fields for next time
    fields_old->copyFrom(fields);

    if(counter%pars_->nwrite==0) {
        printf("Step %d: Time = %f\n", counter, time);
      cudaMemcpyAsync(growth_rates_h, growth_rates, sizeof(cuDoubleComplex)*grids_->NxNyc, cudaMemcpyDeviceToHost);
      print_growth_rates_to_screen();
    }
  } 
  // write time history of phi
  if(pars_->source_option==PHIEXT) {
    writeTimeHistory(fields->phi, time, 1, 0, grids_->Nz/2, timefile);
  }
  // check to see if we should stop simulation
  if(counter%pars_->nwrite==0) {
    stop = checkstop();
  }
  return stop;
}

void Diagnostics::final_diagnostics(Moments* moms, Fields* fields) 
{
  // print final moments and fields
  writeMomOrField(moms->dens_ptr[0], "dens");
  writeMomOrField(moms->upar_ptr[0], "upar");
  writeMomOrField(fields->phi, "phi");

  writeMomOrFieldKpar(moms->dens_ptr[0], "dens");
  writeMomOrFieldKpar(moms->upar_ptr[0], "upar");
  writeMomOrFieldKpar(fields->phi, "phi");

  // write Hermite-Laguerre spectrum |G|**2(l,m)
  if(pars_->source_option==PHIEXT) {
    writeHLspectrum(moms->ghl, 1, 0);
  } else {
    writeHLspectrum(moms->ghl);
  }

  // write geometry coefficient arrays
  writeGeo();

  if(pars_->write_omega) writeGrowthRates();
  fflush(NULL);
}


void Diagnostics::print_growth_rates_to_screen()
{
  int Nx = grids_->Nx;
  int Ny = grids_->Ny;
  int Nyc = grids_->Nyc;

     if(mask) {
  	printf("ky\tkx\t\tomega\t\tgamma\t\tconverged?\n");
  	//for(int i=0; i<1; i++) {
        for(int i=0; i<((Nx-1)/3+1); i++) {
  	  for(int j=0; j<((Ny-1)/3+1); j++) {
  	    int index = j + (Ny/2+1)*i;
  	    if(index!=0) {
  	      printf("%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_h[i], growth_rates_h[index].x, growth_rates_h[index].y);
  	      printf("\n");
  	    }
  	  }
  	  printf("\n");
  	}
  	//for(int i=2*Nx/3+1; i<2*Nx/3+1; i++) {
        for(int i=2*Nx/3+1; i<Nx; i++) {
            for(int j=0; j<((Ny-1)/3+1); j++) {
  	    int index = j + (Ny/2+1)*i;
  	      printf("%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_h[i], growth_rates_h[index].x, growth_rates_h[index].y);
  	      printf("\n");
  	  }
  	  printf("\n");
  	}
    } else {
  	printf("ky\tkx\t\tomega\t\tgamma\t\tconverged?\n");
  	//for(int i=0; i<1; i++) {
        for(int i=0; i<Nx; i++) {
  	  for(int j=0; j<Nyc; j++) {
  	    int index = j + (Ny/2+1)*i;
  	    if(index!=0) {
  	      printf("%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_h[i], growth_rates_h[index].x, growth_rates_h[index].y);
  	      printf("\n");
  	    }
  	  }
  	  printf("\n");
  	}

    }	
}

void Diagnostics::writeGrowthRates()
{
  int Nx = grids_->Nx;
  int Ny = grids_->Ny;

  char ofilename[2000];
  sprintf(ofilename, "%s.omega.kykx", pars_->run_name);
  FILE* out = fopen(ofilename,"w+");

     if(mask) {
  	fprintf(out, "ky\tkx\t\tomega\t\tgamma\t\tconverged?\n");
  	//for(int i=0; i<1; i++) {
        for(int i=0; i<((Nx-1)/3+1); i++) {
  	  for(int j=0; j<((Ny-1)/3+1); j++) {
  	    int index = j + (Ny/2+1)*i;
  	    if(index!=0) {
  	      fprintf(out, "%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_h[i], growth_rates_h[index].x, growth_rates_h[index].y);
  	      fprintf(out, "\n");
  	    }
  	  }
  	  fprintf(out, "\n");
  	}
  	//for(int i=2*Nx/3+1; i<2*Nx/3+1; i++) {
        for(int i=2*Nx/3+1; i<Nx; i++) {
            for(int j=0; j<((Ny-1)/3+1); j++) {
  	    int index = j + (Ny/2+1)*i;
  	      fprintf(out, "%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_h[i], growth_rates_h[index].x, growth_rates_h[index].y);
  	      fprintf(out, "\n");
  	  }
  	  fprintf(out, "\n");
  	}	
    } else {
  	fprintf(out, "ky\tkx\t\tomega\t\tgamma\t\tconverged?\n");
  	//for(int i=0; i<1; i++) {
        for(int i=0; i<Nx; i++) {
  	  for(int j=0; j<Ny/2+1; j++) {
  	    int index = j + (Ny/2+1)*i;
  	    if(index!=0) {
  	      fprintf(out, "%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_h[i], growth_rates_h[index].x, growth_rates_h[index].y);
  	      fprintf(out, "\n");
  	    }
  	  }
  	  fprintf(out, "\n");
  	}
    }
  fclose(out);
}

void Diagnostics::writeMomOrFieldKpar(cuComplex* m, const char* name) {
  int Nx = grids_->Nx;
  int Ny = grids_->Ny;
  int Nz = grids_->Nz;

  char ofilename[2000];
  sprintf(ofilename, "%s.%s.kpar_field", pars_->run_name, name);
  FILE* out = fopen(ofilename,"w+");
  grad_parallel->fft_only(m, res);
  cudaMemcpy(m_h,res,sizeof(cuComplex)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToHost);
  fprintf(out, "#\tkz (1)\t\t\tky (2)\t\t\tkx (3)\t\t\tRe (4)\t\t\tIm (5)\t\t\t");  
  fprintf(out, "\n");
  int blockid = 0;
  if(mask) {
    for(int i=0; i<(Nx-1)/3+1; i++) {
      for(int j=0; j<(Ny-1)/3+1; j++) {
        fprintf(out, "\n#%d\n\n", blockid);
        blockid++;      
        for(int k=0; k<Nz; k++) {
          int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
          //if(index!=0){
            fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", grids_->kz_h[k], grids_->ky_h[j], grids_->kx_h[i], m_h[index].x, m_h[index].y);    	  
          //}
        }     
      }
    }
    for(int i=2*Nx/3+1; i<Nx; i++) {
      for(int j=0; j<(Ny-1)/3+1; j++) {
        fprintf(out, "\n#%d\n\n", blockid);
        blockid++;
        for(int k=0; k<Nz; k++) {
          int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
          fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", grids_->kz_h[k], grids_->ky_h[j], grids_->kx_h[i], m_h[index].x, m_h[index].y);    	  
        }    
      }
    }
  } else {
    for(int i=0; i<Nx; i++) {
      for(int j=0; j<Ny/2+1; j++) {
        fprintf(out, "\n#%d\n\n", blockid);
        blockid++;      
        for(int k=0; k<Nz; k++) {
          int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
          //if(index!=0){
            fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", grids_->kz_h[k], grids_->ky_h[j], grids_->kx_h[i], m_h[index].x, m_h[index].y);    	  
          //}
        }     
      }
    }

  }
  fclose(out);

}

void Diagnostics::writeMomOrField(cuComplex* m, const char* name) {
  int Nx = grids_->Nx;
  int Ny = grids_->Ny;
  int Nz = grids_->Nz;

  char ofilename[2000];
  sprintf(ofilename, "%s.%s.field", pars_->run_name, name);
  FILE* out = fopen(ofilename,"w+");
  cudaMemcpy(m_h,m,sizeof(cuComplex)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToHost);
  fprintf(out, "#\tz (1)\t\t\tky (2)\t\t\tkx (3)\t\t\tRe (4)\t\t\tIm (5)\t\t\t");  
  fprintf(out, "\n");
  int blockid = 0;
  if(mask) {
    for(int i=0; i<(Nx-1)/3+1; i++) {
      for(int j=0; j<(Ny-1)/3+1; j++) {
        fprintf(out, "\n#%d\n\n", blockid);
        blockid++;      
        for(int k=0; k<=Nz; k++) {
          int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
  	//if(index!=0){
  	  if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -geo_->z_h[0], grids_->ky_h[j], grids_->kx_h[i], m_h[j+(Ny/2+1)*i].x, m_h[j+(Ny/2+1)*i].y); 
  	  else fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", geo_->z_h[k], grids_->ky_h[j], grids_->kx_h[i], m_h[index].x, m_h[index].y);    	  
          //}
        }     
      }
    }
    for(int i=2*Nx/3+1; i<Nx; i++) {
      for(int j=0; j<(Ny-1)/3+1; j++) {
        fprintf(out, "\n#%d\n\n", blockid);
        blockid++;
        for(int k=0; k<=Nz; k++) {
          int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
  	if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -geo_->z_h[0], grids_->ky_h[j], grids_->kx_h[i], m_h[j+(Ny/2+1)*i].x, m_h[j+(Ny/2+1)*i].y); 
  	else fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", geo_->z_h[k], grids_->ky_h[j], grids_->kx_h[i], m_h[index].x, m_h[index].y);    	  
        }    
      }
    }
  } else {

    for(int i=0; i<Nx; i++) {
      for(int j=0; j<Ny/2+1; j++) {
        fprintf(out, "\n#%d\n\n", blockid);
        blockid++;      
        for(int k=0; k<=Nz; k++) {
          int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
  	//if(index!=0){
  	  if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -geo_->z_h[0], grids_->ky_h[j], grids_->kx_h[i], m_h[j+(Ny/2+1)*i].x, m_h[j+(Ny/2+1)*i].y); 
  	  else fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", geo_->z_h[k], grids_->ky_h[j], grids_->kx_h[i], m_h[index].x, m_h[index].y);    	  
          //}
        }     
      }
    }

  }
  fclose(out);
}

// MFM
void Diagnostics::writeGridFile(const char* name) {

  char ofilename[2000], c;
  sprintf(ofilename, "%s.%s", pars_->run_name, name);
  FILE* ifile = fopen(pars_->geofilename,"r");
  FILE* ofile = fopen(ofilename,"w+");

  c = fgetc(ifile);
  while (c != EOF) {
    fputc(c, ofile);
    c = fgetc(ifile);
  }
  fclose(ifile);
  fclose(ofile);
}


void Diagnostics::writeHLspectrum(cuComplex* ghl, int ikx, int iky)
{
  // calculate spectrum  
  HLspectrum(ghl, ikx, iky);
  cudaDeviceSynchronize();

  char ofilename[2000];
  sprintf(ofilename, "%s.hlspectrum.out", pars_->run_name);
  FILE* out = fopen(ofilename,"w+");
  
  for(int l=0; l<grids_->Nhermite; l++) {
    for(int m=0; m<grids_->Nlaguerre; m++) {
      fprintf(out, "%d\t%d\t%e\n", l, m, hlspectrum[m+grids_->Nlaguerre*l]);
    }
    fprintf(out, "\n");
  }
  fclose(out); 
}

// sum over ky, kz, z with flux surface average
__global__ void volume_average(float* res, cuComplex* f, float* jacobian, float fluxDenomInv, int ikx, int iky) {
  // reduction code follows https://github.com/parallel-forall/code-samples/blob/master/posts/parallel_reduction_with_shfl/device_reduce_atomic.h
  // device_reduce_atomic_kernel
  float sum = 0.;
  for(int idxyz=blockIdx.x*blockDim.x+threadIdx.x;idxyz<nx*nyc*nz;idxyz+=blockDim.x*gridDim.x) {
    unsigned int idy = idxyz % (nx*nyc) % nyc; 
    unsigned int idx = idxyz % (nx*nyc) / nyc; 
    unsigned int idz = idxyz / (nx*nyc);
    float fac;
    if(idy==0) fac = 1.;
    else fac = 0.5;
    if(ikx<0 && iky<0) {
      if(idy>0 || idx>0) sum += (f[idxyz].x*f[idxyz].x + f[idxyz].y*f[idxyz].y)*jacobian[idz]*fac*fluxDenomInv;
    } else {
      if(idy==iky && idx==ikx) sum += (f[idxyz].x*f[idxyz].x + f[idxyz].y*f[idxyz].y)*jacobian[idz]*fac*fluxDenomInv;
    }
  }
  atomicAdd(res,sum);
}

void Diagnostics::HLspectrum(cuComplex* ghl, int ikx, int iky)
{
  int threads=256;
  int blocks=min((grids_->NxNycNz+threads-1)/threads,2048);
  cudaMemset(hlspectrum, 0., sizeof(float)*grids_->Nmoms);
  for(int l=0; l<grids_->Nhermite; l++) {
    for(int m=0; m<grids_->Nlaguerre; m++) {
      volume_average<<<blocks,threads>>>
	(&hlspectrum[m+grids_->Nlaguerre*l], 
	 &ghl[grids_->NxNycNz*m + grids_->NxNycNz*grids_->Nlaguerre*l], geo_->jacobian, 1./fluxDenom, ikx, iky);
    }
  }
}

void Diagnostics::writeGeo() 
{
  char ofilename[2000];
  sprintf(ofilename, "%s.geo.z", pars_->run_name);
  FILE* out = fopen(ofilename,"w+");

  fprintf(out, "#\tz:1\t\tbmag:2\t\tbgrad:3\t\tgbd:4\t\tgbd0:5\t\tcvd:6\t\tcvd0:7\t\tgds2:8\t\tgds21:9\t\tgds22:10\tgrho:11\t\tjacobian:12\n");
  for(int i=0; i<grids_->Nz; i++) {
    fprintf(out, "\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n", geo_->z_h[i], geo_->bmag_h[i], geo_->bgrad_h[i], geo_->gbdrift_h[i], geo_->gbdrift0_h[i], geo_->cvdrift_h[i], geo_->cvdrift0_h[i], geo_->gds2_h[i], geo_->gds21_h[i], geo_->gds22_h[i], geo_->grho_h[i], geo_->jacobian_h[i]);
  }
  //periodic point
  int i=0;
  fprintf(out, "\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n", -geo_->z_h[i], geo_->bmag_h[i], geo_->bgrad_h[i], geo_->gbdrift_h[i], geo_->gbdrift0_h[i], geo_->cvdrift_h[i], geo_->cvdrift0_h[i], geo_->gds2_h[i], geo_->gds21_h[i], geo_->gds22_h[i], geo_->grho_h[i], geo_->jacobian_h[i]);
}

void Diagnostics::writeTimeHistory(cuComplex* f, float time, int ikx, int iky, int iz, FILE* out) 
{
  cuComplex val;
  cudaMemcpy(&val, &f[iky + grids_->Nyc*ikx + grids_->NxNyc*iz], sizeof(cuComplex), cudaMemcpyDeviceToHost);
  fprintf(out, "%.4f\t\t%.4f\t\t%.4f\n", time, val.x, val.y);
}
  

bool Diagnostics::checkstop() 
{
  struct stat buffer;   
  // check if stopfile exists
  bool stop = (stat (stopfilename_, &buffer) == 0);   
  // remove it if it does exist
  if(stop) remove(stopfilename_);
  return stop;
}
