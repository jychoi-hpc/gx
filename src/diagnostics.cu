#include "diagnostics.h"
#include "device_funcs.h"
#include "cuda_constants.h"
#include "get_error.h"
#include "gx_lib.h"
#include "netcdf.h"
#include <sys/stat.h>

Diagnostics::Diagnostics(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo)
{  
  red = new Red(grids_->NxNycNz);
  cudaDeviceSynchronize();
  CUDA_DEBUG("Reductions: %s \n");
  
  fields_old = new Fields(grids_);
  cudaDeviceSynchronize();
  CUDA_DEBUG("Fields: %s \n");

  id = new NetCDF_ids(grids_, pars_, geo_);
  cudaDeviceSynchronize(); 
  CUDA_DEBUG("NetCDF_ids: %s \n");
  
  grad_parallel = new GradParallelPeriodic(grids_);
  cudaDeviceSynchronize();
  CUDA_DEBUG("Grad parallel periodic: %s \n");

  if (pars_->write_omega) {
    cudaMalloc    (&growth_rates,   sizeof(cuComplex)*grids_->NxNyc);
    cudaMallocHost(&growth_rates_h, sizeof(cuComplex)*grids_->NxNyc);

    cudaMemset(growth_rates, 0., sizeof(cuComplex)*grids_->NxNyc);
  }  
  
  cudaMallocHost(&amom_h, sizeof(cuComplex)*grids_->NxNycNz);
  cudaMalloc(&amom, sizeof(cuComplex)*grids_->NxNycNz);
  cudaMalloc(&rmom, sizeof(float)*grids_->NxNycNz);

  if (pars_->write_pzt) {
    cudaMallocHost(&primary,   sizeof(float));    primary[0] = 0.;  
    cudaMallocHost(&secondary, sizeof(float));    secondary[0] = 0.;
    cudaMallocHost(&tertiary,  sizeof(float));    tertiary[0] = 0.;
    cudaMalloc(&t_bar, sizeof(cuComplex)*grids_->NxNycNz*grids_->Nspecies);
  }

  if (pars_->write_fluxes) {
    cudaMallocHost(&pflux,  sizeof(float)*grids_->Nspecies);
    cudaMallocHost(&qflux,  sizeof(float)*grids_->Nspecies);
    for (int j=0; j<grids_->Nspecies; j++) qflux[j] = 0.;
  }
  cudaMallocHost(&val,    sizeof(float)*2);
  cudaMalloc(&val1,    sizeof(float));
  cudaMemset(val1, 0., sizeof(float));
    
  cudaDeviceProp prop;
  int dev;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  maxThreadsPerBlock_ = prop.maxThreadsPerBlock;
  // for volume averaging
  fluxDenom = 0.;
  for(int i=0; i<grids_->Nz; i++) {
    fluxDenom += geo_->jacobian_h[i]*geo_->grho_h[i];
  }
  fluxDenomInv = 1./fluxDenom;
  
  // set up stop file
  sprintf(stopfilename_, "%s.stop", pars_->run_name);

  dB_spec = dim3(32, 1, 1);
  dG_spec = dim3(grids_->Nyc/dB_spec.x+1, grids_->Nx/dB_spec.y+1, grids_->Nz/dB_spec.z+1);  
  
}

Diagnostics::~Diagnostics()
{
  delete fields_old;
  delete red;
  if (pars_->write_omega) {
    cudaFree(growth_rates);
    cudaFreeHost(growth_rates_h);
  }
  if (pars_->write_fluxes) {
    cudaFreeHost(pflux);
    cudaFreeHost(qflux);
  }
  if (pars_->write_pzt) {
    cudaFreeHost(primary);
    cudaFreeHost(secondary);
    cudaFreeHost(tertiary);
    cudaFree(t_bar);
  }
  cudaFreeHost(val);
  cudaFree(val1);
}

bool Diagnostics::loop_diagnostics(MomentsG* G, Fields* fields, double dt, int counter, double time) 
{
  int retval;
  bool stop = false;
  int nw;

  nw = pars_->nwrite;
  if(counter%nw == 0) {
    printf("Step %d: Time = %f\n", counter, time);
    fflush(NULL);
    if (retval = nc_put_vara(id->file, id->time, id->time_start, id->time_count,  &time))    ERR(retval);
    id->time_start[0] += 1; 
  }

  // write instantaneous growth rates
  if(pars_->write_omega) {
    if (counter%nw == nw-1 || counter%nw == 0) {
      growthRates <<< grids_->NxNyc/maxThreadsPerBlock_+1, maxThreadsPerBlock_>>>
      	(fields->phi, fields_old->phi, dt, growth_rates);
      // save fields for next time
      fields_old->copyFrom(fields);
    }
    
    if(counter > 0 && counter%nw == 0) {
      CP_TO_CPU (growth_rates_h, growth_rates, sizeof(cuComplex)*grids_->NxNyc);
      print_growth_rates_to_screen();
      writeGrowthRates();
    }
  }

  if (counter%nw == 0 && pars_->write_spec_v_time) {
    if (pars_->write_h_spectrum)  writeHspectrum (G, false);
    if (pars_->write_l_spectrum)  writeLspectrum (G, false);
    if (pars_->write_lh_spectrum) writeLHspectrum(G, false);
  }

  // write time history of phi
  if(counter%nw == 0 && pars_->write_rh) {
    ikx_local = 1; iky_local = 0; iz_local=grids_->Nz/2; // correct values for usual RH tests
    //    ikx_local = 0; iky_local = 2; iz_local=grids_->Nz/2; // test values

    CP_TO_CPU(&valphi, &fields->phi[iky_local + grids_->Nyc*ikx_local + grids_->NxNyc*iz_local], sizeof(cuComplex));
    val[0] = valphi.x; val[1] = valphi.y;
    
    if (retval = nc_put_vara(id->file, id->phi_rh, id->rh_start, id->rh_count,  val))    ERR(retval);
    id->rh_start[0] += 1;
  }

  if( counter%nw == 0 && pars_->write_pzt) {
    pzt(G, fields);  // calculate PZT
    cudaDeviceSynchronize();
    if (retval = nc_put_vara(id->file, id->prim, id->pzt_start, id->pzt_count,  &primary[0]))   ERR(retval);
    if (retval = nc_put_vara(id->file, id->sec,  id->pzt_start, id->pzt_count,  &secondary[0])) ERR(retval);
    if (retval = nc_put_vara(id->file, id->tert, id->pzt_start, id->pzt_count,  &tertiary[0]))  ERR(retval);
    primary[0]=0.; secondary[0]=0.; tertiary[0]=0.;
    id->pzt_start[0] += 1; 
  }

  if( counter%nw == 0 && pars_->write_fluxes) {
    // calculate fluxes
    fluxes(G, fields);
    cudaDeviceSynchronize();
    if (retval = nc_put_vara(id->file, id->qflux, id->flux_start, id->flux_count,  &qflux[0]))    ERR(retval);
    for (int j=0; j<grids_->Nspecies; j++) qflux[j] = 0.;    
    id->flux_start[0] += 1; 
  }

  // sync the netcdf data 
  if (counter%nw == 0) {
    if (retval = nc_sync(id->file)) ERR(retval);
  }

  // check to see if we should stop simulation
  stop = checkstop();
  return stop;
}

void Diagnostics::final_diagnostics(MomentsG* G, Fields* fields) 
{
  // print final moments and fields

  if (pars_->write_moms) {
    writeMomOrField (fields->phi,    id->phi);
    writeMomOrField (G->dens_ptr[0], id->density);
    if (grids_->Nm>1) {writeMomOrField (G->upar_ptr[0], id->upar);}
  }
  
  if (pars_->write_phi_kpar) {

    grad_parallel->fft_only(G->dens_ptr[0], amom, CUFFT_FORWARD);
    writeMomOrField (amom, id->density_kpar);

    grad_parallel->fft_only(fields->phi, amom, CUFFT_FORWARD);
    writeMomOrField (amom, id->phi_kpar);
  }
  
  if (pars_->write_h_spectrum)  writeHspectrum (G, true);
  if (pars_->write_l_spectrum)  writeLspectrum (G, true);
  if (pars_->write_lh_spectrum) writeLHspectrum(G, true);

  id->close_nc_file();
  fflush(NULL);
}


//void Diagnostics::write_something(NetCDF_ids* id)
void Diagnostics::write_something()
{
  printf("The current value of the nx handle is %i \n",id->nx);
}

void Diagnostics::writeGrowthRates()
{
  int retval;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  float *omt_out;
  
  cudaMallocHost((void**) &omt_out, sizeof(float)*Nakx*Naky*2);

  reduce2k(omt_out, growth_rates_h);

  if (retval = nc_put_vara(id->file, id->omega_t, id->omt_start, id->omt_count,  omt_out))    ERR(retval);
  id->omt_start[0] += 1;

  cudaFreeHost(omt_out); 
}

void Diagnostics::write_init(MomentsG* G, Fields* fields) {
  writeMomOrField (G->dens_ptr[0], id->density0);
  writeMomOrField (fields->phi,    id->phi0);  
}

void Diagnostics::writeMomOrField(cuComplex* m, int handle) {

  int retval;
  int Nx  = grids_->Nx;
  int Nyc = grids_->Nyc;
  int Nz  = grids_->Nz;

  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  float *mom_out;
  
  CP_TO_CPU(amom_h, m, sizeof(cuComplex)*Nx*Nyc*Nz);

  cudaMallocHost((void**) &mom_out, sizeof(float)*Nakx*Naky*Nz*2);
  reduce2z(mom_out, amom_h);

  if (retval = nc_put_vara(id->file, handle, id->mom_start, id->mom_count, mom_out)) ERR(retval);
  cudaFreeHost(mom_out); 
}

void Diagnostics::writeLHspectrum(MomentsG* G, bool endrun)
{
  int retval;
  float *lhspectrum, *lhspectrum_h; 
  cudaMalloc    (&lhspectrum,     sizeof(float)*grids_->Nmoms);
  cudaMallocHost(&lhspectrum_h,   sizeof(float)*grids_->Nmoms);

  // calculate spectrum  
  LHspectrum(G, lhspectrum);

  CP_TO_CPU(lhspectrum_h, lhspectrum, sizeof(float)*grids_->Nmoms);
  cudaFree(lhspectrum);

  if (endrun) {
    if (retval=nc_put_vara(id->file, id->lhspec, id->lh_start, id->lh_count, lhspectrum_h)) ERR(retval);
  } else {   
    if (retval=nc_put_vara(id->file, id->lhspec_t, id->lht_start, id->lht_count, lhspectrum_h)) ERR(retval);
    id->lht_start[0] += 1;
  }
  cudaFreeHost(lhspectrum_h);
}

void Diagnostics::writeLspectrum(MomentsG* G, bool endrun)
{
  int retval;
  float *lspectrum, *lspectrum_h; 
  cudaMalloc    ((void**) &lspectrum,     sizeof(float)*grids_->Nl);
  cudaMallocHost((void**) &lspectrum_h,   sizeof(float)*grids_->Nl);

  // calculate spectrum  
  Lspectrum(G, lspectrum);

  CP_TO_CPU(lspectrum_h, lspectrum, sizeof(float)*grids_->Nl);
  cudaFree(lspectrum);
  
  if (endrun) {
    if (retval=nc_put_vara(id->file, id->lspec, id->l_start, id->l_count, lspectrum_h)) ERR(retval);
  } else {
    if (retval=nc_put_vara(id->file, id->lspec_t, id->lt_start, id->lt_count, lspectrum_h)) ERR(retval);
    id->lt_start[0] += 1;
  }
  cudaFreeHost(lspectrum_h);
}

void Diagnostics::writeHspectrum(MomentsG* G, bool endrun)
{
  int retval;
  float *hspectrum, *hspectrum_h; 
  cudaMalloc    ((void**) &hspectrum,     sizeof(float)*grids_->Nm);
  cudaMemset(hspectrum, 0.,               sizeof(float)*grids_->Nm);
  cudaMallocHost((void**) &hspectrum_h,   sizeof(float)*grids_->Nm);
  for (int m=0; m<grids_->Nm; m++) hspectrum_h[m] = 0.;

  // calculate spectrum  
  Hspectrum(G, hspectrum);

  CP_TO_CPU(hspectrum_h, hspectrum, sizeof(float)*grids_->Nm);
  cudaFree(hspectrum);
 
  float *hmom_real, *hmom_real_h, *hmom_imag, *hmom_imag_h; 
  
  cudaMalloc    ((void**) &hmom_real,     sizeof(float)*grids_->Nm);
  cudaMemset(hmom_real, 0.,               sizeof(float)*grids_->Nm);
  cudaMallocHost((void**) &hmom_real_h,   sizeof(float)*grids_->Nm);
  for (int m=0; m<grids_->Nm; m++) hmom_real_h[m] = 0.;
  
  cudaMalloc    ((void**) &hmom_imag,     sizeof(float)*grids_->Nm);
  cudaMemset(hmom_imag, 0.,               sizeof(float)*grids_->Nm);
  cudaMallocHost((void**) &hmom_imag_h,   sizeof(float)*grids_->Nm);
  for (int m=0; m<grids_->Nm; m++) hmom_imag_h[m] = 0.;

  for (int m = 0; m < grids_->Nm; m++) {
      Hmom <<<1,1>>> (G->G(0,m), hmom_real, hmom_imag, m, pars_->forcing_index);
  }

  CP_TO_CPU(hmom_real_h, hmom_real, sizeof(float)*grids_->Nm);
  cudaFree(hmom_real);

  CP_TO_CPU(hmom_imag_h, hmom_imag, sizeof(float)*grids_->Nm);
  cudaFree(hmom_imag);

  if (endrun) {
    if (retval=nc_put_vara(id->file, id->hspec, id->m_start, id->m_count, hspectrum_h)) ERR(retval);
    if (retval=nc_put_vara(id->file, id->hmom_real, id->m_start, id->m_count, hmom_real_h)) ERR(retval);
    if (retval=nc_put_vara(id->file, id->hmom_imag, id->m_start, id->m_count, hmom_imag_h)) ERR(retval);
  } else {
    if (retval=nc_put_vara(id->file, id->hspec_t, id->mt_start, id->mt_count, hspectrum_h)) ERR(retval);
    if (retval=nc_put_vara(id->file, id->hmom_real_t, id->mt_start, id->mt_count, hmom_real_h)) ERR(retval);
    if (retval=nc_put_vara(id->file, id->hmom_imag_t, id->mt_start, id->mt_count, hmom_imag_h)) ERR(retval);
    id->mt_start[0] += 1;
  }    
  cudaFreeHost(hspectrum_h);
  cudaFreeHost(hmom_real_h);
  cudaFreeHost(hmom_imag_h);
}
void Diagnostics::LHspectrum(MomentsG* G, float* lhspectrum)
{
  for(int m=0; m<grids_->Nm; m++) {
    for(int l=0; l<grids_->Nl; l++) {
      vol_summand <<<dG_spec, dB_spec>>> (rmom, G->G(l,m), G->G(l,m), geo_->jacobian, fluxDenomInv);
      red->Sum(rmom, val1, true); 
      cudaMemcpy(&lhspectrum[l + grids_->Nl * m], val1, sizeof(float), cudaMemcpyDeviceToDevice);      
    }
  }
}

void Diagnostics::Lspectrum(MomentsG* G, float* lspectrum)
{
  for(int l=0; l<grids_->Nl; l++) {
    cudaMemset(val1, 0., sizeof(float));
    for(int m=0; m<grids_->Nm; m++) {
      vol_summand <<<dG_spec, dB_spec>>> (rmom, G->G(l,m), G->G(l,m), geo_->jacobian, fluxDenomInv);
      red->Sum(rmom, val1);
    }
    cudaMemcpy(&lspectrum[l], val1, sizeof(float), cudaMemcpyDeviceToDevice);
  }
}

void Diagnostics::Hspectrum(MomentsG* G, float* hspectrum)
{
  for(int m=0; m<grids_->Nm; m++) {
    cudaMemset(val1, 0., sizeof(float));
    for(int l=0; l<grids_->Nl; l++) {
      vol_summand <<<dG_spec, dB_spec>>> (rmom, G->G(l,m), G->G(l,m), geo_->jacobian, fluxDenomInv);
      red->Sum(rmom, val1);
    }
    cudaMemcpy(&hspectrum[m], val1, sizeof(float), cudaMemcpyDeviceToDevice);
  }
}

void Diagnostics::pzt(MomentsG* G, Fields* f)
{
  int threads=256;
  int blocks=(grids_->NxNycNz+threads-1)/threads;

  Tbar <<<blocks, threads>>> (t_bar, G->G(), f->phi, geo_->kperp2);
  get_pzt <<<blocks, threads>>> (&primary[0], &secondary[0], &tertiary[0], f->phi, t_bar);
}

void Diagnostics::fluxes(MomentsG* G, Fields* f)
{
  
  for(int is=0; is<grids_->Nspecies; is++) {
    heat_flux_summand <<<dG_spec, dB_spec>>> (rmom, f->phi, G->G(0,0,is), grids_->ky,
					      geo_->jacobian, fluxDenomInv, geo_->kperp2, pars_->species_h[is].rho2);
    red->Sum(rmom, val1, true);
    cudaMemcpy(&qflux[is], val1, sizeof(float), cudaMemcpyDeviceToDevice);
  }
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

// condense a (ky,kx) object for netcdf output, taking into account the mask
// and changing the type from cuComplex to float
void Diagnostics::reduce2k(float *fk, cuComplex* f) {

  int Nx = grids_->Nx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;

  for(int i=0; i<((Nx-1)/3+1); i++) {
    for(int j=0; j<Naky; j++) {
      int index     = j + Nyc *i; 
      int index_out = j + Naky*i; 
      fk[2*index_out]   = f[index].x;
      fk[2*index_out+1] = f[index].y;
    }
  }
  
  for(int i=2*Nx/3+1; i<Nx; i++) {
    for(int j=0; j<Naky; j++) {
      int index = j + Nyc *i;
      int index_out = j + Naky*( i - 2*Nx/3 + (Nx-1)/3 );
      fk[2*index_out]   = f[index].x;
      fk[2*index_out+1] = f[index].y;
    }
  }	  
}

// condense a (ky,kx,z) object for netcdf output, taking into account the mask
// and changing the type from cuComplex to float
void Diagnostics::reduce2z(float *fz, cuComplex* f) {

  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;
  int Nz   = grids_->Nz;

  for (int k=0; k<Nz; k++) {
    for (int i=0; i<(Nx-1)/3+1; i++) {
      for (int j=0; j<Naky; j++) {
	int index     = j + Nyc *i + Nyc*Nx*k; 
	int index_out = j + Naky*i + Naky*Nakx*k; 
	fz[2*index_out]   = f[index].x;
	fz[2*index_out+1] = f[index].y;
      }
    }
  }
  
  for (int k=0; k<Nz; k++) {
    for(int i=2*Nx/3+1; i<Nx; i++) {
      for(int j=0; j<Naky; j++) {
	int index     = j + Nyc *i + Nyc*Nx*k;
	int index_out = j + Naky*( i - 2*Nx/3 + (Nx-1)/3 ) + Naky*Nakx*k;
	fz[2*index_out]   = f[index].x;
	fz[2*index_out+1] = f[index].y;
      }
    }	
  }	
}

void Diagnostics::print_growth_rates_to_screen()
{
  int Nx = grids_->Nx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;

  printf("ky\tkx\t\tomega\t\tgamma\n");
  
  for(int j=0; j<Naky; j++) {
    for(int i=2*Nx/3+1; i<Nx; i++) {
      int index = j + Nyc*i;
      printf("%.4f\t%.4f\t\t%.6f\t%.6f",
	     grids_->ky_h[j], grids_->kx_outh[i], growth_rates_h[index].x, growth_rates_h[index].y);
      printf("\n");
    }
    for(int i=0; i<((Nx-1)/3+1); i++) {
      int index = j + Nyc*i;
      if(index!=0) {
	printf("%.4f\t%.4f\t\t%.6f\t%.6f",
	       grids_->ky_h[j], grids_->kx_outh[i], growth_rates_h[index].x, growth_rates_h[index].y);
	printf("\n");
      } else {
	printf("%.4f\t%.4f\n", grids_->ky_h[j], grids_->kx_outh[i]);
      }
    }
    if (Nx>1) printf("\n");
  }
}


