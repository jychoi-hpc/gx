#include "diagnostics.h"
#include "get_error.h"
#include "netcdf.h"
#include <sys/stat.h>

Diagnostics_GK::Diagnostics_GK(Parameters* pars, Grids* grids, Geometry* geo, Linear* linear, Nonlinear* nonlinear, NetCDF* ncdf) :
  geo_(geo), linear_(linear), nonlinear_(nonlinear), ncdf_(ncdf), fields_old(nullptr), ncdf_big_(nullptr)
{
  pars_ = pars;
  grids_ = grids;
  
  ncdf_->setup(pars_, grids_, geo_);

  if (pars_->write_fields || pars_->write_moms) {
    ncdf_big_ = new NetCDF(pars_, ".big.nc");
    ncdf_big_->setup(pars_, grids_, geo_);
  }

  // set up spectra calculators
  // Always need allSpectra for Phi2 / A_||^2 below
  // Always need tmpG / tmpf for Phi2
  allSpectra_ = new AllSpectraCalcs(grids_, ncdf_->nc_dims);
  cudaMalloc (&tmpG, sizeof(float) * grids_->NxNycNz * grids_->Nmoms * grids_->Nspecies);
  cudaMalloc (&tmpf, sizeof(float) * grids_->NxNycNz * grids_->Nspecies);

  if(pars_->write_moms) {
    cudaMalloc (&tmpC, sizeof(cuComplex) * grids_->NxNycNz * grids_->Nspecies);
  }

  G_old = (MomentsG**) malloc(sizeof(void*)*grids_->Nspecies);

  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is+grids->is_lo;
    G_old[is] = new MomentsG (pars_, grids_, is_glob);
  }

  fields_old = new Fields(pars_, grids_);

  // initialize energy spectra diagnostics
  // Always turn on Phi2
  spectraDiagnosticList.push_back(std::make_unique<Phi2Diagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
  spectraDiagnosticList.push_back(std::make_unique<Phi2ZonalDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));

  // If fapar > 0.0, log A_||^2
  if( pars_->fapar > 0.0 ) {
    spectraDiagnosticList.push_back(std::make_unique<Apar2Diagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
  }

  if(pars_->write_free_energy) {
    spectraDiagnosticList.push_back(std::make_unique<WgDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
    spectraDiagnosticList.push_back(std::make_unique<WphiDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
    spectraDiagnosticList.push_back(std::make_unique<WaparDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
  }

  // initialize flux spectra diagnostics
  if(pars_->write_fluxes) {
    spectraDiagnosticList.push_back(std::make_unique<HeatFluxDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
    spectraDiagnosticList.push_back(std::make_unique<HeatFluxESDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
    spectraDiagnosticList.push_back(std::make_unique<HeatFluxAparDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
    spectraDiagnosticList.push_back(std::make_unique<HeatFluxBparDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
    spectraDiagnosticList.push_back(std::make_unique<ParticleFluxDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
    spectraDiagnosticList.push_back(std::make_unique<ParticleFluxESDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
    spectraDiagnosticList.push_back(std::make_unique<ParticleFluxAparDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
    spectraDiagnosticList.push_back(std::make_unique<ParticleFluxBparDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
    spectraDiagnosticList.push_back(std::make_unique<TurbulentHeatingDiagnostic>(pars_, grids_, geo_, linear_, ncdf_, allSpectra_));

  }
  
  // initialize growth rate diagnostic
  if(pars_->write_omega) {
    growthRateDiagnostic = new GrowthRateDiagnostic(pars_, grids_, ncdf_);
  }

  // initialize fields diagnostics
  if(pars_->write_fields) {
    fieldsDiagnostic = new FieldsDiagnostic(pars_, grids_, ncdf_big_);
    if(pars_->nonlinear_mode) {
      fieldsXYDiagnostic = new FieldsXYDiagnostic(pars_, grids_, nonlinear_, ncdf_big_);
    }
  }

  // set up moments diagnostics
  if(pars_->write_moms) {
    momentsDiagnosticList.push_back(std::make_unique<DensityDiagnostic>(pars_, grids_, geo_, nonlinear_, ncdf_big_));
    momentsDiagnosticList.push_back(std::make_unique<UparDiagnostic>(pars_, grids_, geo_, nonlinear_, ncdf_big_));
    momentsDiagnosticList.push_back(std::make_unique<TparDiagnostic>(pars_, grids_, geo_, nonlinear_, ncdf_big_));
    if(grids_->Nl>1) momentsDiagnosticList.push_back(std::make_unique<TperpDiagnostic>(pars_, grids_, geo_, nonlinear_, ncdf_big_));
    momentsDiagnosticList.push_back(std::make_unique<ParticleDensityDiagnostic>(pars_, grids_, geo_, nonlinear_, ncdf_big_));
    momentsDiagnosticList.push_back(std::make_unique<ParticleUparDiagnostic>(pars_, grids_, geo_, nonlinear_, ncdf_big_));
    momentsDiagnosticList.push_back(std::make_unique<ParticleUperpDiagnostic>(pars_, grids_, geo_, nonlinear_, ncdf_big_));
    momentsDiagnosticList.push_back(std::make_unique<ParticleTempDiagnostic>(pars_, grids_, geo_, nonlinear_, ncdf_big_));
  }

  // set up stop file
  sprintf(stopfilename_, "%s.stop", pars_->run_name);
}

Diagnostics_GK::~Diagnostics_GK()
{
  if(pars_->write_omega) {
    delete growthRateDiagnostic;
  }
  if(pars_->write_free_energy || pars_->write_fluxes) {
    spectraDiagnosticList.clear();
  }
  if(pars_->write_moms) {
    momentsDiagnosticList.clear();
  }

  delete allSpectra_;

  if(pars_->write_fields) delete fieldsDiagnostic;
  if(pars_->write_fields && pars_->nonlinear_mode) delete fieldsXYDiagnostic;
  if(fields_old) delete fields_old;
  if(ncdf_) delete ncdf_;
  if(ncdf_big_) delete ncdf_big_;
}

bool Diagnostics_GK::loop(MomentsG** G, Fields* fields, double dt, int counter, double time) 
{
  bool stop = false;
  if(counter % pars_->nwrite == 1 || time > pars_->t_max) {
    if(grids_->iproc == 0) printf("%s: Step %7d: Time = %10.5f  dt = %.3e   ", pars_->run_name, counter, time, dt);          // To screen
    for( auto & diagnostic : spectraDiagnosticList ) {
      diagnostic->set_dt_data(G_old, fields_old, dt);
      diagnostic->calculate_and_write(G, fields, tmpG, tmpf);
    }

    if(pars_->write_omega) {
      growthRateDiagnostic->calculate_and_write(fields, fields_old, dt);
    }

    ncdf_->nc_grids->write_time(time);
    ncdf_->sync();

    if(grids_->iproc_m == 0) {
      printf("\n");
    }
    fflush(NULL);
  }

  // write out full grid (big) diagnostics less frequently
  if((counter % pars_->nwrite_big == 1 || time > pars_->t_max) && ( pars_->write_moms || pars_->write_fields) ) {
    if(pars_->write_fields) {
      fieldsDiagnostic->calculate_and_write(fields);
      if(pars_->nonlinear_mode) {
        fieldsXYDiagnostic->calculate_and_write(fields);
      }
    }

    for( auto & diagnostic : momentsDiagnosticList ) {
      diagnostic->calculate_and_write(G, fields, tmpC);
    }

    ncdf_big_->nc_grids->write_time(time);
    ncdf_big_->sync();
  }

  // save fields for growth rate calculation in next timestep
  if(counter % pars_->nwrite == 0 || time + dt > pars_->t_max) {
    fields_old->copyPhiFrom(fields);
    fields_old->copyAparFrom(fields);
    fields_old->copyBparFrom(fields);
    for(int is=0; is<grids_->Nspecies; is++) {
      G_old[is]->copyFrom(G[is]);
    }
  }
  
  // check to see if we should stop simulation
  stop = checkstop();
  return stop;
}

Diagnostics_KREHM::Diagnostics_KREHM(Parameters* pars, Grids* grids, Geometry* geo, Linear* linear, Nonlinear* nonlinear, NetCDF* ncdf) :
  geo_(geo), fields_old(nullptr), ncdf_big_(nullptr), linear_(linear), nonlinear_(nonlinear), ncdf_(ncdf)
{
  pars_ = pars;
  grids_ = grids;

  ncdf_->setup(pars_, grids_, geo_);

  if (pars_->write_fields || pars_->write_moms) {
    ncdf_big_ = new NetCDF(pars_, ".big.nc"); 
    ncdf_big_->setup(pars_, grids_, geo_);
  }

  // set up spectra calculators
  if(pars_->write_free_energy) {
    allSpectra_ = new AllSpectraCalcs(grids_, ncdf_->nc_dims);
    cudaMalloc (&tmpG, sizeof(float) * grids_->NxNycNz * grids_->Nmoms * grids_->Nspecies); 
    cudaMalloc (&tmpf, sizeof(float) * grids_->NxNycNz * grids_->Nspecies);
  }
  if(pars_->write_moms) {
    cudaMalloc (&tmpC, sizeof(cuComplex) * grids_->NxNycNz * grids_->Nspecies);
  }
  if(pars_->write_omega) {
    fields_old = new Fields(pars_, grids_);       
  }

  // initialize energy spectra diagnostics
  spectraDiagnosticList.push_back(std::make_unique<Phi2Diagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
  spectraDiagnosticList.push_back(std::make_unique<Apar2Diagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
  spectraDiagnosticList.push_back(std::make_unique<Phi2ZonalDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
  if(pars_->write_free_energy) {
    spectraDiagnosticList.push_back(std::make_unique<WgDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
    spectraDiagnosticList.push_back(std::make_unique<WphiKrehmDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
    spectraDiagnosticList.push_back(std::make_unique<WaparKrehmDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
  }

  // initialize growth rate diagnostic
  if(pars_->write_omega) {
    growthRateDiagnostic = new GrowthRateDiagnostic(pars_, grids_, ncdf_);
  }

  // initialize fields diagnostics
  if(pars_->write_fields) {
    fieldsDiagnostic = new FieldsDiagnostic(pars_, grids_, ncdf_big_);
    if(pars_->nonlinear_mode) {
      fieldsXYDiagnostic = new FieldsXYDiagnostic(pars_, grids_, nonlinear_, ncdf_big_);
    }
  }

  // set up moments diagnostics
  if(pars_->write_moms) {
    momentsDiagnosticList.push_back(std::make_unique<DensityDiagnostic>(pars_, grids_, geo_, nonlinear_, ncdf_big_));
    momentsDiagnosticList.push_back(std::make_unique<UparDiagnostic>(pars_, grids_, geo_, nonlinear_, ncdf_big_));
    momentsDiagnosticList.push_back(std::make_unique<TparDiagnostic>(pars_, grids_, geo_, nonlinear_, ncdf_big_));
  }

  // set up stop file
  sprintf(stopfilename_, "%s.stop", pars_->run_name);
}

Diagnostics_KREHM::~Diagnostics_KREHM()
{
  if(pars_->write_omega) {
    delete growthRateDiagnostic;
  }
  if(pars_->write_free_energy || pars_->write_fluxes) {
    spectraDiagnosticList.clear();
    delete allSpectra_;
  }
  if(pars_->write_moms) {
    momentsDiagnosticList.clear();
  }

  if(pars_->write_fields) delete fieldsDiagnostic;
  if(pars_->write_fields && pars_->nonlinear_mode) delete fieldsXYDiagnostic;
  if(fields_old) delete fields_old;
  if(ncdf_) delete ncdf_;
  if(ncdf_big_) delete ncdf_big_;
}

bool Diagnostics_KREHM::loop(MomentsG** G, Fields* fields, double dt, int counter, double time) 
{
  bool stop = false;
  if(pars_->write_omega && (counter % pars_->nwrite == 0 || time + dt > pars_->t_max)) {
    fields_old->copyPhiFrom(fields);
  }

  if(counter % pars_->nwrite == 1 || time > pars_->t_max) {
    if(grids_->iproc == 0) printf("%s: Step %7d: Time = %10.5f  dt = %.3e   ", pars_->run_name, counter, time, dt);          // To screen
    for( auto & diagnostic : spectraDiagnosticList ) {
      diagnostic->calculate_and_write(G, fields, tmpG, tmpf);
    }

    if(pars_->write_omega) {
      growthRateDiagnostic->calculate_and_write(fields, fields_old, dt);
    }

    ncdf_->nc_grids->write_time(time);
    ncdf_->sync();

    if(grids_->iproc_m == 0) {
      printf("\n");
    }
    fflush(NULL);
  }

  // write out full grid (big) diagnostics less frequently
  if((counter % pars_->nwrite_big == 1 || time > pars_->t_max) && ( pars_->write_moms || pars_->write_fields) ) {
    if(pars_->write_fields) {
      fieldsDiagnostic->calculate_and_write(fields);
      if(pars_->nonlinear_mode) {
        fieldsXYDiagnostic->calculate_and_write(fields);
      }
    }

    for( auto & diagnostic : momentsDiagnosticList ) {
      diagnostic->calculate_and_write(G, fields, tmpC);
    }

    ncdf_big_->nc_grids->write_time(time);
    ncdf_big_->sync();
  }

  // check to see if we should stop simulation
  stop = checkstop();
  return stop;
}

bool Diagnostics::checkstop() 
{
  struct stat buffer;   
  bool stop = (stat (stopfilename_, &buffer) == 0);
  if (stop) remove(stopfilename_);
  return stop;
}

void Diagnostics::print_growth_rates_to_screen(cuComplex* w)
{
  int Nx = grids_->Nx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;

  printf("ky\tkx\t\tomega\t\tgamma\n");

  for(int j=0; j<Naky; j++) {
    for(int i= 1 + 2*Nx/3; i<Nx; i++) {
      int index = j + Nyc*i;
      printf("%.4f\t%.4f\t\t%.6f\t%.6f",  grids_->ky_h[j], grids_->kx_h[i], w[index].x, w[index].y);
      printf("\n");
    }
    for(int i=0; i < 1 + (Nx-1)/3; i++) {
      int index = j + Nyc*i;
      if(index!=0) {
	printf("%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_h[i], w[index].x, w[index].y);
	printf("\n");
      } else {
	printf("%.4f\t%.4f\n", grids_->ky_h[j], grids_->kx_h[i]);
      }
    }
    if (Nx>1) printf("\n");
  }
}

void Diagnostics::restart_write(MomentsG** G, double *time)
{
  char strb[512];
  int retval;
  int ncres;
  strcpy(strb, pars_->restart_to_file.c_str());
  if (retval = nc_create_par(strb, NC_CLOBBER | NC_NETCDF4, pars_->mpcom, MPI_INFO_NULL, &ncres)) ERR(retval);
  
  int moments_out[7];
  
  int Nspecies_glob = grids_->Nspecies_glob;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nz   = grids_->Nz;
  int Nm_glob = grids_->Nm_glob;
  int Nl   = grids_->Nl;

  // handles
  int id_ri, id_nz, id_Nkx, id_Nky;
  int id_nh, id_nl, id_sp;
  int id_G, id_time;
  int ri = 2;

  if (retval = nc_def_dim(ncres, "Nspecies",  Nspecies_glob,    &id_sp)) ERR(retval);
  if (retval = nc_def_dim(ncres, "ri",  ri,    &id_ri)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nz",  Nz,    &id_nz)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nkx", Nakx,  &id_Nkx)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nky", Naky,  &id_Nky)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nl",  Nl,    &id_nl)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nm",  Nm_glob,    &id_nh)) ERR(retval);

  moments_out[0] = id_sp;
  moments_out[1] = id_nh; 
  moments_out[2] = id_nl; 
  moments_out[3] = id_nz;  
  moments_out[4] = id_Nkx;
  moments_out[5] = id_Nky;
  moments_out[6] = id_ri; 

  if (retval = nc_def_var(ncres, "G",    NC_FLOAT, 7, moments_out, &id_G)) ERR(retval);
  if (retval = nc_def_var(ncres, "time", NC_DOUBLE, 0, 0, &id_time)) ERR(retval);
  if (retval = nc_enddef(ncres)) ERR(retval);

  // write time
  if (retval = nc_put_var(ncres, id_time, time)) ERR(retval);

  // write moments
  for(int is=0; is<grids_->Nspecies; is++) {
    G[is]->restart_write(ncres, id_G);
  }

  if (retval = nc_close(ncres)) ERR(retval);
}
