#include "diagnostic_classes.h"

// base class methods
SpectraDiagnostic::SpectraDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf)
{
  nc_type = NC_FLOAT;
  pars_ = pars;
  grids_ = grids;
  geo_ = geo;
  ncdf_ = ncdf;
  nc_group = ncdf_->nc_diagnostics->diagnostics_id;
}

// add a particular type of spectra to the calculation list
void SpectraDiagnostic::add_spectra(SpectraCalc *spectra)
{
  spectraList.push_back(spectra);
  int varid = spectra->define_nc_variable(varname, nc_group, description, pars_->restart && pars_->append_on_restart);
  spectraIds.push_back(varid);
}

// write all spectra
void SpectraDiagnostic::write_spectra(float* data)
{
  for(size_t i = 0; i < spectraList.size(); i++) {
    spectraList[i]->write(data, spectraIds[i], ncdf_->nc_grids->time_index, nc_group, isMoments, skipWrite);
  }
}

// set kernel launch dimensions for diagnostic calculation kernels
void SpectraDiagnostic::set_kernel_dims()
{
  if(isMoments) {
    int nyx =  grids_->Nyc * grids_->Nx;
    int nlm = grids_->Nmoms;

    int nt1 = 16;
    int nb1 = 1 + (nyx-1)/nt1;

    int nt2 = 16;
    int nb2 = 1 + (grids_->Nz-1)/nt2;
    
    dB = dim3(nt1, nt2, 1);
    dG = dim3(nb1, nb2, nlm);
  } else {
    dB = dim3(min(8, grids_->Nyc), min(8, grids_->Nx), min(8, grids_->Nz));
    dG = dim3(1 + (grids_->Nyc-1)/dB.x, 1 + (grids_->Nx-1)/dB.y, 1 + (grids_->Nz-1)/dB.z);  
  }
}

// |Phi|**2 diagnostic class
Phi2Diagnostic::Phi2Diagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "Phi2";
  isMoments = false;
  set_kernel_dims();

  add_spectra(allSpectra->t_spectra);
  add_spectra(allSpectra->kxt_spectra);
  add_spectra(allSpectra->kyt_spectra);
  add_spectra(allSpectra->kxkyt_spectra);
  add_spectra(allSpectra->zt_spectra);
  //add_spectra(allSpectra->kxkyzt_spectra);
}

void Phi2Diagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  // compute |Phi|**2(ky, kx, z, t)
  Phi2_summand <<<dG, dB>>> (tmpf, f->phi, geo_->vol_fac); 	
  // compute and write spectra of |Phi|**2
  write_spectra(tmpf);

  // get Phi**2(t) data to write to screen
  float *phi2 = spectraList[0]->get_data();

  if(grids_->iproc==0) {
    printf ("Phi**2 = %.3e   ", phi2[0]);
  }
}

// |Phi(ky=0)|**2 diagnostic class
Phi2ZonalDiagnostic::Phi2ZonalDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "Phi2_zonal";
  isMoments = false;
  set_kernel_dims();

  add_spectra(allSpectra->t_spectra);
  add_spectra(allSpectra->kxt_spectra);
  add_spectra(allSpectra->zt_spectra);
}

void Phi2ZonalDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  // compute |Phi|**2(ky, kx, z, t)
  Phi2_zonal_summand <<<dG, dB>>> (tmpf, f->phi, geo_->vol_fac); 	
  // compute and write spectra of |Phi|**2
  write_spectra(tmpf);
}

Apar2Diagnostic::Apar2Diagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "Apar2";
  isMoments = false;
  set_kernel_dims();

  add_spectra(allSpectra->t_spectra);
  add_spectra(allSpectra->kxt_spectra);
  add_spectra(allSpectra->kyt_spectra);
  add_spectra(allSpectra->kxkyt_spectra);
  add_spectra(allSpectra->zt_spectra);
}

void Apar2Diagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  // compute |Apar|**2(ky, kx, z, t)
  Phi2_summand <<<dG, dB>>> (tmpf, f->apar, geo_->vol_fac); 	
  // compute and write spectra of |Apar|**2
  write_spectra(tmpf);
}

WphiDiagnostic::WphiDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "Wphi";
  isMoments = false;
  set_kernel_dims();

  add_spectra(allSpectra->st_spectra);
  add_spectra(allSpectra->kxst_spectra);
  add_spectra(allSpectra->kyst_spectra);
  add_spectra(allSpectra->kxkyst_spectra);
  add_spectra(allSpectra->zst_spectra);
}

void WphiDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is + grids_->is_lo;
    float rho2s = pars_->species_h[is_glob].rho2;
    Wphi_summand <<<dG, dB>>> (&tmpf[grids_->NxNycNz*is], f->phi, geo_->vol_fac, geo_->kperp2, rho2s); 	
  }
  write_spectra(tmpf);
}

WaparDiagnostic::WaparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "Wapar";
  isMoments = false;
  set_kernel_dims();

  add_spectra(allSpectra->st_spectra);
  add_spectra(allSpectra->kxst_spectra);
  add_spectra(allSpectra->kyst_spectra);
  add_spectra(allSpectra->kxkyst_spectra);
  add_spectra(allSpectra->zst_spectra);
}

void WaparDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    Wapar_summand <<<dG, dB>>> (&tmpf[grids_->NxNycNz*is], f->apar, geo_->vol_fac, geo_->kperp2, geo_->bmag);
  }
  write_spectra(tmpf);
}

WgDiagnostic::WgDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "Wg";
  isMoments = true;
  set_kernel_dims();

  add_spectra(allSpectra->st_spectra);
  add_spectra(allSpectra->kxst_spectra);
  add_spectra(allSpectra->kyst_spectra);
  add_spectra(allSpectra->kxkyst_spectra);
  add_spectra(allSpectra->zst_spectra);
  add_spectra(allSpectra->lmst_spectra);
}

void WgDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is + grids_->is_lo;
    float nt = pars_->species_h[is_glob].nt;
    Wg_summand <<<dG, dB>>> (&tmpG[grids_->NxNycNz*grids_->Nmoms*is], G[is]->G(), geo_->vol_fac, nt);
  }
  write_spectra(tmpG);
}

// KREHM electrostatic energy
WphiKrehmDiagnostic::WphiKrehmDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "Wphi";
  isMoments = false;
  set_kernel_dims();

  add_spectra(allSpectra->t_spectra);
  add_spectra(allSpectra->kxt_spectra);
  add_spectra(allSpectra->kyt_spectra);
  add_spectra(allSpectra->kxkyt_spectra);
  add_spectra(allSpectra->zt_spectra);
}

void WphiKrehmDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  Wphi_summand_krehm <<<dG, dB>>> (tmpf, f->phi, geo_->vol_fac, grids_->kx, grids_->ky, pars_->rho_i); 	
  write_spectra(tmpf);
}

WaparKrehmDiagnostic::WaparKrehmDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "Wapar";
  isMoments = false;
  set_kernel_dims();

  add_spectra(allSpectra->t_spectra);
  add_spectra(allSpectra->kxt_spectra);
  add_spectra(allSpectra->kyt_spectra);
  add_spectra(allSpectra->kxkyt_spectra);
  add_spectra(allSpectra->zt_spectra);
}

void WaparKrehmDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  Wapar_summand_krehm <<<dG, dB>>> (tmpf, f->apar, f->apar_ext, geo_->vol_fac, grids_->kx, grids_->ky, pars_->rho_i); 	
  write_spectra(tmpf);
}

HeatFluxDiagnostic::HeatFluxDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "HeatFlux";
  description = "Turbulent heat flux in gyroBohm units"; 
  isMoments = false;
  if(grids_->m_lo>0) skipWrite = true; // procs with higher hermites will have nonsense 
                                       // heat flux data, so skip the write from these procs
  set_kernel_dims();

  add_spectra(allSpectra->st_spectra);
  add_spectra(allSpectra->kxst_spectra);
  add_spectra(allSpectra->kyst_spectra);
  add_spectra(allSpectra->kxkyst_spectra);
  add_spectra(allSpectra->zst_spectra);
  //add_spectra(allSpectra->kxkyzst_spectra);
}

void HeatFluxDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is + grids_->is_lo;
    float rho2s = pars_->species_h[is_glob].rho2;
    float p_s = pars_->species_h[is_glob].nt;
    float vts = pars_->species_h[is_glob].vt;
    float tzs = pars_->species_h[is_glob].tz;
    if(grids_->Nm <= 2) {
      G[is]->sync(true);
    }
    heat_flux_summand <<<dG, dB>>> (&tmpf[grids_->NxNycNz*is], f->phi, f->apar, f->bpar, G[is]->G(), grids_->ky,  geo_->flux_fac, geo_->kperp2, rho2s, p_s, vts, tzs); 	
  }
  write_spectra(tmpf);

  // get Q(t) data to write to screen
  float *fluxes = spectraList[0]->get_data();

  if(!skipWrite) {
    for (int is=0; is<grids_->Nspecies; is++) {
      int is_glob = is + grids_->is_lo;
      const char *spec_string = pars_->species_h[is_glob].type == 1 ? "e" : "i";
      printf ("Q_%s = %.3e   ", spec_string, fluxes[is]);
    }
  }
}

HeatFluxESDiagnostic::HeatFluxESDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "HeatFluxES";
  description = "Electrostatic component of turbulent heat flux in gyroBohm units"; 
  isMoments = false;
  if(grids_->m_lo>0) skipWrite = true; // procs with higher hermites will have nonsense 
                                       // heat flux data, so skip the write from these procs
  set_kernel_dims();

  add_spectra(allSpectra->st_spectra);
  add_spectra(allSpectra->kxst_spectra);
  add_spectra(allSpectra->kyst_spectra);
  add_spectra(allSpectra->kxkyst_spectra);
  add_spectra(allSpectra->zst_spectra);
}

void HeatFluxESDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is + grids_->is_lo;
    float rho2s = pars_->species_h[is_glob].rho2;
    float p_s = pars_->species_h[is_glob].nt;
    float vts = pars_->species_h[is_glob].vt;
    heat_flux_ES_summand <<<dG, dB>>> (&tmpf[grids_->NxNycNz*is], f->phi, G[is]->G(), grids_->ky,  geo_->flux_fac, geo_->kperp2, rho2s, p_s, vts); 	
  }
  write_spectra(tmpf);
}

HeatFluxAparDiagnostic::HeatFluxAparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "HeatFluxApar";
  description = "Electromagnetic (A_parallel) component of turbulent heat flux in gyroBohm units"; 
  isMoments = false;
  if(grids_->m_lo>0) skipWrite = true; // procs with higher hermites will have nonsense 
                                       // heat flux data, so skip the write from these procs
  set_kernel_dims();

  add_spectra(allSpectra->st_spectra);
  add_spectra(allSpectra->kxst_spectra);
  add_spectra(allSpectra->kyst_spectra);
  add_spectra(allSpectra->kxkyst_spectra);
  add_spectra(allSpectra->zst_spectra);
}

void HeatFluxAparDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is + grids_->is_lo;
    float rho2s = pars_->species_h[is_glob].rho2;
    float p_s = pars_->species_h[is_glob].nt;
    float vts = pars_->species_h[is_glob].vt;
    heat_flux_Apar_summand <<<dG, dB>>> (&tmpf[grids_->NxNycNz*is], f->apar, G[is]->G(), grids_->ky,  geo_->flux_fac, geo_->kperp2, rho2s, p_s, vts); 	
  }
  write_spectra(tmpf);
}

HeatFluxBparDiagnostic::HeatFluxBparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "HeatFluxBpar";
  description = "Electromagnetic (dB_parallel) component of turbulent heat flux in gyroBohm units"; 
  isMoments = false;
  if(grids_->m_lo>0) skipWrite = true; // procs with higher hermites will have nonsense 
                                       // heat flux data, so skip the write from these procs
  set_kernel_dims();

  add_spectra(allSpectra->st_spectra);
  add_spectra(allSpectra->kxst_spectra);
  add_spectra(allSpectra->kyst_spectra);
  add_spectra(allSpectra->kxkyst_spectra);
  add_spectra(allSpectra->zst_spectra);
}

void HeatFluxBparDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is + grids_->is_lo;
    float rho2s = pars_->species_h[is_glob].rho2;
    float p_s = pars_->species_h[is_glob].nt;
    float tzs = pars_->species_h[is_glob].tz;
    heat_flux_Bpar_summand <<<dG, dB>>> (&tmpf[grids_->NxNycNz*is], f->bpar, G[is]->G(), grids_->ky,  geo_->flux_fac, geo_->kperp2, rho2s, p_s, tzs); 	
  }
  write_spectra(tmpf);
}

ParticleFluxDiagnostic::ParticleFluxDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "ParticleFlux";
  description = "Turbulent particle flux in gyroBohm units"; 
  isMoments = false;
  if(grids_->m_lo>0) skipWrite = true; // procs with higher hermites will have nonsense 
                                       // particle flux data, so skip the write from these procs
  set_kernel_dims();

  add_spectra(allSpectra->st_spectra);
  add_spectra(allSpectra->kxst_spectra);
  add_spectra(allSpectra->kyst_spectra);
  add_spectra(allSpectra->kxkyst_spectra);
  add_spectra(allSpectra->zst_spectra);
}

void ParticleFluxDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is + grids_->is_lo;
    float rho2s = pars_->species_h[is_glob].rho2;
    float n_s = pars_->nspec>1 ? pars_->species_h[is_glob].dens : 0.;
    float vts = pars_->species_h[is_glob].vt;
    float tzs = pars_->species_h[is_glob].tz;
    particle_flux_summand <<<dG, dB>>> (&tmpf[grids_->NxNycNz*is], f->phi, f->apar, f->bpar, G[is]->G(), grids_->ky,  geo_->flux_fac, geo_->kperp2, rho2s, n_s, vts, tzs); 	
  }
  write_spectra(tmpf);

  // get Gam(t) data to write to screen
  float *fluxes = spectraList[0]->get_data();

  if(!skipWrite) {
    for (int is=0; is<grids_->Nspecies; is++) {
      int is_glob = is + grids_->is_lo;
      const char *spec_string = pars_->species_h[is_glob].type == 1 ? "e" : "i";
      printf ("Gam_%s = %.3e   ", spec_string, fluxes[is]);
    }
  }
}

TurbulentHeatingDiagnostic::TurbulentHeatingDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Linear* linear, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "TurbulentHeating";
  description = "Turbulent heating from collisions in gyroBohm units"; 
  isMoments = false;
  if(grids_->m_lo>0) skipWrite = true; // procs with higher hermites will have nonsense 
                                       // heating data, so skip the write from these procs
  set_kernel_dims();

  add_spectra(allSpectra->st_spectra);
  add_spectra(allSpectra->kxst_spectra);
  add_spectra(allSpectra->kyst_spectra);
  add_spectra(allSpectra->kxkyst_spectra);
  add_spectra(allSpectra->zst_spectra);

  linear_ = linear;
}

void TurbulentHeatingDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    turbulent_heating_summand <<<dG, dB>>> (&tmpf[grids_->NxNycNz*is], f->phi, f->apar, f->bpar, 
                                            f_old_->phi, f_old_->apar, f_old_->bpar, 
                                            G[is]->G(), G_old_[is]->G(), geo_->vol_fac, geo_->kperp2, *(G[is]->species), dt_);
  }
  write_spectra(tmpf);

  // get Heat(t) data to write to screen
  float *heat = spectraList[0]->get_data();

  if(!skipWrite) {
    for (int is=0; is<grids_->Nspecies; is++) {
      int is_glob = is + grids_->is_lo;
      const char *spec_string = pars_->species_h[is_glob].type == 1 ? "e" : "i";
      printf ("Heat_%s = %.3e   ", spec_string, heat[is]);
    }
  }
}

void TurbulentHeatingDiagnostic::set_dt_data(MomentsG** G_old, Fields* f_old, float dt) {
  G_old_ = G_old;
  f_old_ = f_old;
  dt_ = dt;
}

GrowthRateDiagnostic::GrowthRateDiagnostic(Parameters* pars, Grids* grids, NetCDF* ncdf)
{
  nc_type = NC_FLOAT;
  pars_ = pars;
  grids_ = grids;
  ncdf_ = ncdf;
  varname = "omega_kxkyt";
  nc_group = ncdf_->nc_diagnostics->diagnostics_id;
  ndim = 4;

  dims[0] = ncdf_->nc_dims->time;
  dims[1] = ncdf_->nc_dims->ky;
  dims[2] = ncdf_->nc_dims->kx;
  dims[3] = ncdf_->nc_dims->ri;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Naky;
  count[2] = grids->Nakx;
  count[3] = 2;

  N = grids->NxNyc;
  Nwrite = grids->Nakx*grids->Naky*2;

  int retval;
  if (pars_->restart && pars_->append_on_restart) {
    if (retval = nc_inq_varid(nc_group, varname.c_str(), &varid)) ERR(retval);
    if (retval = nc_var_par_access(nc_group, varid, NC_COLLECTIVE)) ERR(retval);
  } else {
    if (retval = nc_def_var(nc_group, varname.c_str(), nc_type, ndim, dims, &varid)) ERR(retval);
    if (retval = nc_var_par_access(nc_group, varid, NC_COLLECTIVE)) ERR(retval);
  }

  cudaMalloc (&omg_d, sizeof(cuComplex) * N);
  omg_h = (cuComplex*) malloc  (sizeof(cuComplex) * N);
  cpu = (float*) malloc  (sizeof(float) * Nwrite);
}

GrowthRateDiagnostic::~GrowthRateDiagnostic()
{
  cudaFree(omg_d);
  free(omg_h);
  free(cpu);
}

// need separate calculate and write methods for growth rates, 
// so that can calculate every step but write less often
void GrowthRateDiagnostic::calculate_and_write(Fields* fields, Fields* fields_old, double dt)
{
  int nt = min(512, grids_->NxNyc) ;
  growthRates <<< 1 + (grids_->NxNyc-1)/nt, nt >>> (fields->phi, fields_old->phi, dt, omg_d);

  // write to ncdf
  CP_TO_CPU(omg_h, omg_d, sizeof(cuComplex)*N);
  dealias_and_reorder(omg_h, cpu);
  
  int retval;
  start[0] = ncdf_->nc_grids->time_index;
  if (retval=nc_put_vara(nc_group, varid, start, count, cpu)) ERR(retval);

  if( pars_->iproc == 0) {
	  // print to screen
	  int Nx = grids_->Nx;
	  int Naky = grids_->Naky;
	  int Nyc  = grids_->Nyc;

	  printf("\nky\tkx\t\tomega\t\tgamma\n");

	  for(int j=0; j<Naky; j++) {
		  for(int i= 1 + 2*Nx/3; i<Nx; i++) {
			  int index = j + Nyc*i;
			  printf("%.4f\t%.4f\t\t%.6f\t%.6f",  grids_->ky_h[j], grids_->kx_h[i], omg_h[index].x, omg_h[index].y);
			  printf("\n");
		  }
		  for(int i=0; i < 1 + (Nx-1)/3; i++) {
			  int index = j + Nyc*i;
			  if(index!=0) {
				  printf("%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_h[i], omg_h[index].x, omg_h[index].y);
				  printf("\n");
			  } else {
				  printf("%.4f\t%.4f\n", grids_->ky_h[j], grids_->kx_h[i]);
			  }
		  }
		  if (Nx>1) printf("\n");
	  }
  }

}

void GrowthRateDiagnostic::dealias_and_reorder(cuComplex* fold, float* fnew)
{
  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;

  int NK = grids_->Nakx/2;
 
  int it = 0;
  int itp = it + NK;
  for (int ik=0; ik<Naky; ik++) {
    int Qp = itp + ik*Nakx;
    int Rp = ik  + it*Nyc;
    fnew[2*Qp  ] = fold[Rp].x;
    fnew[2*Qp+1] = fold[Rp].y;
  }

  for (int it = 1; it < NK+1; it++) {
    int itp = NK + it;
    int itn = NK - it;
    int itm = Nx - it;
    for (int ik=0; ik<Naky; ik++) {
      int Qp = itp + ik*Nakx;
      int Rp = ik  + it*Nyc;

      int Qn = itn + ik*Nakx;
      int Rm = ik  + itm*Nyc;
      fnew[2*Qp  ] = fold[Rp].x;
      fnew[2*Qp+1] = fold[Rp].y;

      fnew[2*Qn  ] = fold[Rm].x;
      fnew[2*Qn+1] = fold[Rm].y;
    }
  }
}

FieldsDiagnostic::FieldsDiagnostic(Parameters* pars, Grids* grids, NetCDF* ncdf)
{
  nc_type = NC_FLOAT;
  pars_ = pars;
  grids_ = grids;
  ncdf_ = ncdf;
  varnames[0] = "Phi";
  varnames[1] = "Apar";
  varnames[2] = "Bpar";
  nc_group = ncdf_->nc_diagnostics->diagnostics_id;
  ndim = 5;

  dims[0] = ncdf_->nc_dims->time;
  dims[1] = ncdf_->nc_dims->ky;
  dims[2] = ncdf_->nc_dims->kx;
  dims[3] = ncdf_->nc_dims->z;
  dims[4] = ncdf_->nc_dims->ri;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Naky;
  count[2] = grids->Nakx;
  count[3] = grids->Nz;
  count[4] = 2;

  N = grids->NxNycNz;
  Nwrite = grids->Nakx*grids->Naky*grids->Nz*2;

  int retval;
  for(int i=0; i<3; i++) {
    if (pars_->restart && pars_->append_on_restart) {
      if (retval = nc_inq_varid(nc_group, varnames[i].c_str(), &varids[i])) ERR(retval);
      if (retval = nc_var_par_access(nc_group, varids[i], NC_COLLECTIVE)) ERR(retval);
    } else {
      if (retval = nc_def_var(nc_group, varnames[i].c_str(), nc_type, ndim, dims, &varids[i])) ERR(retval);
      if (retval = nc_var_par_access(nc_group, varids[i], NC_COLLECTIVE)) ERR(retval);
    }
  }

  f_h = (cuComplex*) malloc  (sizeof(cuComplex) * N);
  cpu = (float*) malloc  (sizeof(float) * Nwrite);
}

FieldsDiagnostic::~FieldsDiagnostic() 
{
  free(f_h);
  free(cpu);
}

void FieldsDiagnostic::calculate_and_write(Fields* f)
{
  int retval;
  start[0] = ncdf_->nc_grids->time_index;

  // write phi to ncdf
  CP_TO_CPU(f_h, f->phi, sizeof(cuComplex)*N);
  dealias_and_reorder(f_h, cpu);
  if (retval=nc_put_vara(nc_group, varids[0], start, count, cpu)) ERR(retval);

  // write apar to ncdf
  CP_TO_CPU(f_h, f->apar, sizeof(cuComplex)*N);
  dealias_and_reorder(f_h, cpu);
  if (retval=nc_put_vara(nc_group, varids[1], start, count, cpu)) ERR(retval);
  
  // write bpar to ncdf
  CP_TO_CPU(f_h, f->bpar, sizeof(cuComplex)*N);
  dealias_and_reorder(f_h, cpu);
  if (retval=nc_put_vara(nc_group, varids[2], start, count, cpu)) ERR(retval);
}

// condense a (ky,kx,z) object for netcdf output, taking into account the mask
// and changing the type from cuComplex to float
// and transposing to put z as fastest index
void FieldsDiagnostic::dealias_and_reorder(cuComplex *f, float *fk)
{
  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;
  int Nz   = grids_->Nz;
 
  int NK = grids_->Nakx/2;
 
  int it = 0;
  int itp = it + NK;
  for (int ik=0; ik<Naky; ik++) {
    int Qp = itp + ik*Nakx;
    int Rp = ik  + it*Nyc;
    for (int k=0; k<Nz; k++) {
      int ig = Rp + Nx*Nyc*k;
      int ir = 0 + 2*(k + Nz*Qp);
      int ii = 1 + 2*(k + Nz*Qp);
      fk[ir] = f[ig].x;
      fk[ii] = f[ig].y;
    }
  }

  for (int it = 1; it < NK+1; it++) {
    int itp = NK + it;
    int itn = NK - it;
    int itm = Nx - it;
    for (int ik=0; ik<Naky; ik++) {
      int Qp = itp + ik*Nakx;
      int Rp = ik  + it*Nyc;

      int Qn = itn + ik*Nakx;
      int Rm = ik  + itm*Nyc;
      for (int k=0; k<Nz; k++) {
        int ip = Rp + Nx*Nyc*k;
        int im = Rm + Nx*Nyc*k;

        int irp = 0 + 2*(k + Nz*Qp);
        int iip = 1 + 2*(k + Nz*Qp);

        int irn = 0 + 2*(k + Nz*Qn);
        int iin = 1 + 2*(k + Nz*Qn);

        fk[irp] = f[ip].x;
        fk[iip] = f[ip].y;

        fk[irn] = f[im].x;
        fk[iin] = f[im].y;
      }
    }
  } 
}

// fields transformed to real (x,y,z) space
FieldsXYDiagnostic::FieldsXYDiagnostic(Parameters* pars, Grids* grids, Nonlinear* nonlinear, NetCDF* ncdf)
{
  nc_type = NC_FLOAT;
  pars_ = pars;
  grids_ = grids;
  nonlinear_ = nonlinear;
  ncdf_ = ncdf;
  varnames[0] = "PhiXY";
  varnames[1] = "AparXY";
  varnames[2] = "BparXY";
  nc_group = ncdf_->nc_diagnostics->diagnostics_id;
  ndim = 4;

  dims[0] = ncdf_->nc_dims->time;
  dims[1] = ncdf_->nc_dims->y;
  dims[2] = ncdf_->nc_dims->x;
  dims[3] = ncdf_->nc_dims->z;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Ny;
  count[2] = grids->Nx;
  count[3] = grids->Nz;
   
  int retval;
  for(int i=0; i<3; i++) {
    if (pars_->restart && pars_->append_on_restart) {
      if (retval = nc_inq_varid(nc_group, varnames[i].c_str(), &varids[i])) ERR(retval);
      if (retval = nc_var_par_access(nc_group, varids[i], NC_COLLECTIVE)) ERR(retval);
    } else {
      if (retval = nc_def_var(nc_group, varnames[i].c_str(), nc_type, ndim, dims, &varids[i])) ERR(retval);
      if (retval = nc_var_par_access(nc_group, varids[i], NC_COLLECTIVE)) ERR(retval);
    }
  }

  N = grids->NxNyNz;
  f_h = (float*) malloc  (sizeof(float) * N);
  cpu = (float*) malloc  (sizeof(float) * N);

  fXY = nonlinear_->get_fXY();
  grad_perp_ = nonlinear_->get_grad_perp_f();
}

FieldsXYDiagnostic::~FieldsXYDiagnostic() 
{
  free(f_h);
  free(cpu);
}

void FieldsXYDiagnostic::calculate_and_write(Fields* f)
{
  int retval;
  start[0] = ncdf_->nc_grids->time_index;

  // write phi to ncdf
  grad_perp_->C2R(f->phi, fXY);
  CP_TO_CPU(f_h, fXY, sizeof(float)*N);
  dealias_and_reorder(f_h, cpu);
  if (retval=nc_put_vara(nc_group, varids[0], start, count, cpu)) ERR(retval);

  // write apar to ncdf
  grad_perp_->C2R(f->apar, fXY);
  CP_TO_CPU(f_h, fXY, sizeof(float)*N);
  dealias_and_reorder(f_h, cpu);
  if (retval=nc_put_vara(nc_group, varids[1], start, count, cpu)) ERR(retval);
  
  // write bpar to ncdf
  grad_perp_->C2R(f->bpar, fXY);
  CP_TO_CPU(f_h, fXY, sizeof(float)*N);
  dealias_and_reorder(f_h, cpu);
  if (retval=nc_put_vara(nc_group, varids[2], start, count, cpu)) ERR(retval);
}

// transpose so that z is fastest index
void FieldsXYDiagnostic::dealias_and_reorder(float *f, float *fr)
{
  int Nx   = grids_->Nx;
  int Ny   = grids_->Ny;
  int Nz   = grids_->Nz;

  for (int iy=0; iy<Ny; iy++) {
    for (int ix=0; ix<Nx; ix++) {
      for (int iz=0; iz<Nz; iz++) {
        int ig = iy + Ny*ix + Nx*Ny*iz;
        int iwrite = iz + ix*Nz + iy*Nx*Nz;
        fr[iwrite] = f[ig];
      }
    }
  }
}

// similar structure to FieldsDiagnostic, but with a species index
MomentsDiagnostic::MomentsDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf, string varname)
{
  nc_type = NC_FLOAT;
  pars_ = pars;
  grids_ = grids;
  geo_ = geo;
  ncdf_ = ncdf;
  nc_group = ncdf_->nc_diagnostics->diagnostics_id;
  ndim = 6;

  dims[0] = ncdf_->nc_dims->time;
  dims[1] = ncdf_->nc_dims->species;
  dims[2] = ncdf_->nc_dims->ky;
  dims[3] = ncdf_->nc_dims->kx;
  dims[4] = ncdf_->nc_dims->z;
  dims[5] = ncdf_->nc_dims->ri;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nspecies;
  count[2] = grids->Naky;
  count[3] = grids->Nakx;
  count[4] = grids->Nz;
  count[5] = 2;

  start[1] = grids->is_lo;

  N = grids->NxNycNz*grids->Nspecies;
  Nwrite = grids->Nakx*grids->Naky*grids->Nz*grids->Nspecies*2;

  int retval;
  if (pars_->restart && pars_->append_on_restart) {
    if (retval = nc_inq_varid(nc_group, varname.c_str(), &varid)) ERR(retval);
    if (retval = nc_var_par_access(nc_group, varid, NC_COLLECTIVE)) ERR(retval);
  } else {
    if (retval = nc_def_var(nc_group, varname.c_str(), nc_type, ndim, dims, &varid)) ERR(retval);
    if (retval = nc_var_par_access(nc_group, varid, NC_COLLECTIVE)) ERR(retval);
  }

  f_h = (cuComplex*) malloc  (sizeof(cuComplex) * N);
  cpu = (float*) malloc  (sizeof(float) * Nwrite);

  skipWrite = false;

  int nn1, nn2, nn3, nt1, nt2, nt3, nb1, nb2, nb3;

  nn1 = grids_->Nyc;        nt1 = min(nn1, 32 );   nb1 = 1 + (nn1-1)/nt1;
  nn2 = grids_->Nx;         nt2 = min(nn2,  4 );   nb2 = 1 + (nn2-1)/nt2;
  nn3 = grids_->Nz;         nt3 = min(nn3,  4 );   nb3 = 1 + (nn3-1)/nt3;

  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3);

  // infrastructure for real space (x,y,z) diagnostic
  if (pars->nonlinear_mode) {
    nonlinear_ = nonlinear;
    string varnameXY = varname + "XY";
    ndimXY = 5;

    dimsXY[0] = ncdf_->nc_dims->time;
    dimsXY[1] = ncdf_->nc_dims->species;
    dimsXY[2] = ncdf_->nc_dims->y;
    dimsXY[3] = ncdf_->nc_dims->x;
    dimsXY[4] = ncdf_->nc_dims->z;

    countXY[0] = 1; // each write is a single time slice
    countXY[1] = grids->Nspecies;
    countXY[2] = grids->Ny;
    countXY[3] = grids->Nx;
    countXY[4] = grids->Nz;

    startXY[1] = grids->is_lo;
     
    NXY = grids->NxNyNz*grids->Nspecies;
    if (pars_->restart && pars_->append_on_restart) {
      if (retval = nc_inq_varid(nc_group, varnameXY.c_str(), &varidXY)) ERR(retval);
      if (retval = nc_var_par_access(nc_group, varidXY, NC_COLLECTIVE)) ERR(retval);
    } else {
      if (retval = nc_def_var(nc_group, varnameXY.c_str(), nc_type, ndimXY, dimsXY, &varidXY)) ERR(retval);
      if (retval = nc_var_par_access(nc_group, varidXY, NC_COLLECTIVE)) ERR(retval);
    }
    fXY_h = (float*) malloc  (sizeof(float) * NXY);
    cpuXY = (float*) malloc  (sizeof(float) * NXY);

    fXY = nonlinear_->get_fXY();
    grad_perp_ = nonlinear_->get_grad_perp_f();
  }
}

void MomentsDiagnostic::calculate_and_write(MomentsG** G, Fields* fields, cuComplex* tmp_d)
{
  int retval;
  if(!skipWrite) calculate(G, fields, f_h, fXY_h, tmp_d);

  start[0] = ncdf_->nc_grids->time_index;

  // write to ncdf
  dealias_and_reorder(f_h, cpu);

  if(skipWrite) { 
    // sometimes we need to skip the write on a particular (set of) proc(s), 
    // but all procs still need to call nc_put_vara. so do an empty dummy write
    if (retval=nc_put_vara(nc_group, varid, dummy_start, dummy_count, cpu)) ERR(retval);
  } else {
    if (retval=nc_put_vara(nc_group, varid, start, count, cpu)) ERR(retval);
  }

  // write XY to ncdf
  if (pars_->nonlinear_mode) {
    startXY[0] = ncdf_->nc_grids->time_index;
    dealias_and_reorder_XY(fXY_h, cpuXY);

    if(skipWrite) { 
      // sometimes we need to skip the write on a particular (set of) proc(s), 
      // but all procs still need to call nc_put_vara. so do an empty dummy write
      if (retval=nc_put_vara(nc_group, varidXY, dummy_startXY, dummy_countXY, cpuXY)) ERR(retval);
    } else {
      if (retval=nc_put_vara(nc_group, varidXY, startXY, countXY, cpuXY)) ERR(retval);
    }
  }
}

// condense a (ky,kx,z) object for netcdf output, taking into account the mask
// and changing the type from cuComplex to float
// and transposing to put z as fastest index
void MomentsDiagnostic::dealias_and_reorder(cuComplex *f, float *fk)
{
  int Nsp  = grids_->Nspecies;
  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;
  int Nz   = grids_->Nz;
 
  int NK = grids_->Nakx/2;
 
  for (int is = 0; is<Nsp; is++) {
    int it = 0;
    int itp = it + NK;
    for (int ik=0; ik<Naky; ik++) {
      int Qp = itp + ik*Nakx;
      int Rp = ik  + it*Nyc;
      for (int k=0; k<Nz; k++) {
        int ig = Rp + Nx*Nyc*k + Nx*Nyc*Nz*is;
        int ir = 0 + 2*(k + Nz*Qp) + 2*Nakx*Naky*Nz*is;
        int ii = 1 + 2*(k + Nz*Qp) + 2*Nakx*Naky*Nz*is;
        fk[ir] = f[ig].x;
        fk[ii] = f[ig].y;
      }
    }
  
    for (int it = 1; it < NK+1; it++) {
      int itp = NK + it;
      int itn = NK - it;
      int itm = Nx - it;
      for (int ik=0; ik<Naky; ik++) {
        int Qp = itp + ik*Nakx;
        int Rp = ik  + it*Nyc;
  
        int Qn = itn + ik*Nakx;
        int Rm = ik  + itm*Nyc;
        for (int k=0; k<Nz; k++) {
          int ip = Rp + Nx*Nyc*k + Nx*Nyc*Nz*is;
          int im = Rm + Nx*Nyc*k + Nx*Nyc*Nz*is;
  
          int irp = 0 + 2*(k + Nz*Qp) + 2*Nakx*Naky*Nz*is;
          int iip = 1 + 2*(k + Nz*Qp) + 2*Nakx*Naky*Nz*is;
  
          int irn = 0 + 2*(k + Nz*Qn) + 2*Nakx*Naky*Nz*is;
          int iin = 1 + 2*(k + Nz*Qn) + 2*Nakx*Naky*Nz*is;
  
          fk[irp] = f[ip].x;
          fk[iip] = f[ip].y;
  
          fk[irn] = f[im].x;
          fk[iin] = f[im].y;
        }
      }
    } 
  }
}

// transpose so that z is fastest index
void MomentsDiagnostic::dealias_and_reorder_XY(float *f, float *fr)
{
  int Nsp  = grids_->Nspecies;
  int Nx   = grids_->Nx;
  int Ny   = grids_->Ny;
  int Nz   = grids_->Nz;

  for (int is=0; is<Nsp; is++) {
    for (int iy=0; iy<Ny; iy++) {
      for (int ix=0; ix<Nx; ix++) {
        for (int iz=0; iz<Nz; iz++) {
          int ig = iy + Ny*ix + Nx*Ny*iz + Nx*Ny*Nz*is;
          int iwrite = iz + ix*Nz + iy*Nx*Nz + Nx*Ny*Nz*is;
          fr[iwrite] = f[ig];
        }
      }
    }
  }
}

DensityDiagnostic::DensityDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf)
 : MomentsDiagnostic(pars, grids, geo, nonlinear, ncdf, "Density")  // call base class constructor
{
  if(grids_->m_lo>0) skipWrite = true;
}

void DensityDiagnostic::calculate(MomentsG** G, Fields* fields, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    CP_TO_CPU(f_h + is*grids_->NxNycNz, G[is]->G(0,0), sizeof(cuComplex)*grids_->NxNycNz);

    if(pars_->nonlinear_mode) {
      grad_perp_->C2R(G[is]->G(0,0), fXY);
      CP_TO_CPU(fXY_h + is*grids_->NxNyNz, fXY, sizeof(float)*grids_->NxNyNz);
    }
  }
}

UparDiagnostic::UparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf)
 : MomentsDiagnostic(pars, grids, geo, nonlinear, ncdf, "Upar")  // call base class constructor
{
  if(grids_->m_lo>1 || grids_->m_up<=1) skipWrite = true;
}

void UparDiagnostic::calculate(MomentsG** G, Fields* fields, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    CP_TO_CPU(f_h + is*grids_->NxNycNz, G[is]->G(0,1 - grids_->m_lo), sizeof(cuComplex)*grids_->NxNycNz);

    if(pars_->nonlinear_mode) {
      grad_perp_->C2R(G[is]->G(0,1 - grids_->m_lo), fXY);
      CP_TO_CPU(fXY_h + is*grids_->NxNyNz, fXY, sizeof(float)*grids_->NxNyNz);
    }
  }
}

TparDiagnostic::TparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf)
 : MomentsDiagnostic(pars, grids, geo, nonlinear, ncdf, "Tpar")  // call base class constructor
{
  if(grids_->m_lo>2 || grids_->m_up<=2) skipWrite = true;
}

void TparDiagnostic::calculate(MomentsG** G, Fields* fields, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    scale_singlemom_kernel <<<grids_->NxNycNz/256+1, 256>>> (tmp_d, G[is]->G(0, 2-grids_->m_lo), sqrtf(2.));
    CP_TO_CPU(f_h + is*grids_->NxNycNz, tmp_d, sizeof(cuComplex)*grids_->NxNycNz);

    if(pars_->nonlinear_mode) {
      grad_perp_->C2R(tmp_d, fXY);
      CP_TO_CPU(fXY_h + is*grids_->NxNyNz, fXY, sizeof(float)*grids_->NxNyNz);
    }
  }
}

TperpDiagnostic::TperpDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf)
 : MomentsDiagnostic(pars, grids, geo, nonlinear, ncdf, "Tperp")  // call base class constructor
{
  if(grids_->m_lo>0) skipWrite = true;
}

void TperpDiagnostic::calculate(MomentsG** G, Fields* fields, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    CP_TO_CPU(f_h + is*grids_->NxNycNz, G[is]->G(1,0), sizeof(cuComplex)*grids_->NxNycNz);

    if(pars_->nonlinear_mode) {
      grad_perp_->C2R(G[is]->G(1,0), fXY);
      CP_TO_CPU(fXY_h + is*grids_->NxNyNz, fXY, sizeof(float)*grids_->NxNyNz);
    }
  }
}

ParticleDensityDiagnostic::ParticleDensityDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf)
 : MomentsDiagnostic(pars, grids, geo, nonlinear, ncdf, "ParticleDensity")  // call base class constructor
{
  if(grids_->m_lo>0) skipWrite = true;
}

void ParticleDensityDiagnostic::calculate(MomentsG** G, Fields* fields, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    calc_n_bar <<<dG, dB>>> (tmp_d, G[is]->G(), fields->phi, fields->bpar, geo_->kperp2, *G[is]->species);
    
    CP_TO_CPU(f_h + is*grids_->NxNycNz, tmp_d, sizeof(cuComplex)*grids_->NxNycNz);

    if(pars_->nonlinear_mode) {
      grad_perp_->C2R(tmp_d, fXY);
      CP_TO_CPU(fXY_h + is*grids_->NxNyNz, fXY, sizeof(float)*grids_->NxNyNz);
    }
  }
}

ParticleUparDiagnostic::ParticleUparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf)
 : MomentsDiagnostic(pars, grids, geo, nonlinear, ncdf, "ParticleUpar")  // call base class constructor
{
  if(grids_->m_lo>1 || grids_->m_up<=1) skipWrite = true;
}

void ParticleUparDiagnostic::calculate(MomentsG** G, Fields* fields, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    calc_upar_bar <<<dG, dB>>> (tmp_d, G[is]->G(), fields->apar, geo_->kperp2, *G[is]->species);
    
    CP_TO_CPU(f_h + is*grids_->NxNycNz, tmp_d, sizeof(cuComplex)*grids_->NxNycNz);

    if(pars_->nonlinear_mode) {
      grad_perp_->C2R(tmp_d, fXY);
      CP_TO_CPU(fXY_h + is*grids_->NxNyNz, fXY, sizeof(float)*grids_->NxNyNz);
    }
  }
}

ParticleUperpDiagnostic::ParticleUperpDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf)
 : MomentsDiagnostic(pars, grids, geo, nonlinear, ncdf, "ParticleUperp")  // call base class constructor
{
  if(grids_->m_lo>0) skipWrite = true;
}

void ParticleUperpDiagnostic::calculate(MomentsG** G, Fields* fields, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    calc_uperp_bar <<<dG, dB>>> (tmp_d, G[is]->G(), fields->phi, fields->bpar, geo_->kperp2, *G[is]->species);
    
    CP_TO_CPU(f_h + is*grids_->NxNycNz, tmp_d, sizeof(cuComplex)*grids_->NxNycNz);

    if(pars_->nonlinear_mode) {
      grad_perp_->C2R(tmp_d, fXY);
      CP_TO_CPU(fXY_h + is*grids_->NxNyNz, fXY, sizeof(float)*grids_->NxNyNz);
    }
  }
}

ParticleTempDiagnostic::ParticleTempDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, Nonlinear* nonlinear, NetCDF* ncdf)
 : MomentsDiagnostic(pars, grids, geo, nonlinear, ncdf, "ParticleTemp")  // call base class constructor
{
  if(grids_->m_lo>0) skipWrite = true;
}

void ParticleTempDiagnostic::calculate(MomentsG** G, Fields* fields, cuComplex* f_h, float* fXY_h, cuComplex* tmp_d)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    calc_T_bar <<<dG, dB>>> (tmp_d, G[is]->G(), fields->phi, fields->bpar, geo_->kperp2, *G[is]->species);
    
    CP_TO_CPU(f_h + is*grids_->NxNycNz, tmp_d, sizeof(cuComplex)*grids_->NxNycNz);

    if(pars_->nonlinear_mode) {
      grad_perp_->C2R(tmp_d, fXY);
      CP_TO_CPU(fXY_h + is*grids_->NxNyNz, fXY, sizeof(float)*grids_->NxNyNz);
    }
  }
}
