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

ParticleFluxESDiagnostic::ParticleFluxESDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "ParticleFluxES";
  description = "Electrostatic component of turbulent particle flux in gyroBohm units"; 
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

void ParticleFluxESDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is + grids_->is_lo;
    float rho2s = pars_->species_h[is_glob].rho2;
    float n_s = pars_->nspec>1 ? pars_->species_h[is_glob].dens : 0.;
    float vts = pars_->species_h[is_glob].vt;
    float tzs = pars_->species_h[is_glob].tz;
    particle_flux_ES_summand <<<dG, dB>>> (&tmpf[grids_->NxNycNz*is], f->phi, G[is]->G(), grids_->ky,  geo_->flux_fac, geo_->kperp2, rho2s, n_s, vts, tzs); 	
  }
  write_spectra(tmpf);
}

ParticleFluxAparDiagnostic::ParticleFluxAparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "ParticleFluxApar";
  description = "Electromagnetic (A_parallel) component of turbulent particle flux in gyroBohm units"; 
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

void ParticleFluxAparDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is + grids_->is_lo;
    float rho2s = pars_->species_h[is_glob].rho2;
    float n_s = pars_->nspec>1 ? pars_->species_h[is_glob].dens : 0.;
    float vts = pars_->species_h[is_glob].vt;
    float tzs = pars_->species_h[is_glob].tz;
    particle_flux_Apar_summand <<<dG, dB>>> (&tmpf[grids_->NxNycNz*is], f->apar, G[is]->G(), grids_->ky,  geo_->flux_fac, geo_->kperp2, rho2s, n_s, vts, tzs); 	
  }
  write_spectra(tmpf);
}

ParticleFluxBparDiagnostic::ParticleFluxBparDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
 : SpectraDiagnostic(pars, grids, geo, ncdf)
{
  varname = "ParticleFluxBpar";
  description = "Electromagnetic (dB_parallel) component of turbulent particle flux in gyroBohm units"; 
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

void ParticleFluxBparDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is + grids_->is_lo;
    float rho2s = pars_->species_h[is_glob].rho2;
    float n_s = pars_->nspec>1 ? pars_->species_h[is_glob].dens : 0.;
    float vts = pars_->species_h[is_glob].vt;
    float tzs = pars_->species_h[is_glob].tz;
    particle_flux_Bpar_summand <<<dG, dB>>> (&tmpf[grids_->NxNycNz*is], f->bpar, G[is]->G(), grids_->ky,  geo_->flux_fac, geo_->kperp2, rho2s, n_s, vts, tzs); 	
  }
  write_spectra(tmpf);
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
  if (pars_->restart && pars_->append_on_restart && nc_inq_varid(nc_group, varname.c_str(), &varid)==NC_NOERR) {
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

  // print to screen (but only on proc 0)
  if( grids_->iproc == 0 ) {
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
    if (pars_->restart && pars_->append_on_restart && nc_inq_varid(nc_group, varnames[i].c_str(), &varids[i])==NC_NOERR) {
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
// and flipping kx index so that kx ranges from [-kx_max, -kx_max+1, ..., 0, ..., kx_max -1, kx_max]
void FieldsDiagnostic::dealias_and_reorder(cuComplex *f, float *fk)
{
  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;
  int Nz   = grids_->Nz;
 
  int NK = grids_->Nakx/2;
  int nshift = Nx-Nakx;

  for (int iky=0; iky<Naky; iky++) {
    for (int ikx=0; ikx<Nakx; ikx++) {
      for (int iz=0; iz<Nz; iz++) {
        int ir = 0 + 2*iz + 2*Nz*ikx + 2*Nz*Nakx*iky;
        int ii = 1 + 2*iz + 2*Nz*ikx + 2*Nz*Nakx*iky;
        int idx = ikx;
	// this flips kx index so that -kx's are first, e.g. ikx = 0 corresponds to kx[idx] = -kx_max
        if (ikx < NK) idx = ikx + nshift + NK + 1;
        else idx = ikx - NK;
        int ig = iky + idx*Nyc + iz*Nx*Nyc;
        fk[ir] = f[ig].x;
        fk[ii] = f[ig].y;
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
    if (pars_->restart && pars_->append_on_restart && nc_inq_varid(nc_group, varnames[i].c_str(), &varids[i])==NC_NOERR ) {
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
  if (pars_->restart && pars_->append_on_restart && nc_inq_varid(nc_group, varname.c_str(), &varid)==NC_NOERR )  {
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
// and flipping kx index so that kx ranges from [-kx_max, -kx_max+1, ..., 0, ..., kx_max -1, kx_max]
void MomentsDiagnostic::dealias_and_reorder(cuComplex *f, float *fk)
{
  int Nsp  = grids_->Nspecies;
  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;
  int Nz   = grids_->Nz;
 
  int NK = grids_->Nakx/2;
  int nshift = Nx-Nakx;

  for (int is = 0; is<Nsp; is++) {
    for (int iky=0; iky<Naky; iky++) {
      for (int ikx=0; ikx<Nakx; ikx++) {
        for (int iz=0; iz<Nz; iz++) {
          int ir = 0 + 2*iz + 2*Nz*ikx + 2*Nz*Nakx*iky + 2*Nz*Nakx*Naky*is;
          int ii = 1 + 2*iz + 2*Nz*ikx + 2*Nz*Nakx*iky + 2*Nz*Nakx*Naky*is;
          int idx = ikx;
	  // this flips kx index so that -kx's are first, e.g. ikx = 0 corresponds to kx[idx] = -kx_max
          if (ikx < NK) idx = ikx + nshift + NK + 1;
          else idx = ikx - NK;
          int ig = iky + idx*Nyc + iz*Nx*Nyc + is*Nx*Nyc*Nz;
          fk[ir] = f[ig].x;
          fk[ii] = f[ig].y;
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

ZonalFlowEnergyTransferDiagnostic::ZonalFlowEnergyTransferDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, NetCDF* ncdf_big)
{
  nc_type = NC_FLOAT;
  pars_ = pars;
  grids_ = grids;
  geo_ = geo;
  ncdf_ = ncdf;
  ncdf_big_ = ncdf_big;
  varname = "ZonalEnergyTransfer";
  
  nc_group = ncdf_->nc_diagnostics->diagnostics_id;
  if (pars_->write_zonal_energy_transfer_full) {
    nc_group_big = ncdf_big_->nc_diagnostics->diagnostics_id;
  }
  
  ndim = 5;

  dims[0] = ncdf_->nc_dims->time;
  dims[1] = ncdf_->nc_dims->z;
  dims[2] = ncdf_->nc_dims->target_kx;
  dims[3] = ncdf_->nc_dims->source_kx;
  dims[4] = ncdf_->nc_dims->source_ky;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nz;
  count[2] = grids->Nakx;
  count[3] = grids->Nakx;
  count[4] = 2*(grids->Naky)-1; // extended ky array including negative values

  N = grids->Nakx * grids->Nakx * (2*grids->Naky-1) * grids->Nz;
  N_ext = grids_->Nakx * (2*grids_->Naky-1) * grids_->Nz; // for extended phi
  Nwrite = N;

  int retval;
  
  std::string description = "Nonlinear energy transfer to zonal flows";

  // Create NetCDF variables for reduced spectra in `.out.nc` file. The reduced
  // spectra are (z, t), (target_kx, t), (source_kx, t) and (source_ky, t)
  if (pars_->write_zonal_energy_transfer) {
    struct ReducedSpectrum {
      const char* name;
      int nc_dim;
      int count_val;
      int* dims;
      size_t* counts;
      int* varid;
    };
    
    ReducedSpectrum spectra[] = {
      {"z", ncdf_->nc_dims->z, grids->Nz, z_dims, z_count, &z_varid},
      {"target_kx", ncdf_->nc_dims->target_kx, grids->Nakx, target_kx_dims, target_kx_count, &target_kx_varid},
      {"source_kx", ncdf_->nc_dims->source_kx, grids->Nakx, source_kx_dims, source_kx_count, &source_kx_varid},
      {"source_ky", ncdf_->nc_dims->source_ky, 2*(grids->Naky)-1, source_ky_dims, source_ky_count, &source_ky_varid}
    };
    
    // Process all spectra using the same pattern
    for (int i = 0; i < 4; i++) {
      spectra[i].dims[0] = ncdf_->nc_dims->time;
      spectra[i].dims[1] = spectra[i].nc_dim;
      spectra[i].counts[0] = 1;
      spectra[i].counts[1] = spectra[i].count_val;
      
      std::string spec_varname = varname + "_" + spectra[i].name;
      
      // Get variable from restart variable or create new variable
      if (pars_->restart && pars_->append_on_restart) {
        bool exists = (nc_inq_varid(nc_group, spec_varname.c_str(), spectra[i].varid) == NC_NOERR);
        
        if (exists) {
          if (retval = nc_var_par_access(nc_group, *(spectra[i].varid), NC_COLLECTIVE)) ERR(retval);
        } else {
          if (retval = nc_def_var(nc_group, spec_varname.c_str(), nc_type, 2, spectra[i].dims, spectra[i].varid)) ERR(retval);
          if (retval = nc_var_par_access(nc_group, *(spectra[i].varid), NC_COLLECTIVE)) ERR(retval);
          if (retval = nc_put_att_text(nc_group, *(spectra[i].varid), "description", 
                                   strlen(description.c_str()), description.c_str())) ERR(retval);
        }
      } else {
        if (retval = nc_def_var(nc_group, spec_varname.c_str(), nc_type, 2, spectra[i].dims, spectra[i].varid)) ERR(retval);
        if (retval = nc_var_par_access(nc_group, *(spectra[i].varid), NC_COLLECTIVE)) ERR(retval);
        if (retval = nc_put_att_text(nc_group, *(spectra[i].varid), "description", 
                                 strlen(description.c_str()), description.c_str())) ERR(retval);
      }
    }
    
    // Allocate memory for all reduced spectra and initialise to zero
    cudaMalloc(&transfer_z_d, sizeof(float) * grids->Nz);
    checkCuda(cudaMemset(transfer_z_d, 0.0, sizeof(float) * grids->Nz));
    cudaMalloc(&transfer_target_kx_d, sizeof(float) * grids->Nakx);
    checkCuda(cudaMemset(transfer_target_kx_d, 0.0, sizeof(float) * grids->Nakx));
    cudaMalloc(&transfer_source_kx_d, sizeof(float) * grids->Nakx);
    checkCuda(cudaMemset(transfer_source_kx_d, 0.0, sizeof(float) * grids->Nakx));
    cudaMalloc(&transfer_source_ky_d, sizeof(float) * (2*(grids->Naky)-1));
    checkCuda(cudaMemset(transfer_source_ky_d, 0.0, sizeof(float) * (2*(grids->Naky)-1)));
    
    transfer_z_h = (float*) malloc(sizeof(float) * grids->Nz);
    transfer_target_kx_h = (float*) malloc(sizeof(float) * grids->Nakx);
    transfer_source_kx_h = (float*) malloc(sizeof(float) * grids->Nakx);
    transfer_source_ky_h = (float*) malloc(sizeof(float) * (2*(grids->Naky)-1));
  }

  // Create NetCDF variable for full transfer in `.big.nc` file. The full
  // transfer is resolved over `(t, z, target_kx, source_kx, source_ky)`.
  if (pars_->write_zonal_energy_transfer_full) {
    if (pars_->restart && pars_->append_on_restart && nc_inq_varid(nc_group_big, varname.c_str(), &varid_big)==NC_NOERR) {
      if (retval = nc_inq_varid(nc_group_big, varname.c_str(), &varid_big)) ERR(retval);
      if (retval = nc_var_par_access(nc_group_big, varid_big, NC_COLLECTIVE)) ERR(retval);
    } else {
      if (retval = nc_def_var(nc_group_big, varname.c_str(), nc_type, ndim, dims, &varid_big)) ERR(retval);
      if (retval = nc_var_par_access(nc_group_big, varid_big, NC_COLLECTIVE)) ERR(retval);
      if (retval = nc_put_att_text(nc_group_big, varid_big, "description", 
                                 strlen(description.c_str()), description.c_str())) ERR(retval);
    }
  }

  cudaMalloc(&transfer_d, sizeof(float) * N);
  checkCuda(cudaMemset(transfer_d, 0.0, sizeof(float) * N));
  cudaMalloc(&phi_ext_d, sizeof(cuComplex) * N_ext);
  checkCuda(cudaMemset(phi_ext_d, 0.0, sizeof(float) * N_ext));
  transfer_h = (float*) malloc(sizeof(float) * Nwrite);

  // Set kernel dimension for transfer calculation based on the loop structure:
  // `z`, `target_kx`, `source_kx`, `source_ky`, where `z` is the outer-most
  // loop and `source_ky` is the inner-most loop. Assign threads to the three
  // inner loops and handle `z` with a loop inside the kernel
  int nn1, nn2, nn3, nt1, nt2, nt3, nb1, nb2, nb3;
  
  nn1 = 2*(grids->Naky)-1;  nt1 = min(nn1, 16);  nb1 = 1 + (nn1-1)/nt1;
  nn2 = grids->Nakx;        nt2 = min(nn2, 8);   nb2 = 1 + (nn2-1)/nt2;
  nn3 = grids->Nakx;        nt3 = min(nn3, 8);   nb3 = 1 + (nn3-1)/nt3;
  
  dB = dim3(nt1, nt2, nt3);
  dG = dim3(nb1, nb2, nb3);

  // Set kernel dimensions for `get_full` calculation (extends phi to include
  // ky < 0)
  nn1 = grids->Naky;        nt1 = min(nn1, 16);  nb1 = 1 + (nn1-1)/nt1;
  nn2 = grids->Nakx;        nt2 = min(nn2, 16);  nb2 = 1 + (nn2-1)/nt2;
  nn3 = grids->Nz;          nt3 = min(nn3, 4);   nb3 = 1 + (nn3-1)/nt3;
  
  dB_gf = dim3(nt1, nt2, nt3);
  dG_gf = dim3(nb1, nb2, nb3);
}

ZonalFlowEnergyTransferDiagnostic::~ZonalFlowEnergyTransferDiagnostic()
{
  cudaFree(transfer_d);
  cudaFree(phi_ext_d);
  free(transfer_h);
  
  if (pars_->write_zonal_energy_transfer) {
    cudaFree(transfer_z_d);
    cudaFree(transfer_target_kx_d);
    cudaFree(transfer_source_kx_d);
    cudaFree(transfer_source_ky_d);
    
    free(transfer_z_h);
    free(transfer_target_kx_h);
    free(transfer_source_kx_h);
    free(transfer_source_ky_h);
  }
}

void ZonalFlowEnergyTransferDiagnostic::calculate_and_write(Fields* f, int counter)
{
  // Extend phi to include negative ky values using get_full
  get_full<<<dG_gf, dB_gf>>>(phi_ext_d, f->phi);
  
  // Calculate zonal energy transfer using the extended phi array
  zonal_energy_transfer_summand <<<dG, dB>>> (
    transfer_d,
    phi_ext_d,
    grids_->kx,
    grids_->source_ky,
    geo_->bmag
  );

  int retval;
  start[0] = ncdf_->nc_grids->time_index;
  
  // Calculate and write reduced spectra to .out.nc file if needed
  if (pars_->write_zonal_energy_transfer) {
    compute_reduced_spectra();
    write_reduced_spectra();
  }
  
  // Write full 5D array to .big.nc file if needed
  if (pars_->write_zonal_energy_transfer_full && (counter % pars_->nwrite_big == 0 || counter == 1)) {
    CP_TO_CPU(transfer_h, transfer_d, sizeof(float) * N);
    if (retval = nc_put_vara(nc_group_big, varid_big, start, count, transfer_h)) ERR(retval);
  }
}

void ZonalFlowEnergyTransferDiagnostic::compute_reduced_spectra()
{
  int nz = grids_->Nz;
  int nkys = 2*(grids_->Naky)-1;
  int nkxs = grids_->Nakx;
  int ntkx = grids_->Nakx;
  
  int dims[4] = {nz, ntkx, nkxs, nkys};
  
  // Define reduction kernels and their output sizes
  struct ReductionSpec {
    const char* name;
    float* data_d;
    float* data_h;
    int count;
    void (*reduce_kernel)(float*, const float*);
    int dim_idx[3];  // index of `dims` kernel is threading over
    int var_idx;     // index of the dimension to loop over within the kernel
  };
  
  // Define the dimension indices being used for thread blocks
  // And the dimension that will be looped over inside the kernel
  // Indices reference the dims array: 0=nz, 1=ntkx, 2=nkxs, 3=nkys
  ReductionSpec reductions[] = {
    {"z", transfer_z_d, transfer_z_h, nz, reduce_to_z, {3, 2, 1}},
    {"target_kx", transfer_target_kx_d, transfer_target_kx_h, ntkx, reduce_to_target_kx, {3, 2, 0}},
    {"source_kx", transfer_source_kx_d, transfer_source_kx_h, nkxs, reduce_to_source_kx, {3, 1, 0}},
    {"source_ky", transfer_source_ky_d, transfer_source_ky_h, nkys, reduce_to_source_ky, {2, 1, 0}}
  };
  
  // Launch the reduction kernels
  for (int i = 0; i < 4; i++) {
    dim3 dB(
      min(8, dims[reductions[i].dim_idx[0]]),
      min(8, dims[reductions[i].dim_idx[1]]),
      min(8, dims[reductions[i].dim_idx[2]])
    );
    
    dim3 dG(
      1 + (dims[reductions[i].dim_idx[0]]-1)/dB.x,
      1 + (dims[reductions[i].dim_idx[1]]-1)/dB.y,
      1 + (dims[reductions[i].dim_idx[2]]-1)/dB.z
    );
    
    reductions[i].reduce_kernel<<<dG, dB>>>(
      reductions[i].data_d, 
      transfer_d
    );
  }

  for (int i = 0; i < 4; i++) {
    CP_TO_CPU(reductions[i].data_h, reductions[i].data_d, sizeof(float) * reductions[i].count);
  }
}

void ZonalFlowEnergyTransferDiagnostic::write_reduced_spectra()
{
  int retval;
  
  struct NetCDFVarSpec {
    const char* name;
    int varid;
    size_t* start;
    size_t* count;
    float* data;
  };
  
  NetCDFVarSpec ncvars[] = {
    {"z", z_varid, z_start, z_count, transfer_z_h},
    {"target_kx", target_kx_varid, target_kx_start, target_kx_count, transfer_target_kx_h},
    {"source_kx", source_kx_varid, source_kx_start, source_kx_count, transfer_source_kx_h},
    {"source_ky", source_ky_varid, source_ky_start, source_ky_count, transfer_source_ky_h}
  };
  
  for (int i = 0; i < 4; i++) {
    ncvars[i].start[0] = ncdf_->nc_grids->time_index;
    if (retval = nc_put_vara(nc_group, ncvars[i].varid, ncvars[i].start, ncvars[i].count, ncvars[i].data)) ERR(retval);
  }
}
