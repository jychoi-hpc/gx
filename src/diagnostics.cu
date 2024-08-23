#include "diagnostics.h"
#include "get_error.h"
#include "netcdf.h"
#include <sys/stat.h>
#define loop_R <<< dG_spectra, dB_spectra >>>
#define GALL <<< dG_all, dB_all >>>
#define GFLA <<< 1 + (grids_->Nx*grids_->Nyc - 1)/grids_->Nakx, grids_->Nakx >>> 
#define KXKY <<< dGk, dBk >>>
#define loop_y <<< dgp, dbp >>> 

Diagnostics_GK::Diagnostics_GK(Parameters* pars, Grids* grids, Geometry* geo, Linear* linear, Nonlinear* nonlinear) :
  geo_(geo), fields_old(nullptr), ncdf_(nullptr), ncdf_big_(nullptr), linear_(linear), nonlinear_(nonlinear)
{
  pars_ = pars;
  grids_ = grids;
  
  ncdf_ = new NetCDF(pars_, grids_, geo_, ".out.nc"); 
  // write input parameters to netcdf
  if(! (pars_->restart && pars_->append_on_restart)) pars->store_ncdf(ncdf_->fileid, ncdf_->nc_dims);

  if (pars_->write_fields || pars_->write_moms) {
    ncdf_big_ = new NetCDF(pars_, grids_, geo_, ".big.nc"); 
  }

  // set up spectra calculators
  // Always need allSpectra for Phi2 / A_||^2 below
  allSpectra_ = new AllSpectraCalcs(grids_, ncdf_->nc_dims);
  if(pars_->write_free_energy || pars_->write_fluxes) {
    cudaMalloc (&tmpG, sizeof(float) * grids_->NxNycNz * grids_->Nmoms * grids_->Nspecies); 
    cudaMalloc (&tmpf, sizeof(float) * grids_->NxNycNz * grids_->Nspecies);
  }
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
  spectraDiagnosticList.push_back(std::make_unique<Phi2Diagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
  spectraDiagnosticList.push_back(std::make_unique<Apar2Diagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
  spectraDiagnosticList.push_back(std::make_unique<Phi2ZonalDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
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
    //spectraDiagnosticList.push_back(std::make_unique<ParticleFluxESDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
    //spectraDiagnosticList.push_back(std::make_unique<ParticleFluxAparDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
    //spectraDiagnosticList.push_back(std::make_unique<ParticleFluxBparDiagnostic>(pars_, grids_, geo_, ncdf_, allSpectra_));
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
    for(int i=0; i<spectraDiagnosticList.size(); i++) {
      spectraDiagnosticList[i]->set_dt_data(G_old, fields_old, dt);
      spectraDiagnosticList[i]->calculate_and_write(G, fields, tmpG, tmpf);
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

    for(int i=0; i<momentsDiagnosticList.size(); i++) {
      momentsDiagnosticList[i]->calculate_and_write(G, fields, tmpC);
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

//  int retval;
//  int nw;
//
//  if(id -> omg -> write_v_time && counter >= 0) {                    // complex frequencies
//    int nt = min(512, grids_->NxNyc) ;
//    growthRates <<< 1 + (grids_->NxNyc-1)/nt, nt >>> (fields->phi, fields_old->phi, dt, omg_d);
//    fields_old->copyPhiFrom(fields);
//  }
//
//  nw = pars_->nwrite;
//
//  //if ((counter % nw == nw-1) && id -> omg -> write_v_time) fields_old->copyPhiFrom(fields);
//    
//  if(counter%nw == 1 || time > pars_->t_max) {
//
//    if (pars_->Reservoir && counter > pars_->nstep-pars_->ResPredict_Steps*pars_->ResTrainingDelta) {
//      id -> write_nc(id -> time, time);
//      //      if (pars_->ResWrite) id -> write_nc( id -> r_time, time);
//    }
//    if (!pars_->Reservoir) {
//      id -> write_nc(id -> time, time);
//    }
//
//    if (pars_->write_xymom) id -> write_nc( id -> z_time, time);
//    
//    if ( id -> qs -> write_v_time && grids_->iproc==0) printf("%s: Step %7d: Time = %10.5f,  dt = %.3e,  ", pars_->run_name, counter, time, dt);          // To screen
//    if (!id -> qs -> write_v_time && grids_->iproc==0) printf("%s: Step %7d: Time = %10.5f,  dt = %.3e\n",  pars_->run_name, counter, time, dt);
//  
//    if ( id -> qs -> write_v_time) {                                                                // heat flux
//      
//      for(int is=0; is<grids_->Nspecies; is++) {
//        int is_glob = is + grids_->is_lo;
//	float rho2s = pars_->species_h[is_glob].rho2;
//	float p_s = pars_->species_h[is_glob].nt;
//	float vt_s = pars_->species_h[is_glob].vt;
//	heat_flux_summand loop_R (P2(is), fields->phi, fields->apar, G[is]->G(),
//				  grids_->ky, flux_fac, geo_->kperp2, rho2s, p_s, vt_s);
//      }
//      id -> write_Qky(P2());
//      id -> write_Qkx(P2());
//      id -> write_Qkxky(P2());
//      id -> write_Qz(P2());
//      id -> write_Q(P2()); 
//    }      
//
//    if ( id -> ps -> write_v_time) {
//
//      for(int is=0; is<grids_->Nspecies; is++) {
//        int is_glob = is + grids_->is_lo;
//	float rho2s = pars_->species_h[is_glob].rho2;
//        float n_s = pars_->nspec>1 ? pars_->species_h[is_glob].dens : 0.;
//	float vt_s = pars_->species_h[is_glob].vt;
//	part_flux_summand loop_R (P2(is), fields->phi, fields->apar, G[is]->G(),
//				  grids_->ky, flux_fac, geo_->kperp2, rho2s, n_s, vt_s);
//      }
//      id -> write_P(P2s); 
//    }
//    if ( id -> qs -> write_v_time && grids_->m_lo == 0) printf("\n");
//
//    if(id -> omg -> write_v_time && counter > 0) {                    // complex frequencies
//      print_omg(omg_d);  id -> write_omg(omg_d);
//    }
//    
//    if (pars_->diagnosing_kzspec) {
//      for (int is=0; is < grids_->Nspecies; is++) {             // P2(s) = (1-G0(s)) |phi**2| for each kinetic species
//        int is_glob = is + grids_->is_lo;
//        grad_par->zft(G[is]); // get G = G(kz)
//        W_summand GALL (G2(is), G[is]->G(), kvol_fac, G[is]->species->nt);
//        grad_par->zft_inverse(G[is]); // restore G
//
//        grad_par->zft(fields->phi, amom_d); // get amom_d = phi(kz)
//      
//	float rho2s = pars_->species_h[is_glob].rho2;
//	Wphi_summand loop_R (P2(is), amom_d, kvol_fac, geo_->kperp2, rho2s);
//	float qfac = pars_->species_h[is_glob].nz*pars_->species_h[is_glob].zt;
//	Wphi_scale loop_R   (P2(is), qfac);
//      }
//
//      if (pars_->add_Boltzmann_species) {
//	if (pars_->Boltzmann_opt == BOLTZMANN_IONS)  Wphi2_summand loop_R (Phi2, amom_d, kvol_fac);
//	
//	if (pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS) {
//	  fieldlineaverage GFLA (favg, df, fields->phi, vol_fac); // favg is a dummy variable
//	  grad_par->zft(df, amom_d); // get df = df(kz)
//	  Wphi2_summand loop_R (Phi2, amom_d, kvol_fac); 	
//	}
//
//	float fac = 1./pars_->tau_fac;
//	Wphi_scale loop_R (Phi2, fac);
//      }
//      
//      id -> write_Wkz(G2());    id -> write_Pkz(P2());    id -> write_Akz(Phi2);      
//    }
//    
//    if (pars_->diagnosing_spectra) {                                        // Various spectra
//      for (int is=0; is < grids_->Nspecies; is++) {  
//        W_summand GALL (G2(is), G[is]->G(), vol_fac, G[is]->species->nt);
//      }
//      
//      if (pars_->gx) {
//	for (int is=0; is < grids_->Nspecies; is++) {       // P2(s) = (1-G0(s)) |phi**2| for each kinetic species
//          int is_glob = is + grids_->is_lo;
//	  float rho2s = pars_->species_h[is_glob].rho2;
//	  Wphi_summand loop_R (P2(is), fields->phi, vol_fac, geo_->kperp2, rho2s);
//	  float qnfac = pars_->species_h[is_glob].nz*pars_->species_h[is_glob].zt;
//	  Wphi_scale loop_R   (P2(is), qnfac);
//	}
//
//	if (pars_->add_Boltzmann_species) {
//	  if (pars_->Boltzmann_opt == BOLTZMANN_IONS)  Wphi2_summand loop_R (Phi2, fields->phi, vol_fac);
//	  
//	  if (pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS) {	  
//	    fieldlineaverage GFLA (favg, df, fields->phi, vol_fac); // favg is a dummy variable
//	    Wphi2_summand loop_R (Phi2, df, vol_fac); 	
//	  }
//	  
//	  float fac = 1./pars_->tau_fac;
//	  Wphi_scale loop_R (Phi2, fac);
//	}
//      }
//
//      if (pars_->ks) {
//	cuComplex * g_h;
//	g_h = (cuComplex*) malloc (sizeof(cuComplex)*grids_->Nyc);
//	CP_TO_CPU(g_h, G[0]->G(), sizeof(cuComplex)*grids_->Nyc);
//	float Dtmp = 0.;
//	for (int i=0; i<grids_->Naky; i++) {
//	  Dtmp += (g_h[i].x*g_h[i].x + g_h[i].y*g_h[i].y)*grids_->ky_h[i]*grids_->ky_h[i];
//	}
//	Dks += Dtmp;
//	printf("<D> = %f \t",Dks/((float) ndiag));
//	ndiag += 1;
//	free(g_h);
//      }
//      
//      id->write_Wm    (G2()   );    id->write_Wl    (G2()   );    id->write_Wlm   (G2()   );    
//      id->write_Wz    (G2()   );    id->write_Wky   (G2()   );    id->write_Wkx   (G2()   );    id->write_Wkxky (G2()  );    
//      id->write_Pz    (P2() );    id->write_Pky   (P2() );    id->write_Pkx   (P2() );    id->write_Pkxky (P2());    
//      id->write_Az    (Phi2 );    id->write_Aky   (Phi2 );    id->write_Akx   (Phi2 );    id->write_Akxky (Phi2);
//      
//      // Do not change the order of these four calls because totW is accumulated in order when it is requested:
//      id->write_Ps(P2());    id->write_Ws(G2());    id->write_As(Phi2);    id->write_Wtot();
//    }
//    
//    // Rosenbluth-Hinton diagnostic
//    if(id -> rh -> write) {get_rh(fields);   id -> write_nc (id->rh, val);}
//    
//    /*
//      if( counter%nw == 0 && id -> Pzt -> write_v_time) {
//      pzt(G, fields);  // calculate each of P, Z, and T (very very rough diagnostic)
//      cudaDeviceSynchronize();
//      
//      write_nc (id -> Pzt, primary);
//      write_nc (id -> pZt, secondary);
//      write_nc (id -> pzT, tertiary);
//      }
//    */
//
//    // Plot ky=kz=0 components of various quantities as functions of x
//    id -> write_moment ( id -> vEy,     fields->phi,    vol_fac);
//    id -> write_moment ( id -> kxvEy,   fields->phi,    vol_fac);
//    for(int is=0; is<grids_->Nspecies; is++) {
//      id -> write_moment ( id -> kden,    G[is]->dens_ptr, vol_fac);
//      id -> write_moment ( id -> kUpar,   G[is]->upar_ptr, vol_fac);
//      id -> write_moment ( id -> kTpar,   G[is]->tpar_ptr, vol_fac);
//      id -> write_moment ( id -> kTperp,  G[is]->tprp_ptr, vol_fac);
//      id -> write_moment ( id -> kqpar,   G[is]->qpar_ptr, vol_fac);
//    }
//
//    // Plot some zonal scalars
//    id -> write_moment ( id -> avg_zvE,     fields->phi,    vol_fac);
//    id -> write_moment ( id -> avg_zkxvEy,  fields->phi,    vol_fac);
//    for(int is=0; is<grids_->Nspecies; is++) {
//      id -> write_moment ( id -> avg_zkden,   G[is]->dens_ptr, vol_fac);
//      id -> write_moment ( id -> avg_zkUpar,  G[is]->upar_ptr, vol_fac);
//      id -> write_moment ( id -> avg_zkTpar,  G[is]->tpar_ptr, vol_fac);
//      id -> write_moment ( id -> avg_zkTperp, G[is]->tprp_ptr, vol_fac);
//      id -> write_moment ( id -> avg_zkqpar,  G[is]->qpar_ptr, vol_fac);
//    }
//
//    // Plot f(x,y,z=0)
//    id -> write_moment ( id -> xyPhi,   fields->phi,    vol_fac);
//    
//    // Plot the non-zonal components as functions of (x, y)
//    id -> write_moment ( id -> xykxvEy, fields->phi,    vol_fac);
//    id -> write_moment ( id -> xyvEy,   fields->phi,    vol_fac);
//    id -> write_moment ( id -> xyvEx,   fields->phi,    vol_fac);
//    for(int is=0; is<grids_->Nspecies; is++) {
//      id -> write_moment ( id -> xyden,   G[is]->dens_ptr, vol_fac);
//      id -> write_moment ( id -> xyUpar,  G[is]->upar_ptr, vol_fac);
//      id -> write_moment ( id -> xyTpar,  G[is]->tpar_ptr, vol_fac);
//      id -> write_moment ( id -> xyTperp, G[is]->tprp_ptr, vol_fac);
//      id -> write_moment ( id -> xyqpar,  G[is]->qpar_ptr, vol_fac);
//    }
//
//    if (pars_->Reservoir && counter > pars_->nstep-pars_->ResPredict_Steps*pars_->ResTrainingDelta) {
//      if (!pars_->ResFakeData) id -> write_ks_data ( id -> g_y, G[0]->G());
//    }
//    if (!pars_->Reservoir) {
//      id -> write_ks_data ( id -> g_y, G[0]->G());
//    }
//
//    nc_sync(id->file);
//    fflush(NULL);
//  }
//  if (pars_->Reservoir && counter%pars_->ResTrainingDelta == 0) {
//    grad_perp->C2R(G[0]->G(), gy_d);
//    if (pars_->ResFakeData) {
//      rc->fake_data(gy_d);
//      id -> write_ks_data( id -> g_y, gy_d);
//    }
//    rc->add_data(gy_d);
//  }
//  if (pars_->fixed_amplitude && (counter % nw == nw-2)) {
//    maxPhi KXKY (phi_max, fields->phi);
//    for(int is=0; is<grids_->Nspecies; is++) {
//      G[is]->rescale(phi_max);
//    }
//    fields->rescale(phi_max);
//  }
//  }
  
  // check to see if we should stop simulation
  stop = checkstop();
  return stop;
}

void Diagnostics_GK::finish(MomentsG** G, Fields* fields, double time) 
{
//  if (pars_->Reservoir && rc->predicting()) {
//    if (pars_->ResFakeData) {
//      rc->fake_data(gy_d);
//    } else {
//      for(int is=0; is<grids_->Nspecies; is++) {
//        grad_perp -> C2R (G[is]->G(), gy_d);
//      }
//    }
//    double *gy_double;
//    cudaMalloc(&gy_double, sizeof(double)*grids_->Ny);
//    promote loop_y (gy_double, gy_d, grids_->Ny);
//    
//    for (int i=0; i<pars_->ResPredict_Steps; i++) {
//      rc->predict(gy_double);
//      time += pars_->dt * pars_->ResTrainingDelta;
//      demote loop_y (gy_d, gy_double, grids_->Ny);
//      id -> write_nc(id -> time, time);
//      id -> write_ks_data (id -> g_y, gy_d);
//    }
//  }
//  if (pars_->write_fields) {
//    id -> write_fields(id -> fields_phi,  fields->phi );
//    id -> write_fields(id -> fields_apar, fields->apar);
//    id -> write_fields(id -> fields_bpar, fields->bpar);
//  }
}


  // For each kx, z, l and m, sum the moments of G**2 + Phi**2 (1-Gamma_0) with weights:
  //
  // The full sum is 
  //
  // sum_s sum_l,m sum_ky sum_kx sum_z J(z) == SUM
  //
  // and there is a denominator for the z-integration, to normalize out the volume,
  // SUM => SUM / sum J(z)
  // 
  // W = SUM n_s tau_s G**2 / 2 + SUM n_s Z_s**2 / (2 tau_s) (1 - Gamma_0) Phi**2
  //
  // Since we store only half of the spatial Fourier coefficients for all but the
  // ky = 0 modes
  //
  // (for which we have the full set of kx modes, kx > 0 and kx < 0, and everything is zero for kx = ky = 0)  
  //
  // we multiply all the ky != 0 modes by a factor of two. This is easiest to see in device_functions/vol_summand.
  //
  // To make later manipulations easiest, all partial summations will be carried such that a simple sum of
  // whatever is leftover will give W.
  //

//void Diagnostics_GK::get_rh(Fields* f)
//{
//  int ikx_local, iky_local, iz_local;
//  ikx_local = 1; iky_local = 0; iz_local=grids_->Nz/2; // correct values for usual RH tests
//
//  int idx = iky_local + grids_->Nyc*ikx_local + grids_->NxNyc*iz_local;
//  
//  CP_TO_CPU(&valphi, &f->phi[idx], sizeof(cuComplex));
//  val[0] = valphi.x;
//  val[1] = valphi.y;
//}

Diagnostics_KREHM::Diagnostics_KREHM(Parameters* pars, Grids* grids, Geometry* geo, Linear* linear, Nonlinear* nonlinear) :
  geo_(geo), fields_old(nullptr), ncdf_(nullptr), ncdf_big_(nullptr), linear_(linear), nonlinear_(nonlinear)
{
  pars_ = pars;
  grids_ = grids;

  ncdf_ = new NetCDF(pars_, grids_, geo_, ".out.nc"); 
  // write input parameters to netcdf
  pars->store_ncdf(ncdf_->fileid, ncdf_->nc_dims);

  if (pars_->write_fields || pars_->write_moms) {
    ncdf_big_ = new NetCDF(pars_, grids_, geo_, ".big.nc"); 
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
    for(int i=0; i<spectraDiagnosticList.size(); i++) {
      spectraDiagnosticList[i]->calculate_and_write(G, fields, tmpG, tmpf);
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

    for(int i=0; i<momentsDiagnosticList.size(); i++) {
      momentsDiagnosticList[i]->calculate_and_write(G, fields, tmpC);
    }

    ncdf_big_->nc_grids->write_time(time);
    ncdf_big_->sync();
  }

  // check to see if we should stop simulation
  stop = checkstop();
  return stop;
}

void Diagnostics_KREHM::finish(MomentsG** G, Fields* fields, double time) 
{
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
  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;
  int Nz   = grids_->Nz;
  int Nm   = grids_->Nm;
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

Diagnostics_cetg::Diagnostics_cetg(Parameters* pars, Grids* grids, Geometry* geo) : geo_(geo), fields_old(nullptr), id(nullptr)
{
  printf(ANSI_COLOR_BLUE);
  pars_ = pars;
  grids_ = grids;
  
//  int nL  = grids_->Nl;
//  int nM  = grids_->Nm;
//  int nS  = grids_->Nspecies;
//  int nX  = grids_->Nx;
//  int nXk = grids_->Nakx;
//  int nY  = grids_->Nyc;
//  int nYk = grids_->Naky;
//  int nZ  = grids_->Nz;
//  int nR  = nX  * nY  * nZ; // nY is *not* the number of grid points in the y-direction. 
//  int nK  = nXk * nYk * nZ;
//  int nG  = nR * grids_->Nmoms;
//
//  assert( (nS == 1) && "number of species must be one \n");
//    
//  G2s         = nullptr;  P2s         = nullptr;  
//  omg_d       = nullptr;  tmp_omg_h   = nullptr;  
//  vEk         = nullptr;
//  vol_fac     = nullptr;
//  flux_fac    = nullptr;
//
//  id         = new NetCDF_ids(grids_, pars_); cudaDeviceSynchronize(); CUDA_DEBUG("NetCDF_ids: %s \n");
//
//  float *vol_fac_h;
//  volDenom = 0.;
//  vol_fac_h = (float*) malloc (sizeof(float) * nZ);
//  cudaMalloc (&vol_fac, sizeof(float) * nZ);
//  for (int i=0; i < nZ; i++) volDenom   += geo_->jacobian_h[i];
//  for (int i=0; i < nZ; i++) vol_fac_h[i]  =  geo_->jacobian_h[i] / volDenom;
//  CP_TO_GPU(vol_fac, vol_fac_h, sizeof(float)*nZ);
//  free(vol_fac_h);
//
//  fluxDenom = 0.;  
//  float *flux_fac_h;
//  flux_fac_h = (float*) malloc (sizeof(float) * nZ);
//  cudaMalloc(&flux_fac, sizeof(float)*nZ);
//  for (int i=0; i<grids_->Nz; i++) fluxDenom   += geo_->jacobian_h[i] * geo_->grho_h[i];
//  for (int i=0; i<grids_->Nz; i++) flux_fac_h[i]  = geo_->jacobian_h[i] / fluxDenom;
//
//  CP_TO_GPU(flux_fac, flux_fac_h, sizeof(float)*nZ);
//  free(flux_fac_h);  
//
//  if (pars_->diagnosing_spectra || pars_->diagnosing_kzspec) cudaMalloc (&G2s, sizeof(float) * nG); 
//  cudaMalloc (&P2s, sizeof(float) * nR);
//
//  if (id -> omg -> write_v_time) {
//    fields_old = new      Fields(pars_, grids_);      cudaDeviceSynchronize(); CUDA_DEBUG("Fields: %s \n");
//    cudaMalloc     (    &omg_d,   sizeof(cuComplex) * nX * nY);//     cudaMemset (omg_d, 0., sizeof(cuComplex) * nX * nY);
//    tmp_omg_h = (cuComplex*) malloc (sizeof(cuComplex) * nX * nY);
//    int nn = nX*nY; int nt = min(nn, 512); int nb = 1 + (nn-1)/nt;  cuComplex zero = make_cuComplex(0.,0.);
//    setval <<< nb, nt >>> (omg_d, zero, nn);
//  }  
//
//  if (id -> kxvEy -> write_v_time || id -> xykxvEy -> write_v_time) {
//    cudaMalloc     (&vEk,        sizeof(cuComplex) * grids_->NxNycNz);
//  }
//     
//  // set up stop file
//  sprintf(stopfilename_, "%s.stop", pars_->run_name);
//
//  //  dB_scale = min(512, nR);
//  //  dG_scale = 1 + (nR-1)/dB_scale.x;
//
//  dB_spectra = dim3(min(8, nY), min(8, nX), min(8, nZ));
//  dG_spectra = dim3(1 + (nY-1)/dB_spectra.x, 1 + (nX-1)/dB_spectra.y, 1 + (nZ-1)/dB_spectra.z);  
//
//  int nyx =  nY * nX;
//  int nlm = nL * nM;
//
//  int nt1 = 16;
//  int nb1 = 1 + (nyx-1)/nt1;
//
//  int nt2 = 16;
//  int nb2 = 1 + (grids_->Nz-1)/nt2;
//  
//  dB_all = dim3(nt1, nt2, 1);
//  dG_all = dim3(nb1, nb2, nlm);
//  
//  nt1 = min(32, grids_->Nyc);
//  nb1 = 1 + (grids_->Nyc-1)/nt1;
//
//  nt2 = min(32, grids_->Nx);
//  nb2 = 1 + (grids_->Nx-1)/nt2;
//
//  dBk = dim3(nt1, nt2, 1);
//  dGk = dim3(nb1, nb2, 1);
//  
//  if (grids_->Nakx > 1024) {printf("Need to redefine GFLA in diagnostics \n"); exit(1);}
//
//  nt1 = min(grids_->Ny, 512);
//  nb1 = 1 + (grids_->Ny-1)/nt1;
//
//  dbp = dim3(nt1, 1, 1);
//  dgp = dim3(nb1, 1, 1);
//
//  printf(ANSI_COLOR_RESET);
//  ndiag = 1;

}

Diagnostics_cetg::~Diagnostics_cetg()
{
//  if (fields_old) delete fields_old;
//  if (id)         delete id;
//
//  if (G2s)        cudaFree      ( G2s       );
//  if (P2s)        cudaFree      ( P2s       );
//  if (omg_d)      cudaFree      ( omg_d     );
//  
//  if (vol_fac)    cudaFree      ( vol_fac   );
//  if (flux_fac)   cudaFree      ( flux_fac  );
//  if (tmp_omg_h)  free  ( tmp_omg_h );
}

bool Diagnostics_cetg::loop(MomentsG** G, Fields* fields, double dt, int counter, double time) 
{
//  int retval;
//  bool stop = false;
//  int nw;
//
//  nw = pars_->nwrite;
//
//  if (counter == 0 && id -> omg -> write_v_time) fields_old->copyPhiFrom(fields);
//
//  if(id -> omg -> write_v_time && (counter == 0 || counter%nw==0)) {  // complex frequencies
//    int nt = min(512, grids_->NxNyc) ;
//    growthRates <<< 1 + (grids_->NxNyc-1)/nt, nt >>> (fields->phi, fields_old->phi, dt, omg_d);
//  }
//
//  if ((counter % nw == nw-1) && id -> omg -> write_v_time) fields_old->copyPhiFrom(fields);
//
//    
//  if(counter%nw == 0 || time > pars_->t_max) {
//
//    fflush(NULL);
//    id -> write_nc(id -> time, time);
//    if (grids_->iproc==0) printf("%s: Step %7d: Time = %10.5f,  dt = %.3e\n",  pars_->run_name, counter, time, dt);
// 
//    //if (pars_->write_phi) id->write_nc(id->phi, phi);
//
//    // Plot f(x,y,z=0)
//    if (pars_->write_xymom) id -> write_nc( id -> z_time, time);
//    id -> write_moment ( id -> xyPhi,   fields->phi,    vol_fac);
//    
//    if ( id -> qs -> write_v_time) {                                                                // heat flux
//      float p_s = pars_->species_h[0].nt;      
//      heat_flux_summand_cetg loop_R (P2(), fields->phi, G[0]->G(), grids_->ky, flux_fac, p_s);
//    }
//    id -> write_Qky(P2());
//    id -> write_Qkx(P2());
//    id -> write_Qkxky(P2());
//    id -> write_Qz(P2());
//    id -> write_Q(P2());   
//    
//    if(id -> omg -> write_v_time && counter > 0) {                    // complex frequencies
//      print_omg(omg_d);  id -> write_omg(omg_d);
//    }
//
//    if (pars_->diagnosing_spectra) {                                        // Various spectra
//      W_summand GALL (G2(), G[0]->G(), vol_fac, G[0]->species->nt);
//
//      Wphi_summand_cetg loop_R (P2(), fields->phi, vol_fac);
//      
//      id->write_Wl    (G2());  
//      id->write_Wz    (G2());    id->write_Wky    (G2() );    id->write_Wkx    (G2()   );    id->write_Wkxky (G2()  );    
//      id->write_Phi2z (P2());    id->write_Phi2ky (P2() );    id->write_Phi2kx (P2()   );    id->write_Phi2kxky (P2());    
//     
//    }
//
//    nc_sync(id->file);
//    nc_sync(id->z_file);
//  }
//
//  // check to see if we should stop simulation
//  stop = checkstop();
//  return stop;
}
//
void Diagnostics_cetg::finish(MomentsG** G, Fields* fields, double time) 
{
//  if (pars_->write_fields) {
//    id -> write_fields(id -> fields_phi,  fields->phi );
//  }
}

void Diagnostics_cetg::print_omg(cuComplex *W)
{
  CP_TO_CPU (tmp_omg_h, W, sizeof(cuComplex)*grids_->NxNyc);
  print_growth_rates_to_screen(tmp_omg_h);
}

