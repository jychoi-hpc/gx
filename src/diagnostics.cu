#include "diagnostics.h"
#include "get_error.h"
#include "netcdf.h"
#include <sys/stat.h>
#define loop_R <<< dG_spectra, dB_spectra >>>
#define GALL <<< dG_all, dB_all >>>
#define GFLA <<< 1 + (grids_->Nx*grids_->Nyc - 1)/grids_->Nakx, grids_->Nakx >>> 
#define KXKY <<< dGk, dBk >>>
#define loop_y <<< dgp, dbp >>> 

Diagnostics_GK::Diagnostics_GK(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo),
  fields_old(nullptr), id(nullptr), grad_par(nullptr), amom_d(nullptr), grad_perp(nullptr)//, grad_phi(nullptr)
{
  printf(ANSI_COLOR_BLUE);
  
  int nL  = grids_->Nl;
  int nM  = grids_->Nm;
  int nS  = grids_->Nspecies;
  int nX  = grids_->Nx;
  int nXk = grids_->Nakx;
  int nY  = grids_->Nyc;
  int nYk = grids_->Naky;
  int nZ  = grids_->Nz;
  int nR  = nX  * nY  * nZ; // nY is *not* the number of grid points in the y-direction. 
  int nK  = nXk * nYk * nZ;
  int nG  = nR * grids_->Nmoms * nS;

  favg        = nullptr;  df          = nullptr;  val         = nullptr;  
  G2          = nullptr;  P2s         = nullptr;  Phi2        = nullptr;
  omg_d       = nullptr;  tmp_omg_h   = nullptr;  t_bar       = nullptr;  
  vEk         = nullptr;  phi_max     = nullptr;
  ry_h        = nullptr;  gy_h        = nullptr;  gy_d        = nullptr;
  vol_fac = nullptr;
  flux_fac = nullptr;
  kvol_fac = nullptr;

  
  id         = new NetCDF_ids(grids_, pars_, geo_); cudaDeviceSynchronize(); CUDA_DEBUG("NetCDF_ids: %s \n");
  fields_old = new     Fields(pars_, grids_);       cudaDeviceSynchronize(); CUDA_DEBUG("Fields: %s \n");

  if (pars_->fixed_amplitude) cudaMalloc (&phi_max, sizeof(float) * nX * nY);

  if (pars_->Reservoir) rc = new Reservoir(pars_, grids_->NxNyNz*grids_->Nmoms);  
  
  volDenom = 0.;  
  float *vol_fac_h;
  vol_fac_h = (float*) malloc (sizeof(float) * nZ);
  cudaMalloc (&vol_fac, sizeof(float) * nZ);
  for (int i=0; i < nZ; i++) volDenom   += geo_->jacobian_h[i]; 
  for (int i=0; i < nZ; i++) vol_fac_h[i]  = geo_->jacobian_h[i] / volDenom;
  CP_TO_GPU(vol_fac, vol_fac_h, sizeof(float)*nZ);
  free(vol_fac_h);

  fluxDenom = 0.;  
  float *flux_fac_h;
  flux_fac_h = (float*) malloc (sizeof(float) * nZ);
  cudaMalloc(&flux_fac, sizeof(float)*nZ);
  for (int i=0; i<grids_->Nz; i++) fluxDenom   += geo_->jacobian_h[i]*geo_->grho_h[i];
  for (int i=0; i<grids_->Nz; i++) flux_fac_h[i]  = geo_->jacobian_h[i] / fluxDenom;
  CP_TO_GPU(flux_fac, flux_fac_h, sizeof(float)*nZ);
  free(flux_fac_h);
  
  if (pars_->diagnosing_spectra || pars_->diagnosing_kzspec) cudaMalloc (&G2, sizeof(float) * nG); 

  if (pars_->diagnosing_kzspec) {
    float *kvol_fac_h;
    kvol_fac_h = (float*) malloc (sizeof(float) * nZ);
    cudaMalloc (&kvol_fac, sizeof(float) * nZ);
    for (int i=0; i < nZ; i++) kvol_fac_h[i] = 1.0;
    CP_TO_GPU(kvol_fac, kvol_fac_h, sizeof(float)*nZ);
    free(kvol_fac_h);

    cudaMalloc (&amom_d, sizeof(cuComplex) * nR * nS); 
    if (pars_->local_limit) {
      // nothing, this is not defined, or could be defined as an identity.
    }
    else if(pars_->boundary_option_periodic) {
      grad_par = new GradParallelPeriodic(grids_);
    }
    else {
      grad_par = new GradParallelLinked(grids_, pars_->jtwist);
    }
  }
  // need if (pars_->write_flux || "diagnosing potential) {
  cudaMalloc (&P2s, sizeof(float) * nR * nS);

  if (id -> rh -> write) val = (float*) malloc(sizeof(float)*2);

  if (!pars_->all_kinetic) {

    if (pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS) {
      cudaMalloc((void**) &favg, sizeof(cuComplex) * nX);  
      cudaMalloc((void**)   &df, sizeof(cuComplex) * nR);
    }

    cudaMalloc (&Phi2, sizeof(float) * nR);  
  }
  
  if (id -> omg -> write_v_time) {
    cudaMalloc     (    &omg_d,   sizeof(cuComplex) * nX * nY);//     cudaMemset (omg_d, 0., sizeof(cuComplex) * nX * nY);
    tmp_omg_h = (cuComplex*) malloc (sizeof(cuComplex) * nX * nY);
    int nn = nX*nY; int nt = min(nn, 512); int nb = 1 + (nn-1)/nt;  cuComplex zero = make_cuComplex(0.,0.);
    setval <<< nb, nt >>> (omg_d, zero, nn);
  }  

  if (id -> kxvEy -> write_v_time || id -> xykxvEy -> write_v_time) {
    cudaMalloc     (&vEk,        sizeof(cuComplex) * grids_->NxNycNz);
  }
  
  /*
  if (pars_->diagnosing_pzt) {
    primary = (float*) malloc (sizeof(float));    primary[0] = 0.;  
    secondary = (float*) malloc (sizeof(float));  secondary[0] = 0.;  
    tertiary = (float*) malloc (sizeof(float));    tertiary[0] = 0.;  
    cudaMalloc     (&t_bar,     sizeof(cuComplex) * nR * nS);
  }
  */
    
  // Remember that delta theta is a constant in this formalism!   

  // set up stop file
  sprintf(stopfilename_, "%s.stop", pars_->run_name);

  //  dB_scale = min(512, nR);
  //  dG_scale = 1 + (nR-1)/dB_scale.x;

  dB_spectra = dim3(min(16, nY), min(8, nX), min(8, nZ));
  dG_spectra = dim3(1 + (nY-1)/dB_spectra.x, 1 + (nX-1)/dB_spectra.y, 1 + (nZ-1)/dB_spectra.z);  

  int nyx =  nY * nX;
  int nslm = nL * nM * nS;

  int nt1 = 32;
  int nb1 = 1 + (nyx-1)/nt1;

  int nt2 = 32;
  int nb2 = 1 + (grids_->Nz-1)/nt2;
  
  dB_all = dim3(nt1, nt2, 1);
  dG_all = dim3(nb1, nb2, nslm);
  
  nt1 = min(32, grids_->Nyc);
  nb1 = 1 + (grids_->Nyc-1)/nt1;

  nt2 = min(32, grids_->Nx);
  nb2 = 1 + (grids_->Nx-1)/nt2;

  dBk = dim3(nt1, nt2, 1);
  dGk = dim3(nb1, nb2, 1);
  
  if (grids_->Nakx > 1024) {printf("Need to redefine GFLA in diagnostics \n"); exit(1);}

  nt1 = min(grids_->Ny, 512);
  nb1 = 1 + (grids_->Ny-1)/nt1;

  dbp = dim3(nt1, 1, 1);
  dgp = dim3(nb1, 1, 1);

  printf(ANSI_COLOR_RESET);
  ndiag = 1;
  Dks = 0.;
  
  if (pars_->ks) {
    cudaMalloc     (&gy_d, sizeof(float)*grids_->Ny);
    gy_h = (float*) malloc (sizeof(float)*grids_->Ny);

    if (pars_->ResWrite) {
      ry_h = (double*) malloc(sizeof(double)*pars_->ResQ*grids_->NxNyNz*grids_->Nmoms);
    }
  }
  
  if (pars_->Reservoir) {
    int nbatch = 1;
    grad_perp = new GradPerp(grids_, nbatch, grids_->Nyc);    
  }    
}

Diagnostics_GK::~Diagnostics_GK()
{
  if (fields_old) delete fields_old;
  if (id)         delete id;

  if (G2)         cudaFree      ( G2        );
  if (P2s)        cudaFree      ( P2s       );
  if (Phi2)       cudaFree      ( Phi2      );
  if (t_bar)      cudaFree      ( t_bar     );
  if (omg_d)      cudaFree      ( omg_d     );
  if (gy_d)       cudaFree      ( gy_d      );
  if (amom_d)     cudaFree      ( amom_d    );
  if (favg)       cudaFree      ( favg      );
  if (df)         cudaFree      ( df        );
  
  if (vEk)        cudaFree      ( vEk       );
  if (phi_max)    cudaFree      ( phi_max   );
  
  if (vol_fac)    cudaFree  ( vol_fac   );
  if (flux_fac)   cudaFree  ( flux_fac  );
  if (kvol_fac)   cudaFree  ( kvol_fac  );
  if (val)        free  ( val       );
  if (tmp_omg_h)  free  ( tmp_omg_h );
  if (gy_h)       free  ( gy_h      );
  if (ry_h)       free  ( ry_h      );
}

bool Diagnostics_GK::loop(MomentsG* G, Fields* fields, double dt, int counter, double time) 
{
  int retval;
  bool stop = false;
  int nw;

  if(id -> omg -> write_v_time && counter > 0) {                    // complex frequencies
    int nt = min(512, grids_->NxNyc) ;
    growthRates <<< 1 + (grids_->NxNyc-1)/nt, nt >>> (fields->phi, fields_old->phi, dt, omg_d);
    fields_old->copyPhiFrom(fields);
  }

  nw = pars_->nwrite;

  //if ((counter % nw == nw-1) && id -> omg -> write_v_time) fields_old->copyPhiFrom(fields);
    
  if(counter%nw == 0) {

    fflush(NULL);
    if (pars_->Reservoir && counter > pars_->nstep-pars_->ResPredict_Steps*pars_->ResTrainingDelta) {
      id -> write_nc(id -> time, time);
      //      if (pars_->ResWrite) id -> write_nc( id -> r_time, time);
    }
    if (!pars_->Reservoir) {
      id -> write_nc(id -> time, time);
    }

    if (pars_->write_xymom) id -> write_nc( id -> z_time, time);
    
    if(id -> omg -> write_v_time && counter > 0) {                    // complex frequencies
      print_omg(omg_d);  id -> write_omg(omg_d);
    }

    if ( id -> qs -> write_v_time) printf("Step %d: Time = %f \t Flux = ", counter, time);          // To screen
    if (!id -> qs -> write_v_time) printf("Step %d: Time = %f\n",          counter, time);
  
    if ( id -> qs -> write_v_time) {                                                                // heat flux
      
      for(int is=0; is<grids_->Nspecies; is++) {
	float rho2s = pars_->species_h[is].rho2;
	float p_s = pars_->species_h[is].nt;
	heat_flux_summand loop_R (P2(is), fields->phi, G->G(0,0,is),
				  grids_->ky, flux_fac, geo_->kperp2, rho2s, p_s);
      }
      id -> write_Q(P2s); 
    }      

    if ( id -> ps -> write_v_time) {

      for(int is=0; is<grids_->Nspecies; is++) {
	float rho2s = pars_->species_h[is].rho2;
	float n_s = pars_->species_h[is].dens;
	part_flux_summand loop_R (P2(is), fields->phi, G->G(0,0,is),
				  grids_->ky, flux_fac, geo_->kperp2, rho2s, n_s);
      }
      id -> write_P(P2s); 
    }
    
    if (pars_->diagnosing_kzspec) {
      grad_par->zft(G); // get G = G(kz)
      W_summand GALL (G2, G->G(), kvol_fac, G->nt());
      grad_par->zft_inverse(G); // restore G

      grad_par->zft(fields->phi, amom_d); // get amom_d = phi(kz)
      
      for (int is=0; is < grids_->Nspecies; is++) {             // P2(s) = (1-G0(s)) |phi**2| for each kinetic species
	float rho2s = pars_->species_h[is].rho2;
	Wphi_summand loop_R (P2(is), amom_d, kvol_fac, geo_->kperp2, rho2s);
	float qfac =  pars_->species_h[is].qneut;
	Wphi_scale loop_R   (P2(is), qfac);
      }

      if (pars_->add_Boltzmann_species) {
	if (pars_->Boltzmann_opt == BOLTZMANN_IONS)  Wphi2_summand loop_R (Phi2, amom_d, kvol_fac);
	
	if (pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS) {
	  fieldlineaverage GFLA (favg, df, fields->phi, vol_fac); // favg is a dummy variable
	  grad_par->zft(df, amom_d); // get df = df(kz)
	  Wphi2_summand loop_R (Phi2, amom_d, kvol_fac); 	
	}

	float fac = 1./pars_->tau_fac;
	Wphi_scale loop_R (Phi2, fac);
      }
      
      id -> write_Wkz(G2);    id -> write_Pkz(P2());    id -> write_Akz(Phi2);      
    }
    
    if (pars_->diagnosing_spectra) {                                        // Various spectra
      W_summand GALL (G2, G->G(), vol_fac, G->nt());
      
      if (pars_->gx) {
	for (int is=0; is < grids_->Nspecies; is++) {       // P2(s) = (1-G0(s)) |phi**2| for each kinetic species
	  float rho2s = pars_->species_h[is].rho2;
	  Wphi_summand loop_R (P2(is), fields->phi, vol_fac, geo_->kperp2, rho2s);
	  float qnfac = pars_->species_h[is].qneut;
	  Wphi_scale loop_R   (P2(is), qnfac);
	}

	if (pars_->add_Boltzmann_species) {
	  if (pars_->Boltzmann_opt == BOLTZMANN_IONS)  Wphi2_summand loop_R (Phi2, fields->phi, vol_fac);
	  
	  if (pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS) {	  
	    fieldlineaverage GFLA (favg, df, fields->phi, vol_fac); // favg is a dummy variable
	    Wphi2_summand loop_R (Phi2, df, vol_fac); 	
	  }
	  
	  float fac = 1./pars_->tau_fac;
	  Wphi_scale loop_R (Phi2, fac);
	}
      }

      if (pars_->ks) {
	cuComplex * g_h;
	g_h = (cuComplex*) malloc (sizeof(cuComplex)*grids_->Nyc);
	CP_TO_CPU(g_h, G->G(), sizeof(cuComplex)*grids_->Nyc);
	float Dtmp = 0.;
	for (int i=0; i<grids_->Naky; i++) {
	  Dtmp += (g_h[i].x*g_h[i].x + g_h[i].y*g_h[i].y)*grids_->ky_h[i]*grids_->ky_h[i];
	}
	Dks += Dtmp;
	printf("<D> = %f \t",Dks/((float) ndiag));
	ndiag += 1;
	free(g_h);
      }
      
      id->write_Wm    (G2   );    id->write_Wl    (G2   );    id->write_Wlm   (G2   );    
      id->write_Wz    (G2   );    id->write_Wky   (G2   );    id->write_Wkx   (G2   );    id->write_Wkxky (G2  );    
      id->write_Pz    (P2() );    id->write_Pky   (P2() );    id->write_Pkx   (P2() );    id->write_Pkxky (P2());    
      id->write_Az    (Phi2 );    id->write_Aky   (Phi2 );    id->write_Akx   (Phi2 );    id->write_Akxky (Phi2);
      
      // Do not change the order of these four calls because totW is accumulated in order when it is requested:
      id->write_Ps(P2s);    id->write_Ws(G2);    id->write_As(Phi2);    id->write_Wtot();
    }
    
    // Rosenbluth-Hinton diagnostic
    if(id -> rh -> write) {get_rh(fields);   id -> write_nc (id->rh, val);}
    
    /*
      if( counter%nw == 0 && id -> Pzt -> write_v_time) {
      pzt(G, fields);  // calculate each of P, Z, and T (very very rough diagnostic)
      cudaDeviceSynchronize();
      
      write_nc (id -> Pzt, primary);
      write_nc (id -> pZt, secondary);
      write_nc (id -> pzT, tertiary);
      }
    */

    // Plot ky=kz=0 components of various quantities as functions of x
    id -> write_moment ( id -> vEy,     fields->phi,    vol_fac);
    id -> write_moment ( id -> kxvEy,   fields->phi,    vol_fac);
    id -> write_moment ( id -> kden,    G->dens_ptr[0], vol_fac);
    id -> write_moment ( id -> kUpar,   G->upar_ptr[0], vol_fac);
    id -> write_moment ( id -> kTpar,   G->tpar_ptr[0], vol_fac);
    id -> write_moment ( id -> kTperp,  G->tprp_ptr[0], vol_fac);
    id -> write_moment ( id -> kqpar,   G->qpar_ptr[0], vol_fac);

    // Plot some zonal scalars
    id -> write_moment ( id -> avg_zvE,     fields->phi,    vol_fac);
    id -> write_moment ( id -> avg_zkxvEy,  fields->phi,    vol_fac);
    id -> write_moment ( id -> avg_zkden,   G->dens_ptr[0], vol_fac);
    id -> write_moment ( id -> avg_zkUpar,  G->upar_ptr[0], vol_fac);
    id -> write_moment ( id -> avg_zkTpar,  G->tpar_ptr[0], vol_fac);
    id -> write_moment ( id -> avg_zkTperp, G->tprp_ptr[0], vol_fac);
    id -> write_moment ( id -> avg_zkqpar,  G->qpar_ptr[0], vol_fac);

    // Plot f(x,y,z=0)
    id -> write_moment ( id -> xyPhi,   fields->phi,    vol_fac);
    
    // Plot the non-zonal components as functions of (x, y)
    id -> write_moment ( id -> xykxvEy, fields->phi,    vol_fac);
    id -> write_moment ( id -> xyvEy,   fields->phi,    vol_fac);
    id -> write_moment ( id -> xyvEx,   fields->phi,    vol_fac);
    id -> write_moment ( id -> xyden,   G->dens_ptr[0], vol_fac);
    id -> write_moment ( id -> xyUpar,  G->upar_ptr[0], vol_fac);
    id -> write_moment ( id -> xyTpar,  G->tpar_ptr[0], vol_fac);
    id -> write_moment ( id -> xyTperp, G->tprp_ptr[0], vol_fac);
    id -> write_moment ( id -> xyqpar,  G->qpar_ptr[0], vol_fac);

    if (pars_->Reservoir && counter > pars_->nstep-pars_->ResPredict_Steps*pars_->ResTrainingDelta) {
      if (!pars_->ResFakeData) id -> write_ks_data ( id -> g_y, G->G());
    }
    if (!pars_->Reservoir) {
      id -> write_ks_data ( id -> g_y, G->G());
    }

    nc_sync(id->file);
  }
  if (pars_->Reservoir && counter%pars_->ResTrainingDelta == 0) {
    grad_perp->C2R(G->G(), gy_d);
    if (pars_->ResFakeData) {
      rc->fake_data(gy_d);
      id -> write_ks_data( id -> g_y, gy_d);
    }
    rc->add_data(gy_d);
  }

  if (pars_->fixed_amplitude && (counter % nw == nw-2)) {
    maxPhi KXKY (phi_max, fields->phi);
    G->rescale(phi_max);
    fields->rescale(phi_max);
  }
  
  // check to see if we should stop simulation
  stop = checkstop();
  return stop;
}

void Diagnostics_GK::finish(MomentsG* G, Fields* fields, double time) 
{
  if (pars_->Reservoir && rc->predicting()) {
    if (pars_->ResFakeData) {
      rc->fake_data(gy_d);
    } else {
      grad_perp -> C2R (G->G(), gy_d);
    }
    double *gy_double;
    cudaMalloc(&gy_double, sizeof(double)*grids_->Ny);
    promote loop_y (gy_double, gy_d, grids_->Ny);
    
    for (int i=0; i<pars_->ResPredict_Steps; i++) {
      rc->predict(gy_double);
      time += pars_->dt * pars_->ResTrainingDelta;
      demote loop_y (gy_d, gy_double, grids_->Ny);
      id -> write_nc(id -> time, time);
      id -> write_ks_data (id -> g_y, gy_d);
    }
  }
}

void Diagnostics_GK::print_omg(cuComplex *W)
{
  CP_TO_CPU (tmp_omg_h, W, sizeof(cuComplex)*grids_->NxNyc);
  print_growth_rates_to_screen(tmp_omg_h);
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

void Diagnostics_GK::get_rh(Fields* f)
{
    ikx_local = 1; iky_local = 0; iz_local=grids_->Nz/2; // correct values for usual RH tests

    int idx = iky_local + grids_->Nyc*ikx_local + grids_->NxNyc*iz_local;
  
    CP_TO_CPU(&valphi, &f->phi[idx], sizeof(cuComplex));
    val[0] = valphi.x;
    val[1] = valphi.y;
}

bool Diagnostics_GK::checkstop() 
{
  struct stat buffer;   
  bool stop = (stat (stopfilename_, &buffer) == 0);
  if (stop) remove(stopfilename_);
  return stop;
}

void Diagnostics_GK::print_growth_rates_to_screen(cuComplex* w)
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


Diagnostics_KREHM::Diagnostics_KREHM(Parameters* pars, Grids* grids) :
  pars_(pars), grids_(grids),
  fields_old(nullptr), id(nullptr), grad_par(nullptr), amom_d(nullptr), grad_perp(nullptr)//, grad_phi(nullptr)
{
  printf(ANSI_COLOR_BLUE);
  
  int nL  = grids_->Nl;
  int nM  = grids_->Nm;
  int nS  = grids_->Nspecies;
  int nX  = grids_->Nx;
  int nXk = grids_->Nakx;
  int nY  = grids_->Nyc;
  int nYk = grids_->Naky;
  int nZ  = grids_->Nz;
  int nR  = nX  * nY  * nZ; // nY is *not* the number of grid points in the y-direction. 
  int nK  = nXk * nYk * nZ;
  int nG  = nR * grids_->Nmoms * nS;

  favg        = nullptr;  df          = nullptr;  val         = nullptr;  
  G2          = nullptr;  P2s         = nullptr;  Phi2        = nullptr;
  omg_d       = nullptr;  tmp_omg_h   = nullptr;  t_bar       = nullptr;  
  vEk         = nullptr;  phi_max     = nullptr;

  id         = new NetCDF_ids(grids_, pars_); cudaDeviceSynchronize(); CUDA_DEBUG("NetCDF_ids: %s \n");
  fields_old = new      Fields(pars_, grids_);      cudaDeviceSynchronize(); CUDA_DEBUG("Fields: %s \n");

  float *vol_fac_h;
  vol_fac_h = (float*) malloc (sizeof(float) * nZ);
  cudaMalloc (&vol_fac, sizeof(float) * nZ);
  for (int i=0; i < nZ; i++) vol_fac_h[i]  = 1;
  CP_TO_GPU(vol_fac, vol_fac_h, sizeof(float)*nZ);
  free(vol_fac_h);

  //if (pars_->diagnosing_kzspec) {
  //  float *kvol_fac_h;
  //  kvol_fac_h = (float*) malloc (sizeof(float) * nZ);
  //  cudaMalloc (&kvol_fac, sizeof(float) * nZ);
  //  for (int i=0; i < nZ; i++) kvol_fac_h[i] = 1.0;
  //  CP_TO_GPU(kvol_fac, kvol_fac_h, sizeof(float)*nZ);
  //  free(kvol_fac_h);

  //  cudaMalloc (&amom_d, sizeof(cuComplex) * nR * nS); 
  //  if (pars_->local_limit) {
  //    // nothing, this is not defined, or could be defined as an identity.
  //  }
  //  else if(pars_->boundary_option_periodic) {
  //    grad_par = new GradParallelPeriodic(grids_);
  //  }
  //  else {
  //    grad_par = new GradParallelLinked(grids_, pars_->jtwist);
  //  }
  //}
  // need if (pars_->write_flux || "diagnosing potential) {
  if (pars_->diagnosing_spectra || pars_->diagnosing_kzspec) cudaMalloc (&G2, sizeof(float) * nG); 
  cudaMalloc (&P2s, sizeof(float) * nR * nS);

  if (id -> omg -> write_v_time) {
    cudaMalloc     (    &omg_d,   sizeof(cuComplex) * nX * nY);//     cudaMemset (omg_d, 0., sizeof(cuComplex) * nX * nY);
    tmp_omg_h = (cuComplex*) malloc (sizeof(cuComplex) * nX * nY);
    int nn = nX*nY; int nt = min(nn, 512); int nb = 1 + (nn-1)/nt;  cuComplex zero = make_cuComplex(0.,0.);
    setval <<< nb, nt >>> (omg_d, zero, nn);
  }  

  if (id -> kxvEy -> write_v_time || id -> xykxvEy -> write_v_time) {
    cudaMalloc     (&vEk,        sizeof(cuComplex) * grids_->NxNycNz);
  }
  
  /*
  if (pars_->diagnosing_pzt) {
    primary = (float*) malloc (sizeof(float));    primary[0] = 0.;  
    secondary = (float*) malloc (sizeof(float));  secondary[0] = 0.;  
    tertiary = (float*) malloc (sizeof(float));    tertiary[0] = 0.;  
    cudaMalloc     (&t_bar,     sizeof(cuComplex) * nR * nS);
  }
  */
    
  // Remember that delta theta is a constant in this formalism!   

  // set up stop file
  sprintf(stopfilename_, "%s.stop", pars_->run_name);

  //  dB_scale = min(512, nR);
  //  dG_scale = 1 + (nR-1)/dB_scale.x;

  dB_spectra = dim3(min(16, nY), min(8, nX), min(8, nZ));
  dG_spectra = dim3(1 + (nY-1)/dB_spectra.x, 1 + (nX-1)/dB_spectra.y, 1 + (nZ-1)/dB_spectra.z);  

  int nyx =  nY * nX;
  int nslm = nL * nM * nS;

  int nt1 = 32;
  int nb1 = 1 + (nyx-1)/nt1;

  int nt2 = 32;
  int nb2 = 1 + (grids_->Nz-1)/nt2;
  
  dB_all = dim3(nt1, nt2, 1);
  dG_all = dim3(nb1, nb2, nslm);
  
  nt1 = min(32, grids_->Nyc);
  nb1 = 1 + (grids_->Nyc-1)/nt1;

  nt2 = min(32, grids_->Nx);
  nb2 = 1 + (grids_->Nx-1)/nt2;

  dBk = dim3(nt1, nt2, 1);
  dGk = dim3(nb1, nb2, 1);
  
  if (grids_->Nakx > 1024) {printf("Need to redefine GFLA in diagnostics \n"); exit(1);}

  nt1 = min(grids_->Ny, 512);
  nb1 = 1 + (grids_->Ny-1)/nt1;

  dbp = dim3(nt1, 1, 1);
  dgp = dim3(nb1, 1, 1);

  printf(ANSI_COLOR_RESET);
  ndiag = 1;
  Dks = 0.;
}

Diagnostics_KREHM::~Diagnostics_KREHM()
{
  if (fields_old) delete fields_old;
  if (id)         delete id;

  if (G2)         cudaFree      ( G2        );
  if (P2s)        cudaFree      ( P2s       );
  if (Phi2)       cudaFree      ( Phi2      );
  if (t_bar)      cudaFree      ( t_bar     );
  if (omg_d)      cudaFree      ( omg_d     );
  if (amom_d)     cudaFree      ( amom_d    );

  if (vEk)        cudaFree      ( vEk       );
  if (phi_max)    cudaFree      ( phi_max   );
  
  if (vol_fac)   cudaFree  ( vol_fac  );
  if (kvol_fac)   cudaFree  ( kvol_fac  );
  if (val)        free  ( val       );
  if (tmp_omg_h)  free  ( tmp_omg_h );
}

bool Diagnostics_KREHM::loop(MomentsG* G, Fields* fields, double dt, int counter, double time) 
{
  int retval;
  bool stop = false;
  int nw;

  nw = pars_->nwrite;

  if(counter%nw == 0) {

    fflush(NULL);
    id -> write_nc(id -> time, time);
 
    //if (pars_->write_phi) id->write_nc(id->phi, phi);

    if (pars_->write_xymom) id -> write_nc( id -> z_time, time);
    
    if(id -> omg -> write_v_time && counter > 0) {                    // complex frequencies
      int nt = min(512, grids_->NxNyc) ;
      growthRates <<< 1 + (grids_->NxNyc-1)/nt, nt >>> (fields->phi, fields_old->phi, dt*nw, omg_d);
      fields_old->copyPhiFrom(fields);
      print_omg(omg_d);  id -> write_omg(omg_d);
    }

    if (pars_->diagnosing_spectra) {                                        // Various spectra
      W_summand GALL (G2, G->G(), vol_fac, G->nt());

      Wphi_summand_krehm loop_R (P2(), fields->phi, vol_fac, grids_->kx, grids_->ky, pars_->rho_i);
      
      id->write_Wm    (G2   );    id->write_Wl    (G2   );    id->write_Wlm   (G2   );    
      id->write_Wz    (G2   );    id->write_Wky   (G2   );    id->write_Wkx   (G2   );    id->write_Wkxky (G2  );    
      id->write_Pz    (P2() );    id->write_Pky   (P2() );    id->write_Pkx   (P2() );    id->write_Pkxky (P2());    
      
      // Do not change the order of these four calls because totW is accumulated in order when it is requested:
      //id->write_Ps(P2s);    id->write_Ws(G2);   // id->write_As(Phi2);    id->write_Wtot();
    }

    nc_sync(id->file);
  }

  // check to see if we should stop simulation
  stop = checkstop();
  return stop;
}

void Diagnostics_KREHM::finish(MomentsG* G, Fields* fields, double time) 
{
}

void Diagnostics_KREHM::print_omg(cuComplex *W)
{
  CP_TO_CPU (tmp_omg_h, W, sizeof(cuComplex)*grids_->NxNyc);
  print_growth_rates_to_screen(tmp_omg_h);
}

bool Diagnostics_KREHM::checkstop() 
{
  struct stat buffer;   
  bool stop = (stat (stopfilename_, &buffer) == 0);
  if (stop) remove(stopfilename_);
  return stop;
}

void Diagnostics_KREHM::print_growth_rates_to_screen(cuComplex* w)
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

