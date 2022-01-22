#include <random>
#include <algorithm>
#include <vector>
#include "moments.h"
#define GALL <<< dG_all, dB_all >>>

MomentsG::MomentsG(Parameters* pars, Grids* grids) : 
  grids_(grids), pars_(pars)
{
  G_lm       = nullptr;  dens_ptr   = nullptr;  upar_ptr   = nullptr;  tpar_ptr   = nullptr;
  tprp_ptr   = nullptr;  qpar_ptr   = nullptr;  qprp_ptr   = nullptr;

  size_t lhsize = grids_->size_G;
  // printf("nspecies = %d and size_G = %d \n",grids_->Nspecies, (int) grids_->size_G);
  checkCuda(cudaMalloc((void**) &G_lm, lhsize)); 
  checkCuda(cudaMemset(G_lm, 0., lhsize));

  float * vts_h = (float*) malloc(sizeof(float) * grids_->Nspecies );
  float * tzs_h = (float*) malloc(sizeof(float) * grids_->Nspecies );
  float * zts_h = (float*) malloc(sizeof(float) * grids_->Nspecies );
  float * nts_h = (float*) malloc(sizeof(float) * grids_->Nspecies );  
  float * nzs_h = (float*) malloc(sizeof(float) * grids_->Nspecies );
  float * r2s_h = (float*) malloc(sizeof(float) * grids_->Nspecies );
  float * tps_h = (float*) malloc(sizeof(float) * grids_->Nspecies );
  float * fps_h = (float*) malloc(sizeof(float) * grids_->Nspecies );
  float * ups_h = (float*) malloc(sizeof(float) * grids_->Nspecies );
  float * aps_h = (float*) malloc(sizeof(float) * grids_->Nspecies );
  float * qns_h = (float*) malloc(sizeof(float) * grids_->Nspecies );
  float * nu_ss_h = (float*) malloc(sizeof(float) * grids_->Nspecies );
  int   * typ_h = (int*) malloc(sizeof(int) * grids_->Nspecies );
  
  for (int is=0; is<grids_->Nspecies; is++) {
    vts_h[is] = pars_->species_h[is].vt;
    tzs_h[is] = pars_->species_h[is].tz;
    zts_h[is] = pars_->species_h[is].zt;
    nts_h[is] = pars_->species_h[is].nt;
    nzs_h[is] = pars_->species_h[is].nz;
    r2s_h[is] = pars_->species_h[is].rho2; 
    tps_h[is] = pars_->species_h[is].tprim;
    fps_h[is] = pars_->species_h[is].fprim;     
    ups_h[is] = pars_->species_h[is].uprim;
    aps_h[is] = pars_->species_h[is].as;
    qns_h[is] = pars_->species_h[is].qneut;
    nu_ss_h[is] = pars_->species_h[is].nu_ss;
    typ_h[is] = pars_->species_h[is].type;
  }    

  checkCuda(cudaMalloc( &vts,   sizeof(float) * grids_->Nspecies ) );
  checkCuda(cudaMalloc( &zts,   sizeof(float) * grids_->Nspecies ) );
  checkCuda(cudaMalloc( &tzs,   sizeof(float) * grids_->Nspecies ) );
  checkCuda(cudaMalloc( &nts,   sizeof(float) * grids_->Nspecies ) );
  checkCuda(cudaMalloc( &nzs,   sizeof(float) * grids_->Nspecies ) );
  checkCuda(cudaMalloc( &aps,   sizeof(float) * grids_->Nspecies ) );
  checkCuda(cudaMalloc( &r2s,   sizeof(float) * grids_->Nspecies ) );
  checkCuda(cudaMalloc( &qns,   sizeof(float) * grids_->Nspecies ) );
  checkCuda(cudaMalloc( &tps,   sizeof(float) * grids_->Nspecies ) );
  checkCuda(cudaMalloc( &fps,   sizeof(float) * grids_->Nspecies ) );
  checkCuda(cudaMalloc( &ups,   sizeof(float) * grids_->Nspecies ) );
  checkCuda(cudaMalloc( &nu_ss, sizeof(float) * grids_->Nspecies ) );
  checkCuda(cudaMalloc( &typ,   sizeof(int)   * grids_->Nspecies ) );

  CP_TO_GPU(vts, vts_h, sizeof(float)*grids_->Nspecies);
  CP_TO_GPU(tzs, tzs_h, sizeof(float)*grids_->Nspecies);
  CP_TO_GPU(zts, zts_h, sizeof(float)*grids_->Nspecies);
  CP_TO_GPU(nts, nts_h, sizeof(float)*grids_->Nspecies);
  CP_TO_GPU(nzs, nzs_h, sizeof(float)*grids_->Nspecies);
  CP_TO_GPU(r2s, r2s_h, sizeof(float)*grids_->Nspecies);
  CP_TO_GPU(tps, tps_h, sizeof(float)*grids_->Nspecies);
  CP_TO_GPU(fps, fps_h, sizeof(float)*grids_->Nspecies);
  CP_TO_GPU(ups, ups_h, sizeof(float)*grids_->Nspecies);
  CP_TO_GPU(aps, aps_h, sizeof(float)*grids_->Nspecies);
  CP_TO_GPU(qns, qns_h, sizeof(float)*grids_->Nspecies);
  CP_TO_GPU(nu_ss, nu_ss_h, sizeof(float)*grids_->Nspecies);
  CP_TO_GPU(typ, typ_h, sizeof(int)  *grids_->Nspecies);
  
  free(vts_h);  free(tzs_h);  free(zts_h);  free(nts_h);
  free(nzs_h);  free(r2s_h);  free(tps_h);  free(fps_h);
  free(ups_h);  free(aps_h);  free(qns_h);  free(nu_ss_h);  
  free(typ_h);
  
  dens_ptr = (cuComplex**) malloc(sizeof(cuComplex*) * grids_->Nspecies);
  upar_ptr = (cuComplex**) malloc(sizeof(cuComplex*) * grids_->Nspecies);
  tpar_ptr = (cuComplex**) malloc(sizeof(cuComplex*) * grids_->Nspecies);
  tprp_ptr = (cuComplex**) malloc(sizeof(cuComplex*) * grids_->Nspecies);
  qpar_ptr = (cuComplex**) malloc(sizeof(cuComplex*) * grids_->Nspecies);
  qprp_ptr = (cuComplex**) malloc(sizeof(cuComplex*) * grids_->Nspecies);

  printf("Allocated a G_lm array of size %.2f MB\n", lhsize/1024./1024.);

  int Nm = grids_->Nm;
  int Nl = grids_->Nl;

  for(int s=0; s<grids->Nspecies; s++) {
    // set up pointers for named moments that point to parts of G_lm
    int l,m;
    l = 0, m = 0; // density
    if(l<Nl && m<Nm) dens_ptr[s] = G(l,m,s);
    
    l = 0, m = 1; // u_parallel
    if(l<Nl && m<Nm) upar_ptr[s] = G(l,m,s);
    
    l = 0, m = 2; // T_parallel / sqrt(2)
    if(l<Nl && m<Nm) tpar_ptr[s] = G(l,m,s);
    
    l = 0, m = 3; // q_parallel / sqrt(6)
    if(l<Nl && m<Nm) qpar_ptr[s] = G(l,m,s);

    l = 1, m = 0; // T_perp 
    if(l<Nl && m<Nm) tprp_ptr[s] = G(l,m,s);
    
    l = 1, m = 1; // q_perp
    if(l<Nl && m<Nm) qprp_ptr[s] = G(l,m,s);
  }

  int nn1, nn2, nn3, nt1, nt2, nt3, nb1, nb2, nb3;
  
  if (pars_->ks) {

    printf("initializing Kuramoto-Sivashinsky\n");
    nn1 = grids_->Nyc;                 nt1 = min(nn1, 128);    nb1 = 1 + (nn1-1)/nt1;
    nn2 = 1;                           nt2 = min(nn2,   1);    nb2 = 1 + (nn2-1)/nt2;
    nn3 = 1;                           nt3 = min(nn3,   1);    nb3 = 1 + (nn3-1)/nt3;
    
    dB_all   = dim3(nt1, nt2, nt3);    dG_all   = dim3(nb1, nb2, nb3);
    dimBlock = dim3(nt1, nt2, nt3);    dimGrid  = dim3(nb1, nb2, nb3);
    return;
  } 

  if (pars_->vp) {
    printf("initializing Vlasov-Poisson\n");
    nn1 = grids_->Nyc;                 nt1 = min(nn1, 128);    nb1 = 1 + (nn1-1)/nt1;
    nn2 = 1;                           nt2 = min(nn2,   1);    nb2 = 1 + (nn2-1)/nt2;
    nn3 = 1;                           nt3 = min(nn3,   1);    nb3 = 1 + (nn3-1)/nt3;
    
    dB_all   = dim3(nt1, nt2, nt3);
    dG_all   = dim3(nb1, nb2, nb3);
    return;
  }
  
  //    nn1 = grids_->NxNycNz;      nt1 = min(32, nn1);  nb1 = 1 + (nn1-1)/nt1;
  //    nn2 = 1;                    nt2 = min( 4, Nl);   nb2 = 1 + (nn2-1)/nt2;
  //    nn3 = 1;                    nt3 = min( 4, Nm);   nb3 = 1 + (nn3-1)/nt3;
  
  //    dimBlock = dim3(nt1, nt2, nt3);
  //    dimGrid  = dim3(nb1, nb2, nb3);
  
  //    dimBlock = dim3(32, min(4, Nl), min(4, Nm));
  //    dimGrid  = dim3((grids_->NxNycNz-1)/dimBlock.x+1, 1, 1);
  
  nn1 = grids_->Nyc*grids_->Nx;                   nt1 = min(nn1, 32);    nb1 = (nn1-1)/nt1 + 1;
  nn2 = grids_->Nz;                               nt2 = min(nn2, 32);    nb2 = (nn2-1)/nt2 + 1;
  nn3 = grids_->Nspecies*grids_->Nm*grids_->Nl;   nt3 = min(nn3,  1);    nb3 = (nn3-1)/nt3 + 1;
  
  dB_all = dim3(nt1, nt2, nt3);
  dG_all = dim3(nb1, nb2, nb3);	 
}

MomentsG::~MomentsG() {
  free (dens_ptr);
  free (upar_ptr);
  free (tpar_ptr);
  free (qpar_ptr);
  free (qprp_ptr);

  if ( G_lm     ) cudaFree ( G_lm );
}

void MomentsG::set_zero(void) {
  cudaMemset(G_lm, 0., grids_->size_G);
}

void MomentsG::initVP(double *time) {

  cuComplex *init_h = nullptr;
  init_h = (cuComplex*) malloc(sizeof(cuComplex)*grids_->Nyc*grids_->Nm);

  for (int ig = 0; ig<grids_->Nyc*grids_->Nm; ig++) {
    init_h[ig].x = 0.;
    init_h[ig].y = 0.;
  }
  
  // start with something simple:

  if (!pars_->restart) init_h[0].x = 1.; // This is the Maxwellian background
  init_h[1 + grids_->Nyc * 2].x = pars_->init_amp; // This is a temperature perturbation (up to a factor of sqrt(2)).
  
  CP_TO_GPU(G_lm, init_h, sizeof(cuComplex)*grids_->Nyc*grids_->Nm);
  free(init_h);

  if (pars_->restart) this->restart_read(time);

  cudaDeviceSynchronize();
}

//void MomentsG::initialConditions(double *time) {
//
//  size_t momsize = sizeof(cuComplex)*grids_->NxNycNz;
//  cuComplex *init_h = nullptr;
//  init_h = (cuComplex*) malloc(momsize);
//  
//  std::random_device rd;
//  std::mt19937 gen(rd());
//  std::normal_distribution<float> ramp(0., pars_->init_amp);
//
//  for (int idy = 0; idy<grids_->Nyc; idy++) {
//    init_h[idy].x = 0.;
//    init_h[idy].y = 0.;
//  }
//  
//  for (int idy = 1; idy<grids_->Naky; idy++) {
//    init_h[idy].x = ramp(gen);
//    init_h[idy].y = ramp(gen);
//  }
//
//  //  init_h[1].x =  0.5;
//  //  init_h[2].y = -0.25;
//  
//  CP_TO_GPU(G_lm, init_h, momsize);
//  
//  free(init_h);
//
//  // restart_read goes here, if restart == T
//  // as in gs2, if restart_read is true, we want to *add* the restart values to anything
//  // that has happened above and also move the value of time up to the end of the previous run
//  if(pars_->restart) {
//    DEBUG_PRINT("reading restart file \n");
//    this->restart_read(time);
//  }
//
//  cudaDeviceSynchronize();
//  //  checkCuda(cudaGetLastError());
//
//  //  return cudaGetLastError();
//  
//}

void MomentsG::initialConditions(double* time) {
 
  checkCuda(cudaGetLastError());
  cudaDeviceSynchronize(); // to make sure its safe to operate on host memory

  float *z_h = grids_->z_h;

  size_t momsize = sizeof(cuComplex)*grids_->NxNycNz;
  cuComplex *init_h = nullptr;
  init_h = (cuComplex*) malloc(momsize);

  for (int idx=0; idx<grids_->NxNycNz; idx++) {
    init_h[idx].x = 0.;
    init_h[idx].y = 0.;
  }
  
  if (pars_->ks) {
    init_h[1].x =  0.5;
    init_h[2].y = -0.25;
  } else {
  
    if(pars_->init_single) {
      //initialize single mode
      int iky = pars_->iky_single;
      int ikx = pars_->ikx_single;
      int NKX = 1;
      if (iky == 0 && ikx<1+(grids_->Nx-1)/3) NKX = 2; // reality condition for tertiary tests
      for (int j = 0; j<NKX; j++) {
	if (j==1) ikx = grids_->Nx-ikx;
	DEBUG_PRINT("ikx, iky: %d \t %d \n",ikx, iky);
	//    float fac;
	//    if(pars_->nlpm_test && iky==0) fac = .5;
	//    else fac = 1.;
	//    DEBUG_PRINT("fac = %f \n",fac);
	for(int iz=0; iz<grids_->Nz; iz++) {
	  int index = iky + grids_->Nyc*ikx + grids_->NxNyc*iz;
	  init_h[index].x = pars_->init_amp*cos(pars_->kpar_init*z_h[iz]/pars_->Zp); //*fac;
	  init_h[index].y = pars_->init_amp*cos(pars_->kpar_init*z_h[iz]/pars_->Zp); //*fac;
	  //init_h[index].y = 0.; //init_amp;
	      	    printf("init_h[%d] = (%e, %e) \n",index,init_h[index].x,init_h[index].y);
	}
      }
    } else {
      srand(22);
      float samp;
      int idx;
      //      printf("Hacking the initial condition! \n");
      for(int i=0; i < 1 + (grids_->Nx - 1)/3; i++) {
	for(int j=1; j < 1 + (grids_->Ny - 1)/3; j++) {
	  samp = pars_->init_amp;
	  float ra = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
	  float rb = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
	  for (int js=0; js < 2; js++) {
	    if (i==0) {
	      idx = i;
	    } else {
	      idx = (js==0) ? i : grids_->Nx-i;
	    }
	    for(int k=0; k<grids_->Nz; k++) {
	      int index = j + grids_->Nyc*(idx + grids_->Nx*k);
	      if (js == 0) {
		init_h[index].x = ra;		init_h[index].y = rb;
	      } else {
		init_h[index].x = rb;		init_h[index].y = ra;
	      }
	      if (pars_->kpar_init < 0.) {		
		init_h[index].x *= (cos( -pars_->kpar_init    *z_h[k]/pars_->Zp)
				  + cos((-pars_->kpar_init+1.)*z_h[k]/pars_->Zp));
		init_h[index].y *= (cos( -pars_->kpar_init    *z_h[k]/pars_->Zp)
				  + cos((-pars_->kpar_init+1.)*z_h[k]/pars_->Zp));
	      } else {
		init_h[index].x *= cos(pars_->kpar_init*z_h[k]/pars_->Zp);
		init_h[index].y *= cos(pars_->kpar_init*z_h[k]/pars_->Zp);
	      }
	      	    printf("init_h[%d] = (%e, %e) \n",index,init_h[index].x,init_h[index].y);
	    }
	  }
	  if (pars_->random_init) {
	    for (int k=0; k<grids_->Nz; k++) {
	      int index = j + grids_->Nyc*(idx + grids_->Nx*k);
	      init_h[index].x = 0.;
	      init_h[index].y = 0.;
	    }
	    for (int jj=1; jj<1+(grids_->Nz-1)/3; jj++) {
	      float ka = (float) (samp * rand() / RAND_MAX);
	      float pa = (float) (M_PI * (rand()-RAND_MAX/2) / RAND_MAX);
	      float kb = (float) (samp * rand() / RAND_MAX);
	      float pb = (float) (M_PI * (rand()-RAND_MAX/2) / RAND_MAX);
	      for (int k=0; k<grids_->Nz; k++) {
		int index = j + grids_->Nyc*(idx + grids_->Nx*k);
		
		init_h[index].x += ka*sin((float) jj*z_h[k] + pa);
		init_h[index].y += kb*sin((float) jj*z_h[k] + pb);
	      }
	    }
	  }
	}
      }
    }
  }
  
  // copy initial condition into device memory
  for (int is=0; is<grids_->Nspecies; is++) {
    switch (pars_->initf)
      {
      case inits::density : CP_TO_GPU(dens_ptr[is], init_h, momsize); break;
      case inits::upar    : CP_TO_GPU(upar_ptr[is], init_h, momsize); break;
      case inits::tpar    : CP_TO_GPU(tpar_ptr[is], init_h, momsize); break;
      case inits::tperp   : CP_TO_GPU(tprp_ptr[is], init_h, momsize); break; 
      case inits::qpar    : CP_TO_GPU(qpar_ptr[is], init_h, momsize); break;
      case inits::qperp   : CP_TO_GPU(qprp_ptr[is], init_h, momsize); break;
      }
    checkCuda(cudaGetLastError());    
  }
  free(init_h);     
  // restart_read goes here, if restart == T
  // as in gs2, if restart_read is true, we want to *add* the restart values to anything
  // that has happened above and also move the value of time up to the end of the previous run
  if(pars_->restart) {
    DEBUG_PRINT("reading restart file \n");
    this->restart_read(time);
  }
  cudaDeviceSynchronize();  checkCuda(cudaGetLastError());
  DEBUG_PRINT("initial conditions set \n");  
}

void MomentsG::scale(double    scalar) {scale_kernel GALL (G_lm, scalar);}
void MomentsG::scale(cuComplex scalar) {scale_kernel GALL (G_lm, scalar);}
void MomentsG::mask(void) {maskG GALL (this->G_lm);}

void MomentsG::getH(cuComplex* J0phi) {Hkernel GALL (G_lm, J0phi);}
void MomentsG::getG(cuComplex* J0phi) {Gkernel GALL (G_lm, J0phi);}

void MomentsG::rescale(float * phi_max) {
  rescale_kernel GALL (G_lm, phi_max, grids_->Nspecies*grids_->Nm*grids_->Nl);
}

void MomentsG::add_scaled(double c1, MomentsG* G1,
			  double c2, MomentsG* G2) {
  bool neqfix = !pars_->eqfix;
  add_scaled_kernel GALL (G_lm, c1, G1->G_lm, c2, G2->G_lm, neqfix);
}

void MomentsG::add_scaled(double c1, MomentsG* G1,
			  double c2, MomentsG* G2,
			  double c3, MomentsG* G3) {
  bool neqfix = !pars_->eqfix;
  add_scaled_kernel GALL (G_lm, c1, G1->G_lm, c2, G2->G_lm, c3, G3->G_lm, neqfix);
}

void MomentsG::add_scaled(double c1, MomentsG* G1,
			  double c2, MomentsG* G2,
			  double c3, MomentsG* G3,
			  double c4, MomentsG* G4) {
  bool neqfix = !pars_->eqfix;
  add_scaled_kernel GALL (G_lm, c1, G1->G_lm, c2, G2->G_lm, c3, G3->G_lm, c4, G4->G_lm, neqfix);
}

void MomentsG::add_scaled(double c1, MomentsG* G1,
			  double c2, MomentsG* G2, 
			  double c3, MomentsG* G3,
			  double c4, MomentsG* G4,
			  double c5, MomentsG* G5)
{
  bool neqfix = !pars_->eqfix;
  add_scaled_kernel GALL (G_lm, c1, G1->G_lm, c2, G2->G_lm, c3, G3->G_lm, c4, G4->G_lm, c5, G5->G_lm, neqfix);
}

void MomentsG::reality(int ngz) 
{
  dim3 dB;
  dim3 dG;

  int ngx = (grids_->Nx-1)/3 + 1;
  
  dB.x = 32;
  dG.x = (ngx-1)/dB.x + 1;
  
  int ngy = grids_->Nz;

  dB.y = 8;
  dG.y = (ngy-1)/dB.y + 1;
  
  dB.z = 4;
  dG.z = (ngz-1)/dB.z + 1;

  reality_kernel <<< dG, dB >>> (G_lm, ngz);
}

void MomentsG::restart_write(double* time)
{
  float* G_out;
  cuComplex* G_h;

  int retval;
  int ncres;
  int moments_out[7];
  size_t start[7];
  size_t count[7];
  
  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;
  int Nz   = grids_->Nz;
  int nspec = pars_->nspec;
  int Nm   = grids_->Nm;
  int Nl   = grids_->Nl;

  // handles
  int id_ri, id_nz, id_Nkx, id_Nky;
  int id_nh, id_nl, id_ns;
  int id_G, id_time;

  char strb[512];
  strcpy(strb, pars_->restart_to_file.c_str());

  //  if(pars_->restart) {
  // ultimately, appending to an existing file
  // if appending, are the time values consistent?
  // inquire/define the variable names
  //  } else {
  int ri=2;
  if (retval = nc_create(strb, NC_CLOBBER, &ncres)) ERR(retval);

  if (retval = nc_def_dim(ncres, "ri",  ri,    &id_ri)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nz",  Nz,    &id_nz)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nkx", Nakx,  &id_Nkx)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nky", Naky,  &id_Nky)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nl",  Nl,    &id_nl)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nm",  Nm,    &id_nh)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Ns",  nspec, &id_ns)) ERR(retval);

  moments_out[0] = id_ns;  count[0] = nspec;
  moments_out[1] = id_nh;  count[1] = Nm;
  moments_out[2] = id_nl;  count[2] = Nl;
  moments_out[3] = id_nz;  count[3] = Nz;  
  moments_out[4] = id_Nkx; count[4] = Nakx;
  moments_out[5] = id_Nky; count[5] = Naky;
  moments_out[6] = id_ri;  count[6] = ri;

  start[0] = 0; start[1] = 0; start[2] = 0; start[3] = 0; start[4] = 0; start[5] = 0; start[6] = 0;
  if (retval = nc_def_var(ncres, "G",    NC_FLOAT, 7, moments_out, &id_G)) ERR(retval);
  if (retval = nc_def_var(ncres, "time", NC_DOUBLE, 0, 0, &id_time)) ERR(retval);
  if (retval = nc_enddef(ncres)) ERR(retval);

  if (retval = nc_put_var(ncres, id_time, time)) ERR(retval);
  
  unsigned int itot, jtot;
  jtot = Nx   * Nyc  * Nz * Nm * Nl * nspec;
  itot = Nakx * Naky * Nz * Nm * Nl * nspec;
  G_h = (cuComplex*) malloc(sizeof(cuComplex) * jtot); 
  G_out = (float*) malloc(sizeof(float) * itot * 2);

  for (unsigned int index=0; index <   jtot; index++) {G_h[index].x = 0.; G_h[index].y = 0.;}
  for (unsigned int index=0; index < 2*itot; index++) G_out[index] = 0.;
  
  CP_TO_CPU(G_h, G_lm, sizeof(cuComplex)*jtot);
  
  for (int is=0; is < nspec; is++) {
    for (int m=0; m < Nm; m++) {
      for (int l=0; l < Nl; l++) {
	for (int k=0; k < Nz; k++) {	  
	  for (int i=0; i < 1 + (Nx-1)/3; i++) {
	    for (int j=0; j < Naky; j++) {
	      unsigned int index     = j + Nyc *(i + Nx  *(k + Nz*(l + Nl*(m + Nm*is))));
	      unsigned int index_out = j + Naky*(i + Nakx*(k + Nz*(l + Nl*(m + Nm*is))));
	      G_out[2*index_out]   = G_h[index].x; 
	      G_out[2*index_out+1] = G_h[index].y;
	    }
	  }
	  
	  for (int i=2*Nx/3+1; i < Nx; i++) {
	    for (int j=0; j < Naky; j++) {
	      int it = i-2*Nx/3+(Nx-1)/3; // not very clear, depends on arcane integer math rules
	      unsigned int index     = j + Nyc *(i  + Nx  *(k + Nz*(l + Nl*(m + Nm*is))));
	      unsigned int index_out = j + Naky*(it + Nakx*(k + Nz*(l + Nl*(m + Nm*is))));
	      G_out[2*index_out]   = G_h[index].x;
	      G_out[2*index_out+1] = G_h[index].y;
	    }
	  }
	}
      }
    }
  }

  if (retval = nc_put_vara(ncres, id_G, start, count, G_out)) ERR(retval);
      
  free(G_out);
  free(G_h);

  if (retval = nc_close(ncres)) ERR(retval);
}

void MomentsG::restart_read(double* time)
{
  float scale;
  float* G_in;
  cuComplex* G_h;
  cuComplex* G_hold;

  int retval;
  int ncres;
  
  size_t lhsize = grids_->size_G;
  size_t ldum;
  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Ny   = grids_->Ny;
  int Nyc  = grids_->Nyc;
  int Nz   = grids_->Nz;
  int nspec = pars_->nspec;
  int Nm   = grids_->Nm;
  int Nl   = grids_->Nl;
  
  // handles
  int id_nz, id_Nkx, id_Nky;
  int id_nh, id_nl, id_ns;
  int id_G, id_time;

  char stra[NC_MAX_NAME+1];
  char strb[512];
  strcpy(strb, pars_->restart_from_file.c_str());

  if (retval = nc_open(strb, NC_NOWRITE, &ncres)) { printf("file: %s \n",strb); ERR(retval);}
  
  if (retval = nc_inq_dimid(ncres, "Nkx",  &id_Nkx))  ERR(retval);
  if (retval = nc_inq_dimid(ncres, "Nky",  &id_Nky))  ERR(retval);    
  
  if (retval = nc_inq_dimid(ncres, "Nz",   &id_nz))   ERR(retval);
  if (retval = nc_inq_dimid(ncres, "Nl",   &id_nl))   ERR(retval);
  if (retval = nc_inq_dimid(ncres, "Nm",   &id_nh))   ERR(retval);
  if (retval = nc_inq_dimid(ncres, "Ns",   &id_ns))   ERR(retval);
  if (retval = nc_inq_varid(ncres, "G",    &id_G))    ERR(retval);
  if (retval = nc_inq_varid(ncres, "time", &id_time)) ERR(retval);
  
  if (retval = nc_inq_dim(ncres, id_ns, stra, &ldum))  ERR(retval);
  if (nspec-pars_->ns_add != (int) ldum) {
    printf("Cannot restart because of nspec mismatch: %d \t %zu \n", nspec, ldum);
    exit (1);
  }

  if (retval = nc_inq_dim(ncres, id_nh, stra, &ldum))  ERR(retval);
  if (Nm-pars_->nm_add != (int) ldum) {
    printf("Cannot restart because of Nm mismatch: %d \t %zu \n", Nm, ldum);
    exit (1);
  }

  if (retval = nc_inq_dim(ncres, id_nl, stra, &ldum))  ERR(retval);
  if (Nl-pars_->nl_add != (int) ldum) {
    printf("Cannot restart because of Nl mismatch: %d \t %zu \n", Nl, ldum);
    exit (1);
  }

  if (retval = nc_inq_dim(ncres, id_nz, stra, &ldum))  ERR(retval);
  if (Nz != (int) ldum*pars_->ntheta_mult) {
    printf("Cannot restart because of nz mismatch: %d \t %zu \n", Nz, ldum*pars_->ntheta_mult);
    exit (1);
  }
  
  if (retval = nc_inq_dim(ncres, id_Nkx, stra, &ldum))  ERR(retval);
  if (1 + 2*((Nx/pars_->nx_mult-1)/3) != (int) ldum) {
    printf("Cannot restart because of Nkx mismatch: %d \t %zu \n", Nakx, ldum);
    exit (1);
  }
  
  if (retval = nc_inq_dim(ncres, id_Nky, stra, &ldum))  ERR(retval);
  if (1 + (Ny/pars_->ny_mult-1)/3 != (int) ldum) {
    printf("Cannot restart because of Nky mismatch: %d \t %zu \n", Naky, ldum);
    exit (1);
  }
  
  unsigned int itot;
  //  itot = Nakx * Naky * Nz * Nm * Nl * nspec;
  itot = Nx * Nyc * Nz * Nm * Nl * nspec;

  unsigned int iitot = Nakx * Naky * Nz * Nm * Nl * nspec;
  if (pars_->domain_change) {
    int old_Nakx = 1 + 2 * ((Nx/pars_->nx_mult - 1)/3);
    int old_Naky = 1 +     ((Ny/pars_->ny_mult - 1)/3);
    int old_Nz = Nz/pars_->ntheta_mult;
    int old_Nl = Nl - pars_->nl_add;
    int old_Nm = Nm - pars_->nm_add;
    int old_ns = nspec - pars_->ns_add;
    iitot = old_Nakx * old_Naky * old_Nz * old_Nm * old_Nl * old_ns;
  }
  G_hold = (cuComplex*) malloc(lhsize);
  G_h = (cuComplex*) malloc(lhsize);
  G_in = (float*) malloc(sizeof(float) * iitot * 2);
  
  for (unsigned int index=0; index < itot;  index++) {G_hold[index].x = 0.; G_hold[index].y = 0.;}
  for (unsigned int index=0; index < itot;  index++) {G_h[index].x = 0.; G_h[index].y = 0.;}
  for (unsigned int index=0; index<2*iitot; index++) {G_in[index] = 0.;}
  CP_TO_CPU(G_hold, G_lm, sizeof(cuComplex)*itot);
  
  if (retval = nc_get_var(ncres, id_G, G_in)) ERR(retval);
  if (retval = nc_get_var(ncres, id_time, time)) ERR(retval);
  if (retval = nc_close(ncres)) ERR(retval);

  scale = pars_->scale;

  if (!pars_->domain_change) {
    for (int is=0; is < nspec; is++) {
      for (int m=0; m < Nm; m++) {
	for (int l=0; l < Nl; l++) {
	  for (int k=0; k < Nz; k++) {
	    for (int i=0; i < 1 + (Nx-1)/3; i++) {
	      for (int j=0; j < Naky; j++) {
		unsigned int index    = j + Nyc *(i + Nx  *(k + Nz*(l + Nl*(m + Nm*is))));
		unsigned int index_in = j + Naky*(i + Nakx*(k + Nz*(l + Nl*(m + Nm*is))));
		G_h[index].x = scale * G_in[2*index_in]   + G_hold[index].x;
		G_h[index].y = scale * G_in[2*index_in+1] + G_hold[index].y;
	      }
	    }
	    
	    for (int i=2*Nx/3+1; i < Nx; i++) {
	      for (int j=0; j < Naky; j++) {
		int it = i-2*Nx/3+(Nx-1)/3; // not very clear, depends on arcane integer math rules
		unsigned int index    = j + Nyc *(i  + Nx  *(k + Nz*(l + Nl*(m + Nm*is))));
		unsigned int index_in = j + Naky*(it + Nakx*(k + Nz*(l + Nl*(m + Nm*is))));
		G_h[index].x = scale * G_in[2*index_in]   + G_hold[index].x;
		G_h[index].y = scale * G_in[2*index_in+1] + G_hold[index].y;
	      }
	    }
	  }
	}
      }
    }
  } else {
    int old_Naky = 1 +    (Ny/pars_->ny_mult - 1)/3;    int jj; 
    int old_Nakx = 1 + 2*((Nx/pars_->nx_mult - 1)/3);   int ii; 
    int old_Nx = Nx/pars_->nx_mult;
    int old_Nz = Nz/pars_->ntheta_mult; // not yet implemented
    int old_Nm = Nm - pars_->nm_add;
    int old_Nl = Nl - pars_->nl_add;
    int old_ns = nspec - pars_->ns_add;
    
    for (int is=0; is < min(old_ns, nspec); is++) {
      for (int m=0; m < min(old_Nm, Nm); m++) {
	for (int l=0; l < min(old_Nl, Nl); l++) {
	  for (int k=0; k < Nz; k++) {
	    
	    for (int i=0; i < 1 + old_Nakx/2; i++) {
	      ii = i * pars_->x0_mult;
	      if (ii < 1 + Nakx/2) {
		
		for (int j=0; j < old_Naky; j++) {
		  jj = j * pars_->y0_mult;
		  if (jj < Naky) {
		    
		    unsigned int index    = jj +     Nyc *(ii +     Nx  *(k + Nz*(l +     Nl*(m +     Nm*is))));
		    unsigned int index_in = j  + old_Naky*(i  + old_Nakx*(k + Nz*(l + old_Nl*(m + old_Nm*is))));
		    
		    G_h[index].x = scale * G_in[2*index_in]   + G_hold[index].x;
		    G_h[index].y = scale * G_in[2*index_in+1] + G_hold[index].y;
		    
		  }
		}
	      }
	    }
	    
	    for (int i=2*old_Nx/3+1; i < old_Nx; i++) {
	      ii =(i-old_Nx) * pars_->x0_mult + Nx;
	      if ((i-old_Nx) * pars_->x0_mult + 1 + Nakx/2 > 0) {
		
		for (int j=0; j < old_Naky; j++) {
		  jj = j * pars_->y0_mult;
		  if (jj < Naky) {
		    
		    int it = i-2*old_Nx/3+(old_Nx-1)/3; // not very clear, depends on arcane integer math rules
		    
		    unsigned int index    = jj +     Nyc *(ii +     Nx  *(k + Nz*(l +     Nl*(m +     Nm*is))));
		    unsigned int index_in = j  + old_Naky*(it + old_Nakx*(k + Nz*(l + old_Nl*(m + old_Nm*is))));
		    G_h[index].x = scale * G_in[2*index_in]   + G_hold[index].x;
		    G_h[index].y = scale * G_in[2*index_in+1] + G_hold[index].y;
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  free(G_in);
  free(G_hold);
  
  unsigned int jtot = Nx * Nyc * Nz * Nm * Nl * nspec;
  CP_TO_GPU(G_lm, G_h, sizeof(cuComplex)*jtot);
  
  free(G_h);
}

void MomentsG::qvar(int N)
{
  cuComplex* G_h;
  //  int Nk = grids_->Nyc;
  //  Nk = 1;
  int Nk = grids_->NxNycNz;
  G_h = (cuComplex*) malloc (sizeof(cuComplex)*N);
  for (int i=0; i<N; i++) {G_h[i].x = 0.; G_h[i].y = 0.;}
  CP_TO_CPU (G_h, G_lm, N*sizeof(cuComplex));
  printf("\n");
  // for (int i=0; i<N; i++) printf("var(%d,%d) = (%e, %e) \n", i%Nk, i/Nk, G_h[i].x, G_h[i].y);
  //  for (int i=N-20; i<N; i++) printf("var(%d) = (%e, %e) \n", i, G_h[i].x, G_h[i].y);
  for (int i=0; i<N; i++) printf("m var(%d,%d) = (%e, %e) \n", i%Nk, i/Nk, G_h[i].x, G_h[i].y);
  printf("\n");

  free (G_h);
}

void MomentsG::update_tprim(double time) {

  // this is a proof-of-principle hack. typically nothing will happen here
  
  // for one species (or the first species in the species list):
  // adjust tprim according to the function 
  // if t < t0:
  // tprim = tprim_0
  // if t > t0: 
  //    if (t < tf) tprim = tprim_0 + (tprim_0 - tprim_f)/(t0-tf)*(t-t0)
  //    else tprim = tprim_f

  if (pars_->tp_t0 > -0.5) {
    if (time < (double) pars_->tp_t0) {
      float tp = pars_->tprim0;
      CP_TO_GPU (tps, &tp, sizeof(float));
    } else {
      if (time < (double) pars_->tp_tf) {
	float tfac = (float) time;
	float tprim0 = pars_->tprim0;
	float tprimf = pars_->tprimf;
	float t0 = pars_->tp_t0;
	float tf = pars_->tp_tf;
	float tp = tprim0 + (tprim0-tprimf)/(t0-tf)*(tfac-t0);
	CP_TO_GPU (tps, &tp, sizeof(float));
      } else {
	float tp = pars_->tprimf;
	CP_TO_GPU (tps, &tp, sizeof(float));
      }
    }
  }
}
