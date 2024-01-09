#include "collisions.h"

AbelCollisionOperator::AbelCollisionOperator(Parameters* pars, Grids* grids, Geometry* geo, specie sp, int is_glob) :
  pars_(pars), grids_(grids), geo_(geo), Nlm(grids->Nmoms), Nk(grids_->NxNycNz), sp_a(sp), is_a(is_glob)
{
  cublasCreate(&cublasHandle);

  cudaMalloc ((void**) &collMat, sizeof(cuComplex)*Nlm*Nlm);
  cuComplex *collMat_h = (cuComplex*) malloc(sizeof(cuComplex)*Nlm*Nlm);

  cudaMalloc ((void**) &collFLRMat, sizeof(cuComplex)*Nlm*Nlm);
  cuComplex *collFLRMat_h = (cuComplex*) malloc(sizeof(cuComplex)*Nlm*Nlm);

  init_collision_matrix(collMat_h, collFLRMat_h);
  CP_TO_GPU (collMat, collMat_h, sizeof(cuComplex)*Nlm*Nlm);
  CP_TO_GPU (collFLRMat, collFLRMat_h, sizeof(cuComplex)*Nlm*Nlm);

  if (pars_->coll_conservation) {
    energyConsG = new MomentsG (pars, grids);
    uparLConsG = new MomentsG (pars, grids);
    uparDConsG = new MomentsG (pars, grids);
    uperpLConsG = new MomentsG (pars, grids);
    uperpDConsG = new MomentsG (pars, grids);
    tmpG = new MomentsG (pars, grids);
    cudaMalloc ((void**) &tmpF, sizeof(cuComplex)*grids_->NxNycNz);
    init_conservation(uparLConsG, uparDConsG, uperpLConsG, uperpDConsG, energyConsG);
  }

  if (collMat_h) free(collMat_h);
  if (collFLRMat_h) free(collFLRMat_h);

  int nn1 = grids_->NxNycNz;   int nt1 = 32;  int nb1 = 1 + (nn1-1)/nt1;
  int nn2 = grids_->Nl;        int nt2 = 32;  int nb2 = 1 + (nn2-1)/nt2;
  int nn3 = 1;                 int nt3 = 1;   int nb3 = 1 + (nn3-1)/nt3;

  dimBlock = dim3(nt1, nt2, nt3);
  dimGrid  = dim3(nb1, nb2, nb3);
}

AbelCollisionOperator::~AbelCollisionOperator()
{
  if (collMat) cudaFree(collMat);
  if (collFLRMat) cudaFree(collFLRMat);
  if (reduce) delete reduce;
  if (energyConsG) delete energyConsG;
  if (uparLConsG) delete uparLConsG;
  if (uparDConsG) delete uparDConsG;
  if (uperpLConsG) delete uperpLConsG;
  if (uperpDConsG) delete uperpDConsG;
  if (tmpG) delete tmpG;
  if (tmpF) cudaFree(tmpF);
  cublasDestroy(cublasHandle);
}

void AbelCollisionOperator::init_collision_matrix(cuComplex* collMat, cuComplex* collFLRMat)
{
  // read from netcdf
  char fname[1000];
  sprintf(fname, "%s/collision_data/lorentz_collision_matrices.nc", GX_PATH);
  int NL_lorentz, NM_lorentz;
  int ncid, varid;
  nc_open(fname, NC_NOWRITE, &ncid);
  nc_inq_varid(ncid, "NL", &varid);
  nc_get_var(ncid, varid, &NL_lorentz);
  nc_inq_varid(ncid, "NM", &varid);
  nc_get_var(ncid, varid, &NM_lorentz);
  nc_inq_varid(ncid, "nuDLorentz", &varid);
  float nuDLorentz_data[NL_lorentz*NM_lorentz][NL_lorentz*NM_lorentz];
  nc_get_var(ncid, varid, &nuDLorentz_data);
  nc_inq_varid(ncid, "nuDFLR", &varid);
  float nuDFLR_data[NL_lorentz*NM_lorentz][NL_lorentz*NM_lorentz];
  nc_get_var(ncid, varid, &nuDFLR_data);
  nc_inq_varid(ncid, "invU3Lorentz", &varid);
  float invU3Lorentz_data[NL_lorentz*NM_lorentz][NL_lorentz*NM_lorentz];
  nc_get_var(ncid, varid, &invU3Lorentz_data);
  nc_inq_varid(ncid, "invU3FLR", &varid);
  float invU3FLR_data[NL_lorentz*NM_lorentz][NL_lorentz*NM_lorentz];
  nc_get_var(ncid, varid, &invU3FLR_data);

  sprintf(fname, "%s/collision_data/diffusion_collision_matrices.nc", GX_PATH);
  nc_open(fname, NC_NOWRITE, &ncid);
  int NL_diffusion, NM_diffusion;
  nc_inq_varid(ncid, "NL", &varid);
  nc_get_var(ncid, varid, &NL_diffusion);
  nc_inq_varid(ncid, "NM", &varid);
  nc_get_var(ncid, varid, &NM_diffusion);
  nc_inq_varid(ncid, "nuParDiffT0", &varid);
  float nuParDiffT0_data[NL_diffusion*NM_diffusion][NL_diffusion*NM_diffusion];
  nc_get_var(ncid, varid, &nuParDiffT0_data);
  nc_inq_varid(ncid, "nuParFLR", &varid);
  float nuParFLR_data[NL_diffusion*NM_diffusion][NL_diffusion*NM_diffusion];
  nc_get_var(ncid, varid, &nuParFLR_data);
  
  for(int j=0; j<Nlm; j++) {
    for(int i=0; i<Nlm; i++) {
      int il = i%grids_->Nl; 
      int im = i/grids_->Nl;
      int jl = j%grids_->Nl;
      int jm = j/grids_->Nl;
      collMat[i+Nlm*j].x = 0.;
      collMat[i+Nlm*j].y = 0.;
      collFLRMat[i+Nlm*j].x = 0.;
      collFLRMat[i+Nlm*j].y = 0.;
      // this collision object is being defined for species a. loop over all species b to get contributions.
      for(int is_b=0; is_b<pars_->nspec_in; is_b++) {
        // like-species
        if (is_b == is_a) {
          // lorentz (pitch-angle scattering)
          if(pars_->abel_include_lorentz) {
            collMat[i+Nlm*j].x += sp_a.nu[is_a]*nuDLorentz_data[il + NL_lorentz*im][jl + NL_lorentz*jm];
            // FLR
            if(pars_->abel_include_flr) {
              collFLRMat[i+Nlm*j].x -= sp_a.nu[is_a]*sp_a.rho2*nuDFLR_data[il + NL_lorentz*im][jl + NL_lorentz*jm];
            }
          }

          // diffusion
          if(pars_->abel_include_ediff) {
            collMat[i+Nlm*j].x += sp_a.nu[is_a]*nuParDiffT0_data[il + NL_diffusion*im][jl + NL_diffusion*jm];
            // FLR
            if(pars_->abel_include_flr) {
              collFLRMat[i+Nlm*j].x -= sp_a.nu[is_a]*sp_a.rho2*nuParFLR_data[il + NL_diffusion*im][jl + NL_diffusion*jm];
            }
          }
        } else if(sp_a.type == 1) { // electron-ion
          if(pars_->abel_include_lorentz) {
            collMat[i+Nlm*j].x += sp_a.nu[is_b]*invU3Lorentz_data[il + NL_lorentz*im][jl + NL_lorentz*jm];
            // FLR
            if(pars_->abel_include_flr) {
              collFLRMat[i+Nlm*j].x -= sp_a.nu[is_b]*sp_a.rho2*invU3FLR_data[il + NL_lorentz*im][jl + NL_lorentz*jm];
            }
          }
        }
        // only like-species ion collisions. no ion-electron or ion-other collisions
      }
    }
  }
}

void AbelCollisionOperator::init_conservation(MomentsG* uparLConsG, MomentsG* uparDConsG, MomentsG* uperpLConsG, MomentsG* uperpDConsG, MomentsG* energyConsG)
{
  cudaMalloc ((void**) &alphaE, sizeof(float)*Nlm*grids_->Nl);
  float *alphaE_h = (float*) malloc(sizeof(float)*Nlm*grids_->Nl);
  cudaMalloc ((void**) &alphaParL, sizeof(float)*Nlm*grids_->Nl);
  float *alphaParL_h = (float*) malloc(sizeof(float)*Nlm*grids_->Nl);
  cudaMalloc ((void**) &alphaParD, sizeof(float)*Nlm*grids_->Nl);
  float *alphaParD_h = (float*) malloc(sizeof(float)*Nlm*grids_->Nl);
  cudaMalloc ((void**) &alphaPerpL, sizeof(float)*Nlm*grids_->Nl);
  float *alphaPerpL_h = (float*) malloc(sizeof(float)*Nlm*grids_->Nl);
  cudaMalloc ((void**) &alphaPerpD, sizeof(float)*Nlm*grids_->Nl);
  float *alphaPerpD_h = (float*) malloc(sizeof(float)*Nlm*grids_->Nl);

  char fname[1000];
  int ncid, varid;
  sprintf(fname, "%s/collision_data/conservation_matrices.nc", GX_PATH);
  nc_open(fname, NC_NOWRITE, &ncid);
  int NL, NM;
  nc_inq_varid(ncid, "NL", &varid);
  nc_get_var(ncid, varid, &NL);
  nc_inq_varid(ncid, "NM", &varid);
  nc_get_var(ncid, varid, &NM);
  nc_inq_varid(ncid, "alphaE", &varid);
  float alphaE_data[NL][NM][NL];
  nc_get_var(ncid, varid, &alphaE_data);
  nc_inq_varid(ncid, "alphaParL", &varid);
  float alphaParL_data[NL][NM][NL];
  nc_get_var(ncid, varid, &alphaParL_data);
  nc_inq_varid(ncid, "alphaParD", &varid);
  float alphaParD_data[NL][NM][NL];
  nc_get_var(ncid, varid, &alphaParD_data);
  nc_inq_varid(ncid, "alphaPerpL", &varid);
  float alphaPerpL_data[NL][NM][NL];
  nc_get_var(ncid, varid, &alphaPerpL_data);
  nc_inq_varid(ncid, "alphaPerpD", &varid);
  float alphaPerpD_data[NL][NM][NL];
  nc_get_var(ncid, varid, &alphaPerpD_data);
  for(int j=0; j<grids_->Nl; j++) {
    for(int k=0; k<grids_->Nm; k++) {
      for(int l=0; l<grids_->Nl; l++) {
        alphaE_h[j + grids_->Nl*k + Nlm*l] = alphaE_data[j][k][l];
        alphaParL_h[j + grids_->Nl*k + Nlm*l] = alphaParL_data[j][k][l];
        alphaParD_h[j + grids_->Nl*k + Nlm*l] = alphaParD_data[j][k][l];
        alphaPerpL_h[j + grids_->Nl*k + Nlm*l] = alphaPerpL_data[j][k][l];
        alphaPerpD_h[j + grids_->Nl*k + Nlm*l] = alphaPerpD_data[j][k][l];
      }
    }
  }

  energy_norm = 1.5*alphaE_data[0][0][0] + alphaE_data[1][0][0] + alphaE_data[0][2][0]/sqrt(2);
  momentumL_norm = alphaParL_data[0][1][0];
  momentumD_norm = -alphaParD_data[0][1][0];

  CP_TO_GPU(alphaE, alphaE_h, sizeof(float)*Nlm*grids_->Nl);
  CP_TO_GPU(alphaParL, alphaParL_h, sizeof(float)*Nlm*grids_->Nl);
  CP_TO_GPU(alphaParD, alphaParD_h, sizeof(float)*Nlm*grids_->Nl);
  CP_TO_GPU(alphaPerpL, alphaPerpL_h, sizeof(float)*Nlm*grids_->Nl);
  CP_TO_GPU(alphaPerpD, alphaPerpD_h, sizeof(float)*Nlm*grids_->Nl);

  int nn1 = grids_->NxNycNz;    int nt1 = min(grids_->NxNycNz, 32) ; int nb1 = 1 + (nn1-1)/nt1;
  int nn2 = grids_->Nl;         int nt2 = min(grids_->Nl, 4 )      ; int nb2 = 1 + (nn2-1)/nt2;
  int nn3 = grids_->Nm;         int nt3 = min(grids_->Nm, 4 )      ; int nb3 = 1 + (nn3-1)/nt3;
  dim3 dB = dim3(nt1, nt2, nt3);
  dim3 dG  = dim3(nb1, nb2, nb3);
  abel_conservation_moment<<<dG, dB>>> (energyConsG->G(), alphaE, geo_->kperp2, sp_a);
  abel_conservation_moment<<<dG, dB>>> (uparLConsG->G(), alphaParL, geo_->kperp2, sp_a);
  abel_conservation_moment<<<dG, dB>>> (uparDConsG->G(), alphaParD, geo_->kperp2, sp_a);
  abel_conservation_moment<<<dG, dB>>> (uperpLConsG->G(), alphaPerpL, geo_->kperp2, sp_a);
  abel_conservation_moment<<<dG, dB>>> (uperpDConsG->G(), alphaPerpD, geo_->kperp2, sp_a);

  std::vector<int32_t> full_modes{'y', 'x', 'z', 'l', 'm'};
  std::vector<int32_t> reduced_modes{'y', 'x', 'z'};
  reduce = new Reduction<cuComplex>(grids_, full_modes, reduced_modes);

  free(alphaE_h);
  free(alphaParL_h);
  free(alphaPerpL_h);
  free(alphaParD_h);
  free(alphaPerpD_h);
}
  
void AbelCollisionOperator::apply_collision_matrix(cuComplex* H_in, cuComplex* collMat, cuComplex* H_res, bool accumulate)
{
  cuComplex alpha; alpha.x = 1.0; alpha.y = 0.0;
  cuComplex beta; 
  if(accumulate) {
    beta.x = 1.0; beta.y = 0.0;
  } else {
    beta.x = 0.0; beta.y = 0.0;
  }

  int m = Nk;
  int n = Nlm;
  int k = Nlm;
  
  int lda = m;
  int ldb = n;
  int ldc = m;

  int status = cublasCgemm3m(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
     m, n, k, &alpha,
     H_in, lda,
     collMat, ldb,
     &beta, H_res, ldc);
}

void AbelCollisionOperator::conservation(MomentsG* H_in, MomentsG* consG, MomentsG* H_res, float norm)
{
  // tmpG = H_in . consG
  tmpG->multiply(H_in, consG);

  // partial reduction: tmpF = sum_l,m tmpG
  reduce->Sum(tmpG->G(), tmpF);

  // tmpG = tmpF . consG
  tmpG->scale_by_k(tmpF, consG);
  
  // accumulate result. H_res += 1/norm*tmpG
  H_res->add_scaled(1.0, H_res, 1.0/norm, tmpG);
}

void AbelCollisionOperator::rhs(MomentsG* G, Fields* f, MomentsG* GRhs, bool accumulate)
{
  GtoH<<<dimGrid, dimBlock>>>(G->G(), f->phi, f->apar, f->bpar, geo_->kperp2, *(G->species));

  apply_collision_matrix(G->G(), collMat, GRhs->G(), accumulate);
  if(pars_->coll_conservation) {
    if(pars_->abel_include_lorentz) {
      conservation(G, uparLConsG, GRhs, momentumL_norm);
    }
    if(pars_->abel_include_ediff) {
      conservation(G, energyConsG, GRhs, energy_norm);
      conservation(G, uparDConsG, GRhs, momentumD_norm);
    }
  }

  // FLR
  if(pars_->abel_include_flr) {
    G->scale_by_k(geo_->kperp2, G);
    apply_collision_matrix(G->G(), collFLRMat, GRhs->G(), true);

    if(pars_->coll_conservation) {
      if(pars_->abel_include_lorentz) {
        conservation(G, uperpLConsG, GRhs, momentumL_norm);
      }
      if(pars_->abel_include_ediff) {
        conservation(G, uperpDConsG, GRhs, momentumD_norm);
      }
    }
    G->inv_scale_by_k(geo_->kperp2, G);
  }

  HtoG<<<dimGrid, dimBlock>>>(G->G(), f->phi, f->apar, f->bpar, geo_->kperp2, *(G->species));
}
