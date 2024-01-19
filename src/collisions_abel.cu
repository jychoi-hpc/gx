#include "collisions.h"
#define CAL_CHECK(call)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        calError_t status = call;                                                                                      \
        if (status != CAL_OK)                                                                                          \
        {                                                                                                              \
            fprintf(stderr, "CAL error at %s:%d : %d\n", __FILE__, __LINE__, status);                                  \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

int indxl2g(int iloc, int nb, int iproc, int isrcproc, int nprocs) {
  return nprocs * nb * ((iloc-1)/nb) + (iloc-1)%nb + ((nprocs+iproc-isrcproc)%nprocs)*nb + 1;
}

AbelCollisionOperator::AbelCollisionOperator(Parameters* pars, Grids* grids, Geometry* geo, specie sp, int is_glob) :
  pars_(pars), grids_(grids), geo_(geo), Nlm(grids->Nmoms), Nlm_glob(grids->Nl*grids->Nm_glob), Nk(grids_->NxNycNz), sp_a(sp), is_a(is_glob)
{
  m = Nk;
  n = Nlm_glob;
  k = Nlm_glob;
#ifdef USE_CUBLASMP
  mbA = Nk;
  nbA = Nlm;
  mbB = Nlm;
  nbB = Nlm;
  mbC = Nk;
  nbC = Nlm;
  global_m_a = m;
  global_n_a = k;
  global_m_b = k;
  global_n_b = n;
  global_m_c = m;
  global_n_c = n;

  nprow = 1;
  npcol = grids_->nprocs_m;
  myprow = 1;
  mypcol = grids_->iproc_m;

  llda = cublasMpNumroc(global_m_a, mbA, myprow, 0, nprow);
  loc_n_a = cublasMpNumroc(global_n_a, nbA, mypcol, 0, npcol);

  lldb = cublasMpNumroc(global_m_b, mbB, myprow, 0, nprow);
  loc_n_b = cublasMpNumroc(global_n_b, nbB, mypcol, 0, npcol);

  lldc = cublasMpNumroc(global_m_c, mbC, myprow, 0, nprow);
  loc_n_c = cublasMpNumroc(global_n_c, nbC, mypcol, 0, npcol);

  CUBLAS_CHECK(cublasMpCreate(&cublasHandle, 0));
  CUBLAS_CHECK(cublasMpGridCreate(cublasHandle, nprow, npcol, CUBLASMP_GRID_LAYOUT_COL_MAJOR, grids_->cal_comm, &cublasGrid));

  CUBLAS_CHECK(
      cublasMpMatrixDescriptorCreate(cublasHandle, global_m_a, global_n_a, mbA, nbA, 0, 0, llda, CUDA_C_32F, cublasGrid, &descA));
  CUBLAS_CHECK(
      cublasMpMatrixDescriptorCreate(cublasHandle, global_m_b, global_n_b, mbB, nbB, 0, 0, lldb, CUDA_C_32F, cublasGrid, &descB));
  CUBLAS_CHECK(
      cublasMpMatrixDescriptorCreate(cublasHandle, global_m_c, global_n_c, mbC, nbC, 0, 0, lldc, CUDA_C_32F, cublasGrid, &descC));

#else
  cublasCreate(&cublasHandle);
  llda = m;
  lldb = k;
  lldc = m;
  loc_n_a = k;
  loc_n_b = n;
  loc_n_c = n;
#endif

  cudaMalloc ((void**) &collMat, sizeof(cuComplex)*loc_n_b*lldb);
  cuComplex *collMat_h = (cuComplex*) malloc(sizeof(cuComplex)*loc_n_b*lldb);

  cudaMalloc ((void**) &collFLRMat, sizeof(cuComplex)*loc_n_b*lldb);
  cuComplex *collFLRMat_h = (cuComplex*) malloc(sizeof(cuComplex)*loc_n_b*lldb);

  init_collision_matrix(collMat_h, collFLRMat_h);
  CP_TO_GPU (collMat, collMat_h, sizeof(cuComplex)*loc_n_b*lldb);
  CP_TO_GPU (collFLRMat, collFLRMat_h, sizeof(cuComplex)*loc_n_b*lldb);

  if (pars_->coll_conservation) {
    energyConsG = new MomentsG (pars, grids);
    uparLConsG = new MomentsG (pars, grids);
    uparDConsG = new MomentsG (pars, grids);
    uperpLConsG = new MomentsG (pars, grids);
    uperpDConsG = new MomentsG (pars, grids);
    eiConsG = new MomentsG (pars, grids);
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
  if (eiConsG) delete eiConsG;
  if (tmpG) delete tmpG;
  if (tmpF) cudaFree(tmpF);
#ifdef USE_CUBLASMP
  cublasMpMatrixDescriptorDestroy(cublasHandle, descA);
  cublasMpMatrixDescriptorDestroy(cublasHandle, descB);
  cublasMpMatrixDescriptorDestroy(cublasHandle, descC);
  cublasMpGridDestroy(cublasHandle, cublasGrid);
  cublasMpDestroy(cublasHandle);
  if(d_work) cudaFree(d_work);
  if(h_work) free(h_work);
#else
  cublasDestroy(cublasHandle);
#endif
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
  
  for(int j=0; j<loc_n_b; j++) {
    for(int i=0; i<lldb; i++) {
#ifdef USE_CUBLASMP
      int ig = indxl2g(i, mbB, myprow, 0, nprow);
      int jg = indxl2g(j, nbB, mypcol, 0, npcol);
#else
      int ig = i;
      int jg = j;
#endif
      int il = ig%grids_->Nl; 
      int im = ig/grids_->Nl;
      int jl = jg%grids_->Nl;
      int jm = jg/grids_->Nl;
      collMat[i+lldb*j].x = 0.;
      collMat[i+lldb*j].y = 0.;
      collFLRMat[i+lldb*j].x = 0.;
      collFLRMat[i+lldb*j].y = 0.;
      // this collision object is being defined for species a. loop over all species b to get contributions.
      for(int is_b=0; is_b<pars_->nspec_in; is_b++) {
        // like-species
        if (is_b == is_a) {
          // lorentz (pitch-angle scattering)
          if(pars_->abel_include_lorentz) {
            collMat[i+lldb*j].x += sp_a.nu[is_a]*nuDLorentz_data[jl + NL_lorentz*jm][il + NL_lorentz*im];
            // FLR
            if(pars_->abel_include_flr) {
              collFLRMat[i+lldb*j].x -= sp_a.nu[is_a]*sp_a.rho2*nuDFLR_data[jl + NL_lorentz*jm][il + NL_lorentz*im];
            }
          }

          // diffusion
          if(pars_->abel_include_ediff) {
            collMat[i+lldb*j].x += sp_a.nu[is_a]*nuParDiffT0_data[jl + NL_diffusion*jm][il + NL_diffusion*im];
            // FLR
            if(pars_->abel_include_flr) {
              collFLRMat[i+lldb*j].x -= sp_a.nu[is_a]*sp_a.rho2*nuParFLR_data[jl + NL_diffusion*jm][il + NL_diffusion*im];
            }
          }
        } else if(sp_a.type == 1 && pars_->ei_colls) { // electron-ion
          if(pars_->abel_include_lorentz) {
            collMat[i+lldb*j].x += sp_a.nu[is_b]*invU3Lorentz_data[jl + NL_lorentz*jm][il + NL_lorentz*im];
            // FLR
            if(pars_->abel_include_flr) {
              collFLRMat[i+lldb*j].x -= sp_a.nu[is_b]*sp_a.rho2*invU3FLR_data[jl + NL_lorentz*jm][il + NL_lorentz*im];
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
  cudaMalloc ((void**) &alphaEi, sizeof(float)*Nlm*grids_->Nl);
  float *alphaEi_h = (float*) malloc(sizeof(float)*Nlm*grids_->Nl);

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
  nc_inq_varid(ncid, "alphaEi", &varid);
  float alphaEi_data[NL][NM][NL];
  nc_get_var(ncid, varid, &alphaEi_data);
  for(int j=0; j<grids_->Nl; j++) {
    for(int k=0; k<grids_->Nm; k++) {
      for(int l=0; l<grids_->Nl; l++) {
        alphaE_h[j + grids_->Nl*k + Nlm*l] = alphaE_data[j][k][l];
        alphaParL_h[j + grids_->Nl*k + Nlm*l] = alphaParL_data[j][k][l];
        alphaParD_h[j + grids_->Nl*k + Nlm*l] = alphaParD_data[j][k][l];
        alphaPerpL_h[j + grids_->Nl*k + Nlm*l] = alphaPerpL_data[j][k][l];
        alphaPerpD_h[j + grids_->Nl*k + Nlm*l] = alphaPerpD_data[j][k][l];
        alphaEi_h[j + grids_->Nl*k + Nlm*l] = alphaEi_data[j][k][l];
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
  CP_TO_GPU(alphaEi, alphaEi_h, sizeof(float)*Nlm*grids_->Nl);

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
  abel_conservation_moment<<<dG, dB>>> (eiConsG->G(), alphaEi, geo_->kperp2, sp_a);

  std::vector<int32_t> full_modes{'y', 'x', 'z', 'l', 'm'};
  std::vector<int32_t> reduced_modes{'y', 'x', 'z'};
  reduce = new Reduction<cuComplex>(grids_, full_modes, reduced_modes);

  free(alphaE_h);
  free(alphaParL_h);
  free(alphaPerpL_h);
  free(alphaParD_h);
  free(alphaPerpD_h);

  cudaFree(alphaE);
  cudaFree(alphaParL);
  cudaFree(alphaPerpL);
  cudaFree(alphaParD);
  cudaFree(alphaPerpD);
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

#ifdef USE_CUBLASMP
  int64_t ia = 1;
  int64_t ja = 1;
  int64_t ib = 1;
  int64_t jb = 1;
  int64_t ic = 1;
  int64_t jc = 1;
  
  if(first) {
    CUBLAS_CHECK(cublasMpGemm_bufferSize(
      cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
      m, n, k, 
      &alpha, H_in, ia, ja, descA,
      collMat, ib, jb, descB, 
      &beta, H_res, ic, jc, descC,
      CUDA_C_32F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost)
    );
    checkCudaErrors(cudaMalloc(&d_work, workspaceInBytesOnDevice));
    h_work = malloc(workspaceInBytesOnHost);
    first = false;
  }

  CUBLAS_CHECK(cublasMpGemm(
     cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
     m, n, k, 
     &alpha, H_in, ia, ja, descA,
     collMat, ib, jb, descB, 
     &beta, H_res, ic, jc, descC,
     CUDA_C_32F, d_work, workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost)
  );
     
#else
  int lda = m;
  int ldb = n;
  int ldc = m;

  int status = cublasCgemm3m(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
     m, n, k, &alpha,
     H_in, lda,
     collMat, ldb,
     &beta, H_res, ldc);
#endif
}

void AbelCollisionOperator::conservation(MomentsG* H_in, MomentsG* consG, MomentsG* H_res, float norm)
{
  // tmpG = H_in . consG
  tmpG->multiply(H_in, consG);

  // partial reduction: tmpF = sum_l,m tmpG
  reduce->Sum(tmpG->G(), tmpF);

  // tmpG = tmpF . consG
  tmpG->scale_by_k(tmpF, consG);
  
  // accumulate result. H_res += nu_aa/norm*tmpG
  H_res->add_scaled(1.0, H_res, sp_a.nu[is_a]/norm, tmpG);
}

void AbelCollisionOperator::ei_drag(MomentsG* H_in, MomentsG* consG, Fields* f_in, MomentsG* H_res)
{
  // tmpF = upar_i
  if(grids_->m_lo <= 1 && grids_->m_up > 1) { // only compute current on procs with m=1
    calc_uparbar_i_from_He_and_apar<<<256, grids_->NxNycNz/256 + 1>>>(tmpF, H_in->G(), f_in->apar, geo_->kperp2, geo_->bmag, *(H_in->species), pars_->species_h[pars_->is_ion]);
  }
  if(grids_->nprocs_m > 1) {
    // broadcast current to other procs with other m's
    checkCuda(ncclBroadcast((void*) tmpF, (void*) tmpF, grids_->NxNycNz*2, ncclFloat, 0, grids_->ncclComm_m, 0));
  }

  // tmpG = tmpF . consG
  tmpG->scale_by_k(tmpF, consG);
  
  // accumulate result. H_res += nu_ei*tmpG
  H_res->add_scaled(1.0, H_res, 1.0, tmpG);
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

  // electron-ion conservation
  if(sp_a.type == 1 && pars_->ei_colls && pars_->coll_conservation) {
    ei_drag(G, eiConsG, f, GRhs);
  }

  HtoG<<<dimGrid, dimBlock>>>(G->G(), f->phi, f->apar, f->bpar, geo_->kperp2, *(G->species));
}
