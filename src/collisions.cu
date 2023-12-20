#include "collisions.h"

LorentzCollisionOperator::LorentzCollisionOperator(Parameters* pars, Grids* grids, Geometry* geo, specie sp, int is_glob) :
  pars_(pars), grids_(grids), geo_(geo), Nlm(grids->Nmoms), Nk(grids_->NxNycNz), sp_a(sp), is_a(is_glob)
{
  cublasCreate (&handle);
  cudaMalloc ((void**) &collMat, sizeof(cuComplex)*Nlm*Nlm);
  cuComplex *collMat_h = (cuComplex*) malloc(sizeof(cuComplex)*Nlm*Nlm);

  initCollisionMatrix(collMat_h);
  CP_TO_GPU (collMat, collMat_h, sizeof(cuComplex)*Nlm*Nlm);

  tmpG = new MomentsG (pars, grids);

  if (collMat_h) free(collMat_h);

  int nn1 = grids_->NxNycNz;   int nt1 = 32;  int nb1 = 1 + (nn1-1)/nt1;
  int nn2 = grids_->Nl;        int nt2 = 32;  int nb2 = 1 + (nn2-1)/nt2;
  int nn3 = 1;                 int nt3 = 1;   int nb3 = 1 + (nn3-1)/nt3;

  dimBlock = dim3(nt1, nt2, nt3);
  dimGrid  = dim3(nb1, nb2, nb3);
}

LorentzCollisionOperator::~LorentzCollisionOperator()
{
  if (collMat) cudaFree(collMat);
  if (tmpG) delete tmpG;
  cublasDestroy(handle);
}

void LorentzCollisionOperator::initCollisionMatrix(cuComplex* testMat)
{
  // read from netcdf
  char fname[1000];
  sprintf(fname, "%s/collision_data/lorentz_collision_matrices.nc", GX_PATH);
  int ncid, varid;
  nc_open(fname, NC_NOWRITE, &ncid);
  nc_inq_varid(ncid, "nuDLorentz", &varid);
  float nuDLorentz_data[1024][1024];
  nc_get_var(ncid, varid, &nuDLorentz_data);
  nc_inq_varid(ncid, "invU3Lorentz", &varid);
  float invU3Lorentz_data[1024][1024];
  nc_get_var(ncid, varid, &invU3Lorentz_data);
  
  for(int i=0; i<Nlm; i++) {
    for(int j=0; j<Nlm; j++) {
      int il = i%grids_->Nl; 
      int im = i/grids_->Nl;
      int jl = j%grids_->Nl;
      int jm = j/grids_->Nl;
      testMat[i+Nlm*j].x = 0.;
      testMat[i+Nlm*j].y = 0.;
      if(sp_a.type == 1) { // electrons
        for(int is_b=0; is_b<pars_->nspec_in; is_b++) {
          // electron-electron
          if (is_b == is_a) testMat[i+Nlm*j].x += sp_a.nu[is_a]*nuDLorentz_data[il + 16*im][jl + 16*jm];
          // electron-ion
          else testMat[i+Nlm*j].x += sp_a.nu[is_b]*invU3Lorentz_data[il + 16*im][jl + 16*jm];
        }
      } else if(sp_a.type == 0) { // ions
        // only same-species ion collisions. no ion-electron or ion-other collisions
        testMat[i+Nlm*j].x = sp_a.nu[is_a]*nuDLorentz_data[il + 16*im][jl + 16*jm];
      }
    }
  }

  // set up tensors
  // C = H_res
  // A = H_in
  // B = collMat
  
  int nmodeA = modeA.size();
  int nmodeB = modeB.size();
  int nmodeC = modeC.size();

  // define extents
  std::unordered_map<int, int64_t> extent;
  extent['x'] = grids_->NxNycNz;
  extent['l'] = grids_->Nl;
  extent['m'] = grids_->Nm;
  extent['j'] = grids_->Nl;
  extent['k'] = grids_->Nm;

  // create vector of extents for each tensor
  std::vector<int64_t> extentC;
  for(auto mode : modeC)
      extentC.push_back(extent[mode]);
  std::vector<int64_t> extentA;
  for(auto mode : modeA)
      extentA.push_back(extent[mode]);
  std::vector<int64_t> extentB;
  for(auto mode : modeB)
      extentB.push_back(extent[mode]);

  // initialize cuTENSOR library
  HANDLE_ERROR( cutensorInit(&handleT) );

  // Create Tensor Descriptors
  HANDLE_ERROR( cutensorInitTensorDescriptor( &handleT,
              &descA,
              nmodeA,
              extentA.data(),
              NULL,/*stride*/
              typeA, CUTENSOR_OP_IDENTITY ) );

  HANDLE_ERROR( cutensorInitTensorDescriptor( &handleT,
              &descB,
              nmodeB,
              extentB.data(),
              NULL,/*stride*/
              typeB, CUTENSOR_OP_IDENTITY ) );

  HANDLE_ERROR( cutensorInitTensorDescriptor( &handleT,
              &descC,
              nmodeC,
              extentC.data(),
              NULL,/*stride*/
              typeC, CUTENSOR_OP_IDENTITY ) );


  
}
  
void LorentzCollisionOperator::applyCollisionMatrix_cublas(cuComplex* G_in, cuComplex* G_res, bool accumulate)
{
  int m = Nk;
  int n = Nlm;
  int k = Nlm;
  
  cuComplex alpha; alpha.x = 1.0; alpha.y = 0.0;
  cuComplex beta; 
  if(accumulate) {
    beta.x = 1.0; beta.y = 0.0;
  } else {
    beta.x = 0.0; beta.y = 0.0;
  }
  int lda = m;
  int ldb = n;
  int ldc = m;

  int status = cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
     m, n, k, &alpha,
     G_in, lda,
     collMat, ldb,
     &beta, G_res, ldc);
}

void LorentzCollisionOperator::applyCollisionMatrix_cutensor(cuComplex* G_in, cuComplex* G_res, bool accumulate)
{
  cuComplex alpha; alpha.x = 1.0; alpha.y = 0.0;
  cuComplex beta; 
  if(accumulate) {
    beta.x = 1.0; beta.y = 0.0;
  } else {
    beta.x = 0.0; beta.y = 0.0;
  }

  if(first) {
    //Retrieve the memory alignment for each tensor
    HANDLE_ERROR( cutensorGetAlignmentRequirement( &handleT,
               G_in,
               &descA,
               &alignmentRequirementA) );

    HANDLE_ERROR( cutensorGetAlignmentRequirement( &handleT,
               collMat,
               &descB,
               &alignmentRequirementB) );

    HANDLE_ERROR( cutensorGetAlignmentRequirement( &handleT,
               G_res,
               &descC,
               &alignmentRequirementC) );

    // Create the Contraction Descriptor
    HANDLE_ERROR( cutensorInitContractionDescriptor( &handleT,
                &desc,
                &descA, modeA.data(), alignmentRequirementA,
                &descB, modeB.data(), alignmentRequirementB,
                &descC, modeC.data(), alignmentRequirementC,
                &descC, modeC.data(), alignmentRequirementC,
                typeCompute) );

    // Set the algorithm to use
    HANDLE_ERROR( cutensorInitContractionFind(
                &handleT, &find,
                CUTENSOR_ALGO_DEFAULT) );

    // Query workspace
    HANDLE_ERROR( cutensorContractionGetWorkspace(&handleT,
                &desc,
                &find,
                CUTENSOR_WORKSPACE_RECOMMENDED, &worksize ) );

    // Allocate workspace
    if(worksize > 0)
    {
        if( cudaSuccess != cudaMalloc(&work, worksize) ) // This is optional!
        {
            work = nullptr;
            worksize = 0;
        }
    }

    // Create Contraction Plan
    HANDLE_ERROR( cutensorInitContractionPlan(&handleT,
                                              &plan,
                                              &desc,
                                              &find,
                                              worksize) );
    first = false;
  }

  // Execute the tensor contraction
  cutensorStatus_t err = cutensorContraction(&handleT,
                            &plan,
                     (void*)&alpha, G_in,
                                    collMat,
                     (void*)&beta,  G_res,
                                    G_res,
                            work, worksize, 0 /* stream */);

  // Check for errors
  if(err != CUTENSOR_STATUS_SUCCESS)
  {
      printf("ERROR: %s\n", cutensorGetErrorString(err));
  }
}

void LorentzCollisionOperator::rhs(MomentsG* G, Fields* f, MomentsG* GRhs, bool accumulate)
{
  GtoH<<<dimGrid, dimBlock>>>(G->G(), f->phi, f->apar, f->bpar, geo_->kperp2, *(G->species));

  applyCollisionMatrix_cutensor(G->G(), GRhs->G(), accumulate);

  HtoG<<<dimGrid, dimBlock>>>(G->G(), f->phi, f->apar, f->bpar, geo_->kperp2, *(G->species));
}
