#include <random>
#include <algorithm>
#include <vector>
#include "reservoir.h"

#define loop_MN  <<< blocks_mn, threads_mn >>> 
#define loop_N2  <<< blocks_n2, threads_n2 >>>
#define loop_NK  <<< blocks_nk, threads_nk >>>
#define loops_MN <<< blocks_MN, threads_MN >>>
#define loops_NN <<< blocks_NN, threads_NN >>>
#define loops_QM <<< blocks_QM, threads_QM >>>
#define loop_N   <<< blocks_n,  threads_n  >>>
#define loop_M   <<< blocks_m,  threads_m  >>>


Reservoir::Reservoir(Parameters* pars, int Min) :
  pars_(pars), R(nullptr), V(nullptr), W(nullptr), W_in(nullptr), A_in(nullptr), A_col(nullptr),
  R2(nullptr), x(nullptr), invWork(nullptr), B(nullptr), info(nullptr), P(nullptr), fake_G(nullptr)
{

  M_         = Min;
  ResQ_      = pars_->ResQ;    N_ = ResQ_ * M_;
  K_         = pars_->ResK;
  sigma_     = (double) pars_->ResSigma;
  ResRadius_ = (double) pars_->ResSpectralRadius;
  beta_      = (double) pars_->ResReg;
  nT_        = pars_->ResTrainingSteps;
  iT_        = 0;
  sigNoise_  = pars_->ResSigmaNoise;
  addNoise_  = pars_->add_noise;
  
  int N = N_;    // size of reservoir
  int K = K_;    // number of non-zero columns in each row of A
  int M = M_;    // number of floats in the solution vector 
  int Q = ResQ_; // number of reservoir elements for each element of solution
  
  assert( (N==M*Q) && "Should be exactly Q reservoir elements for each element of G");
							    
  unsigned int nnz = K * N;

  // verify the loop bounds
  
  int nn1, nn2, nt1, nt2, nb1, nb2;
  
  nn1 = N;           nt1 = min(nn1, 512 );   nb1 = 1 + (nn1-1)/nt1;

  threads_n = dim3(nt1, 1, 1);
  blocks_n  = dim3(nb1, 1, 1);

  nn1 = nnz;         nt1 = min(nn1, 512 );   nb1 = 1 + (nn1-1)/nt1;
  
  threads_nk = dim3(nt1, 1, 1);
  blocks_nk  = dim3(nb1, 1, 1);

  nn1 = M;           nt1 = min(nn1, 32 );   nb1 = 1 + (nn1-1)/nt1;
  nn2 = N;           nt2 = min(nn2, 32 );   nb2 = 1 + (nn2-1)/nt2;
  
  threads_MN = dim3(nt1, nt2, 1);
  blocks_MN  = dim3(nb1, nb2, 1);

  nn1 = N;           nt1 = min(nn1, 32 );   nb1 = 1 + (nn1-1)/nt1;
  nn2 = N;           nt2 = min(nn2, 32 );   nb2 = 1 + (nn2-1)/nt2;
  
  threads_NN = dim3(nt1, nt2, 1);
  blocks_NN  = dim3(nb1, nb2, 1);

  nn1 = N*N;         nt1 = min(nn1, 1024 );   nb1 = 1 + (nn1-1)/nt1;
  
  threads_n2 = dim3(nt1, 1, 1);
  blocks_n2  = dim3(nb1, 1, 1);

  nn1 = N*M;         nt1 = min(nn1, 1024 );   nb1 = 1 + (nn1-1)/nt1;
  
  threads_mn = dim3(nt1, 1, 1);
  blocks_mn  = dim3(nb1, 1, 1);

  nn1 = Q;           nt1 = min(nn1, 32 );   nb1 = 1 + (nn1-1)/nt1;
  nn2 = M;           nt2 = min(nn2, 32 );   nb2 = 1 + (nn2-1)/nt2;
  
  threads_QM = dim3(nt1, nt2, 1);
  blocks_QM  = dim3(nb1, nb2, 1);

  threads_m = dim3(nt2, 1, 1);
  blocks_m  = dim3(nb2, 1, 1);
  
  if (pars_->ResFakeData) {
    checkCuda(cudaMalloc((void**) &fake_G, sizeof(double)*M  ) );
    init_Fake_G loop_M (fake_G);
  }

  checkCuda(cudaMalloc((void**) &dG, sizeof(double)*M  ) );  
  checkCuda(cudaMalloc((void**) &R,  sizeof(double)*N  ) );  
  checkCuda(cudaMalloc((void**) &R2, sizeof(double)*N  ) );  
  checkCuda(cudaMalloc((void**) &x,  sizeof(double)*N*K) ); setval loop_NK (x, 0., N*K);
  checkCuda(cudaMalloc((void**) &V,  sizeof(double)*M*N) ); setval loop_MN (V, 0., N*M);
  checkCuda(cudaMalloc((void**) &W,  sizeof(double)*N*N) ); setval loop_N2 (W, 0., N*N);

  bool first = true;
  
  double * A_h;  int * A_j;  double * W_h; 
  A_h = (double*) malloc(sizeof(double) * nnz);
  A_j = (int*) malloc(sizeof(int)   * nnz);
  W_h = (double*) malloc(sizeof(double) * N);
  
  std::random_device rd;
  std::mt19937 gen(rd()); 
  std::uniform_real_distribution<> unif(0., ResRadius_*2./((double) K));
  std::uniform_real_distribution<> win(-sigma_, sigma_);
  std::uniform_real_distribution<> r0(-1., 1.);
  
  std::vector<int> col(N);     std::iota(col.begin(), col.end(), 0);
  std::vector<int> cin(K);     
  
  for (int n=0; n<N; n++) {

    W_h[n] = win(gen);
    std::shuffle(col.begin(), col.end(), gen);
    for (int k=0; k<K; k++) cin[k] = col[k];
    std::sort(cin.begin(), cin.end());

    for (int k=0; k<K; k++) {
      A_j[k + K*n] = cin[k];
      A_h[k + K*n] = unif(gen);
    }    
  }

  checkCuda(cudaMalloc((void**) &W_in,  sizeof(double) * N) );   
  checkCuda(cudaMalloc((void**) &A_in,  sizeof(double)*nnz) ); 
  checkCuda(cudaMalloc((void**) &A_col, sizeof(int)   *nnz) ); 

  CP_TO_GPU (W_in,  W_h, sizeof(double) * N  );
  CP_TO_GPU (A_in,  A_h, sizeof(double) * nnz);
  CP_TO_GPU (A_col, A_j, sizeof(int)    * nnz);
  
  free(W_h);
  free(A_h);
  free(A_j);

  if (false) {
    red = new dBlock_Reduce(N); cudaDeviceSynchronize();
    double *y;      cudaMalloc( &y,      sizeof(double)*N );
    double *x2norm; cudaMalloc( &x2norm, sizeof(double)   );
    double *y2norm; cudaMalloc( &y2norm, sizeof(double)   );
    double *xynorm; cudaMalloc( &xynorm, sizeof(double)   );
    double *x2;     cudaMalloc( &x2,     sizeof(double)*N );
    double *y2;     cudaMalloc( &y2,     sizeof(double)*N );
    double *xy;     cudaMalloc( &xy,     sizeof(double)*N );
    
    setval loop_N  (R, 1., N);
    setval loop_N  (y, 1., N);
    setval loop_NK (x, 1., nnz);    

    double eval = 0.; double eval_old = 10.;  double tol = 1.e-6;  double ex = 0.;   double ey = 0.;
    
    while (abs(eval-eval_old)/abs(eval_old) > tol) {    
      
      eval = eval_old;
      
      myPrep loop_NK (x, R, A_col, nnz);
      mySpMV loop_N  (x2, xy, y2, y, x, A_in, R, K, N);
      red->Sum(y2, y2norm);    red->Sum(x2, x2norm);    red->Sum(xy, xynorm);
      
      inv_scale_kernel loop_N (R, y, y2norm, N); 
      CP_TO_CPU(&ex, x2norm, sizeof(double));
      CP_TO_CPU(&ey, xynorm, sizeof(double));
      eval_old  = ey/ex;
      
      printf("eval = %f \t",eval_old);
    }
    printf("\n Spectral radius is %f \n",eval_old);

    // print the residual
    myPrep loop_NK (x, R, A_col, nnz);
    mySpMV loop_N  (x2, xy, y2, y, x, A_in, R, K, N);  
    eig_residual loop_N (y, A_in, x, R, x2, eval_old, K, N);
    red->Sum(x2, x2norm);  CP_TO_CPU(&ex, x2norm, sizeof(double));
    printf(ANSI_COLOR_YELLOW);  printf("RMS residual = %f \n",sqrt(ex));  printf(ANSI_COLOR_RESET);

    double factor = ResRadius_/eval_old;
    setA loop_NK (A_in, factor, N*K);
    
    cudaFree(x2norm); cudaFree(y2norm); cudaFree(xynorm);
    cudaFree(x2); cudaFree(y2); cudaFree(xy); cudaFree(y);
    delete (red);
    exit(1);
  }
  
  setval loop_N (R, 1., N_);
  dense = new DenseM (N_, M_); // to calculate G = V r2 and also V = V W
}

Reservoir::~Reservoir() 
{
  if (dG)    cudaFree(dG);
  if (R2)    cudaFree(R2);
  if (R)     cudaFree(R);
  if (P)     cudaFree(P);
  if (x)     cudaFree(x);
  if (W_in)  cudaFree(W_in);
  if (A_in)  cudaFree(A_in);
  if (A_col) cudaFree(A_col);
}
  
void Reservoir::fake_data(float* G)
{
  update_Fake_G loop_M (G, iT_);
  printf("iT_ = %d \t",iT_);
}

void Reservoir::add_data(float* G)
{
  int M = M_;  int N = N_;
  if (iT_ < nT_+1) {
    // add the option to add some noise in the course of the promotion.
    if (addNoise_) {
      float *Gnoise_h, *Gnoise;
      checkCuda(cudaMalloc(    (void**) &Gnoise,   sizeof(float)*M ));
      Gnoise_h = (float*) malloc(sizeof(float)*M);
      // specialize to KS for now
      // get a normally-distributed random number with mean = 1 and sigma=sigNoise_, 
      // make sure k=0 component has no noise;
      std::random_device rd;      std::mt19937 gen(rd()); std::normal_distribution<float> noise(0.0, sigNoise_);
      for (int m=0; m<M; m++) Gnoise_h[m] = noise(gen);
      float sum = 0.;
      for (int m=0; m<M; m++) sum += Gnoise_h[m];
      sum = sum/((float) M);
      for (int m=0; m<M; m++) Gnoise_h[m] = 1. + Gnoise_h[m] - sum;
      
      CP_TO_GPU(Gnoise, Gnoise_h, sizeof(cuComplex)*M);
      free(Gnoise_h);
      promote loop_M (dG, G, Gnoise, M);
      cudaFree(Gnoise);      
    } else {
      promote loop_M (dG, G, M);
    }

    // Cannot predict initial condition. And R starts out = 0. 
    // Need to have taken at least one timestep before beginning training. 
    
    // At the end of all other timesteps, use existing R and new G to build V, W
    // Use new G to build new R

    // when predicting:

    // Use existing R to predict G
    // use new G to build new R

    // Skip a few steps to allow warm-up. The early fits are probably bad anyway. 
    // After the first 10 timestep, prepare to predict new G from current R
    if (iT_ > 10) {
      getV loops_MN (V, dG, R2, M, N); 
      getW loops_NN (W,     R2,    N);
    }
    // Use new G (from eqns) to get new R
    update_reservoir (dG);

    /*
    double * W_h;
    W_h = (double*) malloc(sizeof(double)*M*N);
    CP_TO_GPU (W_h,  V, sizeof(double) *M*N  );

    for (int n1=0; n1<N; n1++) {
      for (int n2=0; n2<M; n2++) {
	printf("W[%d, %d] = %e \n", n2, n1, W_h[n2+M*n1]);
      }
    }
    free(W_h);
    exit(1);
    */
    
    if (iT_ == nT_) {
      this->conclude_training();
      printf("Training phase completed. \n");
    }
    iT_ += 1;        
  }    
}
/*
 At the end of the training phase, use regularized least squares
 to find the weights W_out that minimize the difference between the
 observed G and the predicted G = W_out R2
 where R2 is a hidden state built from squaring every other element of R
 and R is a hidden state that is calculated from R = tanh(A R + W_in G).

 Here, A and W_in are matrices of size NxN and NxM, respectively.
 A is sparse, with K elements in each row. The K columns are chosen randomly,
 with all columns equally likely.

 The N*K non-zero elements of A are chosen from a uniform distribution [0, 1) and
 adjusted by multiplying A by a scalar such that the largest eigenvalue of the
 scaled A matrix is the prescribed spectral_radius.

 The elements of W_in are selected randomly from a uniform distribution from [-sigma, sigma)
 and sigma is an input parameter typically ~ 0.5.

 While W_in could be dense, standard practice for the K-S equation seems to be to
 choose it so that each of the M elements of G are individually mapped
 to Q elements of R. Thus W_in G is typically a spread of the G values by a factor of Q
 and then a dot product with a random vector of length Q*M.

 For now, this is the only way to specify W_in that is supported.

 In the training phase, two matrices have been created, W and V, 
 with W = R2 R2_transpose and V = G R2_transpose. The elements of V and W
 are accumulated (over the specified number of training steps) without 
 subtracting the average values of the elements of R2. This might be a bad idea, but 
 it seems to be standard practice. 

 W_out is found from W_out = V (W + beta I)**-1

 where beta is a regularization parameter. In the literature people seem to use small 
 values of beta ~ 1.e-4, but there is not much clarity on why. In particular, if the 
 number of training steps is very large, the W matrix will typically have elements with 
 large values. 

 It would seem to be wiser to form W from delta R2 = R2 - R2_average (element-wise) 
 and also to divide by the number of training steps (and similarly for V). Then one could 
 perhaps understand a standard value of beta. Presently, the model coded here 
 folllows the literature but it would be good to reconsider these decisions.

 For no particular reason, W_out is stored as the matrix P for use in the 
 prediction phase. 

 I use the cusolver library to calculate B**-1 with an LU-decomposition, 
 including pivots. I first tried a Cholesky-based solver but while B was 
 symmetric with positive elements, it was not positive-definite. It might be useful to 
 go back and verify that there were not other problems when I ran those tests.

 I use the cutensor library to calculate P = V B**-1. I expected the cutensor
 library to work easily with mixed precision, but there are evidently subtleties. 
 For now, as noted above, I am using double precision here instead of mixed precision.

 Finally, the cutensor API specifies that the inputs (V and B**-1 here) are assumed to 
 be constants, so I am not trying to reuse the space with something like V = V B**-1. 
*/
void Reservoir::conclude_training(void)
{  
  cusolverDnCreate(&handle);  invSize = 0;
  cudaMalloc(&info, sizeof(int));    cudaMemset(info, 0, sizeof(int));

  // Set B = W + beta I
  getB loop_N (W, beta_, N_);  

  int *ipiv;  cudaMalloc(&ipiv, sizeof(int)*N_);
  
  cusolverDnDgetrf_bufferSize(handle, N_, N_, W, N_, &invSize);

  cudaMalloc(&invWork, sizeof(double)*invSize);

  double *Binv;  cudaMalloc(&Binv, sizeof(double)*N_*N_);
  setval loop_N2 (Binv, 0., N_*N_);  setI loop_N (Binv, N_);
  
  cusolverDnDgetrf(handle, N_, N_, W, N_, invWork, ipiv, info);  
  cusolverDnDgetrs(handle, CUBLAS_OP_N, N_, N_, W, N_, ipiv, Binv, N_, info);

  cudaFree(W);  cudaFree(ipiv); cudaFree(info); cudaFree(invWork);

  cudaMalloc(&P, sizeof(double)*N_*M_);

  dense->MatMat(P, V, Binv);   cudaFree(V);  cudaFree(Binv);
}

bool Reservoir::predicting(void)
{
  bool b;
  b = (iT_ >= nT_) ? true : false ;
  return b;
}

// Use hidden reservoir information to update G
void Reservoir::predict(double* dG)
{
  dense->MatVec(dG, P, R2); // Use hidden information R2 to find new G = P R2
  update_reservoir(dG);     // Use G to find new values for R and R2 (both hidden)
}

// Use current G to update R and R2
void Reservoir::update_reservoir(double* dG)
{
  int K = K_;  int M = M_;  int N = N_;  int NK = N*K;  int Q = ResQ_;

  assert(N==M*Q);
  
  myPrep       loop_NK  (x,  R,    A_col, NK);   // Move current R to x for efficient use later
  WinG         loops_QM (R,  W_in, dG,    Q, M); // Set R = W_in G
  update_state loop_N   (R,  A_in, x,     K, N); // Set R = tanh(A_in x + R)
  getr2        loop_N   (R2, R,           N);    // Calculate R2 from R  

}
