#include "mirror_magma.h"
#include "magma_v2.h"      // also includes cublas_v2.h
#include "magma_lapack.h"

#include "stdio.h"

Mirror_magma::Mirror_magma(Parameters *pars, Grids *grids, Geometry *geo, double p, double r, double u, bool sdirk, double dt_in, double vte):
	p_(p), r_(r), u_(u), sdirk_(sdirk), grids_(grids), pars_(pars), geo_(geo), A_bounce(nullptr), dt_(dt_in), vte_(vte)
{

    magma_init();
    
    magma_queue_t my_queue;    // magma queue variable, internally holds a cuda stream and a cublas handle
    magma_device_t cdev;       // variable to indicate current gpu id

    magma_getdevice( &cdev );
    magma_queue_create( cdev, &my_queue );     // create a queue on this cdev

    int N = pars_->nl_in * pars_->nm_in;
    int Nband = 5;
    int KL = pars_->nm_in - 1;
    int KU = pars_->nm_in - 1;
    int nrhs = grids_->Nyc*grids_->Nx;
    //int nrhs = 1;
    int batchCount = grids_->Nz;
    int ldda = 2*KL+KU+1;
    int lddb = N;
    int lda = ldda;
    int ldb = lddb;
    int info = 0;
    int h_offsets[5] = {-pars_->nm_in+1, -1, 0, 1, pars_->nm_in-1};

    int sizeA = lda*N*batchCount;
    int sizeB = ldb*nrhs*batchCount;
    int sizeBand = Nband*N*batchCount;

    cuComplex *h_B, *h_X, *d_A, *h_diags;
    cuComplex *h_X_test;
    float error, Rnorm, Anorm, Xnorm, *work;
    int* ipiv, *cpu_info;
    int *dipiv, *dinfo_array;

    cuComplex c_one     = MAGMA_C_ONE;
    cuComplex c_neg_one = MAGMA_C_NEG_ONE;

    cuComplex** d_B = NULL;
    cuComplex **dA_array = NULL;
    cuComplex **dB_array = NULL;
    int     **dipiv_array = NULL;

    float tol = 1e-6;
    int status = 0;


    if (sdirk_){
        num_coeff = 1;
    }
    else{
        if (p_ != 0){
            num_coeff = 2;
        }
        else{
            num_coeff = 3;
        }
    }

    double* dcoeff = (double*) malloc(sizeof(double)*num_coeff);
    if (num_coeff == 1){
        dcoeff[0] = r_;
    }
    else if (num_coeff == 2){
        dcoeff[0] = r_;
        dcoeff[1] = u_;
    }
    else{
        dcoeff[0] = p_;
        dcoeff[1] = r_;
        dcoeff[2] = u_;

    }

    //TESTING_CHECK( magma_cmalloc_cpu( &h_A, sizeA ));
    h_A = (cuComplex*)malloc(sizeof(cuComplex)*sizeA);
    A_bounce = (cuComplex*)malloc(sizeof(cuComplex)*N*N);
    TESTING_CHECK( magma_cmalloc_cpu( &h_B, sizeB ));
    TESTING_CHECK( magma_cmalloc_cpu( &h_X, sizeB ));
    TESTING_CHECK( magma_cmalloc_cpu( &h_X_test, ldb ));
    //TESTING_CHECK( magma_cmalloc_cpu( &h_diags, sizeBand ));
    TESTING_CHECK( magma_smalloc_cpu( &work, N ));
    TESTING_CHECK( magma_imalloc_cpu( &ipiv, batchCount*N ));
    TESTING_CHECK( magma_imalloc_cpu( &cpu_info, batchCount ));

    TESTING_CHECK( magma_cmalloc( &d_A, ldda*N*batchCount    ));
    //TESTING_CHECK( magma_cmalloc( &d_B, lddb*nrhs*batchCount ));
    TESTING_CHECK( magma_imalloc( &dipiv, N * batchCount ));
    TESTING_CHECK( magma_imalloc( &dinfo_array, batchCount ));

    TESTING_CHECK( magma_malloc( (void**) &dA_array,    batchCount * sizeof(cuComplex*) ));
    //TESTING_CHECK( magma_malloc( (void**) &dB_array,    batchCount * sizeof(cuComplex*) ));
    TESTING_CHECK( magma_malloc( (void**) &dipiv_array, batchCount * sizeof(int*) ));


    int nn1, nt1, nb1, nn2, nt2, nb2, nn3, nt3, nb3;

    nn1 = N;		    nt1 = min(nn1, 512 );   nb1 = 1 + (nn1-1)/nt1;
    nn2 = 1;         	    nt2 = min(nn2,  1 );   nb2 = 1 + (nn2-1)/nt2;
    nn3 = 1;         	    nt3 = min(nn3,  1 );   nb3 = 1 + (nn3-1)/nt3;
    
    int nn4 = grids_->Nyc;             int nt4 = min(nn4, 16);   int nb4 = 1 + (nn4-1)/nt4;
    int nn5 = grids_->Nx;              int nt5 = min(nn5,  4);   int nb5 = 1 + (nn5-1)/nt5;
    int nn8 = grids_->Nz;   	     int nt8 = min(nn8,  8);   int nb8 = 1 + (nn8-1)/nt8;
    int nn6 = pars_->nm_in * pars_->nl_in; int nt6 = min(nn6, 4); int nb6 = 1 + (nn6-1)/nt6;
    int nn7 = grids_->Nx*grids_->Nz;   int nt7 = min(nn7,  8);   int nb7 = 1 + (nn7-1)/nt7;
    
    int nn9 = pars_->nm_in; 	     int nt9 = min(nn9, 16); int nb9 = 1 + (nn9-1)/nt9;
    int nn10 = pars_->nl_in;   	     int nt10 = min(nn10,  4);   int nb10 = 1 + (nn10-1)/nt10;


    dB = dim3(nt1, nt2, nt3);
    dG = dim3(nb1, nb2, nb3); 

    dB_bd = dim3(nt4, nt7, nt6);
    dG_bd = dim3(nb4, nb7, nb6);

    dB_lu_sm = dim3(nt9, nt10, nt8);
    dG_lu_sm = dim3(nb9, nb10, nb8);

    h_B = (cuComplex*) malloc(sizeof(cuComplex)*ldb*nrhs*batchCount);
    d_B = (cuComplex**) malloc(sizeof(cuComplex*)*batchCount);
    cuComplex* d_B_i = (cuComplex*) malloc(sizeof(cuComplex)*ldb*nrhs);
    checkCuda(cudaMalloc((void**) &dB_array, sizeof(cuComplex*)*batchCount));
    
    for(int i = 0; i < batchCount; i++){
        checkCuda(cudaMalloc((void**) &d_B[i], sizeof(cuComplex)*ldb*nrhs));
    }

    checkCuda(cudaMemcpy(dB_array, d_B, sizeof(cuComplex*)*grids_->Nz, cudaMemcpyHostToDevice));

    /* Initialize the matrices */
    int ione     = 1;
    h_diags = (cuComplex*)malloc(sizeof(cuComplex)*sizeBand);
    //cudaMemset(h_diags, 0.0f, sizeof(cuComplex)*sizeBand);
    fill_diags(h_diags, N, pars_->nm_in, pars_->nl_in, batchCount, Nband, dcoeff[0]);
    fill_A(h_A, h_diags, h_offsets, N, lda, batchCount, KL, KU, grids_->Nm, Nband);

    /*Initialize RHS and copy over to host*/
    fill_dB_array(dB_array, N, nrhs, ldb, batchCount);
    checkCuda(cudaMemcpy(dB_array, d_B, sizeof(cuComplex*)*grids_->Nz, cudaMemcpyDeviceToHost));
    for(int i = 0; i < batchCount; i++){
        checkCuda(cudaMemcpy(d_B[i], d_B_i, sizeof(cuComplex)*ldb*nrhs, cudaMemcpyDeviceToHost));
        for(int j = 0; j < ldb*nrhs; j++){
            h_B[i*ldb*nrhs + j] = d_B_i[j];
        }
    }
    //fill_B(h_B, N, nrhs, ldb, batchCount);

    save_rhs(h_B, h_X_test, ldb, 11, 0, nrhs);

    printf("PRINTING X\n");
    for(int b = 11; b < 12; b++){
        printf("\nBatch  = %d\n", b);
        for (int i =0; i < N; i++){
            for(int j = 0; j < nrhs; j++){
                printf("%f ", h_B[b*ldb*nrhs + j*N + i].x);
            }
            printf("\n");
        }
    }

    printf("magma coeff is %f\n", dcoeff[0]); 
    printf("magma bgrad is %f\n", geo_->bgrad_h[11]);
    printf("magma dt_ is %f\n", dt_);
    printf("magma vte is %f\n", vte);


    printf("PRINTING DIAGS\n");
    for(int i = 0; i < Nband; i++){
        for(int j = 0; j < N; j++){
            printf("%f ", h_diags[11 * Nband * N + i*N + j].x);
        }
        printf("\n");
    }

    save_matrix(h_A, A_bounce, h_offsets, KL, KU, Nband, N, lda, batchCount, 11);

    printf("PRINTING A\n");
    for (int i =0; i < lda; i++){
        for(int j = 0; j < N; j++){
            printf("%f ", h_A[11*lda*N + i+j*lda].x);
        }
        printf("\n");
    }

    cudaMemcpy(d_A, h_A, sizeof(cuComplex)*N*lda*batchCount, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_B, h_B, sizeof(cuComplex)*ldb*nrhs*batchCount, cudaMemcpyHostToDevice);

    /* ====================================================================
        Performs operation using MAGMA
        =================================================================== */
    magma_cset_pointer( dA_array, d_A, ldda, 0, 0, ldda*N,    batchCount, my_queue );
    //magma_cset_pointer( dB_array, d_B, lddb, 0, 0, lddb*nrhs, batchCount, my_queue );
    magma_iset_pointer( dipiv_array, dipiv, 1, 0, 0, N, batchCount, my_queue );

    // synchronous api with ptr array
    //gpu_time = magma_sync_wtime( opts.queue );

    info = magma_cgbsv_batched(
            N, KL, KU, nrhs,
            dA_array, ldda, dipiv_array,
            dB_array, lddb, dinfo_array,
            batchCount, my_queue);
    //gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;


    //magma_cgetmatrix( N, nrhs*batchCount, d_B, lddb, h_X, ldb, my_queue );

    checkCuda(cudaMemcpy(dB_array, d_B, sizeof(cuComplex*)*grids_->Nz, cudaMemcpyDeviceToHost));
    for(int i = 0; i < batchCount; i++){
        checkCuda(cudaMemcpy(d_B[i], d_B_i, sizeof(cuComplex)*ldb*nrhs, cudaMemcpyDeviceToHost));
        for(int j = 0; j < ldb*nrhs; j++){
            h_X[i*ldb*nrhs + j] = d_B_i[j];
        }
    }

    save_rhs(h_X, h_X_test, ldb, 11, 1, nrhs);
//
    printf("PRINTING X\n");
    for(int b = 11; b < 12; b++){
        printf("\nBatch  = %d\n", b);
        for (int i =0; i < N; i++){
            for(int j = 0; j < nrhs; j++){
                printf("%f ", h_X[b*ldb*nrhs + j*N + i].x);
            }
            printf("\n");
        }
    }

    error = 0;
    for (magma_int_t s=0; s < batchCount; s++) {
        cuComplex* hA = h_A + s * lda * N + KL;
        cuComplex* hX = h_X + s * ldb * nrhs;
        cuComplex* hB = h_B + s * ldb * nrhs;

        Anorm = lapackf77_clangb("I", &N, &KL, &KU, hA, &lda, work);
        Xnorm = lapackf77_clange("I", &N, &nrhs, hX, &ldb, work);

        for(magma_int_t j = 0; j < nrhs; j++) {
            blasf77_cgbmv( MagmaNoTransStr, &N, &N, &KL, &KU,
                            &c_one, hA           , &lda,
                                    hX  + j * ldb, &ione,
                        &c_neg_one, hB  + j * ldb, &ione);
        }

        Rnorm = lapackf77_clange("I", &N, &nrhs, hB, &ldb, work);

        double err = Rnorm/(N*Anorm*Xnorm);
        if (std::isnan(err) || std::isinf(err)) {
            error = err;
            break;
        }
        error = max( err, error );
    }
    printf("error is %.3e\n", error);
    bool okay = (error < tol);
    status += ! okay;

    printf("status is %d\n", status);


}

Mirror_magma::~Mirror_magma()
{

}

void Mirror_magma::fill_dB_array(cuComplex** dB_array,int N, int nrhs, int ldb, int batchCount){
    copy_brhs_from_g_d_id<<<dG_bd, dB_bd>>>(dB_array);
}

void Mirror_magma::save_rhs(cuComplex* h_B, cuComplex* h_X_test, int ldb, int ind, int id, int nrhs){
    for(int i = 0; i < ldb; i++){
        h_X_test[i] = h_B[ind*nrhs*ldb + i];
    }

    FILE* file;

    if(id == 0){
        file = fopen("rhs.bin", "wb");
    }
    else{
        file = fopen("rhs_sol.bin", "wb");
    }
    fwrite(h_X_test, sizeof(cuComplex), ldb, file);
    fclose(file);
}

void Mirror_magma::save_matrix(cuComplex* A, cuComplex* A_full, int* h_offsets, int KL, int KU, int num_diags, int N, int lda, int batchCount, int ind){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            A_full[i+j*N] = make_cuComplex(0.0f, 0.0f);
        }
    }

    int offset;
    // for(int j = 0; j < N; j++){
    //     for(int i = max(0,j-KU); i <= min(N-1,j+KL); ++i){
    //         A_full[i + j*N] = A[ind*lda*N + (KL + KU + i - j) + j*lda];
    //     }
    // }

    for(int i = 0; i < lda; i++){
        offset = KL + KU - i;
        for(int r = 0; r < N; r++){
            for(int c = 0; c < N; c++){
                if (c - r == offset){
                    A_full[r + c*N] = A[ind*lda*N + i + r*lda];
                }
            }
        }
    }
    FILE* file = fopen("magma_matrix.bin", "wb");
    fwrite(A_full, sizeof(cuComplex), N*N, file);
    fclose(file);
}

void Mirror_magma::fill_diags(cuComplex* diags, int N, int M, int L, int batchCount, int Nband, double coeff){
    for(int iz = 0; iz < grids_->Nz; iz++){
        for(int j = 0; j < N; j++){
            for(int i = 0; i < Nband; i++){
            int il = int(j / M);
            int im = j % M;
            double prefac = geo_->bgrad_h[iz] * dt_ * vte_ * coeff;
            //double prefac = 1.0f;

            cuComplex lp1mp1 = make_cuComplex(-(il+1)*sqrtf(im+1),0.0)*prefac;
            cuComplex lm = make_cuComplex(il*sqrtf(im),0.0)*prefac;
            cuComplex lp1m = make_cuComplex((il+1)*sqrtf(im),0.0)*prefac;
            cuComplex lmp1 = make_cuComplex(-il*sqrtf(im+1),0.0)*prefac;
            if (i == 0){
                if (im < M-1 and il > 0){
                diags[iz*Nband*L*M + i*L*M + j] = lmp1;
                }
            }
            else if (i == 1){
                if (im > 0){
                diags[iz*Nband*L*M + i*L*M + j] = lm;
                }
            }
            else if (i == 2){
                diags[iz*Nband*L*M + i*L*M + j] = make_cuComplex(1.0f,0.0f);
            }
            else if (i == 3){
                if (im < M-1){
                diags[iz*Nband*L*M + i*L*M + j] = lp1mp1;
                }
            }
            else if (i == 4){
                if (im > 0 and il < L-1){
                diags[iz*Nband*L*M + i*L*M + j] = lp1m;
                }
            }
            }
        }
    }

}

void Mirror_magma::fill_A(cuComplex* A, cuComplex* h_diags, int* h_offsets, int N, int lda, int batchCount, int KL, int KU, int M, int num_diags){
    int offset;
    int ind;

    //A[0] = make_cuComplex(0.0f, 0.0f);
    for(int b = 0; b < batchCount; b++){
        for(int i = 0; i < 2*KL + KU + 1; i++){
            for(int j = 0; j < N; j++){
                A[b*lda*N + j+i*N] = make_cuComplex(0.0f, 0.0f);
            }
        }
    }

    for(int b = 0; b < batchCount; b++){
        for(int d = 0; d < num_diags; d++){
            int offset = h_offsets[d];
            if(offset > 0){
                for (int j = 0; j < N; j++){
                    int A_offset = max(0, offset);
                    A[b*lda*N + KL + KU - offset + (j + A_offset)*lda] = h_diags[b*num_diags*N + d*N + j];
                }
            }
            else{
                for (int j = 0; j < N + offset; j++){
//                    int A_offset = max(0, -offset);
                    A[b*lda*N + KL + KU - offset + j*lda] = h_diags[b*num_diags*N + d*N + j-offset];
                }
            }

        }
    }
    

    // for(int b = 0; b < batchCount; b++){
    //     for(int j = 0; j < N; j++){
    //         for(int i = max(0,j-KU); i <min(N,j+KL); i++){
    //             offset = j - i;
    //             for(int d = 0; d < num_diags; d++){
    //                 if(offset == h_offsets[d]){
    //                     ind = d;
    //                     A[b*lda*N + (KL + KU + 1 + i - j) + j*lda] = h_diags[ind*N+i];
    //                 }
    //             }
    //         }
    //     }
    // }


}

void Mirror_magma::fill_B(cuComplex* B, int N, int nrhs, int ldb, int batchCount){
    for(int i = 0; i < batchCount; i++){
        for(int j = 0; j < nrhs; j++){
            for(int k = 0; k < N; k++){
                if(i == 11){
                    B[i*ldb*nrhs + j*N + k] = make_cuComplex((float) (k+0.5f), 0.0f);
                }
                else{
                    B[i*ldb*nrhs + j*N + k] = make_cuComplex((float) (k+1.0f), 0.0f);
                }
            }
        }
    }
}

// Mirror_magma::Mirror_magma(Parameters *pars, Grids *grids, Geometry *geo, double p, double r, double u, bool sdirk, double dt_in, double vte):
// 	p_(p), r_(r), u_(u), sdirk_(sdirk), grids_(grids), pars_(pars), geo_(geo), A_bounce(nullptr), dt_(dt_in), vte_(vte)
// {

//     magma_init();
    
//     magma_queue_t my_queue;    // magma queue variable, internally holds a cuda stream and a cublas handle
//     magma_device_t cdev;       // variable to indicate current gpu id

//     magma_getdevice( &cdev );
//     magma_queue_create( cdev, &my_queue );     // create a queue on this cdev

//     int N = 4;
//     int Nband = 3;
//     int KL = 1;
//     int KU = 1;
//     int nrhs = 2;
//     int batchCount = 2;
//     int ldda = 2*KL+KU+1;
//     int lddb = N;
//     int lda = ldda;
//     int ldb = lddb;
//     int info = 0;

//     int sizeA = lda*N*batchCount;
//     int sizeB = ldb*nrhs*batchCount;

//     cuComplex *h_A, *h_B, *h_X, *d_A, *d_B;
//     float error, Rnorm, Anorm, Xnorm, *work;
//     int* ipiv, *cpu_info;
//     int *dipiv, *dinfo_array;

//     cuComplex c_one     = MAGMA_C_ONE;
//     cuComplex c_neg_one = MAGMA_C_NEG_ONE;

//     cuComplex **dA_array = NULL;
//     cuComplex **dB_array = NULL;
//     int     **dipiv_array = NULL;

//     float tol = 1e-6;
//     int status = 0;

//     TESTING_CHECK( magma_cmalloc_cpu( &h_A, sizeA ));
//     TESTING_CHECK( magma_cmalloc_cpu( &h_B, sizeB ));
//     TESTING_CHECK( magma_cmalloc_cpu( &h_X, sizeB ));
//     TESTING_CHECK( magma_smalloc_cpu( &work, N ));
//     TESTING_CHECK( magma_imalloc_cpu( &ipiv, batchCount*N ));
//     TESTING_CHECK( magma_imalloc_cpu( &cpu_info, batchCount ));

//     TESTING_CHECK( magma_cmalloc( &d_A, ldda*N*batchCount    ));
//     TESTING_CHECK( magma_cmalloc( &d_B, lddb*nrhs*batchCount ));
//     TESTING_CHECK( magma_imalloc( &dipiv, N * batchCount ));
//     TESTING_CHECK( magma_imalloc( &dinfo_array, batchCount ));

//     TESTING_CHECK( magma_malloc( (void**) &dA_array,    batchCount * sizeof(cuComplex*) ));
//     TESTING_CHECK( magma_malloc( (void**) &dB_array,    batchCount * sizeof(cuComplex*) ));
//     TESTING_CHECK( magma_malloc( (void**) &dipiv_array, batchCount * sizeof(int*) ));

//     /* Initialize the matrices */
//     int ione     = 1;
//     fill_A(h_A, N, lda, batchCount, KL, KU);
//     fill_B(h_B, N, nrhs, ldb, batchCount);

//     for(int b = 0; b < batchCount; b++){
//         printf("\nBatch  = %d\n", b);
//         for (int i =0; i < 2*KL + KU + 1; i++){
//             for(int j = 0; j < N; j++){
//                 printf("%f ", h_A[b*lda*N + i+j*N].x);
//             }
//             printf("\n");
//         }
//     }

//     cudaMemcpy(d_A, h_A, sizeof(cuComplex)*N*lda*batchCount, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, sizeof(cuComplex)*ldb*nrhs*batchCount, cudaMemcpyHostToDevice);

//     /* ====================================================================
//         Performs operation using MAGMA
//         =================================================================== */
//     magma_cset_pointer( dA_array, d_A, ldda, 0, 0, ldda*N,    batchCount, my_queue );
//     magma_cset_pointer( dB_array, d_B, lddb, 0, 0, lddb*nrhs, batchCount, my_queue );
//     magma_iset_pointer( dipiv_array, dipiv, 1, 0, 0, N, batchCount, my_queue );

//     // synchronous api with ptr array
//     //gpu_time = magma_sync_wtime( opts.queue );

//     info = magma_cgbsv_batched(
//             N, KL, KU, nrhs,
//             dA_array, ldda, dipiv_array,
//             dB_array, lddb, dinfo_array,
//             batchCount, my_queue);
//     //gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;


//     magma_cgetmatrix( N, nrhs*batchCount, d_B, lddb, h_X, ldb, my_queue );
// //
//     for(int b = 0; b < batchCount; b++){
//         printf("\nBatch  = %d\n", b);
//         for (int i =0; i < N; i++){
//             for(int j = 0; j < nrhs; j++){
//                 printf("%f ", h_X[b*ldb*nrhs + j*N + i].x);
//             }
//             printf("\n");
//         }
//     }

//     error = 0;
//     for (magma_int_t s=0; s < batchCount; s++) {
//         cuComplex* hA = h_A + s * lda * N + KL;
//         cuComplex* hX = h_X + s * ldb * nrhs;
//         cuComplex* hB = h_B + s * ldb * nrhs;

//         Anorm = lapackf77_clangb("I", &N, &KL, &KU, hA, &lda, work);
//         Xnorm = lapackf77_clange("I", &N, &nrhs, hX, &ldb, work);

//         for(magma_int_t j = 0; j < nrhs; j++) {
//             blasf77_cgbmv( MagmaNoTransStr, &N, &N, &KL, &KU,
//                             &c_one, hA           , &lda,
//                                     hX  + j * ldb, &ione,
//                         &c_neg_one, hB  + j * ldb, &ione);
//         }

//         Rnorm = lapackf77_clange("I", &N, &nrhs, hB, &ldb, work);

//         double err = Rnorm/(N*Anorm*Xnorm);
//         if (std::isnan(err) || std::isinf(err)) {
//             error = err;
//             break;
//         }
//         error = max( err, error );
//     }
//     printf("error is %f\n", error);
//     bool okay = (error < tol);
//     status += ! okay;

//     printf("status is %d\n", status);


// }

// Mirror_magma::~Mirror_magma()
// {
// }

// void Mirror_magma::fill_A(cuComplex* A, int N, int lda, int batchCount, int KL, int KU){
//     cuComplex onec = make_cuComplex(1.0f, 0.0f);
//     cuComplex zeroc = make_cuComplex(0.0f, 0.0f);
//     cuComplex twoc = make_cuComplex(2.0f, 0.0f);
//     cuComplex threec = make_cuComplex(3.0f, 0.0f);
//     cuComplex fourc = make_cuComplex(4.0f, 0.0f);

//     cuComplex h_diags [3*N] = {zeroc,onec,twoc,threec,onec,twoc,threec,fourc,threec,twoc,onec,zeroc};

//     int num_diags = 3;
//     int offsets [3] = {-1,0,1};

//     int offset;
//     int ind;

//     for(int b = 0; b < batchCount; b++){
//         for(int i = 0; i < 2*KL + KU + 1; i++){
//             for(int j = 0; j < N; j++){
//                 A[b*lda*N + i+j*N] = make_cuComplex(0.0f, 0.0f);
//             }
//         }
//     }

//     for(int b = 0; b < batchCount; b++){
//         for(int j = 0; j < N; j++){
//             for(int i = max(0,j-KU); i <min(N,j+KL+1); i++){
//                 offset = j - i;
//                 for(int d = 0; d < num_diags; d++){
//                     if(offset == offsets[d]){
//                         ind = d;
//                     }
//                 }
//                 A[b*lda*N + (KL + KU + i - j) + j*N] = h_diags[ind*N+i];
//             }
//         }
//     }
// }

// void Mirror_magma::fill_B(cuComplex* B, int N, int nrhs, int ldb, int batchCount){
//     for(int i = 0; i < batchCount; i++){
//         for(int j = 0; j < nrhs; j++){
//             for(int k = 0; k < N; k++){
//                 B[i*ldb*nrhs + j*N + k] = make_cuComplex((float) (k+1.0f), 0.0f);
//             }
//         }
//     }
// }

