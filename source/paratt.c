/*
 * C Source file for Parallel Tensor Train by Bram Rodgers.
 * Original Draft Dated: 16, June 2022
 */

/*
 * Macros and Includes go here: (Some common ones included)
 */
#include "paratt.h"
#include "math.h"
#include "string.h"
#include<sys/types.h>
#include<sys/stat.h>
#include<unistd.h>
/*
 * Locally used helper functions:
 */


extern void dgemv_(const char* trans,
                   const int*  m,
                   const int*  n,
                   double*     alpha,
                   double*     a,
                   const int*  lda,
                   double*     x,
                   const int*  incx,
                   double*     beta,
                   double*     y,
                   const int*  incy);

/*
 * Static Local Variables:
 */

//given a full tensor multiindex, get an array that tells you which
//mpi rank index the tensor's slices are.
void full_idx_2_proc_idx(paratt* x_ptt, int* mIdx, int* procIdx){
    int k, dimRem, chunkSize;
    for(k=0;k<x_ptt->numDim;k++){
        //step 1: is mIdx in pre-remainder zone?
        dimRem = x_ptt->dim[k] % x_ptt->mpi_sz;
        chunkSize = x_ptt->dim[k] / x_ptt->mpi_sz;
        if(mIdx[k] < dimRem*(chunkSize+1)){
            procIdx[k] = mIdx[k] / (chunkSize+1);
        }else{
            procIdx[k] = dimRem + 
                         ((mIdx[k] - dimRem*(chunkSize+1)) / chunkSize);
        }
    }
}
//given a full tensor multiindex, get an array that tells you the offset
//for the locally stored TT tensor slices
void full_idx_2_dist_idx(paratt* x_ptt, int* mIdx, int* distIdx){
    int k, dimRem, chunkSize;
    for(k=0;k<x_ptt->numDim;k++){
        //step 1: is mIdx in pre-remainder zone?
        dimRem = x_ptt->dim[k] % x_ptt->mpi_sz;
        chunkSize = x_ptt->dim[k] / x_ptt->mpi_sz;
        if(mIdx[k] < dimRem*(chunkSize+1)){
            distIdx[k] = mIdx[k] % (chunkSize+1);
        }else{
            distIdx[k] = (mIdx[k] - dimRem*(chunkSize+1)) % chunkSize;
        }
    }
}

/*
 * Function Implementations:
 */


attac_tt* alloc_tt(int ndims, int* d, int* ttR){
    int k=0;
    attac_tt* tt  = (attac_tt*) malloc(sizeof(attac_tt));
    tt->numDim  = ndims;
    tt->dim = (int*) malloc(sizeof(int)*ndims);
    tt->ttRank    = (int*) malloc(sizeof(int)*(ndims+1));
    tt->coreList  = (double**) malloc(sizeof(double*)*ndims);
    tt->coreMem   = (int*) malloc(sizeof(int)*ndims);
    k = 0;
    tt->dim[k] = d[k];
    tt->ttRank[k]    = ttR[k];
    tt->coreList[k]  = (double*) 
                            malloc(sizeof(double)*d[k]*ttR[k+1]);
    tt->coreMem[k] = d[k]*ttR[k+1];
    for(k=1;k<ndims;k++){
        tt->dim[k] = d[k];
        tt->ttRank[k]    = ttR[k];
        tt->coreList[k]  = (double*) 
                             malloc(sizeof(double)*d[k]*ttR[k]*ttR[k+1]);
        tt->coreMem[k] = d[k]*ttR[k]*ttR[k+1];
    }
    k = ndims;
    tt->ttRank[k]    = ttR[k];
    tt->wk = NULL;
    tt->wksz = 0;
    return tt;
}

paratt* alloc_ptt(int ndims, int* d, int* ttR){
    paratt* ptt  = (paratt*) malloc(sizeof(paratt));
    int* global_d = (int*) malloc(sizeof(int)*(ndims));
    int* local_d;
    int* alloc_tt_dim;
    int k, j,rem;
    MPI_Comm_rank(MPI_COMM_WORLD, &(ptt->mpi_rk));
    MPI_Comm_size(MPI_COMM_WORLD, &(ptt->mpi_sz));
    alloc_tt_dim = (int*) malloc(sizeof(int)*(ndims+1));
    local_d  = (int*) malloc(sizeof(int)*(ndims*ptt->mpi_sz));
    for(k = 0;k < ndims;k++){
        global_d[k] = d[k];
        rem  = d[k] % ptt->mpi_sz;
        for(j=0;j<ptt->mpi_sz;j++){
            local_d[j+ptt->mpi_sz*k] = d[k] / ptt->mpi_sz;
            if(j < rem){
                local_d[j+ptt->mpi_sz*k]  += 1;
            }
        }
        alloc_tt_dim[k] = local_d[ptt->mpi_rk + ptt->mpi_sz*k];
    }
    ptt->dim = global_d;
    ptt->numDim = ndims;
    ptt->x = alloc_tt(ndims, alloc_tt_dim, ttR);
    ptt->distrib_dim_mat = local_d;
    ptt->iwk = alloc_tt_dim;
    ptt->iwksz = ndims+1;
    ptt->orthog_wk = alloc_tsqr_tree(MPI_COMM_WORLD);
    return ptt;
}


attac_tt* rank_change_tt(attac_tt* x_tt, int* ttR){
    int k=0;
    for(k=0;k < x_tt->numDim;k++){
        if(x_tt->coreMem[k] < x_tt->dim[k]*ttR[k]*ttR[k+1]){
            x_tt->coreList[k]  = (double*) realloc(x_tt->coreList[k],
                                sizeof(double)*
                                x_tt->dim[k]*ttR[k]*ttR[k+1]);
            x_tt->coreMem[k] = x_tt->dim[k]*ttR[k]*ttR[k+1];
        }
        x_tt->ttRank[k] = ttR[k];
    }
    k = x_tt->numDim;
    x_tt->ttRank[k] = ttR[k];
    return x_tt;
}

void copy_rank_change_tt(attac_tt* dest_tt,attac_tt* src_tt){
    int k=0;
    int numCols;
    dest_tt = rank_change_tt(dest_tt, src_tt->ttRank);
    for(k=0;k<(dest_tt->numDim);k++){
        numCols = src_tt->dim[k] * src_tt->ttRank[k+1];
        dlacpy_("A", &(src_tt->ttRank[k]),
                     &(numCols),
                     src_tt->coreList[k],
                     &(src_tt->ttRank[k]),
                     dest_tt->coreList[k],
                     &(dest_tt->ttRank[k]));
    }
}

void free_tt(attac_tt* x_tt){
    int k=0;
    free(x_tt->dim);
    free(x_tt->ttRank);
    for(k=0;k<x_tt->numDim;k++){
        free(x_tt->coreList[k]);
    }
    free(x_tt->coreList);
    free(x_tt->coreMem);
    if(x_tt->wk != NULL){
        free(x_tt->wk);
    }
    free(x_tt);
}


void free_ptt(paratt* x_ptt){
    free_tt(x_ptt->x);
    free(x_ptt->distrib_dim_mat);
    if(x_ptt->iwk != NULL){
        free(x_ptt->iwk);
    }
    free(x_ptt->dim);
    free_tsqr_tree(x_ptt->orthog_wk);
    free(x_ptt);
}


void send_recv_above(void* send_buff,
                     void* recv_buff,
                     int msg_sz,
                     MPI_Datatype dat_t,
                     MPI_Request *sendReq,
                     MPI_Request *recvReq){
    int dest_proc, source_proc;
    int mpi_rk, mpi_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &(mpi_rk));
    MPI_Comm_size(MPI_COMM_WORLD, &(mpi_sz));
    dest_proc = mpi_rk + 1;
    source_proc = mpi_rk - 1;
    if(dest_proc >= mpi_sz){
        dest_proc -= mpi_sz;
    }
    if(source_proc < 0){
        source_proc += mpi_sz;
    }
    MPI_Isend(send_buff, msg_sz, dat_t, dest_proc,0,MPI_COMM_WORLD, sendReq);
    MPI_Irecv(recv_buff, msg_sz, dat_t,source_proc,0,MPI_COMM_WORLD,recvReq);
}

void send_recv_below(void* send_buff,
                     void* recv_buff,
                     int msg_sz,
                     MPI_Datatype dat_t,
                     MPI_Request *sendReq,
                     MPI_Request *recvReq){
    int dest_proc, source_proc;
    int mpi_rk, mpi_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &(mpi_rk));
    MPI_Comm_size(MPI_COMM_WORLD, &(mpi_sz));
    dest_proc = mpi_rk - 1;
    source_proc = mpi_rk + 1;
    if(dest_proc < 0){
        dest_proc += mpi_sz;
    }
    if(source_proc >= mpi_sz){
        source_proc -= mpi_sz;
    }
    MPI_Isend(send_buff, msg_sz, dat_t, dest_proc,0,MPI_COMM_WORLD, sendReq);
    MPI_Irecv(recv_buff, msg_sz, dat_t,source_proc,0,MPI_COMM_WORLD,recvReq);
}


void attac_tt_axpby(double a, attac_tt* x_tt, 
                    double b, attac_tt* y_tt, attac_tt* z_tt){
    int k;
    int newRank = 0;
    int newRank_p1 = 0;
    for(k=0;k<z_tt->numDim;k++){
        if(k < 1){
            newRank = 1;
            newRank_p1 = x_tt->ttRank[k+1]+y_tt->ttRank[k+1];
        }else if(k<z_tt->numDim-1){
            newRank = x_tt->ttRank[k]+y_tt->ttRank[k];
            newRank_p1 = x_tt->ttRank[k+1]+y_tt->ttRank[k+1];
        }else{
            newRank = x_tt->ttRank[k]+y_tt->ttRank[k];
            newRank_p1 = 1;
        }
        if(z_tt->coreMem[k] < z_tt->dim[k]*newRank*newRank_p1){
            z_tt->coreList[k] = (double*) realloc(z_tt->coreList[k],
                               sizeof(double)*z_tt->dim[k]*newRank*newRank_p1);
            z_tt->coreMem[k] = z_tt->dim[k]*newRank*newRank_p1;
        }
        z_tt->ttRank[k] = newRank;
    }
    z_tt->ttRank[z_tt->numDim] = 1;
    attacFormalAXPBY(x_tt->numDim,
                     x_tt->dim,
                     a, x_tt->ttRank,
                     x_tt->coreList,
                     b, y_tt->ttRank,
                     y_tt->coreList,
                     z_tt->ttRank, z_tt->coreList);
}


void attac_tt_scale(double a, attac_tt* x_tt){
    int i,j,k;
    for(i=0;i<x_tt->ttRank[0];i++){
        for(j=0;j<x_tt->dim[0];j++){
            for(k=0;k<x_tt->ttRank[1];k++){
                x_tt->coreList[0][i+
                         x_tt->ttRank[0]*(j+x_tt->dim[0]*k)] *= a;
            }
        }
    }
}

void attac_tt_truncate(attac_tt* x_tt, double tol, int* maxRank){
    int rMax, coreMaxSz,i;
    char ortho = 'R';
    //Step 1: calculate the work size needed to apply the truncaiton operation.
    //Follows formula wksz = max(x_tt->ttRank[i]*
    //                           x_tt->dim[i]*
    //                           x_tt->ttRank[i + 1]) + 3 rMax^2 + 2*rMax
    //where rMax = max(x_tt->ttRank[i])
    rMax = 0;
    coreMaxSz = 0;
    for(i=0;i<x_tt->numDim;i++){    
        coreMaxSz = (coreMaxSz > 
                        (x_tt->ttRank[i]*x_tt->dim[i]*x_tt->ttRank[i + 1])) ?
                    coreMaxSz : 
                        (x_tt->ttRank[i]*x_tt->dim[i]*x_tt->ttRank[i + 1]);
        rMax = (rMax > x_tt->ttRank[i]) ? rMax : x_tt->ttRank[i];
    }
    coreMaxSz += rMax*(3*rMax + 2);
    if(x_tt->wksz < coreMaxSz){
        x_tt->wksz = coreMaxSz;
        x_tt->wk = (double*) realloc(x_tt->wk,sizeof(double)*coreMaxSz);
    }
    //Step 2: call the attacCompress function of Hussam's library.
    attacTruncate(x_tt->numDim,//const int d,
                  x_tt->dim,//const int *n,
                  x_tt->ttRank,//int *rx,
                  x_tt->coreList,//double **x,
                  maxRank,
                  tol,//const double threshold,
                  ortho,//const char ortho,
                  x_tt->wk);//double *work)
}

void paratt_truncate(paratt* x_ptt, double tol, int* maxRank){
    int rMax, i;
    char ortho = 'R';
    rMax = 1;
    for(i=0; i < x_ptt->numDim; i++){
        rMax = (rMax > x_ptt->x->ttRank[i]*x_ptt->x->dim[i]) ?
                rMax : x_ptt->x->ttRank[i]*x_ptt->x->dim[i];
    }
    //get the needed size of the integer work vector
    if(x_ptt->iwksz < (x_ptt->mpi_sz + (2*rMax))){
        x_ptt->iwksz = x_ptt->mpi_sz + (2*rMax);
        x_ptt->iwk = (int*) realloc(x_ptt->iwk,sizeof(int)*x_ptt->iwksz);
    }
    pattacTruncate(MPI_COMM_WORLD,
                   x_ptt->numDim, 
                   x_ptt->distrib_dim_mat,
                   x_ptt->x->ttRank,// *rx,
                   x_ptt->x->coreList,// double **x,
                   maxRank,
                   tol,
                   tol,
                   ortho,
                   x_ptt->orthog_wk,
                   x_ptt->iwk,       //assumed size = MPI num procs
                   &(x_ptt->x->wk),// double** flt_wrk_ptr, //work query
                   &(x_ptt->x->wksz));
}


double attac_tt_norm(attac_tt* x_tt){
    int rMax=0, maxCoreSz=0, i=0;
    double nrm;
    for(i=0;i<x_tt->numDim;i++){
        rMax = (rMax < x_tt->ttRank[i]) ? x_tt->ttRank[i] : rMax;
        maxCoreSz = (maxCoreSz < 
                     x_tt->ttRank[i]*x_tt->dim[i]*x_tt->ttRank[i+1]) ?
                    (x_tt->ttRank[i]*x_tt->dim[i]*x_tt->ttRank[i+1]) :
                     maxCoreSz; 
    }
    if(x_tt->wksz < maxCoreSz + (2*rMax*rMax)){
        x_tt->wksz = maxCoreSz + (2*rMax*rMax);
        x_tt->wk = (double*) realloc(x_tt->wk,
                                sizeof(double)*x_tt->wksz);
    }
    nrm = attacNorm(x_tt->numDim,//const int d,
                    x_tt->dim,// const int *n,
                    x_tt->ttRank,// const int *rx,
                    x_tt->coreList,//double **x,
                    x_tt->wk);
    return nrm;
}


double paratt_norm(paratt* x_ptt){
    int rMax=0, i,j;
    rMax = 1;
    double nrm;
    for(i=0; i < x_ptt->numDim; i++){
        rMax = (rMax > x_ptt->x->ttRank[i]*x_ptt->x->dim[i]) ?
                rMax : x_ptt->x->ttRank[i]*x_ptt->x->dim[i];
    }
    //get the needed size of the integer work vector
    if(x_ptt->iwksz < (x_ptt->mpi_sz + (12*rMax))){
        x_ptt->iwksz = x_ptt->mpi_sz + (12*rMax);
        x_ptt->iwk = (int*) realloc(x_ptt->iwk,sizeof(int)*x_ptt->iwksz);
    }
    for(i = 0; i < x_ptt->numDim; i++){
	    for(j=0; j < x_ptt->x->ttRank[i]*
	                 x_ptt->x->dim[i]*x_ptt->x->ttRank[i+1]; j++){
	        x_ptt->x->coreList[i][j] /= sqrt((double)x_ptt->dim[i]);
	    }
	}
    pattacLOrthogonalization(MPI_COMM_WORLD, x_ptt->numDim,
                             x_ptt->distrib_dim_mat,
                             x_ptt->x->ttRank, x_ptt->x->coreList,
                             x_ptt->orthog_wk, x_ptt->iwk);
    // **< Frobenius norm of the last TT core
    nrm = dlange_("F", &(x_ptt->x->ttRank[x_ptt->numDim - 1]),
                    &(x_ptt->distrib_dim_mat[x_ptt->mpi_rk+
                                    (x_ptt->numDim-1)*x_ptt->mpi_sz]),
                    x_ptt->x->coreList[x_ptt->numDim - 1],
                    &(x_ptt->x->ttRank[x_ptt->numDim - 1]), NULL);
    // < Squared norm 
    nrm *= nrm;
    MPI_Allreduce(MPI_IN_PLACE, &nrm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    nrm = sqrt(nrm);
    for(i = 0; i < x_ptt->numDim; i++){
	    for(j=0; j < x_ptt->x->ttRank[i]*
	                 x_ptt->x->dim[i]*x_ptt->x->ttRank[i+1]; j++){
	       x_ptt->x->coreList[i][j] *= sqrt((double)x_ptt->dim[i]);
	    }
	}
    return nrm;
}


double paratt_dotprod(paratt* x_ptt, paratt* y_ptt){
    int i,j,k, maxRank, maxDim;
    double val = 0.0, sqrt_dim;
    maxRank = 0;
    maxDim  = 0;
    for(k=0;k<x_ptt->numDim;k++){
        maxRank = (maxRank > x_ptt->x->ttRank[k]) ?
                            maxRank : x_ptt->x->ttRank[k];
        maxRank = (maxRank > y_ptt->x->ttRank[k]) ?
                            maxRank : y_ptt->x->ttRank[k];
        maxDim  = (maxDim > x_ptt->x->dim[k]) ?
                            maxDim : x_ptt->x->dim[k];
        x_ptt->iwk[k] = 0;
    }
    if(x_ptt->x->wksz < maxRank*(maxRank + maxRank*maxDim)){
        x_ptt->x->wksz = maxRank*(maxRank + maxRank*maxDim);
        x_ptt->x->wk = (double*) realloc(x_ptt->x->wk,
                                sizeof(double)*x_ptt->x->wksz);
    }
    for(i = 0; i < x_ptt->numDim; i++){
        sqrt_dim = sqrt((double)x_ptt->dim[i]);
	    for(j=0; j < x_ptt->x->ttRank[i]*
	                 x_ptt->x->dim[i]*x_ptt->x->ttRank[i+1]; j++){
	        x_ptt->x->coreList[i][j] /= sqrt_dim;
	    }
	}
    if(y_ptt != x_ptt){
        for(i = 0; i < y_ptt->numDim; i++){
            sqrt_dim = sqrt((double)y_ptt->dim[i]);
            for(j=0; j < y_ptt->x->ttRank[i]*
                        y_ptt->x->dim[i]*y_ptt->x->ttRank[i+1]; j++){
                y_ptt->x->coreList[i][j] /= sqrt_dim;
            }
        }
    }
    val = pattacInnerProduct(MPI_COMM_WORLD, x_ptt->iwk, x_ptt->numDim,
                             x_ptt->x->dim, x_ptt->x->ttRank,
                             x_ptt->x->coreList, y_ptt->x->ttRank,
                             y_ptt->x->coreList, x_ptt->x->wk);
    
    for(i = 0; i < x_ptt->numDim; i++){
	    sqrt_dim = sqrt((double)x_ptt->dim[i]);
	    for(j=0; j < x_ptt->x->ttRank[i]*
	                 x_ptt->x->dim[i]*x_ptt->x->ttRank[i+1]; j++){
	        x_ptt->x->coreList[i][j] *= sqrt_dim;
	    }
	}
    if(y_ptt != x_ptt){
        for(i = 0; i < y_ptt->numDim; i++){
            sqrt_dim = sqrt((double)y_ptt->dim[i]);
            for(j=0; j < y_ptt->x->ttRank[i]*
                        y_ptt->x->dim[i]*y_ptt->x->ttRank[i+1]; j++){
                y_ptt->x->coreList[i][j] *= sqrt_dim;
            }
        }
    }
    return val;
}

double paratt_entry(paratt* x_ptt, int* mIdx){
    int k,maxRank,*local_mIdx,*proc_holder;
    double val = 0.0;
    if(x_ptt->iwksz < (2*x_ptt->numDim)){
        x_ptt->iwksz = (2*x_ptt->numDim);
        x_ptt->iwk = (int*) realloc(x_ptt->iwk,
                                sizeof(int)*x_ptt->iwksz);
    }
    maxRank = 0;
    for(k=0;k<x_ptt->numDim;k++){
        maxRank = (maxRank > x_ptt->x->ttRank[k]) ?
                            maxRank : x_ptt->x->ttRank[k];
    }
    if(x_ptt->x->wksz < 2*maxRank){
        x_ptt->x->wksz = (2*maxRank);
        x_ptt->x->wk = (double*) realloc(x_ptt->x->wk,
                                sizeof(double)*x_ptt->x->wksz);
    }
    
    
    local_mIdx = x_ptt->iwk;
    proc_holder = &(x_ptt->iwk[x_ptt->numDim]);
    full_idx_2_dist_idx(x_ptt, mIdx, local_mIdx);
    full_idx_2_proc_idx(x_ptt, mIdx, proc_holder);
    val = pattacValue(MPI_COMM_WORLD, local_mIdx, proc_holder,
                x_ptt->numDim, x_ptt->x->dim, x_ptt->x->ttRank,
                x_ptt->x->coreList,x_ptt->x->wk);
    return val;
}


void attac_set_rank1(attac_tt** x_tt_ptr, double** vec, int ndims, int* d){
    int i,j;
    int* ttR_for_alloc;
    ttR_for_alloc = (int*) malloc(sizeof(int)*(ndims+1));
    for(i=0;i<ndims+1;i++){
        ttR_for_alloc[i] = 1;
    }
    *x_tt_ptr = alloc_tt(ndims, d,ttR_for_alloc);
    for(i=0;i<ndims;i++){
        for(j=0;j<d[i];j++){
            (*x_tt_ptr)->coreList[i][j] = vec[i][j];
        }
    }
    free(ttR_for_alloc);
}


attac_tt* attac_sum_of_product( double*** vec,
                                   int ndims, int* d, int all_ttR){
    attac_tt* L_sum;//The left argument of the sum. Changes rank during loop.
    attac_tt* R_sum;//The right argument of the sum. Stays rank 1 during loop.
    attac_tt* S_out;//the final output of the sum.
    int* ttR_for_alloc;//rank vector for allocating memory to above tt tensors
    int i,k;//loop iterators
    if(all_ttR == 1){
        attac_set_rank1(&S_out, *vec, ndims, d);
        return S_out;
    }
    ttR_for_alloc = (int*)  malloc(sizeof(int)*(ndims+1));
    for(i=0;i<ndims+1;i++){
        ttR_for_alloc[i] = 1;
    }
    attac_set_rank1(&L_sum, vec[0], ndims, d);
    attac_set_rank1(&R_sum, vec[1], ndims, d); 
    ttR_for_alloc[0]=1;
    for(i=1;i<ndims;i++){
        ttR_for_alloc[i] = 2;
    }
    ttR_for_alloc[ndims]=1;
    S_out = alloc_tt(ndims, d,ttR_for_alloc);
    //Loop over all the terms in the sum of rank one tensors.
    for(k=1;k<all_ttR;k++){
        attacFormalSum(ndims,//number of modes
                       d,//dimension for each mode
                       L_sum->ttRank,//ranks of the left sum
                       L_sum->coreList,//core list of the left sum
                       R_sum->ttRank,//ranks of the right sum
                       R_sum->coreList,//core list of the right sum
                       S_out->ttRank,//resulting ranks of output
                       S_out->coreList);//core list of output.
        if(k < all_ttR-1){//if there are more terms left in sum
            copy_rank_change_tt(L_sum,S_out);//copy the sum result into L
            free_tt(R_sum);                  //delete old sum term
            attac_set_rank1(&R_sum, vec[k+1], ndims, d);//load new sum term
            //The next few lines resize the memory for S_out to call sum again
            for(i=1;i<ndims;i++){
                ttR_for_alloc[i] = L_sum->ttRank[i] + 1;
            }
            S_out = rank_change_tt(S_out, ttR_for_alloc);
        }else{
            free_tt(L_sum);
            free_tt(R_sum);
        }
    }
    free(ttR_for_alloc);
    return S_out;
}


void attac_mult_diag_mat(attac_tt* f_tt, int mode, double* vec){
    int i,j,k;
    for(i=0;i<f_tt->ttRank[mode];i++){
        for(j=0;j<f_tt->dim[mode];j++){
            for(k=0;k<f_tt->ttRank[mode+1];k++){
                f_tt->coreList[mode][i+
                         f_tt->ttRank[mode]*(j+f_tt->dim[mode]*k)] *= vec[j];
            }
        }
    }
}


void attac_apply_periodic_stencil(attac_tt* f_tt, int mode,
                                        double* sten, int slen){
    int i,j,k,a;
    int ghostSize, ghostOffset, coreOffset,slen_m1_o2,slen_bot;
    double *matmul_out;
    slen = (slen % 2) ? (slen) : (slen-1);
    slen_m1_o2 = (slen-1)/2;
    //resize the work vector to hold the top and bottom row for ghost
    //cell storage.
    ghostSize = f_tt->ttRank[mode]*f_tt->ttRank[mode+1];
    if(f_tt->wksz < (slen-1)*ghostSize + f_tt->dim[mode]){
        f_tt->wksz = (slen-1)*ghostSize + f_tt->dim[mode];
        f_tt->wk = (double*) realloc(f_tt->wk,sizeof(double)*f_tt->wksz);
    }
    matmul_out = &(f_tt->wk[(slen-1)*ghostSize]);
    //copy the ghost cells into the work vector
    for(k=0;k<f_tt->ttRank[mode+1];k++){
        for(j=0;j<slen_m1_o2;j++){
            for(i=0;i<f_tt->ttRank[mode];i++){
                //the ghost cells above the top row
                f_tt->wk[i + f_tt->ttRank[mode]*(j + (slen_m1_o2)*k)] =
                        f_tt->coreList[mode][i + 
                              f_tt->ttRank[mode]*(j+f_tt->dim[mode]-slen_m1_o2
                              +(f_tt->dim[mode]*k))];
                //the ghost cells below the bottom row
                f_tt->wk[(slen_m1_o2)*ghostSize +
                            i + f_tt->ttRank[mode]*(j +(slen_m1_o2)*k)] =
                        f_tt->coreList[mode][i + 
                              f_tt->ttRank[mode]*(j +(f_tt->dim[mode]*k))];
            }
        }
    }
    //Apply the stencil to coreList[mode]
    for(k=0;k<f_tt->ttRank[mode+1];k++){
        for(i=0;i<f_tt->ttRank[mode];i++){
            ghostOffset = i + ((f_tt->ttRank[mode]*slen_m1_o2)*k);
            coreOffset  = i + f_tt->ttRank[mode]*f_tt->dim[mode]*k;
            for(a=0;a<f_tt->dim[mode];a++){
                matmul_out[a] = 0.0;
                if(a < slen_m1_o2){//top layer of gohst cells
                    for(j=0;j<(slen_m1_o2)-a;j++){
                        matmul_out[a] += sten[j]*
                                        f_tt->wk[ghostOffset +
                                                   f_tt->ttRank[mode]*(j+a)];
                    }
                    for(j=0;j<=a+(slen_m1_o2);j++){
                        matmul_out[a] += sten[j+(slen_m1_o2)-a]*
                                             f_tt->coreList[mode][coreOffset+
                                                    f_tt->ttRank[mode]*j];
                    }
                }else if(a < f_tt->dim[mode]-(slen_m1_o2) ){
                    //if no ghost cell is needed
                    for(j=0;j<slen;j++)
                    matmul_out[a] += sten[j]*f_tt->coreList[mode][coreOffset+
                                             f_tt->ttRank[mode]*
                                                (a+j-(slen_m1_o2))];
                }else{//bottom layer of gohst cells
                    for(j=a-(slen_m1_o2);j < f_tt->dim[mode];j++){
                        matmul_out[a] += sten[j+(slen_m1_o2)-a]*
                                             f_tt->coreList[mode][coreOffset+
                                             f_tt->ttRank[mode]*j];
                    }
                    slen_bot = slen_m1_o2 - (f_tt->dim[mode]-a)+1;
                    for(j=0;j<slen_bot;j++){
                        matmul_out[a] += sten[slen-slen_bot+j]*
                                        f_tt->wk[ghostOffset + 
                                            (slen_m1_o2)*ghostSize +
                                           (f_tt->ttRank[mode]*j)];
                    }
                }
            }
            for(a=0;a<f_tt->dim[mode];a++){
                f_tt->coreList[mode][coreOffset+ f_tt->ttRank[mode]*a] 
                                                          = matmul_out[a];
            }
        }
    }
}



void pattac_apply_periodic_stencil(paratt* f_ptt, int mode,
                                        double* sten, int slen){
    int i,j,k,a,stencil_coord,gc_coord;
    int ghostSize, ghostOffset, coreOffset,slen_m1_o2;
    double  f_value;
    double *matmul_out;
    double *local_block;
    double *gc_above;
    double *gc_below;
    MPI_Request send_below;
    MPI_Request recv_below;
    MPI_Request send_above;
    MPI_Request recv_above;
    MPI_Status  mpi_stat;
    attac_tt* f_tt = f_ptt->x;
    slen = (slen % 2) ? (slen) : (slen-1);
    slen_m1_o2 = (slen-1)/2;
    //resize the work vector to hold the top and bottom row for ghost
    //cell storage.
    ghostSize = f_tt->ttRank[mode]*f_tt->ttRank[mode+1];
    if(f_tt->wksz < 2*(slen-1)*ghostSize + f_tt->dim[mode]){
        f_tt->wksz = 2*(slen-1)*ghostSize + f_tt->dim[mode];
        f_tt->wk = (double*) realloc(f_tt->wk,sizeof(double)*f_tt->wksz);
    }
    for(k=0;k<f_tt->ttRank[mode+1];k++){
        for(j=0;j<slen_m1_o2;j++){
            for(i=0;i<f_tt->ttRank[mode];i++){
                //the ghost cells above the top row
                f_tt->wk[i + f_tt->ttRank[mode]*(j + (slen_m1_o2)*k)] =
                        f_tt->coreList[mode][i + 
                              f_tt->ttRank[mode]*(j+f_tt->dim[mode]-slen_m1_o2
                              +(f_tt->dim[mode]*k))];
                //the ghost cells below the bottom row
                f_tt->wk[(slen_m1_o2)*ghostSize +
                            i + f_tt->ttRank[mode]*(j +(slen_m1_o2)*k)] =
                        f_tt->coreList[mode][i + 
                              f_tt->ttRank[mode]*(j +(f_tt->dim[mode]*k))];
            }
        }
    }
    //copy the ghost cells into the work vector
    //the ghost cells above the top row
    gc_above = &(f_tt->wk[2*(slen_m1_o2)*ghostSize]);
    gc_below = &(f_tt->wk[3*(slen_m1_o2)*ghostSize]);
    matmul_out = &(f_tt->wk[4*(slen_m1_o2)*ghostSize]);
    if(f_ptt->mpi_sz > 1){
        local_block = &(f_tt->wk[0]);
        send_recv_above(local_block,gc_above,ghostSize*slen_m1_o2,MPI_DOUBLE,
                                        &send_above,&recv_above);
        local_block = &(f_tt->wk[(slen_m1_o2)*ghostSize]);
        send_recv_below(local_block,gc_below,ghostSize*slen_m1_o2,MPI_DOUBLE,
                                        &send_below,&recv_below);
        MPI_Wait(&send_below, &mpi_stat);
        MPI_Wait(&recv_below, &mpi_stat);
        MPI_Wait(&send_above, &mpi_stat);
        MPI_Wait(&recv_above, &mpi_stat);
    }else{
        gc_above = &(f_tt->wk[0]);
        gc_below = &(f_tt->wk[(slen_m1_o2)*ghostSize]);
    }
    //Apply the stencil to coreList[mode]
    for(k=0;k<f_tt->ttRank[mode+1];k++){
        for(i=0;i<f_tt->ttRank[mode];i++){
            ghostOffset = i + ((f_tt->ttRank[mode]*slen_m1_o2)*k);
            coreOffset  = i + f_tt->ttRank[mode]*f_tt->dim[mode]*k;
            for(a=0;a<f_tt->dim[mode];a++){
                matmul_out[a] = 0.0;
                stencil_coord = 0;
                gc_coord = 0;
                for(j = a - (slen_m1_o2); j < a; j++){
                    if(j < 0){
                        f_value = gc_above[ghostOffset+
                                                f_tt->ttRank[mode]*gc_coord];
                        gc_coord++;
                    }else{
                        f_value = f_tt->coreList[mode][coreOffset+
                                             f_tt->ttRank[mode]*j];
                    }
                    matmul_out[a] += sten[stencil_coord]*f_value;
                    stencil_coord++;
                }
                matmul_out[a] +=sten[stencil_coord]*
                                    f_tt->coreList[mode][coreOffset+
                                             f_tt->ttRank[mode]*a];
                stencil_coord++;
                gc_coord = 0;
                for(j = a + 1; j <= a + (slen_m1_o2); j++){
                    if(j < f_tt->dim[mode]){
                        f_value = f_tt->coreList[mode][coreOffset+
                                             f_tt->ttRank[mode]*j];
                    }else{
                        f_value = gc_below[ghostOffset+
                                                f_tt->ttRank[mode]*gc_coord];
                        gc_coord++;
                    }
                    matmul_out[a] += sten[stencil_coord]*f_value;
                    stencil_coord++;
                }
            }
            for(a=0;a<f_tt->dim[mode];a++){
                f_tt->coreList[mode][coreOffset+ f_tt->ttRank[mode]*a] 
                                                          = matmul_out[a];
            }
        }
    }
}
                                      
void pattac_apply_cdf_stencil(paratt* f_ptt, int mode,
                                double* sten, int slen){
    int i,j,k,a,stencil_coord,gc_coord;
    int ghostSize, ghostOffset, coreOffset,slen_m1_o2;
    double  f_value;
    double *matmul_out;
    double *local_block;
    double *gc_above;
    double *gc_below;
    MPI_Request send_below;
    MPI_Request recv_below;
    MPI_Request send_above;
    MPI_Request recv_above;
    MPI_Status  mpi_stat;
    attac_tt* f_tt = f_ptt->x;
    slen = (slen % 2) ? (slen) : (slen-1);
    slen_m1_o2 = (slen-1)/2;
    //resize the work vector to hold the top and bottom row for ghost
    //cell storage.
    ghostSize = f_tt->ttRank[mode]*f_tt->ttRank[mode+1];
    if(f_tt->wksz < 2*(slen-1)*ghostSize + f_tt->dim[mode]){
        f_tt->wksz = 2*(slen-1)*ghostSize + f_tt->dim[mode];
        f_tt->wk = (double*) realloc(f_tt->wk,sizeof(double)*f_tt->wksz);
    }
    for(k=0;k<f_tt->ttRank[mode+1];k++){
        for(j=0;j<slen_m1_o2;j++){
            for(i=0;i<f_tt->ttRank[mode];i++){
                //the ghost cells above the top row
                if(f_ptt->mpi_rk < f_ptt->mpi_sz - 1){
                    f_tt->wk[i + f_tt->ttRank[mode]*(j + (slen_m1_o2)*k)] =
                              f_tt->coreList[mode][i + 
                              f_tt->ttRank[mode]*(j+f_tt->dim[mode]-slen_m1_o2
                              +(f_tt->dim[mode]*k))];
                }
                //the ghost cells below the bottom row
                if(f_ptt->mpi_rk > 0){
                    f_tt->wk[(slen_m1_o2)*ghostSize +
                            i + f_tt->ttRank[mode]*(j +(slen_m1_o2)*k)] =
                            f_tt->coreList[mode][i + 
                              f_tt->ttRank[mode]*(j +(f_tt->dim[mode]*k))];
                }
            }
        }
    }
    //copy the ghost cells into the work vector
    //the ghost cells above the top row
    gc_above = &(f_tt->wk[2*(slen_m1_o2)*ghostSize]);
    gc_below = &(f_tt->wk[3*(slen_m1_o2)*ghostSize]);
    matmul_out = &(f_tt->wk[4*(slen_m1_o2)*ghostSize]);
    if(f_ptt->mpi_rk == 0){
        if(f_ptt->mpi_sz > 1){
            MPI_Isend(f_tt->wk, ghostSize*slen_m1_o2,
                                MPI_DOUBLE, 1,0,MPI_COMM_WORLD,&send_below);
            MPI_Irecv(gc_below, ghostSize*slen_m1_o2, MPI_DOUBLE,1,0,
                                        MPI_COMM_WORLD,&recv_below);
        }
        for(k=0;k<f_tt->ttRank[mode+1];k++){
            for(j=0;j<slen_m1_o2;j++){
                for(i=0;i<f_tt->ttRank[mode];i++){
                    gc_above[i + f_tt->ttRank[mode]*(j + (slen_m1_o2)*k)] =// 0.;
                                f_tt->coreList[mode][i + 
                                      f_tt->ttRank[mode]*(slen_m1_o2-j-1
                                          +(f_tt->dim[mode]*k))];
                }
            }
        }
        if(f_ptt->mpi_sz > 1){
            MPI_Wait(&send_below, &mpi_stat);
            MPI_Wait(&recv_below, &mpi_stat);
        }
    }
    if(f_ptt->mpi_rk == f_ptt->mpi_sz - 1){
        local_block = &(f_tt->wk[(slen_m1_o2)*ghostSize]);
        if(f_ptt->mpi_sz > 1){
            MPI_Isend(local_block, ghostSize*slen_m1_o2, MPI_DOUBLE,
                        f_ptt->mpi_rk-1,0,MPI_COMM_WORLD,&send_above);
            MPI_Irecv(gc_above, ghostSize*slen_m1_o2, MPI_DOUBLE,
                        f_ptt->mpi_rk-1,0,MPI_COMM_WORLD,&recv_above);
        }
        for(k=0;k<f_tt->ttRank[mode+1];k++){
            for(j=0;j<slen_m1_o2;j++){
                for(i=0;i<f_tt->ttRank[mode];i++){
                    gc_below[i + f_tt->ttRank[mode]*(j + (slen_m1_o2)*k)] =
                              f_tt->coreList[mode][i + 
                                    f_tt->ttRank[mode]*(f_tt->dim[mode]-1-j
                                        +(f_tt->dim[mode]*k))];
                }
            }
        }
        if(f_ptt->mpi_sz > 1){
            MPI_Wait(&send_above, &mpi_stat);
            MPI_Wait(&recv_above, &mpi_stat);
        }
    }
    if(f_ptt->mpi_rk > 0 && f_ptt->mpi_rk < f_ptt->mpi_sz - 1){
        local_block = &(f_tt->wk[0]);
        send_recv_above(local_block,gc_above,ghostSize*slen_m1_o2,MPI_DOUBLE,
                                        &send_above,&recv_above);
        local_block = &(f_tt->wk[(slen_m1_o2)*ghostSize]);
        send_recv_below(local_block,gc_below,ghostSize*slen_m1_o2,MPI_DOUBLE,
                                        &send_below,&recv_below);
        MPI_Wait(&send_below, &mpi_stat);
        MPI_Wait(&recv_below, &mpi_stat);
        MPI_Wait(&send_above, &mpi_stat);
        MPI_Wait(&recv_above, &mpi_stat);
    }
    //Apply the stencil to coreList[mode]
    for(k=0;k<f_tt->ttRank[mode+1];k++){
        for(i=0;i<f_tt->ttRank[mode];i++){
            ghostOffset = i + ((f_tt->ttRank[mode]*slen_m1_o2)*k);
            coreOffset  = i + f_tt->ttRank[mode]*f_tt->dim[mode]*k;
            for(a=0;a<f_tt->dim[mode];a++){
                matmul_out[a] = 0.0;
                stencil_coord = 0;
                gc_coord = 0;
                for(j = a - (slen_m1_o2); j < a; j++){
                    if(j < 0){
                        f_value = gc_above[ghostOffset+
                                                f_tt->ttRank[mode]*gc_coord];
                        gc_coord++;
                    }else{
                        f_value = f_tt->coreList[mode][coreOffset+
                                             f_tt->ttRank[mode]*j];
                    }
                    matmul_out[a] += sten[stencil_coord]*f_value;
                    stencil_coord++;
                }
                matmul_out[a] +=sten[stencil_coord]*
                                    f_tt->coreList[mode][coreOffset+
                                             f_tt->ttRank[mode]*a];
                stencil_coord++;
                gc_coord = 0;
                for(j = a + 1; j <= a + (slen_m1_o2); j++){
                    if(j < f_tt->dim[mode]){
                        f_value = f_tt->coreList[mode][coreOffset+
                                             f_tt->ttRank[mode]*j];
                    }else{
                        f_value = gc_below[ghostOffset+
                                                f_tt->ttRank[mode]*gc_coord];
                        gc_coord++;
                    }
                    matmul_out[a] += sten[stencil_coord]*f_value;
                    stencil_coord++;
                }
            }
            for(a=0;a<f_tt->dim[mode];a++){
                f_tt->coreList[mode][coreOffset+ f_tt->ttRank[mode]*a] 
                                                          = matmul_out[a];
            }
        }
    }
}


void pattac_apply_cdf_lf_flux(paratt* f_ptt, int mode,
                                double* wave_speed, double* gc_wave_speed,
                                double* source_coef,
                                double dt, double dx){
    int i,j,k,a,stencil_coord,gc_coord;
    int ghostSize, ghostOffset, coreOffset,slen,slen_m1_o2;
    double  f_value;
    double sten[3];
    double d2_sten[3];
    double *matmul_out;
    double *local_block;
    double *gc_above;
    double *gc_below;
    MPI_Request send_below;
    MPI_Request recv_below;
    MPI_Request send_above;
    MPI_Request recv_above;
    MPI_Status  mpi_stat;
    attac_tt* f_tt = f_ptt->x;
    slen = 3;
    slen_m1_o2 = 1;
    sten[0] = -0.5/dx;
    sten[1] =  0.0;
    sten[2] =  0.5/dx;
    d2_sten[0] = 0.5;
    d2_sten[1] = -1.0;
    d2_sten[2] = 0.5;
    //resize the work vector to hold the top and bottom row for ghost
    //cell storage.
    ghostSize = f_tt->ttRank[mode]*f_tt->ttRank[mode+1];
    if(f_tt->wksz < 2*(slen-1)*ghostSize + f_tt->dim[mode]){
        f_tt->wksz = 2*(slen-1)*ghostSize + f_tt->dim[mode];
        f_tt->wk = (double*) realloc(f_tt->wk,sizeof(double)*f_tt->wksz);
    }
    for(k=0;k<f_tt->ttRank[mode+1];k++){
        for(j=0;j<slen_m1_o2;j++){
            for(i=0;i<f_tt->ttRank[mode];i++){
                //the ghost cells above the top row
                if(f_ptt->mpi_rk < f_ptt->mpi_sz - 1){
                    f_tt->wk[i + f_tt->ttRank[mode]*(j + (slen_m1_o2)*k)] =
                              f_tt->coreList[mode][i + 
                              f_tt->ttRank[mode]*(j+f_tt->dim[mode]-slen_m1_o2
                              +(f_tt->dim[mode]*k))];
                }
                //the ghost cells below the bottom row
                if(f_ptt->mpi_rk > 0){
                    f_tt->wk[(slen_m1_o2)*ghostSize +
                            i + f_tt->ttRank[mode]*(j +(slen_m1_o2)*k)] =
                            f_tt->coreList[mode][i + 
                              f_tt->ttRank[mode]*(j +(f_tt->dim[mode]*k))];
                }
            }
        }
    }
    //copy the ghost cells into the work vector
    //the ghost cells above the top row
    gc_above = &(f_tt->wk[2*(slen_m1_o2)*ghostSize]);
    gc_below = &(f_tt->wk[3*(slen_m1_o2)*ghostSize]);
    matmul_out = &(f_tt->wk[4*(slen_m1_o2)*ghostSize]);
    if(f_ptt->mpi_rk == 0){
        if(f_ptt->mpi_sz > 1){
            MPI_Isend(f_tt->wk, ghostSize*slen_m1_o2,
                                MPI_DOUBLE, 1,0,MPI_COMM_WORLD,&send_below);
            MPI_Irecv(gc_below, ghostSize*slen_m1_o2, MPI_DOUBLE,1,0,
                                        MPI_COMM_WORLD,&recv_below);
        }
        for(k=0;k<f_tt->ttRank[mode+1];k++){
            for(j=0;j<slen_m1_o2;j++){
                for(i=0;i<f_tt->ttRank[mode];i++){
                    gc_above[i + f_tt->ttRank[mode]*(j + (slen_m1_o2)*k)] = //0.;
                               f_tt->coreList[mode][i + 
                                    f_tt->ttRank[mode]*(slen_m1_o2-j-1
                                        +(f_tt->dim[mode]*k))];
                }
            }
        }
        if(f_ptt->mpi_sz > 1){
            MPI_Wait(&send_below, &mpi_stat);
            MPI_Wait(&recv_below, &mpi_stat);
        }
    }
    if(f_ptt->mpi_rk == f_ptt->mpi_sz - 1){
        local_block = &(f_tt->wk[(slen_m1_o2)*ghostSize]);
        
        if(f_ptt->mpi_sz > 1){
            MPI_Isend(local_block, ghostSize*slen_m1_o2, MPI_DOUBLE,
                        f_ptt->mpi_rk-1,0,MPI_COMM_WORLD,&send_above);
            MPI_Irecv(gc_above, ghostSize*slen_m1_o2, MPI_DOUBLE,
                        f_ptt->mpi_rk-1,0, MPI_COMM_WORLD,&recv_above);
        }
        for(k=0;k<f_tt->ttRank[mode+1];k++){
            for(j=0;j<slen_m1_o2;j++){
                for(i=0;i<f_tt->ttRank[mode];i++){
                    gc_below[i + f_tt->ttRank[mode]*(j + (slen_m1_o2)*k)] =
                              f_tt->coreList[mode][i + 
                                    f_tt->ttRank[mode]*(f_tt->dim[mode]-1-j
                                        +(f_tt->dim[mode]*k))];
                }
            }
        }
        if(f_ptt->mpi_sz > 1){
            MPI_Wait(&send_above, &mpi_stat);
            MPI_Wait(&recv_above, &mpi_stat);
        }
    }
    if(f_ptt->mpi_rk > 0 && f_ptt->mpi_rk < f_ptt->mpi_sz - 1){
        local_block = &(f_tt->wk[0]);
        send_recv_above(local_block,gc_above,ghostSize*slen_m1_o2,MPI_DOUBLE,
                                        &send_above,&recv_above);
        local_block = &(f_tt->wk[(slen_m1_o2)*ghostSize]);
        send_recv_below(local_block,gc_below,ghostSize*slen_m1_o2,MPI_DOUBLE,
                                        &send_below,&recv_below);
        MPI_Wait(&send_below, &mpi_stat);
        MPI_Wait(&recv_below, &mpi_stat);
        MPI_Wait(&send_above, &mpi_stat);
        MPI_Wait(&recv_above, &mpi_stat);
    }
    //Apply the stencil to coreList[mode]
    for(k=0;k<f_tt->ttRank[mode+1];k++){
        for(i=0;i<f_tt->ttRank[mode];i++){
            ghostOffset = i + ((f_tt->ttRank[mode]*slen_m1_o2)*k);
            coreOffset  = i + f_tt->ttRank[mode]*f_tt->dim[mode]*k;
            for(a=0;a<f_tt->dim[mode];a++){
                matmul_out[a] = 0.0;
                stencil_coord = 0;
                gc_coord = 0;
                for(j = a - (slen_m1_o2); j < a; j++){
                    if(j < 0){
                        f_value = -gc_wave_speed[gc_coord]*
                                       gc_above[ghostOffset+
                                                f_tt->ttRank[mode]*gc_coord];
                        gc_coord++;
                    }else{
                        f_value = -wave_speed[j]*
                                        f_tt->coreList[mode][coreOffset+
                                                f_tt->ttRank[mode]*j];
                    }
                    matmul_out[a] += sten[stencil_coord]*f_value; 
                    stencil_coord++;
                }

                stencil_coord++;
                gc_coord = 0;
                for(j = a + 1; j <= a + (slen_m1_o2); j++){
                    if(j < f_tt->dim[mode]){
                        f_value = -wave_speed[j]*
                                        f_tt->coreList[mode][coreOffset+
                                                f_tt->ttRank[mode]*j];
                    }else{
                        f_value = -gc_wave_speed[gc_coord+slen_m1_o2]*
                                       gc_below[ghostOffset+
                                                f_tt->ttRank[mode]*gc_coord];
                        gc_coord++;
                    }
                    matmul_out[a] += sten[stencil_coord]*f_value; 
                    stencil_coord++;
                }
            }
            for(a=0;a<f_tt->dim[mode];a++){
                f_tt->coreList[mode][coreOffset+ f_tt->ttRank[mode]*a] 
                                                  = matmul_out[a]
                                   + source_coef[a]*
                                        f_tt->coreList[mode][coreOffset+
                                            f_tt->ttRank[mode]*a];
            }
        }
    }
}


void pattac_apply_cdf_lw_rhs(paratt* f_ptt, int mode,
                                double* wave_speed, double* gc_wave_speed,
                                double* bdry_wave_speed,
                                double* source_coef,
                                double dt, double dx){
    int i,j,k,a,stencil_coord,gc_coord;
    int ghostSize, ghostOffset, coreOffset,slen,slen_m1_o2;
    double  f_value;
    double u_half[2];
    double *matmul_out;
    double *local_block;
    double *gc_above;
    double *gc_below;
    MPI_Request send_below;
    MPI_Request recv_below;
    MPI_Request send_above;
    MPI_Request recv_above;
    MPI_Status  mpi_stat;
    attac_tt* f_tt = f_ptt->x;
    slen = 3;
    slen_m1_o2 = 1;
    //resize the work vector to hold the top and bottom row for ghost
    //cell storage.
    ghostSize = f_tt->ttRank[mode]*f_tt->ttRank[mode+1];
    if(f_tt->wksz < 2*(slen-1)*ghostSize + f_tt->dim[mode]){
        f_tt->wksz = 2*(slen-1)*ghostSize + f_tt->dim[mode];
        f_tt->wk = (double*) realloc(f_tt->wk,sizeof(double)*f_tt->wksz);
    }
    for(k=0;k<f_tt->ttRank[mode+1];k++){
        for(j=0;j<slen_m1_o2;j++){
            for(i=0;i<f_tt->ttRank[mode];i++){
                //the ghost cells above the top row
                if(f_ptt->mpi_rk < f_ptt->mpi_sz - 1){
                    f_tt->wk[i + f_tt->ttRank[mode]*(j + (slen_m1_o2)*k)] =
                              f_tt->coreList[mode][i + 
                              f_tt->ttRank[mode]*(j+f_tt->dim[mode]-slen_m1_o2
                              +(f_tt->dim[mode]*k))];
                }
                //the ghost cells below the bottom row
                if(f_ptt->mpi_rk > 0){
                    f_tt->wk[(slen_m1_o2)*ghostSize +
                            i + f_tt->ttRank[mode]*(j +(slen_m1_o2)*k)] =
                            f_tt->coreList[mode][i + 
                              f_tt->ttRank[mode]*(j +(f_tt->dim[mode]*k))];
                }
            }
        }
    }
    //copy the ghost cells into the work vector
    //the ghost cells above the top row
    gc_above = &(f_tt->wk[2*(slen_m1_o2)*ghostSize]);
    gc_below = &(f_tt->wk[3*(slen_m1_o2)*ghostSize]);
    matmul_out = &(f_tt->wk[4*(slen_m1_o2)*ghostSize]);
    if(f_ptt->mpi_rk == 0){
        if(f_ptt->mpi_sz > 1){
            MPI_Isend(f_tt->wk, ghostSize*slen_m1_o2,
                                MPI_DOUBLE, 1,0,MPI_COMM_WORLD,&send_below);
            MPI_Irecv(gc_below, ghostSize*slen_m1_o2, MPI_DOUBLE,1,0,
                                        MPI_COMM_WORLD,&recv_below);
        }
        for(k=0;k<f_tt->ttRank[mode+1];k++){
            for(j=0;j<slen_m1_o2;j++){
                for(i=0;i<f_tt->ttRank[mode];i++){
                    gc_above[i + f_tt->ttRank[mode]*(j + (slen_m1_o2)*k)] = 
                                f_tt->coreList[mode][i + 
                                      f_tt->ttRank[mode]*(slen_m1_o2-j-1
                                          +(f_tt->dim[mode]*k))];
                }
            }
        }
        if(f_ptt->mpi_sz > 1){
            MPI_Wait(&send_below, &mpi_stat);
            MPI_Wait(&recv_below, &mpi_stat);
        }
    }
    if(f_ptt->mpi_rk == f_ptt->mpi_sz - 1){
        local_block = &(f_tt->wk[(slen_m1_o2)*ghostSize]);
        
        if(f_ptt->mpi_sz > 1){
            MPI_Isend(local_block, ghostSize*slen_m1_o2, MPI_DOUBLE,
                        f_ptt->mpi_rk-1,0,MPI_COMM_WORLD,&send_above);
            MPI_Irecv(gc_above, ghostSize*slen_m1_o2, MPI_DOUBLE,
                        f_ptt->mpi_rk-1,0, MPI_COMM_WORLD,&recv_above);
        }
        for(k=0;k<f_tt->ttRank[mode+1];k++){
            for(j=0;j<slen_m1_o2;j++){
                for(i=0;i<f_tt->ttRank[mode];i++){
                    gc_below[i + f_tt->ttRank[mode]*(j + (slen_m1_o2)*k)] =
                              f_tt->coreList[mode][i + 
                                    f_tt->ttRank[mode]*(f_tt->dim[mode]-1-j
                                        +(f_tt->dim[mode]*k))];
                }
            }
        }
        if(f_ptt->mpi_sz > 1){
            MPI_Wait(&send_above, &mpi_stat);
            MPI_Wait(&recv_above, &mpi_stat);
        }
    }
    if(f_ptt->mpi_rk > 0 && f_ptt->mpi_rk < f_ptt->mpi_sz - 1){
        local_block = &(f_tt->wk[0]);
        send_recv_above(local_block,gc_above,ghostSize*slen_m1_o2,MPI_DOUBLE,
                                        &send_above,&recv_above);
        local_block = &(f_tt->wk[(slen_m1_o2)*ghostSize]);
        send_recv_below(local_block,gc_below,ghostSize*slen_m1_o2,MPI_DOUBLE,
                                        &send_below,&recv_below);
        MPI_Wait(&send_below, &mpi_stat);
        MPI_Wait(&recv_below, &mpi_stat);
        MPI_Wait(&send_above, &mpi_stat);
        MPI_Wait(&recv_above, &mpi_stat);
    }
    //Apply the stencil to coreList[mode]
    for(k=0;k<f_tt->ttRank[mode+1];k++){
        for(i=0;i<f_tt->ttRank[mode];i++){
            ghostOffset = i + ((f_tt->ttRank[mode]*slen_m1_o2)*k);
            coreOffset  = i + f_tt->ttRank[mode]*f_tt->dim[mode]*k;
            u_half[0] = 0.5*(gc_above[ghostOffset] +
                        f_tt->coreList[mode][coreOffset])
                        - ((wave_speed[0]*f_tt->coreList[mode][coreOffset]
                           -gc_above[ghostOffset]*gc_wave_speed[0])*
                           dt/(2.0*dx));
            for(a=0;a<f_tt->dim[mode];a++){
                if(a+1 < f_tt->dim[mode]){
                    u_half[1] = 0.5*( f_tt->coreList[mode][coreOffset+
                                            f_tt->ttRank[mode]*(a+1)] +
                         f_tt->coreList[mode][coreOffset+
                                            f_tt->ttRank[mode]*a])
                        - ((wave_speed[a+1]*f_tt->coreList[mode][coreOffset+
                                            f_tt->ttRank[mode]*(a+1)]
                           -wave_speed[a]*f_tt->coreList[mode][coreOffset+
                                            f_tt->ttRank[mode]*(a)])*
                           dt/(2.0*dx));
                }else{
                    u_half[1] = 0.5*( gc_below[ghostOffset] +
                         f_tt->coreList[mode][coreOffset+
                                            f_tt->ttRank[mode]*a])
                        - (( gc_below[ghostOffset]*gc_wave_speed[1]
                           -wave_speed[a]*f_tt->coreList[mode][coreOffset+
                                            f_tt->ttRank[mode]*(a)])*
                           dt/(2.0*dx));
                }
                matmul_out[a] =  -(bdry_wave_speed[a+1]*u_half[1]
                                   -bdry_wave_speed[a]*u_half[0])/dx
                                   + source_coef[a]*
                                        f_tt->coreList[mode][coreOffset+
                                            f_tt->ttRank[mode]*a];
                u_half[0] = u_half[1];
            }
            for(a=0;a<f_tt->dim[mode];a++){
                f_tt->coreList[mode][coreOffset+ f_tt->ttRank[mode]*a] 
                                                          = matmul_out[a];
            }
        }
    }
}

void pattac_apply_marginal(paratt* f_ptt, paratt** marginal_ptt_ptr, 
                            int mode, double* row_vector){
    int i, j, k,rank_removed, dim_removed, wk_numRow, wk_numCol, message_size;
    const int mpi_sz = f_ptt->mpi_sz;
    attac_tt* marginal_tt;
    
    if(*marginal_ptt_ptr  != NULL){
        fprintf(stderr,"pattac_apply_marginal: Error.\n"
                       "Input marginal_tt_ptr was not NULL.\n");
        exit(1);
    }
    dim_removed = f_ptt->dim[mode];
    if(mode < f_ptt->numDim -1){
        wk_numRow = f_ptt->x->ttRank[mode];
        wk_numCol = f_ptt->x->ttRank[mode+1];
        rank_removed = f_ptt->x->ttRank[mode+1];
        for(i = mode+1; i < f_ptt->numDim; i++){
            f_ptt->x->ttRank[i] = f_ptt->x->ttRank[i+1];
        }
    }else if(mode < f_ptt->numDim){
        wk_numRow = f_ptt->x->ttRank[mode];
        wk_numCol = 1;
        rank_removed = f_ptt->x->ttRank[mode];
        f_ptt->x->ttRank[mode] = 1;
    }
    for(i = mode; i < f_ptt->numDim-1; i++){
        f_ptt->dim[i] = f_ptt->dim[i+1];
    }
    
    *marginal_ptt_ptr = alloc_ptt(f_ptt->numDim-1,
                                    f_ptt->dim, f_ptt->x->ttRank);
    marginal_tt = (*marginal_ptt_ptr)->x;
    
    for(i = f_ptt->numDim-1; i >= mode+1; i--){
        f_ptt->dim[i] = f_ptt->dim[i-1];
    }
    f_ptt->dim[mode] = dim_removed;
    if(mode < f_ptt->numDim -1){
        for( i = f_ptt->numDim; i >= mode+2; i--){
            f_ptt->x->ttRank[i] = f_ptt->x->ttRank[i-1];
        }
        f_ptt->x->ttRank[mode+1] = rank_removed;
    }else if(mode < f_ptt->numDim){
        f_ptt->x->ttRank[mode] = rank_removed;
    }
    //change dim_removed to local coreList dimension
    dim_removed = f_ptt->x->dim[mode];
    
    if(marginal_tt->wksz < wk_numRow*wk_numCol){
        marginal_tt->wksz = wk_numRow*wk_numCol;
        marginal_tt->wk = (double*) realloc(marginal_tt->wk,
                                sizeof(double)*marginal_tt->wksz);
    }
    if(f_ptt->x->wksz < wk_numRow*wk_numCol*mpi_sz ){
        f_ptt->x->wksz = wk_numRow*wk_numCol*mpi_sz;
        f_ptt->x->wk = (double*) realloc(f_ptt->x->wk,
                                sizeof(double)*f_ptt->x->wksz);
    }
    if(mode < f_ptt->numDim - 1){
        for(k = 0; k < f_ptt->x->ttRank[mode+1]; k++){       
            for(i = 0; i < wk_numRow; i++){
                marginal_tt->wk[i + wk_numRow*k] = 0.0;
                for(j = 0 ; j < dim_removed; j++){
                    marginal_tt->wk[i + wk_numRow*k] += row_vector[j]*
                        f_ptt->x->coreList[mode][i +
                                         (j+k*dim_removed)*wk_numRow];
                }
            }
        }
    }else{
        for(i = 0; i < wk_numRow; i++){
            marginal_tt->wk[i] = 0.0;
            for(j = 0 ; j < dim_removed; j++){
                marginal_tt->wk[i] += row_vector[j]*
                    f_ptt->x->coreList[mode][i + rank_removed*j];
            }
        }
    }
        //resize workvector of f_ptt to recieve integration
    message_size = wk_numRow*wk_numCol;
    for(i=0;i < f_ptt->mpi_sz; i++){
        if(i == f_ptt->mpi_rk){
            MPI_Bcast(marginal_tt->wk, message_size, MPI_DOUBLE,
                        i, MPI_COMM_WORLD);
        }else{
            MPI_Bcast(f_ptt->x->wk+i*message_size,
                        message_size, MPI_DOUBLE,
                        i, MPI_COMM_WORLD);
        }
    }
    for(i = 0;i < f_ptt->mpi_sz; i++){
        if(i != f_ptt->mpi_rk){
            for(j = 0; j < message_size; j++){
                marginal_tt->wk[j] += f_ptt->x->wk[j+i*message_size];
            }
        }
    }
    
    if(mode < f_ptt->numDim - 1){
        for(j=0;j<f_ptt->x->dim[mode+1]*f_ptt->x->ttRank[mode+2];j++){
            for(i=0;i<wk_numRow;i++){
                marginal_tt->coreList[mode][i+wk_numRow*j] = 0.0;
                for(k=0;k<wk_numCol;k++){
                    marginal_tt->coreList[mode][i+wk_numRow*j] +=
                        marginal_tt->wk[i+wk_numRow*k]*
                        f_ptt->x->coreList[mode+1][k+wk_numCol*j];
                }
            }
        }
        for(i = 0; i < mode; i++){
            //copying modes before deleted mode 
            memcpy( marginal_tt->coreList[i], f_ptt->x->coreList[i],
                sizeof(double)*(f_ptt->x->ttRank[i]*f_ptt->x->dim[i]*
                                f_ptt->x->ttRank[i+1]));
        }
        for(i = mode+1; i < marginal_tt->numDim; i++){
            //copying modes after deleted mode 
            memcpy( marginal_tt->coreList[i], f_ptt->x->coreList[i+1],
                sizeof(double)*(f_ptt->x->ttRank[i+1]*
                                f_ptt->x->dim[i+1]*
                                f_ptt->x->ttRank[i+2]));
        }
    }else{
        k = f_ptt->x->ttRank[mode-1]*f_ptt->x->dim[mode-1];
        for(i=0;i<k;i++){
            marginal_tt->coreList[mode-1][i] = 0.0;
            for(j=0;j<rank_removed;j++){
                marginal_tt->coreList[mode-1][i] += marginal_tt->wk[j]*
                                    f_ptt->x->coreList[mode-1][i+k*j];
            }
        }
        for(i = 0; i < mode-1; i++){
            //copying modes before deleted mode 
            memcpy( marginal_tt->coreList[i], f_ptt->x->coreList[i],
                sizeof(double)*(f_ptt->x->ttRank[i]*f_ptt->x->dim[i]*
                                f_ptt->x->ttRank[i+1]));
        }
    }
}


void pattac_marginal_cusum(paratt* f_ptt, int mode, double* row_vector){
    const int sz1 = f_ptt->x->ttRank[mode];
    const int sz2 = f_ptt->x->dim[mode];
    const int sz3 = f_ptt->x->ttRank[mode+1];
    const int gsz = sz1*sz3;
    int i, j, k;
    MPI_Request sendReq;
    MPI_Request recvReq;
    MPI_Status  mpi_stat;
    /*
     * Resize work vector to store "bottom" ghost cell slice for
     * integration.
     */
    if(f_ptt->x->wksz < 2*gsz){
        f_ptt->x->wksz = 2*gsz;
        f_ptt->x->wk = (double*) realloc(f_ptt->x->wk,
                                sizeof(double)*f_ptt->x->wksz);
    }
    for(k = 0; k < sz3; k++){
        for(i = 0; i < sz1; i++){
            for(j = 0; j < sz2; j++){
                f_ptt->x->coreList[mode][i+((j+(k*sz2))*sz1)] *= row_vector[j];
            }
            for(j = 1; j < sz2; j++){
                f_ptt->x->coreList[mode][i+((j+(k*sz2))*sz1)] +=
                               f_ptt->x->coreList[mode][i+((j-1+(k*sz2))*sz1)];
            }
            f_ptt->x->wk[i + sz1*k] = 
                               f_ptt->x->coreList[mode][i+((j-1+(k*sz2))*sz1)];
        }
    }
    if(f_ptt->mpi_rk > 0){
        MPI_Irecv(&(f_ptt->x->wk[gsz]), gsz, MPI_DOUBLE ,
                    f_ptt->mpi_rk - 1,0,MPI_COMM_WORLD, &recvReq);
        MPI_Wait(&recvReq, &mpi_stat);
        for(k = 0; k < sz3; k++){
            for(j = 0; j < sz2; j++){
                for(i = 0; i < sz1; i++){
                    f_ptt->x->coreList[mode][i+((j+(k*sz2))*sz1)] += 
                                                   f_ptt->x->wk[gsz+i + sz1*k];
                }
            }
        }
        for(i = 0; i < gsz; i++){
            f_ptt->x->wk[i] += f_ptt->x->wk[gsz+i];
        }
    }
    if(f_ptt->mpi_rk < f_ptt->mpi_sz - 1){
        MPI_Isend(f_ptt->x->wk, gsz, MPI_DOUBLE,
                    f_ptt->mpi_rk + 1,0,MPI_COMM_WORLD, &sendReq);
    }
}


//
void pattac_marginal_trap(paratt* f_ptt, int mode, double* row_vector){
    const int sz1 = f_ptt->x->ttRank[mode];
    const int sz2 = f_ptt->x->dim[mode];
    const int sz3 = f_ptt->x->ttRank[mode+1];
    const int gsz = sz1*sz3;
    double cumulant;
    double delta;
    int i, j, k;
    MPI_Request sendReq;
    MPI_Request recvReq;
    MPI_Status  mpi_stat;
    /*
     * Resize work vector to store "bottom" ghost cell slice for
     * integration.
     */
    if(f_ptt->x->wksz < 4*gsz){
        f_ptt->x->wksz = 4*gsz;
        f_ptt->x->wk = (double*) realloc(f_ptt->x->wk,
                                sizeof(double)*f_ptt->x->wksz);
    }
    
    if(f_ptt->mpi_rk > 0){
        MPI_Irecv(&(f_ptt->x->wk[2*gsz]), 2*gsz, MPI_DOUBLE ,
                    f_ptt->mpi_rk - 1,0,MPI_COMM_WORLD, &recvReq);
    }
    for(k = 0; k < sz3; k++){
        for(i = 0; i < sz1; i++){
            if(f_ptt->mpi_rk > 0){
                cumulant = f_ptt->x->coreList[mode][i+(((k*sz2))*sz1)]*
                            0.5*row_vector[0];
            }else{
                cumulant = 0.0;
            }
            for(j = 1; j < sz2; j++){
                delta = 0.5*row_vector[j]*
                        (f_ptt->x->coreList[mode][i+((j-1+(k*sz2))*sz1)]
                       + f_ptt->x->coreList[mode][i+((j  +(k*sz2))*sz1)]);
                f_ptt->x->coreList[mode][i+((j-1+(k*sz2))*sz1)] = cumulant;
                cumulant += delta;
            }
            if(f_ptt->mpi_rk == 0){
                f_ptt->x->wk[i + sz1*k] = cumulant;
            }
            j = sz2 - 1;
            if(f_ptt->mpi_rk < f_ptt->mpi_sz - 1){
                f_ptt->x->wk[gsz + i + sz1*k] = 
                            f_ptt->x->coreList[mode][i+((j+(k*sz2))*sz1)];
            }
            f_ptt->x->coreList[mode][i+((j+(k*sz2))*sz1)] = cumulant;
        }
    }
    if(f_ptt->mpi_rk > 0){
        MPI_Wait(&recvReq, &mpi_stat);
        for(k = 0; k < sz3; k++){
            for(j = 0; j < sz2; j++){
                for(i = 0; i < sz1; i++){
                    f_ptt->x->coreList[mode][i+((j+(k*sz2))*sz1)] += 
                                   f_ptt->x->wk[2*gsz+i+ sz1*k]+
                                  +f_ptt->x->wk[3*gsz+i+ sz1*k]*
                                    0.5*row_vector[0];
                }
            }
        }
        if(f_ptt->mpi_rk < f_ptt->mpi_sz - 1){
            j = sz2 - 1;
            for(k = 0; k < sz3; k++){
                for(i = 0; i < sz1; i++){
                    f_ptt->x->wk[i+ sz1*k] = f_ptt->x->coreList[mode][i+
                                                        ((j+(k*sz2))*sz1)];
                }
            }
        }
    }
    if(f_ptt->mpi_rk < f_ptt->mpi_sz - 1){
        MPI_Isend(f_ptt->x->wk, 2*gsz, MPI_DOUBLE,
                    f_ptt->mpi_rk + 1,0,MPI_COMM_WORLD, &sendReq);
    }
}

void pattac_scale_weight_sum(paratt* f_ptt, int mode,
                             double* scale, double* sum_w){
    const int sz1 = f_ptt->x->ttRank[mode];
    const int sz2 = f_ptt->x->dim[mode];
    const int sz3 = f_ptt->x->ttRank[mode+1];
    const int gsz = sz1*sz3;
    double* core_copy;
    double cumulant;
    double delta;
    int i, j, k;
    MPI_Request sendReq;
    MPI_Request recvReq;
    MPI_Status  mpi_stat;
    /*
     * Resize work vector to store "bottom" ghost cell slice for
     * integration.
     */
    if(f_ptt->x->wksz < 4*gsz + gsz*sz2){
        f_ptt->x->wksz = 4*gsz + gsz*sz2;
        f_ptt->x->wk = (double*) realloc(f_ptt->x->wk,
                                sizeof(double)*f_ptt->x->wksz);
    }
    core_copy = &(f_ptt->x->wk[4*gsz]);
    if(f_ptt->mpi_rk > 0){
        MPI_Irecv(&(f_ptt->x->wk[2*gsz]), 2*gsz, MPI_DOUBLE ,
                    f_ptt->mpi_rk - 1,0,MPI_COMM_WORLD, &recvReq);
    }
    for(k = 0; k < sz3; k++){
        for(i = 0; i < sz1; i++){
            if(f_ptt->mpi_rk > 0){
                cumulant = f_ptt->x->coreList[mode][i+(((k*sz2))*sz1)]*
                            0.5*sum_w[0];
            }else{
                cumulant = 0.0;
            }
            for(j = 1; j < sz2; j++){
                core_copy[i+((j-1+(k*sz2))*sz1)] = 
                        f_ptt->x->coreList[mode][i+((j-1+(k*sz2))*sz1)];
                delta = 0.5*sum_w[j]*
                        (f_ptt->x->coreList[mode][i+((j-1+(k*sz2))*sz1)]
                       + f_ptt->x->coreList[mode][i+((j  +(k*sz2))*sz1)]);
                f_ptt->x->coreList[mode][i+((j-1+(k*sz2))*sz1)] = cumulant;
                cumulant += delta;
            }
            if(f_ptt->mpi_rk == 0){
                f_ptt->x->wk[i + sz1*k] = cumulant;
            }
            j = sz2 - 1;
            if(f_ptt->mpi_rk < f_ptt->mpi_sz - 1){
                f_ptt->x->wk[gsz + i + sz1*k] = 
                            f_ptt->x->coreList[mode][i+((j+(k*sz2))*sz1)];
            }
            core_copy[i+((j+(k*sz2))*sz1)] = 
                    f_ptt->x->coreList[mode][i+((j+(k*sz2))*sz1)];
            f_ptt->x->coreList[mode][i+((j+(k*sz2))*sz1)] = cumulant;
        }
    }
    if(f_ptt->mpi_rk > 0){
        MPI_Wait(&recvReq, &mpi_stat);
        for(k = 0; k < sz3; k++){
            for(j = 0; j < sz2; j++){
                for(i = 0; i < sz1; i++){
                    f_ptt->x->coreList[mode][i+((j+(k*sz2))*sz1)] += 
                                   f_ptt->x->wk[2*gsz+i+ sz1*k]+
                                  +f_ptt->x->wk[3*gsz+i+ sz1*k]*
                                    0.5*sum_w[0];
                }
            }
        }
        if(f_ptt->mpi_rk < f_ptt->mpi_sz - 1){
            j = sz2 - 1;
            for(k = 0; k < sz3; k++){
                for(i = 0; i < sz1; i++){
                    f_ptt->x->wk[i+ sz1*k] = f_ptt->x->coreList[mode][i+
                                                        ((j+(k*sz2))*sz1)];
                }
            }
        }
    }
    if(f_ptt->mpi_rk < f_ptt->mpi_sz - 1){
        MPI_Isend(f_ptt->x->wk, 2*gsz, MPI_DOUBLE,
                    f_ptt->mpi_rk + 1,0,MPI_COMM_WORLD, &sendReq);
    }
    for(k = 0; k < sz3; k++){
        for(j = 0; j < sz2; j++){
            for(i = 0; i < sz1; i++){
                f_ptt->x->coreList[mode][i+((j+(k*sz2))*sz1)] +=
                         core_copy[i+((j+(k*sz2))*sz1)]*scale[j];
            }
        }
    }
}


void pattac_write_tt_folder(char* folder_name, paratt* f_ptt){
    static const char overwrite_mode[3] = "w+";
    static const char append_mode[3] = "a+";
    static char file_name[128];
    int wait_msg = 1, recv_idx;
    int lr_idx, c_idx, rr_idx;
    int dim_idx, sz1, sz2;
    MPI_Status  mpi_stat;
    FILE*   cr_file;
    //First, check if the specifed folder exists.
    struct stat st = {0};
    //Core 0 immediately checks if the folder already exists. If it doesn't,
    //then the folder is created.
    if(f_ptt->mpi_rk == 0){
        if (stat(folder_name, &st) == -1) {
            mkdir(folder_name, 0777);
        }
        //Core zero creates the files within this folder
        for(dim_idx = 0; dim_idx < f_ptt->numDim; dim_idx++){
            sprintf(&(file_name[0]),"%s/core_%d.cr",folder_name,dim_idx+1);
            cr_file = fopen(file_name, &(overwrite_mode[0]));
            if(cr_file == NULL){
                fprintf(stderr,"\nTT write error: can't open %s\n",file_name);
                exit(1);
            }
            fprintf(cr_file,"%d,%d,%d\n", f_ptt->x->ttRank[dim_idx],
                    f_ptt->dim[dim_idx],f_ptt->x->ttRank[dim_idx+1]);
            fflush(cr_file);
            fclose(cr_file);
        }
    }
    for(dim_idx = 0; dim_idx < f_ptt->numDim; dim_idx++){
        wait_msg = f_ptt->mpi_rk;
        for(rr_idx = 0; rr_idx < f_ptt->x->ttRank[dim_idx+1]; rr_idx++){
            //Each core waits for the core above it in the memory layout
            //to finish writing to disk. Then it writes the contents of
            // each of the core slice
            if(wait_msg > 0){
                recv_idx  = f_ptt->mpi_rk-1;
                recv_idx += (recv_idx < 0) ? f_ptt->mpi_sz : 0;
                MPI_Recv(&wait_msg, 1, MPI_INT, recv_idx,
                         0, MPI_COMM_WORLD, &mpi_stat);
            }
            sprintf(&(file_name[0]),"%s/core_%d.cr",folder_name,dim_idx+1);
            cr_file = fopen(file_name, &(append_mode[0]));
            if(cr_file == NULL){
                fprintf(stderr,"\nTT write error: can't open %s\n",file_name);
                exit(1);
            }
            sz2 = f_ptt->x->dim[dim_idx];
            sz1 = f_ptt->x->ttRank[dim_idx];
            for(c_idx = 0; c_idx < f_ptt->x->dim[dim_idx]; c_idx++){
                for(lr_idx = 0; lr_idx < f_ptt->x->ttRank[dim_idx]; lr_idx++){
                    fprintf(cr_file,"%+1.16e\n",
                                        f_ptt->x->coreList[dim_idx][lr_idx+
                                                ((c_idx+(rr_idx*sz2))*sz1)]);
                }
            }
            fclose(cr_file);
            wait_msg = 1;
            //if there are more cores to write, give the next one the go-ahead
            if(f_ptt->mpi_rk+1 < f_ptt->mpi_sz){
                MPI_Send(&wait_msg, 1, MPI_INT,
                         f_ptt->mpi_rk+1, 0, MPI_COMM_WORLD);
            }else if(rr_idx+1 < f_ptt->x->ttRank[dim_idx+1]){
                //if MPI rank 0 needs to do more writing
                MPI_Send(&wait_msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }
    }
}

paratt* pattac_load_tt_folder(char* folder_name, int ndims){
    static const int  buff_sz = 128;
    static char buff[128];
    int dim_idx, rk_idx;
    int* ttr;
    int* dim;
    int ttr_k, ttr_kp1, dim_beg, dim_end;
    paratt* new_ptt;
    dim = (int*) malloc(sizeof(int)*ndims);
    ttr = (int*) malloc(sizeof(int)*(ndims+1));
    dim_idx = 1;
    sprintf(&(buff[0]),"%s/core_%d.cr",folder_name,dim_idx);
    read_core_size(&(buff[0]),
                  &ttr_k,
                  &(dim[0]),
                  &ttr_kp1);
    if(ttr_k != 1){
        fprintf(stderr, "%s:Line %d %s::pattac_load_tt_folder:: "
	             "Bad left TT rank in file %d.\n",
	             __FILE__, __LINE__, __func__, dim_idx);
    }
    ttr[0] = 1;
    for(dim_idx = 1; dim_idx < ndims; dim_idx++){
        sprintf(&(buff[0]),"%s/core_%d.cr",folder_name,dim_idx+1);
        ttr_k = ttr_kp1;
        read_core_size(&(buff[0]),
                      &(ttr[dim_idx]),
                      &(dim[dim_idx]),
                      &(ttr_kp1));
        //check for compatible core sizes.
        if(ttr[dim_idx] != ttr_k){
            fprintf(stderr, "%s:Line %d %s::pattac_load_tt_folder:: "
	                 "Bad left TT rank in file %d.\n",
	                 __FILE__, __LINE__, __func__, dim_idx);
        }
    }
    if(1 != ttr_kp1){
        fprintf(stderr, "%s:Line %d %s::pattac_load_tt_folder:: "
                 "Bad right TT rank in file %d.\n",
                 __FILE__, __LINE__, __func__, dim_idx);
    }
    ttr[ndims] = ttr_kp1;
    new_ptt = alloc_ptt(ndims, dim, ttr);
    //for each core, compute this MPI processor's subslice of the tensor
    //stored on disk.
    MPI_Barrier(MPI_COMM_WORLD);
    for(dim_idx = 0; dim_idx < ndims; dim_idx++){
        dim_beg = 0;
        for(rk_idx = 0; rk_idx < new_ptt->mpi_rk; rk_idx++){
            dim_beg += new_ptt->distrib_dim_mat[rk_idx+new_ptt->mpi_sz*dim_idx];
        }
        dim_end = dim_beg +
                  new_ptt->distrib_dim_mat[rk_idx+new_ptt->mpi_sz*dim_idx];
        sprintf(&(buff[0]),"%s/core_%d.cr",folder_name,dim_idx+1);
        load_core_slice(&(buff[0]),&(new_ptt->x->coreList[dim_idx]),
                          &(new_ptt->x->coreMem[dim_idx]),
                          &(new_ptt->x->ttRank[dim_idx]),
                          &(new_ptt->x->dim[dim_idx]),
                          &(new_ptt->x->ttRank[dim_idx+1]),
                          dim_beg, dim_end);
    }
    return new_ptt;
    free(dim);
    free(ttr);
}

void read_core_size(char* fname,
                     int* rl,
                     int* N,
                     int* rr){
    static const int  buff_sz = 128;
    static const char read_mode[2] = "r";
    static char buff[128];
    FILE* cr_file = fopen(fname,&(read_mode[0]));
    char* str_stat;
    int s_idx;
    int core_sz[3];
    int load_idx;
    if(cr_file == NULL){
        fprintf(stderr,"\nread_core_size error: can't open %s\n",fname);
        return;
    }
    str_stat = &(buff[0]);
    s_idx = -1;
    do{
        s_idx += 1;
        buff[s_idx] = fgetc(cr_file);
        if(buff[s_idx] == EOF || s_idx == buff_sz){
            fprintf(stderr, "\nread_core_size error: can't read line 1.\n");
            fclose(cr_file);
            return;
        }
    }while(buff[s_idx] != '\n');
    load_idx = 0;
    for(s_idx=0;s_idx<buff_sz;s_idx++){
        if(buff[s_idx] == ',' || buff[s_idx] == '\n'){
            buff[s_idx]        = '\0';
            core_sz[load_idx]  = atoi(str_stat);
            load_idx          += 1;
            str_stat           = (&(buff[s_idx])) + 1;
        }
        if(load_idx >= 3){
            break;
        }
    }
    rl[0]    = core_sz[0];
    N[0]     = core_sz[1];
    rr[0]    = core_sz[2];
    fclose(cr_file);
}

void load_core_slice(char* fname,
                     double** cr,
                     int* cr_sz,
                     int* rl,
                     int* N,
                     int* rr,
                     int N_beg,
                     int N_end){
    static const int  buff_sz = 128;
    static const char read_mode[2] = "r";
    static char buff[128];
    FILE* cr_file = fopen(fname,&(read_mode[0]));
    char* str_stat;
    char* float_end;
    int i,j,k,s_idx;
    int core_sz[3];
    int load_idx, line_idx;
    int num_lines_data;
    if(cr_file == NULL){
        fprintf(stderr,"\nload_core_slice error: can't open %s\n",fname);
        return;
    }
    str_stat = &(buff[0]);
    s_idx = -1;
    do{
        s_idx += 1;
        buff[s_idx] = fgetc(cr_file);
        if(buff[s_idx] == EOF || s_idx == buff_sz){
            fprintf(stderr, "\nload_core_slice error: can't read line 1.\n");
            fclose(cr_file);
            return;
        }
    }while(buff[s_idx] != '\n');
    load_idx = 0;
    for(s_idx=0;s_idx<buff_sz;s_idx++){
        if(buff[s_idx] == ',' || buff[s_idx] == '\n'){
            buff[s_idx]        = '\0';
            core_sz[load_idx]  = atoi(str_stat);
            load_idx          += 1;
            str_stat           = (&(buff[s_idx])) + 1;
        }
        if(load_idx >= 3){
            break;
        }
    }
    if(N_end > core_sz[1] || N_beg < 0){
        fclose(cr_file);
        fprintf(stderr,
                "\nload_core_slice error: slice indexes out of bounds.\n"
                "N_beg = %d, N_end = %d, core_sz[1] = %d\n",
                N_beg,N_end,core_sz[1]);
        return;
    }
    load_idx       = 0;
    num_lines_data = core_sz[0]*core_sz[1]*core_sz[2];
    if(cr_sz[0] < core_sz[0]*(N_end-N_beg)*core_sz[2]){
        cr_sz[0] = core_sz[0]*(N_end-N_beg)*core_sz[2];
        cr[0]    = (double*) realloc(cr[0],sizeof(double)*cr_sz[0]);
    }
    rl[0]    = core_sz[0];
    N[0]     = N_end-N_beg;
    rr[0]    = core_sz[2];
    line_idx = 1;
    for(k=0;k<core_sz[2];k++){
        for(j=0;j<core_sz[1];j++){
            for(i=0;i<core_sz[0];i++){
                s_idx = -1;
                do{
                    s_idx += 1;
                    buff[s_idx] = fgetc(cr_file);
                    if(buff[s_idx] == EOF || s_idx == buff_sz){
                        fprintf(stderr,
                            "\nload_core_slice error: can't read line %d.\n",
                             line_idx);
                        fclose(cr_file);
                        return;
                    }
                }while(buff[s_idx] != '\n');
                buff[s_idx] = '\0';
                if(N_beg <= j && j < N_end){ 
                    cr[0][load_idx] = strtod(&(buff[0]),&float_end);
                    load_idx += 1;
                }
                line_idx += 1;
            }
        }
    }
    fclose(cr_file);
}




