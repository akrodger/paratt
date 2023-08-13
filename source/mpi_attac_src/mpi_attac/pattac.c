/*
 *
 * This software is a fork of MPI ATTAC (the original software) by the original
 * authors listed below. This fork retains the same 2 clause license and
 * disclaimer as the orignal software -- 2022 Abram Rodgers
 *
 * Original software copyright 2021 Hussam Al Daas, Grey Ballard, and 
 * Peter Benner
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,*
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
*/

#include <math.h>
#include <unistd.h>
#include "../serial_attac/utilities.h"
#include "../serial_attac/attac.h"
#include "pattac.h"

/** \fn void pattacScale(const double alpha, int mode, const int d, const int* n, const int* rx, double *const *x);
 * \brief
 * \details
 * \param alpha
 * \param mode
 * \param d
 * \param n
 * \param rx
 * \param x
 * \remarks
 * \warning
*/
void pattacScale(const double alpha, int mode, const int d, const int* n, const int* rx, double **x){
    int i;
	if(mode < 0 || mode > d-1){
		/**< Scale first mode */
    for(i = 0; i < n[0] * rx[1]; i++){
      x[0][i] *= alpha;
    }
	}else{
		i = mode;
		int j;
    for(j = 0; j < rx[i] * n[i] * rx[i + 1]; j++){
      x[i][j] *= alpha;
    }
	}
}

/** \fn double pattacValue(const MPI_Comm comm, const int* isRedundant, const int* index, const int* procHolder, const int d, const int* n, const int* rx, double *const *x, double* work);
 * \brief
 * \details
 * \param comm
 * \param isRedundant
 * \param index
 * \param procHolder
 * \param d
 * \param n
 * \param rx
 * \param x
 * \param work
 * \remarks
 * \warning
*/
double pattacValue(const MPI_Comm comm, const int* index, const int* procHolder, const int d, const int* n, const int* rx, double *const *x, double* work){
	int rank = 0;//, ierr = 0;
	MPI_Comm_rank(comm, &rank);
	MPI_Status status;
	// < Maximum TT rank
    int rmax = 0;
    int i;
    for(i = 0; i < d; i++){
        rmax = (rmax > rx[i]) ? rmax : rx[i];
    }

    // < Temporary variables for computation
    double *a = work, *b = work + rmax, *temp = NULL;

    if(rank == procHolder[d - 1]){
        memcpy(a, x[d - 1] + index[d - 1] * rx[d - 1], rx[d - 1] * sizeof(double));
    }
    int one = 1;
    double done = 1.0;
    double zero = 0.0;
    for(i = d - 2; i > -1; i--){
    int rxn = rx[i] * n[i];
        //  < procHolder[i + 1] sends the vector to procHolder[i]
        if(rank == procHolder[i + 1] && rank != procHolder[i]){
           MPI_Send(a, rx[i + 1], MPI_DOUBLE, procHolder[i], 0, comm);
        }
        // < procHolder[i] receives the vector from procHolder[i + 1]
        if(rank == procHolder[i] && rank != procHolder[i + 1]){
            MPI_Recv(a, rx[i + 1], MPI_DOUBLE, procHolder[i + 1], 0, comm, &status);
		}
		if(rank == procHolder[i]){
      dgemv_("N", &rx[i], &rx[i + 1], &done, x[i] + index[i] * rx[i], &rxn, a, &one, &zero, b, &one);
			temp = a;
			a = b;
			b = temp;
		}
	}
	MPI_Bcast(a, 1, MPI_DOUBLE, procHolder[0], comm);
	return a[0];
}

/** \fn void pattacFormalSum(const int d, const int *n, const int *rx, double *const *x, const int *ry, double *const *y, int* rz, double **z);
 * \brief
 * \details
 * \param d
 * \param n
 * \param rx
 * \param x
 * \param ry
 * \param y
 * \param rz
 * \param z
 * \remarks
 * \warning
*/
void pattacFormalSum(const int d, const int *n, const int *rx, double *const *x, const int *ry, double *const *y, int* rz, double **z){
	attacFormalSum(d, n, rx, x, ry, y, rz, z);
}

/** \fn void pattacFormalAXPBY(const int d, const int *n, const double alpha, const int *rx, double * const *x, const double beta, const int *ry, double *const *y, int* rz, double **z);
 * \brief
 * \details
 * \param d
 * \param n
 * \param alpha
 * \param rx
 * \param x
 * \param beta
 * \param ry
 * \param y
 * \param rz
 * \param z
 * \remarks
 * \warning
*/
void pattacFormalAXPBY(const int d, const int *n, const double alpha, const int *rx, double * const *x, const double beta, const int *ry, double *const *y, int* rz, double **z){
	attacFormalAXPBY(d, n, alpha, rx, x, beta, ry, y, rz, z);
}

/** \fn void pattacHadamard(const int d, const int *n, const int *rx, double *const *x, const int *ry, double *const *y, int* rz, double **z);
 * \brief
 * \details
 * \param d
 * \param n
 * \param rx
 * \param x
 * \param ry
 * \param y
 * \param rz
 * \param z
 * \remarks
 * \warning
*/
void pattacHadamard(const int d, const int *n, const int *rx, double *const *x, const int *ry, double *const *y, int* rz, double **z){
	attacHadamard(d, n, rx, x, ry, y, rz, z);
}



/** \fn double pattacNorm(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, const int *rx, double **x, double* work);
 * \brief
 * \details
 * \param comm
 * \param isRedundant
 * \param d
 * \param n
 * \param rx
 * \param x
 * \param work
 * \remarks
 * \warning
*/
double pattacNorm(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, const int *rx, double **x, double* work){
	double norm = 0;
	int rank = -1;
	MPI_Comm_rank(comm, &rank);

  // Avoid TSQR 
  // if x[0]     is not distributed
  // if x[d - 1] is not distributed and """"N[0] * rx[1] * rx[1] < """"N[d - 1] * rx[d - 1] * rx[d-1]  requires the global N
  if(isRedundant[0] == 1){
	  // **< Left orthogonalization of the TT tensor
	  //pattacBFLOrthogonalization(comm, isRedundant, d, n, rx, x, work);
	  // **< Frobenius norm of the last TT core
    norm = dlange_("F", &rx[d - 1], &n[d - 1], x[d - 1], &rx[d - 1], NULL);
    if(isRedundant[d - 1] == 0){
	    // < Squared norm 
	    norm *= norm;
	    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
	    // < Norm 
	    norm = sqrt(norm);
    }
  }else{
	  // < Right orthogonalization of the TT tensor 
	  //pattacBFROrthogonalization(comm, isRedundant, d, n, rx, x, work);
	  // < Frobenius norm of the first TT core
    norm = dlange_("F", &n[0], &rx[1], x[0], &n[0], NULL);
    if(isRedundant[d - 1] == 0){
	    // < Squared norm
	    norm *= norm;
	    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
	    // < Norm 
	    norm = sqrt(norm);
    }
  }
	return norm;
}


/** \fn void pattacLOrthogonalization(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, const int *rx, double **x, double* work);
 * \brief
 * \details
 * \param comm
 * \param isRedundant
 * \param d
 * \param n
 * \param rx   input/output ranks of x
 * \param x    input/output tensor x to be left orthogonalized on output
 * \param work minimal space: \f$2 (1 + CLog2P) r^2 + r^2 + 2 r^2 + \max(n[i] rx[i] rx[i+1])\f$, where \f$r = \max(rx[i])\f$ and P is number of procs
 * \remarks
 * \warning
*/
void pattacLOrthogonalization(MPI_Comm comm,
                              int d,
                              int* distrib_dim_mat,
                              int *rx,
                              double **x,
                              tsqr_tree* qrTree,
                              int* int_wrk){
    int size = 0, rank = 0;
    int core_right_numCols, i, j;
    double* R;
    int_list local_dim;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    local_dim.arr = int_wrk;
    local_dim.len = size;
	for(i = 0; i < d - 1; i++){
		/**
		 * \remarks Compute a QR factorization of x[i]_{(2)}^T.
		 */
		for(j = 0; j < size; j++){
		    local_dim.arr[j] = rx[i]*distrib_dim_mat[j+i*size];
		}
		fill_tsqr_tree(qrTree,local_dim,rx[i+1]);
		tsqr_factorize(qrTree, x[i]);
        realloc_to_max(&qrTree->flt_wrk,&qrTree->wrk_sz,sizeof(double),
                       qrTree->q_numCol[0]*qrTree->r_numCol[0]);
        R = qrTree->flt_wrk;
        upper_tri_unpack(R,qrTree->q_numCol[0],qrTree->q_numCol[0],
                           qrTree->r_numCol[0],qrTree->R[0]);
		/**
		 * \remarks Multiplies R by the left TT core.
		 */
		/**< x[i + 1] = R x[i + 1] */

        core_right_numCols = distrib_dim_mat[rank+(i+1)*size] * rx[i + 2];
    //dtrmm_("L", "U", "N", "N", 
    //        &rx[i + 1], &nprxp, &done, R, &rx[i + 1], x[i + 1], &rx[i + 1]);
        trapezoid_mult('l',
                        R,
                        qrTree->q_numCol[0],
                        qrTree->r_numCol[0],
                        core_right_numCols,
                        x[i + 1],
                        rx[i + 1],
                        &rx[i + 1],//int*    result_rows,
                        &core_right_numCols);//int*    result_cols);

        /**< Generate Q of x[i]_{(2)}^T*/
        tsqr_make_q(qrTree, x[i]);
	}
}


void pattacLTruncate(MPI_Comm comm,
                     const int d, 
                     int* distrib_dim_mat,
                     int *rx,
                     double **x,
                     const int *maxr,
                     const double abs_tol,
                     tsqr_tree* qrTree,
                     int* int_wrk,
                     double** flt_wrk_ptr,
                     int*     flt_sz_ptr){
    static const double      done   = 1.0;
    static const double      dzero  = 0.0;
    static const char no_transpose  = 'N';
    static const char copy_all      = 'A';
    static const char strong_trunc  = 'A';
    static const char do_pivot      = 'P';
    static const char reduced_svd   = 'R';
    const double abs_tol_div_d = abs_tol/sqrt(d-1.0);
    int size = 0, rank = 0;
    int i, j, k;
    int_list local_dim;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    //int num_rows_full_sVector;
	int ierr = 0;
	int num_sv_found;
	// < Maximum TT rank
	int rmax       = 0;
	int local_dmax = 0;
	double work_sizes[3];
	double sigSum;
	double *sig = NULL;
	double *VT = NULL;
	double *svd_wk = NULL;
	double *xTemp = NULL;
	double *R = NULL;
	double *Rwork = NULL;
    int    *iwork = int_wrk + size;
    int liwork;
    int rwork_sz;
	int rankSVD = 0;
	int R_sz, sig_sz, VT_sz, svd_wk_sz, xTemp_sz;
	for(i = 0; i < d; i++){
	    local_dmax = (local_dmax > distrib_dim_mat[rank+i*size]) ?
	                  local_dmax : distrib_dim_mat[rank+i*size];
		rmax = (rmax > rx[i]) ? rmax : rx[i];
	}
    local_dim.arr = int_wrk;
    local_dim.len = size;
    xTemp_sz  = rmax*rmax*local_dmax;
	for(i = d - 1; i > 0; i--){
		// < Compute the R factor and implicit Q factor
		for(j = 0; j < size; j++){
		    local_dim.arr[j] = rx[i+1]*distrib_dim_mat[j+i*size];
		}
		//printf("\nmode = %d\n", i);
		//getc(stdin);
		fill_tsqr_tree(qrTree,local_dim,rx[i]);
		//printf("\nFILLED TSQR TREE. ABOUT TO WFLQ\n");
		//getc(stdin);
		wflq_factorize(qrTree, x[i]);
		//printf("\nDONE WITH WFLQ\n");
		//getc(stdin);
		R_sz = qrTree->r_numCol[0]*qrTree->q_numCol[0];
		sig_sz = (qrTree->q_numCol[0] < qrTree->r_numCol[0]) ? 
		          qrTree->q_numCol[0] : qrTree->r_numCol[0];
		num_sv_found = sig_sz;

		VT_sz  = sig_sz*qrTree->q_numCol[0];
		svd_wk_sz = -1;
		liwork    = -1;
        rwork_sz  = -1;
        dgesvdq_(&strong_trunc,
                 &do_pivot,
                 &no_transpose,
                 &reduced_svd,
                 &reduced_svd,
                 qrTree->r_numCol,
                 qrTree->q_numCol,
                 R,
                 qrTree->r_numCol,
                 sig,
                 xTemp,
                 qrTree->r_numCol,
                 VT,
                 &sig_sz,
                 &num_sv_found,
                 iwork,
                 &liwork,
                 &(work_sizes[0]),
                 &svd_wk_sz,
                 &(work_sizes[2]),
                 &rwork_sz,
                 &ierr);
		/*
        dgesdd_(&copy_all,
                qrTree->r_numCol, qrTree->q_numCol,
                    R, qrTree->r_numCol,
                    sig,
                    xTemp,  qrTree->r_numCol,
                    VT, &sig_sz,
                    &sigSum, &svd_wk_sz,
                    iwork,
                    &ierr);
        */
        svd_wk_sz = ((int) work_sizes[0])+1;//(int)(sigSum);
        rwork_sz  = ((int) work_sizes[2])+1;
        liwork    = iwork[0];
        realloc_to_max(flt_wrk_ptr,flt_sz_ptr,sizeof(double),
                        R_sz+sig_sz+VT_sz+svd_wk_sz+rwork_sz+xTemp_sz);
        //
        // \remark: R on proc 0 is already unpacked and has 0 in its lower triangular part
        //
        R      = *flt_wrk_ptr;
        sig    = R      + R_sz;
        VT     = sig    + sig_sz;
        svd_wk = VT     + VT_sz;
        Rwork  = svd_wk + svd_wk_sz;
        xTemp  = Rwork  + rwork_sz;
        lower_tri_unpack(R,
                         qrTree->r_numCol[0],
                         qrTree->r_numCol[0],
                         qrTree->q_numCol[0],
                         qrTree->R[0]);
        //printf("\nDone with lower tri unpack.\n");
        //getc(stdin);
/*
        dgesdd_(&copy_all,
                qrTree->r_numCol, qrTree->q_numCol,
                    R, qrTree->r_numCol,
                    sig, xTemp,  qrTree->r_numCol, VT, &sig_sz,
                    svd_wk, &svd_wk_sz, iwork, &ierr);
*/
        dgesvdq_(&strong_trunc,
                 &do_pivot,
                 &no_transpose,
                 &reduced_svd,
                 &reduced_svd,
                 qrTree->r_numCol,
                 qrTree->q_numCol,
                 R,
                 qrTree->r_numCol,
                 sig,
                 xTemp,
                 qrTree->r_numCol,
                 VT,
                 &sig_sz,
                 &num_sv_found,
                 iwork,
                 &liwork,
                 svd_wk,
                 &svd_wk_sz,
                 Rwork,
                 &rwork_sz,
                 &ierr);
		if(ierr != 0){
			fprintf(stderr, "%s:Line %d %s::LAPACKE_dgesdd:: "
			                "ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
			fprintf(stderr, "core %d compression, r[%d]  = %d "
			                "\n r[%d] = %d \n n[%d] = %d\n",
			                 i, i, rx[i], i + 1, rx[i + 1],
			                 i,distrib_dim_mat[rank+size*i]);
			MPI_Abort(MPI_COMM_WORLD, ierr);
			exit(1);
		}
		/*
		num_rows_full_sVector = 0;
	    for(j = 0; j < size; j++){
	        num_rows_full_sVector += distrib_dim_mat[j+i*size];
	    }*/
        sigSum = 0.0;
        rankSVD = num_sv_found;
	    for(j = num_sv_found-1; j >= 0; j--){
            sigSum += (((sig[j]*sig[j])));
		    if( sqrt(sigSum) < abs_tol_div_d){	
			    rankSVD = j;
		    }else{
			    break;
		    }
	    }
        rankSVD = (qrTree->q_numCol[0] < rankSVD) ?
                   qrTree->q_numCol[0] : rankSVD;
        rankSVD = (maxr[i] < rankSVD) ? maxr[i] : rankSVD;
        rankSVD = (rankSVD > 1) ? rankSVD : 1;
        // < Scale the column j in U by the j^th singular value
        for(j = 0; j < rankSVD; j++){
            for(k = 0; k < qrTree->r_numCol[0]; k++){
                R[j*qrTree->r_numCol[0] + k] =
                                     sig[j]*xTemp[j*qrTree->r_numCol[0] + k];
            }
        }
        wflq_make_q(qrTree, xTemp);
        dgemm_(	&no_transpose,//character 	TRANSA,
                &no_transpose,//character 	TRANSB,
                &rankSVD,//integer 	M,
                &(qrTree->q_numRow[qrTree->leafIdx[rank]]),//integer 	N,
                &(qrTree->q_numCol[0]),//integer 	K,
                &done,//double precision 	ALPHA,
                VT,//double precision, dimension(lda,*) 	A,
                &sig_sz,//integer 	LDA,
                xTemp,//double precision, dimension(ldb,*) 	B,
                &(qrTree->q_numCol[0]),//integer 	LDB,
                &dzero,//double precision 	BETA,
                x[i],//double precision, dimension(ldc,*) 	C,
                &rankSVD);//integer 	LDC
        //size of the core one left of current core.
        int rxmnm = rx[i-1]*distrib_dim_mat[rank+size*(i-1)];
        //Multiply singular value weighted R into 1 core left. Save in temp.
        dgemm_(	&no_transpose,
                &no_transpose,
                &rxmnm,         //# rows left core reshaping
                &rankSVD,       //truncated SVD rank, new num columns.
                &rx[i],         //shared index of matrix product, old num cols
                &done,          //64 bit floating point 1.0
                x[i - 1],       //address of left core
                &rxmnm,         //leading dimension of left core
                R,              //Contains sing. vects weighted by sig
                &qrTree->r_numCol[0],         //leading dimension of R 
                &dzero,         //64 bit floating point 0.0
                xTemp,          //temporary variable.
                &rxmnm);        //leading dimension of result.
        dlacpy_(&copy_all, &rxmnm, &rankSVD, xTemp, &rxmnm, x[i - 1], &rxmnm);
		rx[i] = rankSVD;
	}
}

/** \fn void pattacROrthogonalization(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, const int *rx, double **x, double* work);
 * \brief
 * \details
 * \param comm
 * \param isRedundant
 * \param d
 * \param n
 * \param rx   input/output ranks of x
 * \param x    input/output tensor to be right orthogonalized
 * \param work minimum space: \f$2 (1 + CLog2P) r^2 + r^2 + 2r^2 + \max(n[i] rx[i] rx[i + 1])\f$, where \f$P\f$ is the number of processors and \f$r\f$ is \f$\max(rx[i])\f$
 * \remarks
 * \warning
*/
void pattacROrthogonalization(MPI_Comm comm,
                              int d,
                              int* distrib_dim_mat,
                              int *rx,
                              double **x,
                              tsqr_tree* qrTree,
                              int* int_wrk){
    int size = 0, rank = 0;
    int core_left_numRows, i, j;
    double* R;
    int_list local_dim;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    local_dim.arr = int_wrk;
    local_dim.len = size;
	for(i = d-1; i > 0; i--){
		/**
		 * \remarks Compute a QR factorization of x[i]_{(2)}^T.
		 */
		for(j = 0; j < size; j++){
		    local_dim.arr[j] = rx[i+1]*distrib_dim_mat[j+i*size];
		}
		fill_tsqr_tree(qrTree,local_dim,rx[i]);
		wflq_factorize(qrTree, x[i]);
        realloc_to_max(&qrTree->flt_wrk,&qrTree->wrk_sz,sizeof(double),
                       qrTree->q_numCol[0]*qrTree->r_numCol[0]);
        R = qrTree->flt_wrk;
        lower_tri_unpack(R,
                         qrTree->r_numCol[0],
                         qrTree->r_numCol[0],
                         qrTree->q_numCol[0],
                         qrTree->R[0]);
		//for(int jj = 0; jj < qrTree->r_numCol[0]; jj++){
        //    printf("\n");
        //    for(int ii = 0; ii <  qrTree->q_numCol[0]; ii++){
        //        printf(" %+2.1le",R[jj+ii*qrTree->r_numCol[0]]);
        //    }
        //}
        core_left_numRows = rx[i - 1]*distrib_dim_mat[rank+(i-1)*size];
        trapezoid_mult('r',
                        R,
                        qrTree->r_numCol[0],
                        qrTree->q_numCol[0],
                        core_left_numRows,
                        x[i - 1],
                        core_left_numRows,
                        &core_left_numRows,//int*    result_rows,
                        &rx[i]);//int*    result_cols);
        wflq_make_q(qrTree, x[i]);
	}
}


void pattacTruncate(MPI_Comm comm,
                    const int d, 
                    int *distrib_dim_mat,
                    int *rx,
                    double **x,
                    const int *maxr,
                    const double abs_tol,
                    const double rel_tol,
                    const char ortho,
                    tsqr_tree* qrTree,
                    int* int_wrk,       //assumed size = MPI num procs
                    double** flt_wrk_ptr, //work query always performed.
                    int*     flt_sz_ptr){
	int rank = 0, size=1, i, j, full_tensor_mode;
	double tol;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	
	for(i = 0; i < d; i++){
	    full_tensor_mode = 0;
	    for(j = 0; j < size; j++){
	        full_tensor_mode += distrib_dim_mat[j+i*size];
	    }
	    for(j=0; j < rx[i]*distrib_dim_mat[rank+i*size]*rx[i+1]; j++){
	        x[i][j] /= sqrt((double)full_tensor_mode);
	    }
	}
	
	if(ortho == 'L'){
		/**< Right orthogonalization */
		pattacROrthogonalization(comm, d, distrib_dim_mat,
                                   rx, x, qrTree, int_wrk);
		/**< Compress a right orthogonal TT tensor */
		//pattacRTruncate(comm, isRedundant, d, n, rx, x, maxr, abs_tol, work);
	}else if(ortho == 'R'){
		/**< Left orthogonalization */
		pattacLOrthogonalization(comm, d, distrib_dim_mat,
                                   rx, x, qrTree, int_wrk);
	  // **< Frobenius norm of the last TT core
        /*
        tol = dlange_("F", &rx[d - 1], &distrib_dim_mat[rank+(d-1)*size],
                        x[d - 1], &rx[d - 1], NULL);
	    // < Squared norm 
	    tol *= tol;
	    MPI_Allreduce(MPI_IN_PLACE, &tol, 1, MPI_DOUBLE, MPI_SUM, comm);
	    //tol contains the norm.
        tol = sqrt(tol);
	    for(i = 0; i < d; i++){
	        full_tensor_mode = 0;
	        for(j = 0; j < size; j++){
	            full_tensor_mode += distrib_dim_mat[j+i*size];
	        }
	        if(i < d-1){
	            for(j=0; j < rx[i]*distrib_dim_mat[rank+i*size]*rx[i+1]; j++){
	                x[i][j] *= pow(tol,1.0/d);
	            }
	        }else{
	            for(j=0; j < rx[i]*distrib_dim_mat[rank+i*size]*rx[i+1]; j++){
	                x[i][j] *= pow(tol,(1.0-d)/d);
	            }
	        }
	    }
	    tol = tol*rel_tol;
	    tol = abs_tol < tol ? abs_tol : tol;
	    */
	    tol = abs_tol;
		/**< Compress a left orthogonal TT tensor */
        pattacLTruncate(comm, d, distrib_dim_mat, rx, x, maxr, tol, qrTree,
		                int_wrk, flt_wrk_ptr, flt_sz_ptr);

	}else{
		fprintf(stderr, "%s:Line %d %s::Not known orthogonalization side\n", __FILE__, __LINE__, __func__);
		MPI_Abort(comm, -99);
	}
	
	for(i = 0; i < d; i++){
	    full_tensor_mode = 0;
	    for(j = 0; j < size; j++){
	        full_tensor_mode += distrib_dim_mat[j+i*size];
	    }
	    for(j=0; j < rx[i]*distrib_dim_mat[rank+i*size]*rx[i+1]; j++){
	        x[i][j] *= sqrt((double)full_tensor_mode);
	    }
	}
	
}


/** \fn double pattacInnerProduct(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, const int *rx, double *const *x, const int *ry, double *const *y, double *work)
 * \brief Inner product of two TT tensors
 * \details
 * \param comm
 * \param isRedundant
 * \param d
 * \param n
 * \param rx
 * \param x
 * \param ry
 * \param y
 * \param work
 * \remarks
 * \warning
*/
double pattacInnerProduct(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, const int *rx, double *const *x, const int *ry, double *const *y, double *work){
	int rank = 0, ierr = 0;
	MPI_Comm_rank(comm, &rank);
	int rmax = 0;
	int i;
	for(i = 0; i < d; i++){
		rmax = (rmax > rx[i]) ? rmax : rx[i];
		rmax = (rmax > ry[i]) ? rmax : ry[i];
	}
  double *a = work;
  double *b = work + rmax * rmax;

  int rxn = rx[0] * n[0];
  int ryn = ry[0] * n[0];
  int nry = n[0] * ry[1];

  double done = 1.0;
  double zero = 0.0;
  /**< initiate the first core product */

  dgemm_("T", "N", &rx[1], &ry[1], &n[0], &done, x[0], &rxn, y[0], &ryn, &zero, a, &rx[1]);

#if TIME_BREAKDOWN
    if(isRedundant[0] == 0) MPI_Barrier(comm);
#endif
  /**< sum up along the mode */
  ierr = MPI_Allreduce(MPI_IN_PLACE, a, rx[1] * ry[1], MPI_DOUBLE, MPI_SUM, comm);
  if(ierr != 0){
    fprintf(stderr, "%s:Line %d %s::MPI_Allreduce:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
  }

    for(i = 1; i < d; i++){
        rxn = rx[i] * n[i];
        ryn = ry[i] * n[i];
        nry = n[i] * ry[i + 1];
        /**< H(B) = M H(Y_{i}) */
        dgemm_("N", "N", &rx[i], &nry, &ry[i], &done, a, &rx[i], y[i], &ry[i], &zero, b, &rx[i]);
        /**< V(X_{i})^T V(B)  */
        dgemm_("T", "N", &rx[i + 1], &ry[i + 1], &rxn, &done, x[i], &rxn, b, &rxn, &zero, a, &rx[i + 1]);

        ierr = MPI_Allreduce(MPI_IN_PLACE, a, rx[i + 1] * ry[i + 1], MPI_DOUBLE, MPI_SUM, comm);
        
        if(ierr != 0){
	        fprintf(stderr, "%s:Line %d %s::MPI_Allreduce:: ierr = %d\n",
	                                 __FILE__, __LINE__, __func__, ierr);
        }
    }
	return a[0];
}

