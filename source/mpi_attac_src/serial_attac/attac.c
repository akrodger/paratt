/** \copyright
 * Copyright 2021 Hussam Al Daas, Grey Ballard, and Peter Benner
 * 
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/** \file attac.c
 * \author Hussam Al Daas
 * \brief Algorithms for Tensor Train Arithmetic and Computations
 * \date 16/07/2019
 * \details 
*/
#include <math.h>
#include <string.h>
#include "utilities.h"
#include "attac.h"

/** \fn void attacFormalSum(const int d, const int *n, const int *rx, double * const *x, const int *ry, double * const *y, int* rz, double **z);
 * \brief Puts z = x + y in formal representation
 * \details
 * \param d  number of modes
 * \param n  mode sizes
 * \param rx input ranks of x
 * \param x  input tensor x
 * \param ry input ranks of y
 * \param y  input tensor y
 * \param rz output ranks of z
 * \param z  output tensor z = x + y
 * \remarks
 * \warning
*/
void attacFormalSum(const int d, const int *n, const int *rx, double * const *x, const int *ry, double * const *y, int* rz, double **z){
    int i, j;
	rz[0] = 1;
	rz[d] = 1;
	for(i = 1; i < d; ++i)
		rz[i] = rx[i] + ry[i];
	
	/**< copy cores 0 of x and y*/
  dlacpy_("A", &n[0], &rx[1], x[0], &n[0], z[0], &n[0]);
  dlacpy_("A", &n[0], &ry[1], y[0], &n[0], z[0] + n[0] * rx[1], &n[0]);

	/**< copy cores d-1 of x and y*/
  dlacpy_("A", &rx[d - 1], &n[d - 1], x[d - 1], &rx[d - 1], z[d - 1], &rz[d - 1]);
  dlacpy_("A", &ry[d - 1], &n[d - 1], y[d - 1], &ry[d - 1], z[d - 1] + rx[d - 1], &rz[d - 1]);
	for(i = 1; i < d - 1; ++i){
		memset(z[i], 0, rz[i] * n[i] * rz[i + 1] * sizeof(double));
		
		for(j = 0; j < n[i]; ++j){
      int rxn = rx[i] * n[i];
      int ryn = ry[i] * n[i];
      int rzn = rz[i] * n[i];
      dlacpy_("A", &rx[i], &rx[i + 1], x[i] + j * rx[i], &rxn, z[i] + j * rz[i], &rzn);
      dlacpy_("A", &ry[i], &ry[i + 1], y[i] + j * ry[i], &ryn, z[i] + j * rz[i] + rx[i + 1] * rz[i] * n[i] + rx[i], &rzn);
		}
	}

}

/** \fn void attacFormalAXPBY(const int d, const int *n, const double alpha, const int *rx, double * const *x, const double beta, const int *ry, double * const *y, int* rz, double **z);
 * \brief Puts z = alpha x +  beta y in formal representation
 * \details
 * \param d     number of modes
 * \param n     mode sizes
 * \param alpha input scalar
 * \param rx    input ranks of x
 * \param x     input tensor x
 * \param beta  input scalar
 * \param ry    input ranks of y
 * \param y     input tensor y
 * \param rz    output ranks of z
 * \param z     output tensor z = alpha x + beta y
 * \remarks
 * \warning
*/
void attacFormalAXPBY(const int d, const int *n, const double alpha, const int *rx, double * const *x, const double beta, const int *ry, double * const *y, int* rz, double **z){
    int i, j;
	rz[0] = 1;
	rz[d] = 1;
	for( i = 1; i < d; ++i)
		rz[i] = rx[i] + ry[i];
	
	/**< Scale and copies cores 0 of x and y*/
  for(i = 0; i < n[0] * rx[1]; i++){
    z[0][i] = x[0][i] * alpha;
  }
  for(i = 0; i < n[0] * ry[1]; i++){
    z[0][rx[1] * n[0] + i] = y[0][i] * beta;
  }

	/**< Copy cores d-1 of x and y*/
  dlacpy_("A", &rx[d - 1], &n[d - 1], x[d - 1], &rx[d - 1], z[d - 1], &rz[d - 1]);
  dlacpy_("A", &ry[d - 1], &n[d - 1], y[d - 1], &ry[d - 1], z[d - 1] + rx[d - 1], &rz[d - 1]);

	for(i = 1; i < d - 1; ++i){
		memset(z[i], 0, rz[i] * n[i] * rz[i + 1] * sizeof(double));
		
		for( j = 0; j < n[i]; ++j){
      int rxn = rx[i] * n[i];
      int ryn = ry[i] * n[i];
      int rzn = rz[i] * n[i];
      dlacpy_("A", &rx[i], &rx[i + 1], x[i] + j * rx[i], &rxn, z[i] + j * rz[i], &rzn);
      dlacpy_("A", &ry[i], &ry[i + 1], y[i] + j * ry[i], &ryn, z[i] + j * rz[i] + rx[i + 1] * rz[i] * n[i] + rx[i], &rzn);
		}
	}

}

/** \fn void attacHadamard(const int d, const int *n, const int *rx, double * const *x, const int *ry, double * const *y, int* rz, double **z);
 * \brief Puts the formal representation of Hadamard product of x and y in z
 * \details
 * \param d  number of modes
 * \param n  modes size
 * \param rx input ranks of x
 * \param x  input tensor x
 * \param ry input ranks of y
 * \param y  input tensor y
 * \param rz output ranks of z
 * \param z  output tensor z, z = x Hadamard y
 * \remarks
 * \warning
*/
void attacHadamard(const int d, const int *n, const int *rx, double * const *x, const int *ry, double * const *y, int* rz, double **z){
    int i, j;
	for(i = 0; i < d; ++i){
		for(j = 0; j < n[i]; ++j){
			attacKron(rx[i], rx[i + 1], x[i] + j * rx[i], rx[i] * n[i], ry[i], ry[i + 1], y[i] + j * ry[i], ry[i] * n[i], z[i] + j * rz[i], rz[i] * n[i]);
		}
	}

}

/** \fn double attacNormFromHadamard(const int d, const int *n, const int *rx, double * const *x, double* work);
 * \brief  Computes the Frobenius norm a tensor y s.t. x = Hadamard(y,y)
 * \details
 * \param d    number of modes
 * \param n    modes size
 * \param rx   input ranks of x
 * \param x    input tensor x (supposed to contain the Hadamard product of a tensor by itself)
 * \param work minimal size n[d - 1] + 2r
 * \remarks
 * \warning
*/
double attacNormFromHadamard(const int d, const int *n, const int *rx, double * const *x, double* work){
    int i, j;
	double norm = 0.0;
	int rmax = 0;
	for(i = 0; i < d; ++i){
		rmax = (rmax > rx[i]) ? rmax : rx[i];
	}
	double* b = work;
	double* a = work + rmax;
	for(i = 0; i < n[d - 1]; ++i)
		a[i] = 1.;

  int rxn = rx[d - 1] * n[d - 1];
  int one = 1;
  double done = 1.0;
  double zero = 0.0;

	//cblas_dgemv(CblasColMajor, CblasNoTrans, rx[d - 1], n[d - 1], 1., x[d - 1], rx[d - 1], a, 1, 0, b, 1);
  dgemv_("N", &rx[d - 1], &n[d - 1], &done, x[d - 1], &rx[d - 1], a, &one, &zero, b, &one);
	
	for(i = d - 2; i > -1; --i){
    rxn = rx[i] * n[i];
		double* swipe = a;
		a = b;
		b = swipe;
		memset(b, 0, rx[i] * sizeof(double));
		for(j = 0; j < n[i]; ++j){
			//cblas_dgemv(CblasColMajor, CblasNoTrans, rx[i], rx[i + 1], 1., x[i] + j * rx[i], n[i] * rx[i], a, 1, 1., b, 1);
      dgemv_("N", &rx[i], &rx[i + 1], &done, x[i] + j * rx[i], &rxn, a, &one, &done, b, &one);
		}
	}
	norm = sqrt(b[0]);

	return norm;
}

/** \fn double attacNorm_2(const int d, const int *n, const int *rx, double * const *x, double* workd, int* worki);
 * \brief Computes the Frobenius norm of a TT Tensor x by computing its Hadamard product
 * \details
 * \param d    number of modes
 * \param n    modes size
 * \param rx   input ranks of x
 * \param x    input tensor x
 * \param work minimal size \sum_{i=0}^{d-1} rx[i]^2 n[i] rx[i+1]^2 + 2 r^2 + n[d - 1]
 * \remarks
 * \warning
*/
double attacNorm_2(const int d, const int *n, const int *rx, double * const *x, double* workd, int* worki){
    int i;
	double norm = 0;

	double** xHx = NULL;
	xHx = (double**) malloc(d * sizeof(double*));
	int* rxHx = worki;
	int offset = 0;
	for(i = 0; i < d; ++i){
		rxHx[i] = rx[i] * rx[i];
		rxHx[i + 1] = rx[i + 1] * rx[i + 1];
		xHx[i] = workd + offset;
		offset += rxHx[i] * n[i] * rxHx[i + 1];
	}

	attacHadamard(d, n, rx, x, rx, x, rxHx, xHx);
	
	norm = attacNormFromHadamard(d, n, rxHx, xHx, workd + offset);

	free(xHx);

	return norm;
}

/** \fn double attacNorm(const int d, const int *n, const int *rx, double **x, double* work);
 * \brief Computes the Frobenius norm of a tt Tensor x without computing its Hadamard product
 * \details
 * \param d    number of modes
 * \param n    modes size
 * \param rx   input ranks of x
 * \param x    input tensor x
 * \param work 2 r^2 + max(rx[i] n[i] rx[i+1])
 * \remarks
 * \warning
*/
double attacNorm(const int d, const int *n, const int *rx, double **x, double* work){

	double norm = 0;
	attacLOrthogonalization(d, n, rx, x, work);
  norm = dlange_("F", &rx[d - 1], &n[d - 1], x[d - 1], &rx[d - 1], NULL);

	return norm;
}

/** \fn void attacROrthogonalization(const int d, const int *n, const int *rx, double **x, double *work){
 * \brief Right to left orthogonalization of the TT tensor x
 * \details
 * \param d    number of modes
 * \param n    modes size
 * \param rx   input ranks of x
 * \param x    input tensor x
 * \param work 2 r^2 + max(rx[i] n[i] rx[i+1])
 * \remarks
 * \warning
*/
void attacROrthogonalization(const int d, const int *n, const int *rx, double **x, double *work){
    int i, k;
	int ierr = 0;
  int rmax = 1;
  //block size bug fix
  int blksz = 1;
  for(i = 1; i < d; i++){
    rmax = (rmax > rx[i]) ? rmax : rx[i];
  }
  int offset = 0;
  offset = rmax * rmax;
	double* tau = work;
  double *w = work + offset;
	for(i = d - 1; i > 0; --i){
    offset = rmax * rmax;
		/**
		 * \remarks We want to compute an LQ factorization of x[i]_{(0)} (unfolding in mode 0). To do so
		 *	        we unfold in the 0 mode in order to perform a QR decomposition 
		 *          with row major layout of x[i]. Note that the data in memory are in comlumn major layout
		 */
    int nrx = n[i] * rx[i + 1];
    blksz = (rx[i] < nrx ) ? rx[i] : nrx;
   // printf("\n%d\n%d\n%d\n",rx[i],nrx,blksz);
    dgelqt_(&rx[i], &nrx, &blksz, x[i], &rx[i], tau, &rx[i], w, &ierr);
		if(ierr != 0){
			fprintf(stderr, "%s:Line %d %s::dgelqt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
			fprintf(stderr, "core %d orthogonalization, r[%d]  = %d \n r[%d] = %d \n n[%d] = %d\n", i, i, rx[i], i + 1, rx[i + 1], i, n[i]);
			exit(1);
		}
		
		/**< x[i - 1] = x[i - 1] R */
		//cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, rx[i - 1] * n[i - 1], rx[i], 1., x[i], rx[i], x[i - 1], rx[i - 1] * n[i - 1]);
    int rxmnm = rx[i - 1] * n[i - 1];
    double done = 1.0;
    dtrmm_("R", "L", "N", "N", &rxmnm, &rx[i], &done, x[i], &rx[i], x[i - 1], &rxmnm);

		/**
		 * \remarks Forms the orthogonal factor
		 */
    dlacpy_("A", &rx[i], &nrx, x[i], &rx[i], w, &rx[i]);
    offset += rx[i] * n[i] * rx[i + 1];
    memset(x[i], 0, rx[i] * n[i] * rx[i + 1] * sizeof(double));
    for(k = 0; k < rx[i]; k++){
      x[i][k + k * rx[i]] = 1.;
    }
    dgemlqt_("R", "N", &rx[i], &nrx, &blksz, &blksz, w, &blksz, tau, &blksz, x[i], &rx[i], work + offset, &ierr);
		if(ierr != 0){
			fprintf(stderr, "%s:Line %d %s::dgemlqt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
			exit(1);
		}
	}

}

/** \fn void attacRCompress(const int d, const int *n, int *rx, double **x, const double threshold, double *work);
 * \brief   Compresses the TT tensor x which is supposed to be right to left orthogonal
 * \details
 * \param d         number of modes
 * \param n         modes size
 * \param rx        input ranks of x
 * \param x         input tensor x
 * \param threshold Truncation threshold
 * \param work      minimal size: max(rx[i] n[i] rx[i + 1]) + 3 r^2 + 2 r
 * \remarks
 * \warning
*/
void attacRCompress(const int d, const int *n, int *rx, double **x, const double threshold, double *work){
    int i, j, k;
	int ierr = 0;
	int rmax = 0;
	for(i = 0; i < d; ++i){
		rmax = (rmax > rx[i]) ? rmax : rx[i];
	}
	double *R = work + rmax * rmax;
	double *tau = R + rmax * rmax;
	double *sig = tau + rmax * rmax;
	double *superb = sig + rmax;
	double *xTemp = superb + rmax;

	int rank = 0;
	int nrx = n[0] * rx[1];
	int rxn = rx[0] * n[0];

  dgeqrt_(&n[0], &rx[1], &rx[1], x[0], &n[0], tau, &rx[1], work, &ierr);
	if(ierr != 0){
		fprintf(stderr, "%s:Line: %d %s::dgeqrt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
		exit(1);
	}
  dlacpy_("U", &rx[1], &rx[1], x[0], &n[0], R, &rx[1]);
  setLowerZero('C', rx[1], rx[1], R, rx[1]);
//	ierr = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'O', rx[1], rx[1], R, rx[1], sig, work, rx[1], NULL, 1, superb);
	if(ierr != 0){
		fprintf(stderr, "%s:Line %d %s::LAPACKE_dgesvd:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
		exit(1);
	}

	for(j = 0; j < rx[1]; ++j){
		if(sig[j] > threshold){
			rank = j + 1;
		}else{
			break;
		}
	}

	/**< scale the column j in V by the singular value */
	for(k = 0; k < rank; ++k){
		for(j = 0; j < rx[1]; ++j){
			R[k + j * rx[1]] *= sig[k];
		}
	}
  memset(xTemp, 0, rxn * rank * sizeof(double));
  dlacpy_("A", &rx[1], &rank, work, &rx[1], xTemp, &rxn);
  dgemqrt_("L", "N", &rxn, &rank, &rx[1], &rx[1], x[0], &rxn, tau, &rx[1], xTemp, &rxn, work, &ierr);
	if(ierr != 0){
		fprintf(stderr, "%s:Line: %d %s::dgemlqt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
		exit(1);
	}
  dlacpy_("A", &rxn, &rx[1], xTemp, &rxn, x[0], &rxn);

	//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rank, n[1] * rx[2], rx[1], 1., R, rx[1], x[1], rx[1], 0., xTemp, rank);
  int nprxp = n[1] * rx[2];
  double done = 1.0;
  double zero = 0.0;
  dgemm_("N", "N", &rank, &nprxp, &rx[1], &done, R, &rx[1], x[1], &rx[1], &zero, xTemp, &rank);
  rx[1] = rank;
	for(i = 1; i < d - 1; i++){
    nrx = n[i] * rx[i + 1];
    rxn = rx[i] * n[i];
    dgeqrt_(&rxn, &rx[i + 1], &rx[i + 1], xTemp, &rxn, tau, &rx[i + 1], work, &ierr);
		if(ierr != 0){
			fprintf(stderr, "%s:Line: %d %s::dgelqt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
			exit(1);
		}
    dlacpy_("U", &rx[i + 1], &rx[i + 1], xTemp, &rxn, R, &rx[i + 1]);
    setLowerZero('C', rx[i + 1], rx[i + 1], R, rx[i + 1]);
//		ierr = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'O', rx[i + 1], rx[i + 1], R, rx[i + 1], sig, work, rx[i + 1], NULL, 1, superb);
		if(ierr != 0){
			fprintf(stderr, "%s:Line %d %s::LAPACKE_dgesvd:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
			exit(1);
		}

		for(j = 0; j < rx[i + 1]; ++j){
			if(sig[j] > threshold){
				rank = j + 1;
			}else{
				break;
			}
		}

		/**< scale the column j in u by the singular value */
		for(k = 0; k < rank; ++k){
			for(j = 0; j < rx[i + 1]; ++j){
				R[k + j * rx[i + 1]] *= sig[k];
			}
		}
    memset(x[i], 0, rxn * rank * sizeof(double));
    dlacpy_("A", &rx[i + 1], &rank, work, &rx[i + 1], x[i], &rxn);
    dgemqrt_("L", "N", &rxn, &rank, &rx[i + 1], &rx[i + 1], xTemp, &rxn, tau, &rx[i + 1], x[i], &rxn, work, &ierr);
	  if(ierr != 0){
	  	fprintf(stderr, "%s:Line: %d %s::dgemlqt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
	  	exit(1);
	  }

		//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rank, n[i + 1] * rx[i + 2], rx[i + 1], 1., R, rx[i + 1], x[i + 1], rx[i + 1], 0., xTemp, rank);
    nprxp = n[i + 1] * rx[i + 2];
    dgemm_("N", "N", &rank, &nprxp, &rx[i + 1], &done, R, &rx[i + 1], x[i + 1], &rx[i + 1], &zero, xTemp, &rank);
    rx[i + 1] = rank;
	}
  dlacpy_("A", &rx[d - 1], &n[d - 1], xTemp, &rx[d - 1], x[d - 1], &rx[d - 1]);

}


/** \fn void attacRTruncate(const int d, const int *n, int *rx, double **x, const double threshold, double *work);
 * \brief   Compresses the TT tensor x which is supposed to be right to left orthogonal
 * \details
 * \param d         number of modes
 * \param n         modes size
 * \param rx        input ranks of x
 * \param x         input tensor x
 * \param abs_tol   ensure the 2 norm error is smaller than abs_tol
 * \param work      minimal size: max(rx[i] n[i] rx[i + 1]) + 3 r^2 + 2 r
 * \remarks
 * \warning
*/
void attacRTruncate(const int d, const int *n, int *rx, double **x, const int *maxr, const double abs_tol, double *work){
    int i, j, k;
	int ierr = 0;
	int rmax = 0;
	for(i = 0; i < d; ++i){
		rmax = (rmax > rx[i]) ? rmax : rx[i];
	}
	double *R = work + rmax * rmax;
	double *tau = R + rmax * rmax;
	double *sig = tau + rmax * rmax;
	double *superb = sig + rmax;
	double *xTemp = superb + rmax;

    double sigSum;
	int rank = 0;
	int nrx = n[0] * rx[1];
	int rxn = rx[0] * n[0];
    int blksz = (n[0] < rx[1]) ? n[0] : rx[1];
  dgeqrt_(&n[0], &rx[1], &blksz, x[0], &n[0], tau, &rx[1], work, &ierr);
	if(ierr != 0){
		fprintf(stderr, "%s:Line: %d %s::dgeqrt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
		exit(1);
	}
  dlacpy_("U", &rx[1], &rx[1], x[0], &n[0], R, &rx[1]);
  setLowerZero('C', rx[1], rx[1], R, rx[1]);
//	ierr = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'O', rx[1], rx[1], R, rx[1], sig, work, rx[1], NULL, 1, superb);
	if(ierr != 0){
		fprintf(stderr, "%s:Line %d %s::LAPACKE_dgesvd:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
		exit(1);
	}
	for(j=1; j <= d; j++){
	    if(maxr[j] > rx[j] || maxr[j] < 0){
	        fprintf(stderr, "%s:Line %d %s::maxRank Check:: mode %d, %d > %d\n", __FILE__, __LINE__, __func__, j, maxr[j],rx[j]);
		    exit(1);
	    }
	}
	
	sigSum = 0.0;
	rank = rx[1]-1;
    for(j = rx[1]-1; j >= 0; j--){
        sigSum += sig[j]*sig[j];
	    if( sqrt(sigSum) < abs_tol/sqrt(d-1.0)){
		    rank = j;
	    }else{
		    break;
	    }
    }
    rank = (maxr[1] < rank) ? maxr[1] : rank;
    rank = (rank > 1) ? rank : 1;
	/**< scale the column j in V by the singular value */
	for(k = 0; k < rank; ++k){
		for(j = 0; j < rx[1]; ++j){
			R[k + j * rx[1]] *= sig[k];
		}
	}
  memset(xTemp, 0, rxn * rank * sizeof(double));
  dlacpy_("A", &rx[1], &rank, work, &rx[1], xTemp, &rxn);
  dgemqrt_("L", "N", &rxn, &rank, &blksz, &blksz, x[0], &rxn, tau, &rx[1], xTemp, &rxn, work, &ierr);
	if(ierr != 0){
		fprintf(stderr, "%s:Line: %d %s::dgemqrt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
		exit(1);
	}
  dlacpy_("A", &rxn, &rx[1], xTemp, &rxn, x[0], &rxn);

	//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rank, n[1] * rx[2], rx[1], 1., R, rx[1], x[1], rx[1], 0., xTemp, rank);
  int nprxp = n[1] * rx[2];
  double done = 1.0;
  double zero = 0.0;
  dgemm_("N", "N", &rank, &nprxp, &rx[1], &done, R, &rx[1], x[1], &rx[1], &zero, xTemp, &rank);
  rx[1] = rank;
	for(i = 1; i < d - 1; i++){
    nrx = n[i] * rx[i + 1];
    rxn = rx[i] * n[i];
    blksz = (rxn < rx[i+1]) ? rxn : rx[i+1];
    dgeqrt_(&rxn, &rx[i + 1], &blksz, xTemp, &rxn, tau, &rx[i + 1], work, &ierr);
		if(ierr != 0){
			fprintf(stderr, "%s:Line: %d %s::dgeqrt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
			exit(1);
		}
    dlacpy_("U", &rx[i + 1], &rx[i + 1], xTemp, &rxn, R, &rx[i + 1]);
    setLowerZero('C', rx[i + 1], rx[i + 1], R, rx[i + 1]);
//		ierr = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'O', rx[i + 1], rx[i + 1], R, rx[i + 1], sig, work, rx[i + 1], NULL, 1, superb);
		if(ierr != 0){
			fprintf(stderr, "%s:Line %d %s::LAPACKE_dgesvd:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
			exit(1);
		}
		
	    sigSum = 0.0;
	    rank = rx[i+1]-1;
	    for(j = rx[i+1]-1; j >= 0; j--){
	        sigSum += sig[j]*sig[j];
		    if( sqrt(sigSum) < abs_tol/sqrt(d-1.0)){
			    rank = j;
		    }else{
			    break;
		    }
	    }
        rank = (maxr[i+1] < rank) ? maxr[i+1] : rank;
    rank = (rank > 1) ? rank : 1;
		/**< scale the column j in u by the singular value */
		for(k = 0; k < rank; ++k){
			for(j = 0; j < rx[i + 1]; ++j){
				R[k + j * rx[i + 1]] *= sig[k];
			}
		}
    memset(x[i], 0, rxn * rank * sizeof(double));
    dlacpy_("A", &rx[i + 1], &rank, work, &rx[i + 1], x[i], &rxn);
    dgemqrt_("L", "N", &rxn, &rank, &blksz, &blksz, xTemp, &rxn, tau, &rx[i + 1], x[i], &rxn, work, &ierr);
	  if(ierr != 0){
	  	fprintf(stderr, "%s:Line: %d %s::dgemlqt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
	  	exit(1);
	  }

		//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rank, n[i + 1] * rx[i + 2], rx[i + 1], 1., R, rx[i + 1], x[i + 1], rx[i + 1], 0., xTemp, rank);
    nprxp = n[i + 1] * rx[i + 2];
    dgemm_("N", "N", &rank, &nprxp, &rx[i + 1], &done, R, &rx[i + 1], x[i + 1], &rx[i + 1], &zero, xTemp, &rank);
    rx[i + 1] = rank;
	}
  dlacpy_("A", &rx[d - 1], &n[d - 1], xTemp, &rx[d - 1], x[d - 1], &rx[d - 1]);

}

/** \fn void attacLOrthogonalization(const int d, const int *n, const int *rx, double **x, double *work);
 * \brief Left to right orthogonalization of the TT tensor x
 * \details
 * \param d    number of modes
 * \param n    modes size
 * \param rx   input ranks of x
 * \param x    input tensor x
 * \param work 2 r^2 + max(rx[i] n[i] rx[i+1])
 * \remarks
 * \warning
*/
void attacLOrthogonalization(const int d, const int *n, const int *rx, double **x, double *work){
    int i, k;
	int ierr = 0;
  int rmax = 1;
  for(i = 1; i < d; i++){
    rmax = (rmax > rx[i]) ? rmax : rx[i];
  }
	double* tau = work;
	double* w = work + rmax * rmax;

	for(i = 0; i < d - 1; ++i){
    int offset = rmax * rmax;
    int rxn = rx[i] * n[i];
		/**
		 * \remarks We want to compute a QR factorization of x[i]_{(2)}^\top (horizontal unfolding).
		 */
    dgeqrt_(&rxn, &rx[i + 1], &rx[i + 1], x[i], &rxn, tau, &rx[i + 1], w, &ierr);
		if(ierr != 0){
			fprintf(stderr, "%s:Line %d %s::dgeqrt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
			fprintf(stderr, "core %d orthogonalization, r[%d]  = %d \n r[%d] = %d \n n[%d] = %d\n", i, i, rx[i], i + 1, rx[i + 1], i, n[i]);
			exit(1);
		}
		
		/**< x[i + 1] = R x[i + 1]*/
		//cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, rx[i + 1], rx[i + 2] * n[i + 1], 1., x[i], rx[i] * n[i], x[i + 1], rx[i + 1]);
    int nprxp = n[i + 1] * rx[i + 2];
    double done = 1.0;
    dtrmm_("L", "U", "N", "N", &rx[i + 1], &nprxp, &done, x[i], &rxn, x[i + 1], &rx[i + 1]);

		/**
		 * \remarks Forms the orthogonal factor
		 */
    dlacpy_("A", &rxn, &rx[i + 1], x[i], &rxn, w, &rxn);
    offset += rx[i] * n[i] * rx[i + 1];
    memset(x[i], 0, rx[i] * n[i] * rx[i + 1] * sizeof(double));
    for(k = 0; k < rx[i + 1]; k++){
      x[i][k + k * rxn] = 1.;
    }
    dgemqrt_("L", "N", &rxn, &rx[i + 1], &rx[i + 1], &rx[i + 1], w, &rxn, tau, &rx[i + 1], x[i], &rxn, work + offset, &ierr);
		if(ierr != 0){
			fprintf(stderr, "%s:Line %d %s::dgemqrt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
			exit(1);
		}
	}

}

/** \fn void attacLCompress(const int d, const int *n, int *rx, double **x, const double threshold, double *work);
 * \brief   Compresses the TT tensor x which is supposed to be left to right orthogonal
 * \details
 * \param d         number of modes
 * \param n         modes size
 * \param rx        input ranks of x
 * \param x         input tensor x
 * \param threshold Truncation threshold
 * \param work      minimal size: max(rx[i] n[i] rx[i + 1]) + 3 r^2 + 2 r
 * \remarks
 * \warning
*/
void attacLCompress(const int d, const int *n, int *rx, double **x, const double threshold, double *work){
    int i, j, k;
	int ierr = 0;
	int rmax = 0;
	for(i = 0; i < d; ++i){
		rmax = (rmax > rx[i]) ? rmax : rx[i];
	}
	double *R = work + rmax * rmax;
	double *tau = R + rmax * rmax;
	double *sig = tau + rmax * rmax;
	double *superb = sig + rmax;
	double *xTemp = superb + rmax;

	int rank = 0;
	int nrx = n[d - 1] * rx[d];

  dgelqt_(&rx[d - 1], &nrx, &rx[d - 1], x[d - 1], &rx[d - 1], tau, &rx[d - 1], work, &ierr);
	if(ierr != 0){
		fprintf(stderr, "%s:Line: %d %s::dgelqt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
		exit(1);
	}
  dlacpy_("L", &rx[d - 1], &rx[d - 1], x[d - 1], &rx[d - 1], R, &rx[d - 1]);
  setLowerZero('R', rx[d - 1], rx[d - 1], R, rx[d - 1]);
//	ierr = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'S', rx[d - 1], rx[d - 1], R, rx[d - 1], sig, NULL, 1, work, rx[d - 1], superb);
	if(ierr != 0){
		fprintf(stderr, "%s:Line %d %s::LAPACKE_dgesvd:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
		exit(1);
	}

	for(j = 0; j < rx[d - 1]; ++j){
		if(sig[j] > threshold){
			rank = j + 1;
		}else{
			break;
		}
	}

	/**< scale the column j in u by the singular value */
	for(j = 0; j < rank; ++j){
		for(k = 0; k < rx[d - 1]; ++k){
			R[k + j * rx[d - 1]] *= sig[j];
		}
	}
  memset(xTemp, 0, rank * nrx * sizeof(double));
  dlacpy_("A", &rank, &rx[d - 1], work, &rx[d - 1], xTemp, &rank);
  dgemlqt_("R", "N", &rank, &nrx, &rx[d - 1], &rx[d - 1], x[d - 1], &rx[d - 1], tau, &rx[d - 1], xTemp, &rank, work, &ierr);
	if(ierr != 0){
		fprintf(stderr, "%s:Line: %d %s::dgemlqt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
		exit(1);
	}
  dlacpy_("A", &rank, &nrx, xTemp, &rank, x[d - 1], &rank);

	//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rx[d - 2] * n[d - 2], rank, rx[d - 1], 1., x[d - 2], rx[d - 2] * n[d - 2], R, rx[d - 1], 0., xTemp, rx[d - 2] * n[d - 2]);
  int rxmnm = rx[d - 2] * n[d - 2];
  double done = 1.0;
  double zero = 0.0;
  dgemm_("N", "N", &rxmnm, &rank, &rx[d - 1], &done, x[d - 2], &rxmnm, R, &rx[d - 1], &zero, xTemp, &rxmnm);
  rx[d - 1] = rank;
	for(i = d - 2; i > 0; --i){
    nrx = n[i] * rx[i + 1];
    dgelqt_(&rx[i], &nrx, &rx[i], xTemp, &rx[i], tau, &rx[i], work, &ierr);
		if(ierr != 0){
			fprintf(stderr, "%s:Line: %d %s::dgelqt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
			exit(1);
		}
    dlacpy_("L", &rx[i], &rx[i], xTemp, &rx[i], R, &rx[i]);
    setLowerZero('R', rx[i], rx[i], R, rx[i]);
//		ierr = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'S', rx[i], rx[i], R, rx[i], sig, NULL, 1, work, rx[i], superb);
		if(ierr != 0){
			fprintf(stderr, "%s:Line %d %s::LAPACKE_dgesvd:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
			exit(1);
		}

		for(j = 0; j < rx[i]; ++j){
			if(sig[j] > threshold){
				rank = j + 1;
			}else{
				break;
			}
		}

		/**< scale the column j in u by the singular value */
		for(j = 0; j < rank; ++j){
			for(k = 0; k < rx[i]; ++k){
				R[k + j * rx[i]] *= sig[j];
			}
		}
    memset(x[i], 0, rank * nrx * sizeof(double));
    dlacpy_("A", &rank, &rx[i], work, &rx[i], x[i], &rank);
    dgemlqt_("R", "N", &rank, &nrx, &rx[i], &rx[i], xTemp, &rx[i], tau, &rx[i], x[i], &rank, work, &ierr);
	  if(ierr != 0){
	  	fprintf(stderr, "%s:Line: %d %s::dgemlqt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
	  	exit(1);
	  }

		//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n[i - 1] * rx[i - 1], rank, rx[i], 1., x[i - 1], n[i - 1] * rx[i - 1], R, rx[i], 0., xTemp, n[i - 1] * rx[i - 1]);
    rxmnm = rx[i - 1] * n[i - 1];
    dgemm_("N", "N", &rxmnm, &rank, &rx[i], &done, x[i - 1], &rxmnm, R, &rx[i], &zero, xTemp, &rxmnm);
    rx[i] = rank;
	}
  dlacpy_("A", &n[0], &rx[1], xTemp, &n[0], x[0], &n[0]);

}



/** \fn void attacLTruncate(const int d, const int *n, int *rx, double **x, const double threshold, double *work);
 * \brief   Compresses the TT tensor x which is supposed to be left to right orthogonal
 * \details
 * \param d         number of modes
 * \param n         modes size
 * \param rx        input ranks of x
 * \param x         input tensor x
 * \param abs_tol   ensure the 2 norm error is smaller than abs_tol
 * \param work      minimal size: max(rx[i] n[i] rx[i + 1]) + 3 r^2 + 2 r
 * \remarks
 * \warning
*/
void attacLTruncate(const int d, const int *n, int *rx, double **x, const int *maxr, const double abs_tol, double *work){
    int i, j, k;
	int ierr = 0;
	int rmax = 0;
	for(i = 0; i < d; ++i){
		rmax = (rmax > rx[i]) ? rmax : rx[i];
	}
	double *R = work + rmax * rmax;
	double *tau = R + rmax * rmax;
	double *sig = tau + rmax * rmax;
	double *superb = sig + rmax;
	double *xTemp = superb + rmax;

    double sigSum;
	int rank = 0;
	int nrx = n[d - 1] * rx[d];

  dgelqt_(&rx[d - 1], &nrx, &rx[d - 1], x[d - 1], &rx[d - 1], tau, &rx[d - 1], work, &ierr);
	if(ierr != 0){
		fprintf(stderr, "%s:Line: %d %s::dgelqt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
		exit(1);
	}
  dlacpy_("L", &rx[d - 1], &rx[d - 1], x[d - 1], &rx[d - 1], R, &rx[d - 1]);
  setLowerZero('R', rx[d - 1], rx[d - 1], R, rx[d - 1]);
//	ierr = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'S', rx[d - 1], rx[d - 1], R, rx[d - 1], sig, NULL, 1, work, rx[d - 1], superb);
	if(ierr != 0){
		fprintf(stderr, "%s:Line %d %s::LAPACKE_dgesvd:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
		exit(1);
	}
	for(j=1; j <= d; j++){
	    if(maxr[j] > rx[j] || maxr[j] < 0){
	        fprintf(stderr, "%s:Line %d %s::maxRank Check:: mode %d, %d > %d\n", __FILE__, __LINE__, __func__, j, maxr[j],rx[j]);
		    exit(1);
	    }
	}
	
    sigSum = 0.0;
    rank = rx[d-1]-1;
	for(j = rx[d-1]-1; j >= 0; j--){
	    sigSum += sig[j]*sig[j];
		if( sqrt(sigSum) < abs_tol/sqrt(d-1.0)){
			rank = j;
		}else{
			break;
		}
	}
    rank = (maxr[d-1] < rank) ? maxr[d-1] : rank;
    rank = (rank > 1) ? rank : 1;
	/**< scale the column j in u by the singular value */
	for(j = 0; j < rank; ++j){
		for(k = 0; k < rx[d - 1]; ++k){
			R[k + j * rx[d - 1]] *= sig[j];
		}
	}
  memset(xTemp, 0, rank * nrx * sizeof(double));
  dlacpy_("A", &rank, &rx[d - 1], work, &rx[d - 1], xTemp, &rank);
  dgemlqt_("R", "N", &rank, &nrx, &rx[d - 1], &rx[d - 1], x[d - 1], &rx[d - 1], tau, &rx[d - 1], xTemp, &rank, work, &ierr);
	if(ierr != 0){
		fprintf(stderr, "%s:Line: %d %s::dgemlqt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
		exit(1);
	}
  dlacpy_("A", &rank, &nrx, xTemp, &rank, x[d - 1], &rank);

	//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rx[d - 2] * n[d - 2], rank, rx[d - 1], 1., x[d - 2], rx[d - 2] * n[d - 2], R, rx[d - 1], 0., xTemp, rx[d - 2] * n[d - 2]);
  int rxmnm = rx[d - 2] * n[d - 2];
  double done = 1.0;
  double zero = 0.0;
  dgemm_("N", "N", &rxmnm, &rank, &rx[d - 1], &done, x[d - 2], &rxmnm, R, &rx[d - 1], &zero, xTemp, &rxmnm);
  rx[d - 1] = rank;
	for(i = d - 2; i > 0; --i){
    nrx = n[i] * rx[i + 1];
    
    dgelqt_(&rx[i], &nrx, &rx[i], xTemp, &rx[i], tau, &rx[i], work, &ierr);
		if(ierr != 0){
			fprintf(stderr, "%s:Line: %d %s::dgelqt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
			exit(1);
		}
    dlacpy_("L", &rx[i], &rx[i], xTemp, &rx[i], R, &rx[i]);
    setLowerZero('R', rx[i], rx[i], R, rx[i]);
//		ierr = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'S', rx[i], rx[i], R, rx[i], sig, NULL, 1, work, rx[i], superb);
		if(ierr != 0){
			fprintf(stderr, "%s:Line %d %s::LAPACKE_dgesvd:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
			exit(1);
		}

        sigSum = 0.0;
        rank = rx[i]-1;
	    for(j = rx[i]-1; j >= 0; j--){
	        sigSum += sig[j]*sig[j];
		    if( sqrt(sigSum) < abs_tol/sqrt(d-1.0)){
			    rank = j;
		    }else{
			    break;
		    }
	    }
        rank = (maxr[i] < rank) ? maxr[i] : rank;
        rank = (rank > 1) ? rank : 1;
		/**< scale the column j in u by the singular value */
		for(j = 0; j < rank; ++j){
			for(k = 0; k < rx[i]; ++k){
				R[k + j * rx[i]] *= sig[j];
			}
		}
    memset(x[i], 0, rank * nrx * sizeof(double));
    dlacpy_("A", &rank, &rx[i], work, &rx[i], x[i], &rank);
    dgemlqt_("R", "N", &rank, &nrx, &rx[i], &rx[i], xTemp, &rx[i], tau, &rx[i], x[i], &rank, work, &ierr);
	  if(ierr != 0){
	  	fprintf(stderr, "%s:Line: %d %s::dgemlqt:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
	  	exit(1);
	  }

		//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n[i - 1] * rx[i - 1], rank, rx[i], 1., x[i - 1], n[i - 1] * rx[i - 1], R, rx[i], 0., xTemp, n[i - 1] * rx[i - 1]);
    rxmnm = rx[i - 1] * n[i - 1];
    dgemm_("N", "N", &rxmnm, &rank, &rx[i], &done, x[i - 1], &rxmnm, R, &rx[i], &zero, xTemp, &rxmnm);
    rx[i] = rank;
	}
  dlacpy_("A", &n[0], &rx[1], xTemp, &n[0], x[0], &n[0]);

}

/** \fn void attacCompress(const int d, const int *n, int *rx, double **x, const double threshold, const char ortho, double *work);
 * \brief   Compresses the TT tensor x
 * \details
 * \param d         number of modes
 * \param n         modes size
 * \param rx        input ranks of x
 * \param x         input tensor x
 * \param threshold Truncation threshold
 * \param ortho     Orthogonalization state of the tensor at the end of the procedure
 * \param work      minimal size: max(rx[i] n[i] rx[i + 1]) + 3 r^2 + 2 r
 * \remarks
 * \warning
*/
void attacCompress(const int d, const int *n, int *rx, double **x, const double threshold, const char ortho, double *work){

	double norm = 0;
	if(ortho == 'R'){
		attacLOrthogonalization(d, n, rx, x, work);
    norm = dlange_("F", &rx[d - 1], &n[d - 1], x[d - 1], &rx[d - 1], NULL);
		attacLCompress(d, n, rx, x, threshold/norm, work);
	}else if(ortho == 'L'){
		attacROrthogonalization(d, n, rx, x, work);
    norm = dlange_("F", &n[0], &rx[1], x[0], &n[0], NULL);
		attacRCompress(d, n, rx, x, threshold/norm, work);
	}else{
		fprintf(stderr, "%s:Line %d %s::Not known orthogonalization side\n", __FILE__, __LINE__, __func__);
	}

}

/** \fn void attacCompress(const int d, const int *n, int *rx, double **x, const double threshold, const char ortho, double *work);
 * \brief   Compresses the TT tensor x
 * \details
 * \param d         number of modes
 * \param n         modes size
 * \param rx        input ranks of x
 * \param x         input tensor x
 * \param abs_tol   Guarantee accuracy up to specified absolute tolerance.
 * \param ortho     Orthogonalization state of the tensor at the end of the procedure
 * \param work      minimal size: max(rx[i] n[i] rx[i + 1]) + 3 r^2 + 2 r
 * \remarks
 * \warning
*/
void attacTruncate(const int d, const int *n, int *rx, double **x, const int *maxr, const double abs_tol, const char ortho, double *work){

	double norm = 0;
	if(ortho == 'R'){
		attacLOrthogonalization(d, n, rx, x, work);
    norm = dlange_("F", &rx[d - 1], &n[d - 1], x[d - 1], &rx[d - 1], NULL);
		attacLTruncate(d, n, rx, x, maxr, abs_tol, work);//threshold/norm, work);
	}else if(ortho == 'L'){
		attacROrthogonalization(d, n, rx, x, work);
    norm = dlange_("F", &n[0], &rx[1], x[0], &n[0], NULL);
		attacRTruncate(d, n, rx, x, maxr, abs_tol, work);//threshold/norm, work);
	}else{
		fprintf(stderr, "%s:Line %d %s::Not known orthogonalization side\n", __FILE__, __LINE__, __func__);
	}

}


/** \fn double attacValue(const int* index, const int d, const int* n, const int* rx, double *const *x, double* work);
 * \brief
 * \details
 * \param index     input indices of the element
 * \param d         number of modes
 * \param n         modes size
 * \param rx        input ranks of x
 * \param x         input tensor x
 * \param work      2 r
 * \remarks
 * \warning
*/
double attacValue(const int* index, const int d, const int* n, const int* rx, double *const *x, double* work){
    /**< Maximum TT rank */
    int rmax = 0;
    int i;
    for(i = 0; i < d; ++i){
	    rmax = (rmax > rx[i]) ? rmax : rx[i];
    }
    int rxn = rx[d - 1] * n[d - 1];
    int one = 1;
    double done = 1.0;
    double zero = 0.0;

    /**< Temporary variables for computation */
    double* a = work, *b = work + rmax, *temp = NULL;

    memcpy(a, x[d - 1] + index[d - 1] * rx[d - 1], rx[d - 1] * sizeof(double));

    for(i = d - 2; i > -1; --i){
    rxn = rx[i] * n[i];
	    //cblas_dgemv(CblasColMajor, CblasNoTrans, rx[i], rx[i + 1], 1., x[i] + index[i] * rx[i], n[i] * rx[i], a, 1, 0.0, b, 1);
    dgemv_("N", &rx[i], &rx[i + 1], &done, x[i] + index[i] * rx[i], &rxn, a, &one, &zero, b, &one);
	    temp = a;
	    a = b;
	    b = temp;
    }
    return a[0];
}

/** \fn void attacScale(const double alpha, int mode, const int d, const int* n, const int* rx, double **x);
 * \brief
 * \details
 * \param alpha     input scalar
 * \param mode      which mode to be scaled
 * \param d         number of modes
 * \param n         modes size
 * \param rx        input ranks of x
 * \param x         input tensor x
 * \remarks
 * \warning
*/
void attacScale(const double alpha, int mode, const int d, const int* n, const int* rx, double **x){
    int i, j;
	if(mode < 0 || mode > d-1){
		/**< Scale first mode */
        for(i = 0; i < n[0] * rx[1]; i++){
            x[0][i] *= alpha;
        }
	}else{
		i = mode;
        for(j = 0; j < rx[i] * n[i] * rx[i + 1]; j++){
            x[i][j] *= alpha;
        }
	}

}


/** \fn double attacNormSquaredHadamardSymOpt(const int d, const int *n, const int *rx, double *const *x, double *work, int *worki)
 * \brief Norm squared of a TT tensor
 * \details Compute norm squared of a TT tensor exploiting the symmetry
 * \param d         number of modes
 * \param n         modes size
 * \param rx        input ranks of x
 * \param x         input tensor x
 * \param work      minimal space: 2r^2 + max(rx[i] n[i] rx[i+1])
 * \remarks
 * \warning
*/
double attacNormSquaredHadamardSymOpt(const int d, const int *n, const int *rx, double *const *x, double *work){
	int ierr = 0;
	int rmax = 0;
	int i, j, k;
	for(i = 0; i < d; i++){
		rmax = (rmax > rx[i]) ? rmax : rx[i];
	}
  double *a = work;
  double *b = work + rmax * rmax;

  int rxn = rx[0] * n[0];
  int nrx = n[0] * rx[1];
  double done = 1.0;
  double zero = 0.0;
  /**< initiate the first core product */
  //cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, rx[1], rx[0] * n[0], 1., x[0], rx[0] * n[0], 0, a, rx[1]);
  dsyrk_("L", "T", &rx[1], &rxn, &done, x[0], &rxn, &zero, a, &rx[1]);

	for(i = 1; i < d; i++){
    rxn = rx[i] * n[i];
    nrx = n[i] * rx[i + 1];
    for(j = 0; j < rx[i]; j++){
      for(k = 0; k < j; k++){
        a[k + j * rx[i]] = a[j + k * rx[i]];
      }
    }
//    ierr = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'O', rx[i], rx[i], a, rx[i], b, NULL, 1, NULL, 1, b + rx[i]);
		if(ierr != 0){
			fprintf(stderr, "%s:Line %d %s::LAPACKE_dgesvd:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
			exit(1);
		}
    for(j = 0; j < rx[i]; j++){
      for(k = 0; k < rx[i]; k++){
        a[k + j * rx[i]] *= sqrt(b[k]);
      }
    }
    int lb = rx[i] * rx[i];
    dgeqlf_(&rx[i], &rx[i], a, &rx[i], b, b + rx[i], &lb, &ierr);
		if(ierr != 0){
			fprintf(stderr, "%s:Line %d %s::dgeqlf:: ierr = %d\n", __FILE__, __LINE__, __func__, ierr);
			exit(1);
		}
    // GEMM Usage
		setLowerZero('R', rx[i], rx[i], a, rx[i]); // upper part of a is lower part of a^T (row major viewed)
    //cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans, rx[i], rx[i + 1] * n[i], rx[i], 1., a, rx[i], x[i], rx[i], 0, b, rx[i]);
    dgemm_("N", "N", &rx[i], &nrx, &rx[i], &done, a, &rx[i], x[i], &rx[i], &zero, b, &rx[i]);
    /**< V(B)^T V(B) */
    //cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, rx[i + 1], rx[i] * n[i], 1., b, rx[i] * n[i], 0, a, rx[i + 1]);
    dsyrk_("L", "T", &rx[i + 1], &rxn, &done, b, &rxn, &zero, a, &rx[i + 1]);
    /********** End SVDChol  **********/
	}
	return a[0];
}

/** \fn double attacInnerProduct(const int d, const int *n, const int *rx, double *const *x, const int *ry, double *const *y, double *work)
 * \brief Inner product of two TT tensors
 * \details
 * \param d         number of modes
 * \param n         modes size
 * \param rx        input ranks of x
 * \param x         input tensor x
 * \param ry        input ranks of y
 * \param y         input tensor y
 * \param work      minimal space: r^2 + max(rx[i] n[i] ry[i + 1])
 * \remarks
 * \warning
*/
double attacInnerProduct(const int d, const int *n, const int *rx, double *const *x, const int *ry, double *const *y, double *work){
	int rmax = 0;
	int i;
	for(i = 0; i < d; i++){
		rmax = (rmax > rx[i]) ? rmax : rx[i];
		rmax = (rmax > ry[i]) ? rmax : ry[i];
	}
  double *a = work;
  double *b = work + rmax * rmax;

  int rxn = rx[0] * n[0];
  int nrx = n[0] * rx[1];
  int ryn = ry[0] * n[0];
  int nry = n[0] * ry[1];

  double done = 1.0;
  double zero = 0.0;
  /**< initiate the first core product */
  //cblas_dgemm (CblasColMajor, CblasTrans, CblasNoTrans, rx[1], ry[1], n[1], 1., x[0], n[0], y[0], n[0], 0, a, rx[1]);
  dgemm_("T", "N", &rx[1], &ry[1], &n[0], &done, x[0], &rxn, y[0], &ryn, &zero, a, &rx[1]);

	for(i = 1; i < d; i++){
    rxn = rx[i] * n[i];
    nrx = n[i] * rx[i + 1];
    ryn = ry[i] * n[i];
    nry = n[i] * ry[i + 1];
    /**< H(B) = M H(X_{i}) */
    //cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans, rx[i], ry[i + 1] * n[i], ry[i], 1., a, rx[i], y[i], ry[i], 0, b, ry[i]);
    dgemm_("N", "N", &rx[i], &nry, &ry[i], &done, a, &rx[i], y[i], &ry[i], &zero, b, &rx[i]);
    /**< V(X_{i})^T V(B)  */
    //cblas_dgemm (CblasColMajor, CblasTrans, CblasNoTrans, rx[i + 1], ry[i + 1], rx[i] * n[i], 1., x[i], rx[i] * n[i], b, rx[i] * n[i], 0, a, rx[i + 1]);
    dgemm_("T", "N", &rx[i + 1], &ry[i + 1], &rxn, &done, x[i], &rxn, b, &rxn, &zero, a, &rx[i + 1]);
	}
	return a[0];
}
