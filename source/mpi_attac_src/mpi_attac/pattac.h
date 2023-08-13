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
/** \file pattac.h
 * \author Hussam Al Daas
 * \brief MPI Algorithms for Tensor Train Arithmetic and Computations
 * \date 16/07/2019
 * \details 
*/
#ifndef PATTAC_H
#define PATTAC_H

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "tsqr.h"

#define lapack_int int

extern void dgeqlf_(const int*, const int*, double*, const int*, double*, double*, const int*, int*);
extern double dlange_(const char*, const int*, const int*, const double*, const int*, double*);
extern void dlacpy_(const char*, const int*, const int*, const double*, const int*, double*, const int*);
extern void dtrmm_(const char*, const char*, const char*, const char*, const int*, const int*, const double*, double*, const int*, double*, const int*);
extern void dgemv_(const char*, const int*, const int*, double*, double*, const int*, double*, const int*, double*, double*, const int*);
extern void dgemm_(const char*, const char*, const int*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*);
extern void dsyrk_(const char*, const char*, const int*, const int*, const double*, const double*, const int*, const double*, double*, const int*);

lapack_int LAPACKE_dgesdd_work( int matrix_layout,
                                char jobz,
                                lapack_int m,
                                lapack_int n,
                                double* a,
                                lapack_int lda,
                                double* s,
                                double* u,
                                lapack_int ldu,
                                double* vt,
                                lapack_int ldvt,
                                double* work,
                                lapack_int lwork,
                                lapack_int* iwork );

extern void dgesdd_(char const* jobz,
                    const int* m, const int* n,
                    double* A, const int* lda,
                    double* S,
                    double* U,  const int* ldu,
                    double* VT, const int* ldvt,
                    double* work, const int* lwork,
                    int* iwork,
                    int* info );


extern void dgesvdq_(const char* JOBA,
                    const char* JOBP,
                    const char* JOBR,
                    const char* JOBU,
                    const char* JOBV,
                    const int*  M,
                    const int*  N,
                    double*     A,
                    const int*  LDA,
                    double*     S,
                    double*     U,
                    const int*  LDU,
                    double*     V,
                    const int*  LDV,
                    int*        NUMRANK,
                    int*        IWORK,
                    const int*  LIWORK,
                    double*     WORK,
                    int*        LWORK,
                    double*     RWORK,
                    const int*  LRWORK,
                    int*        INFO);
#define LAPACK_COL_MAJOR 102

void pattacScale(const double alpha, int mode, const int d, const int* n, const int* rx, double **x);
double pattacValue(const MPI_Comm comm, const int* index, const int* procHolder, const int d, const int* n, const int* rx, double *const *x, double* work);

void pattacFormalSum(const int d, const int *n, const int *rx, double *const *x, const int *ry, double *const *y, int* rz, double **z);
void pattacFormalAXPBY(const int d, const int *n, const double alpha, const int *rx, double * const *x, const double beta, const int *ry, double *const *y, int* rz, double **z);
void pattacHadamard(const int d, const int *n, const int *rx, double *const *x, const int *ry, double *const *y, int* rz, double **z);

double pattacNormFromHadamard(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, const int *rx, double *const *x, double* work);

double pattacNorm(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, const int *rx, double **x, double* work);

void pattacLOrthogonalization(MPI_Comm comm,
                              int d,
                              int* distrib_dim_mat,
                              int *rx,
                              double **x,
                              tsqr_tree* qrTree,
                              int* int_wrk);

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
                     int*     flt_sz_ptr);

void pattacROrthogonalization(MPI_Comm comm,
                              int d,
                              int* distrib_dim_mat,
                              int *rx,
                              double **x,
                              tsqr_tree* qrTree,
                              int* int_wrk);
void pattacRCompress(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, int *rx, double **x, const double tau, double *work);
void pattacRTruncate(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, const int *n_global, int *rx, double **x, const int *maxr, const double abs_tol, double *work);

void pattacCompress(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, int *rx, double **x, const double tau, const char ortho, double *work);
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
                    int*     flt_sz_ptr); //size of flt wrk vector


double pattacInnerProduct(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, const int *rx, double *const *x, const int *ry, double *const *y, double *work);
double pattacNormSquaredHadamardOpt(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, const int *rx, double *const *x, double *work);
double pattacNormSquaredHadamardSymOpt(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, const int *rx, double *const *x, double *work, int *worki);

void pattacGramRL(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, const int *rx, double *const *x, double *work, double *GR);
void pattacGramLR(const MPI_Comm comm, const int* isRedundant, const int d, const int *n, const int *rx, double *const *x, double *work, double *GL);
#endif

