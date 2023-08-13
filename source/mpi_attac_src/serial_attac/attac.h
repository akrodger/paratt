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
/** \file attac.h
 * \author Hussam Al Daas
 * \brief Algorithms for Tensor Train Arithmetic and Computations
 * \date 16/07/2019
 * \details 
*/
#ifndef ATTAC_H
#define ATTAC_H

#include <stdlib.h>
#include <stdio.h>

extern void dgeqrt_(const int*, const int*, const int*, double*, const int*, double*, const int*, double*, int*);
extern void dgelqt_(const int*, const int*, const int*, double*, const int*, double*, const int*, double*, int*);
extern void dgemlqt_(const char*, const char*, const int*, const int*, const int*, const int*, const double*, const int*, const double*, const int*, double*, const int*, double*, int*);
extern void dgemqrt_(const char*, const char*, const int*, const int*, const int*, const int*, const double*, const int*, const double*, const int*, double*, const int*, double*, int*);
extern void dgeqlf_(const int*, const int*, double*, const int*, double*, double*, const int*, int*);
extern double dlange_(const char*, const int*, const int*, const double*, const int*, double*);
extern void dlacpy_(const char*, const int*, const int*, const double*, const int*, double*, const int*);
extern void dtrmm_(const char*, const char*, const char*, const char*, const int*, const int*, const double*, double*, const int*, double*, const int*);
extern void dgemv_(const char*, const int*, const int*, double*, double*, const int*, double*, const int*, double*, double*, const int*);
extern void dgemm_(const char*, const char*, const int*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*);
extern void dsyrk_(const char*, const char*, const int*, const int*, const double*, const double*, const int*, const double*, double*, const int*);
extern int LAPACKE_dgesvd(const int, const char, const char, const int, const int, double*, const int, double*, double*, int, double*, const int, double*);
#define LAPACK_COL_MAJOR 102

void attacFormalSum(const int d, const int *n, const int *rx, double * const *x, const int *ry, double * const *y, int* rz, double **z) ;
void attacFormalAXPBY(const int d, const int *n, const double alpha, const int *rx, double * const *x, const double beta, const int *ry, double * const *y, int* rz, double **z);
void attacHadamard(const int d, const int *n, const int *rx, double * const *x, const int *ry, double * const *y, int* rz, double **z);
double attacNormFromHadamard(const int d, const int *n, const int *rx, double * const *x, double* work);
double attacNorm_2(const int d, const int *n, const int *rx, double * const *x, double* workd, int* worki);
double attacNorm(const int d, const int *n, const int *rx, double **x, double* work);

void attacROrthogonalization(const int d, const int *n, const int *rx, double **x, double* work);
void attacRCompress(const int d, const int *n, int *rx, double **x, const double threshold, double *work);
void attacRTruncate(const int d, const int *n, int *rx, double **x, const int *maxr, const double abs_tol, double *work);

void attacLOrthogonalization(const int d, const int *n, const int *rx, double **x, double* work);
void attacLCompress(const int d, const int *n, int *rx, double **x, const double threshold, double *work);
void attacLTruncate(const int d, const int *n, int *rx, double **x, const int *maxr, const double abs_tol, double *work);

void attacCompress(const int d, const int *n, int *rx, double **x, const double threshold, const char ortho, double *work);
void attacTruncate(const int d, const int *n, int *rx, double **x, const int *maxr, const double abs_tol, const char ortho, double *work);

void attacROrthogonalizationImplicitQs(const int d, const int *n, const int *rx, double **x, double *v, double *work);
void attacRCompressImplicitQs(const int d, const int *n, int *rx, double **x, const double threshold, double *v, double *work);
void attacLOrthogonalizationImplicitQs(const int d, const int *n, const int *rx, double **x, double *v, double *work);
void attacLCompressImplicitQs(const int d, const int *n, int *rx, double **x, const double threshold, double *v, double *work);
void attacCompressImplicitQs(const int d, const int *n, int *rx, double **x, const double threshold, const char ortho, double *work);

double attacValue(const int* index, const int d, const int* n, const int* rx, double *const *x, double* work);
void attacScale(const double alpha, int mode, const int d, const int* n, const int* rx, double **x);

double attacInnerProduct(const int d, const int *n, const int *rx, double *const *x, const int *ry, double *const *y, double *work);
double attacNormSquaredHadamardOpt(const int d, const int *n, const int *rx, double *const *x, double *work);
double attacNormSquaredHadamardSymOpt(const int d, const int *n, const int *rx, double *const *x, double *work);
#endif
