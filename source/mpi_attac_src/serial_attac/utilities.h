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
/** \file utilities.h
 * \author Hussam Al Daas
 * \brief Utility functions
 * \date 16/07/2019
 * \details 
*/
#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>



extern void dtrmm_(const char*, const char*, const char*, const char*, const int*, const int*, const double*, double*, const int*, double*, const int*);
extern void dgemm_(const char*, const char*, const int*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*);

void sumVecs(const int n, const double* x, double* y);

void printMatrix(const char layout, const int m, const int n, const double* a, const int lda);

void attacKron(const int ma, const int na, const double* a, const int lda, const int mb, const int nb, const double* b, const int ldb, double* c, const int ldc);

void packR(const char layout, const int n, const double* a, const int lda, double* pa);

void unpackR(const char layout, const int n, const double* pa, double* a, const int lda);

void getTreeDepth(const int size, int* treeDepth);

void setLowerZero(const char layout, const int m, const int n, double* a, const int lda);

void unpad_rows(double* aMat,
                  int numRow,//new leading dim
                  int numCol,//number of columns
                  int delRows);//old leading dim = numRow+delRows

void trapezoid_mult(char    side,
                    double* trapezoid_mat,
                    int     numRow_trapezoid,
                    int     numCol_trapezoid,
                    int     numInd_rectangle,
                    double* rectangle_mat,
                    int     ldrect,
                    int*    result_rows,
                    int*    result_cols);
#endif
