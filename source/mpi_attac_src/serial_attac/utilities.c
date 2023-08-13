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
/** \file utilities.c
 * \author Hussam Al Daas
 * \brief Utility functions
 * \date 16/07/2019
 * \details 
*/
#include <math.h>
#include <unistd.h>
#include "utilities.h"

/** \fn void sumVecs(const int n, const double* x, double* y);
 * \brief
 * \details
 * \param n
 * \param x
 * \param y
 * \remarks
 * \warning
*/
void sumVecs(const int n, const double* x, double* y){
    int i;
	for(i = 0; i < n; i++){
		y[i] += x[i];
	}
}

/** \fn void printMatrix(const char layout, const int m, const int n, const double* a, const int lda);
 * \brief
 * \details
 * \param layout
 * \param m
 * \param n
 * \param a
 * \param lda
 * \remarks
 * \warning
*/
void printMatrix(const char layout, const int m, const int n, const double* a, const int lda){
    int i, j;
	if(layout == 'C'){
		for(i = 0; i < m; i++){
			for(j = 0; j < n; j++){
				printf("%.16f\t ", a[i + j * lda]);
			}
			printf("\n");
		}
	}else{
		for(i = 0; i < m; i++){
			for(j = 0; j < n; j++){
				printf("%.16f\t ", a[i * lda + j]);
			}
			printf("\n");
		}
	}
}

/** \fn void setLowerZero(const char layout, const int m, const int n, double* a, const int lda);
 * \brief
 * \details
 * \param layout
 * \param m
 * \param n
 * \param a
 * \param lda
 * \remarks
 * \warning
*/
void setLowerZero(const char layout, const int m, const int n, double* a, const int lda){
    int i, j;
	if(layout == 'C'){
		for(i = 0; i < m; i++){
			for(j = 0; j < i; j++){
				a[i + j * lda] = 0;
			}
		}
	}else{
		for(i = 0; i < m; i++){
			for(j = 0; j < i; j++){
				a[i * lda + j] = 0;
			}
		}
	}
}

/** \fn void attacKron(const int ma, const int na, const double* a, const int lda, const int mb, const int nb, const double* b, const int ldb, double* c, const int ldc);
 * \brief
 * \details
 * \param ma
 * \param na
 * \param a
 * \param lda
 * \param mb
 * \param nb
 * \param b
 * \param ldb
 * \param c
 * \param ldc
 * \remarks
 * \warning
*/
void attacKron(const int ma, const int na, const double* a, const int lda, const int mb, const int nb, const double* b, const int ldb, double* c, const int ldc){
#ifdef DEBUG
	if(ldc < mb * ma){
		fprintf(stderr, "%s:Line %d %s::LDC is smaller than necessary\n", __FILE__, __LINE__, __func__);
		fprintf(stderr, "mA = %d, mB = %d, ldC = %d < %d x %d\n", ma, mb, ldc, ma, mb);
	}
#endif
    int i, j, k, l;
	for(j = 0; j < na; j++){
		for(i = 0; i < ma; i++){
      for(k = 0; k < mb; k++){
        for(l = 0; l < nb; l++){
          c[mb * i + j * ldc * nb + k + l * ldc] = a[i + j * lda] * b[k + l * ldb];
        }
      }
		}
	}
}

/** \fn void packR(const char layout, const int n, const double* a, const int lda, double* pa);
 * \brief Pack a triangular matrix
 * \details
 * \param layout
 * \param n
 * \param a
 * \param lda
 * \param pa
 * \remarks
 * \warning
*/
void packR(const char layout, const int n, const double* a, const int lda, double* pa){
	int counter = 0, i, j;
	if(layout == 'C'){
		for(j = 0; j < n; j++){
			for(i = 0; i < j + 1; i++){
				pa[counter++] = a[i + j * lda];
			}
		}
	}else if(layout == 'R'){
		for(i = 0; i < n; i++){
			for(j = i; j < n; j++){
				pa[counter++] = a[i * lda + j];
			}
		}
	}else{
		fprintf(stderr, "%s:Line %d %s::Not recognized layout\n", __FILE__, __LINE__, __func__);
#ifdef ATTACMPI
		MPI_Abort(MPI_COMM_WORLD, -99);
#else
		exit(0);
#endif
	}
}

/** \fn void unpackR(const char layout, const int n, const double* pa, double* a, const int lda){
 * \brief Unpack a triangular matrix
 * \details
 * \param n
 * \param layout
 * \param pa
 * \param a
 * \param lda
 * \remarks
 * \warning
*/
void unpackR(const char layout, const int n, const double* pa, double* a, const int lda){
	int counter = 0, i, j;
	if(layout == 'C'){
		for(j = 0; j < n; j++){
			for(i = 0; i < j + 1; i++){
				a[i + j * lda] = pa[counter++];
			}
		}
	}else if(layout == 'R'){
		for(i = 0; i < n; i++){
			for(j = i; j < n; j++){
				a[i * lda + j] = pa[counter++];
			}
		}
	}else{
		fprintf(stderr, "%s:Line %d %s::Not recognized layout\n", __FILE__, __LINE__, __func__);
#ifdef ATTACMPI
		MPI_Abort(MPI_COMM_WORLD, -99);
#else
		exit(0);
#endif
	}
}

/** \fn void getTreeDepth(const int size, int* treeDepth){
 * \brief Compute the depth of the binary tree with size leaves
 * \details
 * \param size
 * \param treeDepth
 * \remarks
 * \warning
*/
void getTreeDepth(const int size, int* treeDepth){
	int log2Size = 0;
	int temp = 1;
	while(size > temp){
		log2Size++;
		temp *= 2;
	}
	treeDepth[0] = log2Size;
}


void unpad_rows(double* aMat,
                  int numRow,
                  int numCol,
                  int delRows){
    int j,k;
    for(k=1;k<numCol;k++){
        for(j=0;j<numRow;j++){
            aMat[j+(numRow*k)] = aMat[j+((numRow+delRows)*k)];
        }
    }
}


void trapezoid_mult(char    side,
                    double* trapezoid_mat,
                    int     numRow_trapezoid,
                    int     numCol_trapezoid,
                    int     numInd_rectangle,
                    double* rectangle_mat,
                    int     ldrect,
                    int*    result_rows,
                    int*    result_cols){
    static const char transa = 'n';
    static const char   diag = 'n';
    static const double done = 1.0;
    double* left_term;
    double* right_term;
    int minRowCol, delInds, numRow_c, numCol_c, ldL, ldR;
    char uplo;
    minRowCol = (numRow_trapezoid < numCol_trapezoid)?
                 numRow_trapezoid : numCol_trapezoid;
    uplo =  (side == 'r' || side == 'R') ? 'l' :
           ((side == 'l' || side == 'L') ? 'u' : 'N');
    if(side == 'l' || side == 'L'){
        uplo      = 'u';
        numRow_c  = minRowCol;
        numCol_c  = numInd_rectangle;
        delInds   = numCol_trapezoid - minRowCol;
        if(delInds > 0){
            left_term  = &(trapezoid_mat[numRow_trapezoid*minRowCol]);
            right_term = &(rectangle_mat[minRowCol]);
            ldL        = numRow_trapezoid;
            ldR        = ldrect; 
        }
    }else if(side == 'r' || side == 'R'){
        uplo = 'l';
        numRow_c = numInd_rectangle;
        numCol_c = minRowCol;
        delInds  = numRow_trapezoid - minRowCol;
        if(delInds > 0){
            left_term  = &(rectangle_mat[ldrect*minRowCol]);
            right_term = &(trapezoid_mat[minRowCol]);
            ldL        = ldrect;
            ldR        = numRow_trapezoid;
        }
    }else{
        fprintf(stderr, "\ntrapezoid_mult() illegal side: '%c'\n", side);
        exit(1);
    }
    dtrmm_(&side,             //SIDE
            &uplo,            //UPLO
            &transa,          //TRANSA
            &diag,            //DIAG
            &numRow_c,        //M
            &numCol_c,        //N
            &done,            //ALPHA
            trapezoid_mat,    //A
            &numRow_trapezoid,//LDA
            rectangle_mat,    //B
            &ldrect);         //LDB
    if(delInds > 0){
        dgemm_(&transa,       //TRANSA
               &transa,       //TRANSB
               &numRow_c,     //M
               &numCol_c,     //N
               &delInds,      //K
               &done,         //ALPHA
               left_term,     //A
               &ldL,          //LDA
               right_term,    //B
               &ldR,          //LDB
               &done,         //BETA
               rectangle_mat, //C
               &ldrect);      //LDC
    }
    unpad_rows(rectangle_mat,numRow_c,numCol_c,ldrect-numRow_c);
    *result_rows = numRow_c;
    *result_cols = numCol_c;
}
