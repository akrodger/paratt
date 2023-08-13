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
/** \file putilities.h
 * \author Hussam Al Daas
 * \brief Utility functions - parallel
 * \date 16/07/2019
 * \details 
*/
#ifndef PUTILITIES_H
#define PUTILITIES_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define ATTACMPI
extern void dgeqrt_(const int*, const int*, const int*, double*, const int*, double*, const int*, double*, int*);
extern void dgelqt_(const int*, const int*, const int*, double*, const int*, double*, const int*, double*, int*);
extern void dgemlqt_(const char*, const char*, const int*, const int*, const int*, const int*, const double*, const int*, const double*, const int*, double*, const int*, double*, int*);
extern void dgemqrt_(const char*, const char*, const int*, const int*, const int*, const int*, const double*, const int*, const double*, const int*, double*, const int*, double*, int*);
extern void dtpqrt_(const int*, const int*, const int*, const int*, double*, const int*, double*, const int*, double*, const int*, double*, int*);
extern void dtplqt_(const int*, const int*, const int*, const int*, double*, const int*, double*, const int*, double*, const int*, double*, int*);
extern void dtpmqrt_(const char*, const char*, const int*, const int*, const int*, const int*, const int*, const double*, const int*, const double*, const int*, double*, const int*, double*, const int*, double*, int*);
extern void dtpmlqt_(const char*, const char*, const int*, const int*, const int*, const int*, const int*, const double*, const int*, const double*, const int*, double*, const int*, double*, const int*, double*, int*);
extern void dlacpy_(const char*, const int*, const int*, const double*, const int*, double*, const int*);


void laplace1d(const MPI_Comm comm, const int n, const double *x, double *Ax);
#endif
