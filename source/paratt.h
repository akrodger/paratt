#ifndef PARATT_PARALLEL_TT_H
#define PARATT_PARALLEL_TT_H

/*
 * C Header file for Parallel Tensor Train by Bram Rodgers.
 * Original Draft Dated: 16, June 2022
 */

/*
 * Header File Body:
 */

/*
 * Macros and Includes go here.
 */
#include<stdlib.h>
#include "./mpi_attac_src/serial_attac/attac.h"
#include "./mpi_attac_src/serial_attac/utilities.h"
#include "./mpi_attac_src/mpi_attac/pattac.h"

/*
 * Struct Definitions:
 */

typedef struct{
//numModes is the number of indexes of the uncompressed tensor.
int       numDim;
//dim is an array of integers which describe the size of the
//fully uncompressed tt tensor.
int*      dim;
//ttRank is an array of integers which describe the sizes of of the
//arrays stored within coreList. There are numModes+1 many elements in
//this array. The first and last entries hold the number 1.
int*      ttRank;
//coreList is an array of contiguously allocated 3-way tensors.
//coreList[i] points to the [0,0,0] element of the i index TT tensor core.
//there are numModes many cores.
/*
coreList[0]          is        1           by     modeDepth[0]      by ttRank[1]
coreList[i]          is    ttRank[i]       by     modeDepth[i]      by ttRank[i+1]
coreList[numModes-1] is ttRank[numModes]   by modeDepth[numModes]   by    1
*/
double**  coreList;
//An array which says exactly how much memory is currently allocated to
//each core. This is given as the number of doubles allocated per core.
int* coreMem;
//A work vector used for computing the the attac and pattac subroutines.
double* wk;
//number of elements allocated to the work vector.
int wksz;
} attac_tt;

typedef struct{
attac_tt*  x;
int        mpi_rk;
int        mpi_sz;
int*       distrib_dim_mat;
int*       dim;
int        numDim;
int*       iwk;
int        iwksz;
tsqr_tree* orthog_wk;
} paratt;

/*
 * Function Declarations:
 */

/*
 * Return a pointer to an allocated tt tensor using malloc() several times
 * the contents of the arrays d and ttR are copied into the output pointer's
 * corresponding struct.
 * The work vector is initialized to NULL and wksz is set to 0.
 */
attac_tt* alloc_tt(int ndims, int* d, int* ttR);


paratt* alloc_ptt(int ndims, int* d, int* ttR);


/*
 * Change the ranks of a x_tt tensor which is assumed to be allocated by
 * alloc_tt(). We call realloc() on the x_tt->coreList[k] memory blocks
 * and copy the values of ttR[k] into x_tt->ttRank[k] for each k.
 * No other values of x_tt are modified.
 * ttR is assumed to have x_tt->numModes many elements.
 * Returns the pointer x_tt
 */
attac_tt* rank_change_tt(attac_tt* x_tt, int* ttR);

/*
 * Call rank_change_tt on dest_tt using src_tt as the input tensor train ranks.
 * Then copy the src_tt cores into the dest_tt cores.
 * We assume dest_tt and src_tt are two tensors with the same mode depths but
 * different ranks.
 */
void copy_rank_change_tt(attac_tt* dest_tt,attac_tt* src_tt);


/*
 * Free a tt tensor which has been created with alloc_tt()
 * A NULL check is done on the work vector before freeing it.
 */
void free_tt(attac_tt* x_tt);


void free_ptt(paratt* x_ptt);

//MPI helper function for passing ghost cells in a cylcic pattern:
// 0 <- 1 <- ... <- num_proc <- 0
//This is done using nonblocking sends and recieves.
void send_recv_below(void* send_buff,
                     void* recv_buff,
                     int msg_sz,
                     MPI_Datatype dat_t,
                     MPI_Request *sendReq,
                     MPI_Request *recvReq);

//MPI helper function for passing ghost cells in a cylcic pattern:
// 0 -> 1 -> ... -> num_proc -> 0
//This is done using nonblocking sends and recieves.
void send_recv_above(void* send_buff,
                     void* recv_buff,
                     int msg_sz,
                     MPI_Datatype dat_t,
                     MPI_Request *sendReq,
                     MPI_Request *recvReq);
/*
 * use attacFormalAXPBY() to compute a linear combination of x_tt and y_tt.
 * We resize z_tt if needed.
 */
void attac_tt_axpby(double a, attac_tt* x_tt, 
                    double b, attac_tt* y_tt, attac_tt* z_tt);

/*
 * Scale the input tensor by the given scalar.
 *
 */
void attac_tt_scale(double a, attac_tt* x_tt);

/*
 * Use the low-level routines of mpi-attac to implement an easier to use
 * tt-rounding function. It does the work size computation, reallocs
 * the workvector if needed, then uses the struct elements to call
 * attacTruncate(), result is stored in x_tt->coreList.
 */
void attac_tt_truncate(attac_tt* x_tt, double tol, int* maxRank);



void paratt_truncate(paratt* x_ptt, double tol, int* maxRank);

/*
 * resize work vector then call attacNorm()
 */
double attac_tt_norm(attac_tt* x_tt);


/*
 * Compute norm of TT tensor using TSQR.
 */
double paratt_norm(paratt* x_ptt);

/*
 * Compute the dot product of the parallel TT tensors.
 */
double paratt_dotprod(paratt* x_ptt, paratt* y_ptt);

/*
 * compute a value at a given index of the MPI distributed memory tensor.
 */
double paratt_entry(paratt* x_ptt, int* mIdx);

/*
 * Make a rank 1 tt tensor out of a 2d array. Save that tt tensor in *x_tt.
 */
void attac_set_rank1(attac_tt** x_tt_ptr, double** vec, int ndims, int* d);

/*
 * Make a sum of rank 1 tensors into a tt tensor.
 * does an alloc_tt, then copies the 3-way array of vectors into a tt tensor.
 * Let % denote the tensor product. Let : dnote an array slice.
 * The final result is
 * x =
 *vec[0][0][:]        %    vec[0][1][:]       % ... % vec[0][ndims-1][:]
 *                             +
 *vec[1][0][:]        %    vec[1][1][:]       % ... % vec[1][ndims-1][:]
 *                             +
 *                             .
 *                             .
 *                             .
 *                             +
 *vec[all_ttR-1][0][:] % vec[all_ttR-1][1][:] % ... % vec[all_ttR-1][ndims-1][:]
 *
 * The tensor train rank array gets set uniformly to the values all_ttR
 */
attac_tt* attac_sum_of_product(double*** vec,
                                   int ndims, int* d, int all_ttR);

/*
 * Apply a linear map defined by a vector containing a matrix's diagonal.
 * for example, if vec[i] = (i+1)^2 for each entry, then this is the same
 * as multiplying the matrix diag(1,4,9,...,n^2) into the specified mode.
 *
 * f_tt:    a tensor train struct you want to apply a diagonal matrix to
 * mode:    the tensor core index you want the matrix to be applied to.
 * vec:     a pointer to an array containing f_tt->dim[mode] many entries.
 */
void attac_mult_diag_mat(attac_tt* f_tt, int mode, double* vec);

/*
 * Apply a linear map defined by a given centered stecil.
 * For example, if sten = [-1, 0, 1] and mode = 0 or mode = f_tt->numDim-1,
 * then this is equivalent to multiplying a periodic centered finite difference
 * stencil into the non-rank index of the first/last core of f_tt respectively.
 *
 * f_tt:    a tensor train struct you want to apply a stencil to.
 * mode:    the tensor core index you want the stencil to be applied to.
 * sten:    a pointer to an array containing a centered stencil.
 * slen:    number of elements in sten. assumed odd. if slen is even, 
 *          then we decrement it by 1 and continue. Then only first slen-1
 *          elements of sten are accessed.
 *
 */
void attac_apply_periodic_stencil(attac_tt* f_tt, int mode,
                                        double* sten, int slen);

void pattac_apply_periodic_stencil(paratt* f_ptt, int mode,
                                        double* sten, int slen);
  
/*
 * Apply a finite difference stencil with the following boundary conditoins.
 * 
 * Let f[0], f[1], ... , f[N-1] be a vector.
 *
 * if i < 0, then f[i] = 0.
 *
 * if i >= N, then f[i] = f[N-1-i]
 *
 * These are the boundary conditions for a multivariate CDF if the support
 * of its PDF is strictly within the grid indexed by 1, 2, ... , N-2.
 *
 *
 */                                      
void pattac_apply_cdf_stencil(paratt* f_ptt, int mode,
                                double* sten, int slen);

/*
 * Apply a Lax-Frieddrichs finite difference stencil with CDF type BCs
 * 
 *
 */                                      
void pattac_apply_cdf_lf_flux(paratt* f_ptt, int mode,
                                double* wave_speed, double* gc_wave_speed,
                                double* source_coef,
                                double dt, double dx);
/*
 * Apply a Lax-Wendroff stencil with CDF type BCs
 * 
 *
 */  
void pattac_apply_cdf_lw_rhs(paratt* f_ptt, int mode,
                                double* wave_speed, double* gc_wave_speed,
                                double* bdry_wave_speed,
                                double* source_coef,
                                double dt, double dx);

/*
 * Multiply a row vector by the tt core labeled by mode.
 * When f_ptt represents a multivariate probability density and row_vector
 * is an array of integration weights, then this function can be used
 * to compute a marginal probability density.
 *
 * paratt** marginal_tt_ptr is used to allocate a new tt tensor with 1 fewer
 * mode. If not null, an error and exit(1) is thrown briefly after entry to
 * discourage likely memory leaks.
 *
 * row_vector is expected to be split up in parallel in the same
 * pattern as the parallel TT cores.
 */
void pattac_apply_marginal(paratt* f_ptt, paratt** marginal_ptt_ptr, 
                            int mode, double* row_vector);

/*
 * Compute a weighted cumulative sum on the given TT mode. Can be used for
 * computing indefinite integrals or Cumulative Distribution Functions.
 * Set the row vector to be the grid spacing in this case.
 * Assumes pattern row_vector[k] = x[k] - x[k-1]. (Periodic bc assumption.) 
 */
void pattac_marginal_cusum(paratt* f_ptt, int mode, double* row_vector);

/*
 * Compute a weighted cumulative sum on the given TT mode by averaging
 * successive points along the tensor dimension.
 * Can be used for computing indefinite integrals or Cumulative Distribution
 * Functions via the trapezoidal rule.
 * Assumes pattern row_vector[k] = x[k] - x[k-1]
 * On MPI rank 0, k=0 not referenced. On MPI Rank k > 0, contains the 
 * integration weight for the space between the ranks. E.g.:
 * (rank 0: x[0], x[1]), row_vector[1] referenced only.
 * (rank 1: x[2], x[3]), row_vector[0] indicates between x[1] and x[2]
 *                       row_vector[1] indicated between x[2] and x[3]
 */
void pattac_marginal_trap(paratt* f_ptt, int mode, double* row_vector);

void pattac_scale_weight_sum(paratt* f_ptt, int mode,
                             double* scale, double* sum_w);
/*
 * Write the tensor train cores of f_ptt to the file system in the folder
 * named folder_name. Inside this folder, the cores are expected to be named
 * core_1.cr, core_2.cr, ... , core_[f_ptt->numDim].cr
 *
 *
 */
void pattac_write_tt_folder(char* folder_name, paratt* f_ptt);

/*
 * Loads tensor train cores from the file system in the folder named
 * folder_name. Inside this folder, the cores are expected to be named
 * core_1.cr, core_2.cr, ... , core_[ndims].cr
 *
 * Where ndims is the number of dimensions of a newly allocated paratt.
 * A pointer to the newly allocated and loaded tensor is returned.
 *
 */
paratt* pattac_load_tt_folder(char* folder_name, int ndims);
/*
 * Read the first line of the given core file and store the
 * sizes of that core in the variables *rl, *N, and *rr.
 */
void read_core_size(char* fname,
                     int* rl,
                     int* N,
                     int* rr);
/*
 * Load a subarray of a tt tensor core into the array *cr, possibly
 * reallocating if there isnt enough memory.
 */
void load_core_slice(char* fname,
                     double** cr,
                     int* cr_sz,
                     int* rl,
                     int* N,
                     int* rr,
                     int N_beg,
                     int N_end);
#endif
