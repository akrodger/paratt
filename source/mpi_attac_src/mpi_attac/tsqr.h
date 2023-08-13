#ifndef BRAMS_TSQR_H
#define BRAMS_TSQR_H

/*
 * Binary Tree Tall Skinny QR by Bram Rodgers.
 * Original Draft Dated: 4, Sep 2022
 */

/* Copyright 2022 Bram Rodgers
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
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

/*
 * Macros and Includes go here.
 */
#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<math.h>
#include"mpi.h"

/*
 * Struct Definitions:
 */
 
///A struct used to navigate an array which stores the data contained in a tree.
typedef struct{
    int t_id;///index of this node in tree array.
    int p_id;///index to parent node.
    int l_id;///index of left child node.
    int r_id;///index of right child node.
} tree_node;

///A struct used for storing arrays of integers.
typedef struct{
    int *arr;
    int  len;
} int_list;

typedef struct{
    tree_node* n;
    int        numNodes;
    int_list*  addr;
    int*       leafIdx;
    double**   Q;
    int*       Q_memSize;
    double**   R;
    int*       R_memSize;
    double**   tau;
    int*       tau_numel;
    int*       q_numRow;
    int*       q_numCol;
    int*       r_numCol;
    double*    flt_wrk;
    int        wrk_sz;
    MPI_Comm   comm;
} tsqr_tree;

/*
 * Function Declarations:
 */
void realloc_to_max(double** ptrptr,
                    int* memLenPtr,
                    size_t s,
                    int compare);
///Check if the the tree node at the given array index is a leaf.
int isLeaf(tree_node* n, int i);
///Check if the the tree node at the given array index is a root.
int isRoot(tree_node* n, int i);
///Check if the the tree node at the given array index is neither root, leaf.
int isIntr(tree_node* n, int i);
///Check if the the tree node at the given array index is a left child.
int isLchild(tree_node* n, int i);
///Check if the the tree node at the given array index is a right child.
int isRchild(tree_node* n, int i);
///Find the side of a triangular/trapezoidal part of a rectangular matrix.
///uplo = 'u' or 'U' is upper. uplo = 'l'  or 'L' is lower. Otherwise return 0.
int get_tri_pack_size(int numRow, int numCol, char uplo);
///Pack an upper triangular/trapezoidal matrix.
int upper_tri_pack(double* dest,
                    int numRow,
                    int numCol,
                    double* src,
                    int ld_src);
///Unpack an upper triangular/trapezoidal matrix.
int upper_tri_unpack(double* dest,
                     int ld_dest,
                     int numRow,
                     int numCol,
                     double* src);
///Pack a lower triangular/trapezoidal matrix.
int lower_tri_pack(double* dest,
                    int numRow,
                    int numCol,
                    double* src,
                    int ld_src);
///Unpack a lower triangular/trapezoidal matrix.
int lower_tri_unpack(double* dest,
                     int ld_dest,
                     int numRow,
                     int numCol,
                     double* src);
/**
 * Allocate an array of tree_nodes to *treePtr, with the number of nodes
 * stored at *numNodesPtr. The tree is defined as a nearly balanced binary
 * tree, where whenever the number of total decendant nodes is odd, then
 * give the left side of the tree more children.
 * Examples of trees computed by this algorithm:
 *    numLeaves=4 | numLeaves=3  |  numLeaves=6  |  numLeaves=9
 *        /\             /\             /\                /\
 *       /  \           /\2            /  \              /  \
 *      /\  /\         0 1            /\  /\            /    \
 *     0 1 2 3                       /\2 /\5           /      \
 *                                  0 1 3 4           /\      /\
 *                                                   /  \    /  \
 *                                                  /\  /\  /\  /\
 *                                                 /\2 3 4 5 6 7 8
 *                                                0 1
 * tree_node** treePtr: A pointer to the memory location where we will
 *                      save the tree memory structure.
 *                      malloc() is called for this.
 * int*    numNodesPtr: A pointer to an integer which will say how many
 *                      nodes the tree will have.
 * int       numLeaves: The number of leaves in the nearly balanced tree,
 *                      See the above examples. If this parameter is set
 *                      less than or equal to 2, we compute a tree with 2
 *                      leaves. 
 */
void tree_balanced_init(tree_node **treePtr,
                        int  *numNodesPtr,
                        int   numLeaves);

/**
 * Allocate the memory for a TSQR tree, using tree_balanced_init as the
 * message passing layout. We first call tree_balanced_init to find out
 * how much memory is required. Then we traverse the tree, calling malloc
 * to fill the tsqr_tree.addr entries with integer addresses representing
 * the parallel memory layout of the QR factorization process.
 * 
 * MPI_Comm comm:  The communicator which will be used to compute QR factors.
 *
 * Returns:
 *  a struct pointer retn_tree where
 *  retn_tree->n contains a tree allocated with tree_balanced_init
 *  retn_tree->numNodes contains the number of nodes on the balanced tree
 *  retn_tree->addr is an array of tree addresses defining the parallel
 *      tree traversal pattern.
 *  retn_tree->leafIdx is an array which gives the tree index of a wanted leaf.
 *                      For example, retn_tree->leafIdx[2] gives a leaf
 *                      with the label 2 as defined in the examples of
 *                      tree_balanced_init()
 *  retn_tree->Q is an array of memory locations used to point to work
 *               vector locations in TSQR compuations. One pointer per
 *               tree node. The numRow and numCol arrays are the same size.
 *
 *  retn_tree->q_numRow: an array of integers saying the number of rows of
 *                       the various orthogonal matrices Q[]
 *  retn_tree->q_numCol: an array of integers saying the number of cols of
 *                       the various orthogonal matrices Q[]
 *  retn_tree->r_numCol: an array of integers saying the number of cols of
 *                       the upper triangular or upper trapezoidal matrices R.
 *  
 */
tsqr_tree* alloc_tsqr_tree(MPI_Comm comm);

void free_tsqr_tree(tsqr_tree* qrTree);

/**
 * Populate a tsqr_tree with row and column information for QR factorization.
 * These rules are selected so that the final QR factorization can
 * be done one a row-wise distributed memory matrix with wide blocks in each
 * processor. This makes the memory layout different from that described by
 * demmel et. al., see readme for a reference.
 *
 * To fill a tsqr_tree with row and column information for an LQ factorization,
 * set the numRow_distrib variable to contain the distributed columns
 * information instead. Set the numCol variable to be the number of rows of
 * of the wide matrix for wflq.
 *
 * Our memory layout is defined in a leaves-to-root manner.
 *
 * If this node is a leaf: 
 *   q_numRow on leaf with address i = numRow_distrib.arr[i]
 *   q_numCol on leaf with address i = min(numRow_distrib.arr[i],numCol)
 *   r_numCol on leaf with address i = numCol
 * Otherwise:
 *   q_numRow on current node = q_numCol on left plus q_numCol on right
 *   q_numCol on current node = min(q_numCol on left plus q_numCol on right,
 *                                                                      numCol)
 *   q_numCol on current node = numCols
 *
 * tsqr_tree* qrTree:       A tsqr_tree which has been allocated with 
 *                          alloc_tsqr_tree
 * int_list numRow_distrib: An int_list struct which says how many rows of
 *                          the QR matrix are stored on each processor. For LQ,
 *                          this is the number of columns on each processor.
 * int numCol:              The number columns of the matrix we do a QR factor
 *                          on. In the case of LQ factor, this the number of
 *                          rows.
 */
void fill_tsqr_tree(tsqr_tree* qrTree,
                    int_list numRow_distrib,
                    int numCol);

/*
 * Use the  MPI communicator stored in qrTree communicator to factorize a
 * matrix of the form
 *
 *  A_distrib =   [[   A_0   ];
 *                 [   A_1   ];
 *                      .
 *                      .
 *                      .
 *                 [ A_{p-1} ]]
 *
 * Where each A_i block is stored on the processor with MPI rank i.
 * Each processor calls this function, then treverses a tree structure as
 * defined in alloc_tsqr_tree(), starting a leaf with index equal to the MPI
 * rank of that processor. A QR factorization is then done, storing the
 * result at the pointer qrTree->Q[qrTree->leaf_idx[i]] and also
 * qrTree->R[qrTree->leaf_idx[i]].
 *
 * Each processor moves up the tree to it's parent node
 * and does a check against the addr member variable to identify if an MPI
 * communication should be evaluated. Then another factorization is done
 * after comm, eventually resulting in the R factor of the matrix A_distrib
 * begin stored on the processor with MPI rank 0.
 *
 * At the end of computation, the memory addresses qrTree->Q[i]
 * store packed Householder QR factorizations with reflector coefficients
 * stored in qrTree->tau[i]. The memory addresses qrTree->R[i] store
 * store packed triangular R factors from QR factorization and are used
 * in MPI messaging. These arrays are essentially message passing buffers.
 *
 * Example below for a 4 processor binary tree communication pattern.
 * The general logical flow of left and right children is the same for
 * any general binary tree. The numeric strings after the underscore
 * in each factor contains the qrTree->addr array values
 *
 *  A_0=Q_0 * R_0 //Stored on processor 0
 *              \
 *               \ Recv R_1 from 1
 *                [R_0;
 *                 R_1] = Q_01 * R_01 //Stored on processor 0
 *               / Send R_1 to 0      \
 *              /                      \    
 *  A_1=Q_1 * R_1                       \
 *     //Stored on processor 1           \ Recv R_23 from 2
 *                                        [R_01;
 *     //Stored on processor 2             R_23] = Q_0123 * R_0123 //On proc 0
 *  A_2=Q_2 * R_2                        / Send R_23 to 0
 *              \                       /
 *               \ Recv R_3 from 3     /
 *                [R_2;               /
 *                 R_3] = Q_23 * R_23 //Stored on processor 2
 *               / Send R_3 to 2
 *              /
 *  A_3=Q_3 * R_3 //Stored on processor 3
 *
 * tsqr_tree*    qrTree: A tsqr_tree which has been initialized before calling
 *                       this function by first calling alloc_tsqr_tree.
 * double*    A_mpi_rnk: A column major contiguously allocated matrix of size
 *                       qrTree->q_numRow[qrTree->leafIdx[mpi_rank]]
 *                       number of rows by
 *                       qrTree->r_numCol[qrTree->leafIdx[mpi_rank]]
 *                       number of columns. Note, if number of cols is
 *                       mismatched from one processor to any other, then the 
 *                       concatenation of R_i factors step above will lead
 *                       to an index of out bounds error, possibly crashing.
 */
void tsqr_factorize(tsqr_tree* qrTree, double* A_mpi_rnk);

/*
 * Use the MPI communicator stored in qrTree to generate a column-orthogonal
 * matrix
 * of the form
 *
 *  Q_distrib =   [[   Q_0   ];
 *                 [   Q_1   ];
 *                      .
 *                      .
 *                      .
 *                 [ Q_{p-1} ]]
 *
 * Where each Q_i block is stored on the processor with MPI rank i.
 * Each processor calls this function, then treverses a tree structure as
 * defined in alloc_tsqr_tree(), starting at the root index and ending at a
 * leaf with address equal to the MPI rank of that processor.
 * This is done by multiplying Q factors as defined by tsqr_factorize().
 *
 * In essense, this algorithm is the  tsqr_factorize() algorithm in reverse.
 *
 * tsqr_tree*    qrTree: A tsqr_tree which has been initialized filled with
 *                       QR factorization data as defined in tsqr_factorize()
 * double*    Q_mpi_rnk: A column major contiguously allocated matrix of size
 *                       qrTree->q_numRow[qrTree->leafIdx[mpi_rank]]
 *                       number of rows by
 *                       qrTree->r_numCol[qrTree->leafIdx[mpi_rank]]
 *                       number of columns. This matrix is a block of a column
 *                       orthogonal matrix stored across all the processors.
 */
void tsqr_make_q(tsqr_tree* qrTree, double* Q_mpi_rnk);

/*
 * Use the  MPI communicator stored in qrTree communicator to factorize a
 * matrix of the form
 *
 *  A_distrib = [[ A_0 ], [ A_1 ], ... ,[ A_{p-1} ]]
 *
 * This algorithm is essentially similar to tsqr_factorize(), except
 * every matrix defined in the linear algebra gets transposed.
 *
 * Where each A_i block is stored on the processor with MPI rank i.
 * Each processor calls this function, then treverses a tree structure as
 * defined in alloc_tsqr_tree(), starting a leaf with index equal to the MPI
 * rank of that processor. An LQ factorization is then done, storing the
 * result at the pointer qrTree->Q[qrTree->leaf_idx[i]] and also
 * qrTree->R[qrTree->leaf_idx[i]]. Here, R stores the L factor of LQ.
 *
 * Each processor moves up the tree to it's parent node
 * and does a check against the addr member variable to identify if an MPI
 * communication should be evaluated. Then another factorization is done
 * after comm, eventually resulting in the L factor of the matrix A_distrib
 * begin stored on the processor with MPI rank 0.
 *
 * At the end of computation, the memory addresses qrTree->Q[i]
 * store packed Householder QR factorizations with reflector coefficients
 * stored in qrTree->tau[i]. The memory addresses qrTree->R[i] store
 * store packed triangular L factors from LQ factorization and are used
 * in MPI messaging. These arrays are essentially message passing buffers.
 *
 * An example of the message passing structure is given for the QR variant
 * of this same algorithm.
 *
 * tsqr_tree*    qrTree: A tsqr_tree which has been initialized before calling
 *                       this function by first calling alloc_tsqr_tree.
 * double*    A_mpi_rnk: A column major contiguously allocated matrix of size
 *                       qrTree->r_numCol[qrTree->leafIdx[mpi_rank]]
 *                       number of rows by
 *                       qrTree->q_numRow[qrTree->leafIdx[mpi_rank]]
 *                       number of columns. Note, if number of rows is
 *                       mismatched from one processor to any other, then the 
 *                       concatenation of L_i factors step will lead
 *                       to an index of out bounds error, possibly crashing.
 */
void wflq_factorize(tsqr_tree* qrTree, double* A_mpi_rnk);

/*
 * Use the  MPI communicator stored in qrTree communicator to generate a
 * column-orthogonal matrix of the form
 *
 *  Q_distrib =   [[ Q_0 ], [ Q_1 ], ... , [ Q_{p-1} ]]
 *
 * This algorithm is essentially similar to wflq_make_q(), except
 * every matrix defined in the linear algebra gets transposed.
 *
 * Where each Q_i block is stored on the processor with MPI rank i.
 * Each processor calls this function, then treverses a tree structure as
 * defined in alloc_tsqr_tree(), starting at the root index and ending at a
 * leaf with address equal to the MPI rank of that processor.
 * This is done by multiplying Q factors as defined by wflq_factorize().
 *
 *
 * In essense, this algorithm is the  tsqr_factorize() algorithm in reverse.
 *
 * tsqr_tree*    qrTree: A tsqr_tree which has been initialized filled with
 *                       QR factorization data as defined in tsqr_factorize()
 * double*    Q_mpi_rnk: A column major contiguously allocated matrix of size
 *                       qrTree->r_numCol[qrTree->leafIdx[mpi_rank]]
 *                       number of rows by
 *                       qrTree->q_numRow[qrTree->leafIdx[mpi_rank]]
 *                       number of columns. This matrix is a block of a column
 *                       orthogonal matrix stored across all the processors.
 */
void wflq_make_q(tsqr_tree* qrTree, double* Q_mpi_rnk);

#endif
