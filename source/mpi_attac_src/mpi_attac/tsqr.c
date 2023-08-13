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
 * Macros and Includes go here:
 */
#include "tsqr.h"

extern void dgeqrf_(const int* M,
                    const int* N,
                    double*    A,
                    const int* LDA,
                    double*    TAU,
                    double*    WORK,
                    const int* LWORK,
                    int*       INFO);


extern void dgelqf_(const int* M,
                    const int* N,
                    double*    A,
                    const int* LDA,
                    double*    TAU,
                    double*    WORK,
                    const int* LWORK,
                    int*       INFO);


extern void dorgqr_(const int*    M,
                    const int*    N,
                    const int*    K,
                    double*       A,
                    const int*    LDA,
                    const double* TAU,
                    double*       WORK,
                    const int*    LWORK,
                    int*          INFO);

extern void dorglq_(const int*    M,
                    const int*    N,
                    const int*    K,
                    double*       A,
                    const int*    LDA,
                    const double* TAU,
                    double*       WORK,
                    const int*    LWORK,
                    int*          INFO);

extern void dormqr_(const char*   SIDE,
                    const char*   TRANS,
                    const int*    M,
                    const int*    N,
                    const int*    K,
                    const double* A,
                    const int*    LDA,
                    const double* TAU,
                    double*	      C,
                    const int*    LDC,
                    double*       WORK,
                    const int*    LWORK,
                    int*          INFO);

extern void dormlq_(const char*   SIDE,
                    const char*   TRANS,
                    const int*    M,
                    const int*    N,
                    const int*    K,
                    const double* A,
                    const int*    LDA,
                    const double* TAU,
                    double*	      C,
                    const int*    LDC,
                    double*       WORK,
                    const int*    LWORK,
                    int*          INFO);

/*
 * Locally used helper functions:
 */

void alloc_int_list(int_list *iList, int len){
    iList[0].arr = malloc(sizeof((iList[0].arr[0]))*len);
    iList[0].len = len;
}

void realloc_to_max(double** ptrptr,
                    int* memLenPtr,
                    size_t s,
                    int compare){
    if(*ptrptr != NULL){
        if(*memLenPtr < compare){
                *ptrptr = realloc(*ptrptr,s*compare);
                *memLenPtr = compare;
        }
    }else{
        *ptrptr = malloc(s*compare);
        *memLenPtr = compare;
    }
}

int isLeaf(tree_node* n, int i){
    int ret = 0;
    if((n[i].l_id == -1) && (n[i].r_id == -1)){
        ret = 1;
    }
    return ret;
}

int isRoot(tree_node* n, int i){
    int ret = 0;
    if(n[i].t_id == n[i].p_id){
        ret = 1;
    }
    return ret;
}

int isIntr(tree_node* n, int i){
    int ret = 0;
    if((isLeaf(n,i) == 0) && (isRoot(n,i) == 0)){
        ret = 1;
    }
    return ret;
}

int isLchild(tree_node* n, int i){
    int ret = 0;
    if(n[n[i].p_id].l_id == n[i].t_id){
        ret = 1;
    }
    return ret;
}

int isRchild(tree_node* n, int i){
    int ret = 0;
    if(n[n[i].p_id].r_id == n[i].t_id){
        ret = 1;
    }
    return ret;
}

int get_tri_pack_size(int numRow, int numCol, char uplo){
    int tri_part = 0, rect_part = 0, len_diff, area = 0;
    len_diff = numCol - numRow;
    if(uplo == 'u' || uplo == 'U'){
        if(len_diff > 0){//more cols than rows
            rect_part = numRow*len_diff;
            tri_part  = (numRow*(numRow+1))/2;
        }else{
            tri_part  = (numCol*(numCol+1))/2;
        }
    }else if(uplo == 'l' || uplo == 'L'){
        if(len_diff < 0){//more rows than cols
            rect_part = -numCol*len_diff;
            tri_part  = (numCol*(numCol+1))/2;
        }else{
            tri_part  = (numRow*(numRow+1))/2;
        }
    }
    area = tri_part + rect_part;
    return area;
}


int upper_tri_pack(double* dest,
                    int numRow,
                    int numCol,
                    double* src,
                    int ld_src){
    int i,j,m;
    m=0;
    for(j=0;j<numCol;j++){
        for(i=0;i<numRow;i++){
            if(i <= j){
                dest[m] = src[i+ld_src*j];
                m=m+1;
            }
        }
    }
    return m;
}

int upper_tri_unpack(double* dest,
                     int ld_dest,
                     int numRow,
                     int numCol,
                     double* src){
    int i,j,m;
    m=0;
    for(j=0;j<numCol;j++){
        for(i=0;i<numRow;i++){
            if(i <= j){
                dest[i+ld_dest*j] = src[m];
                m=m+1;
            }else{
                dest[i+ld_dest*j] = 0.0;
            }
        }
    }
    return m;
}

int lower_tri_pack(double* dest,
                    int numRow,
                    int numCol,
                    double* src,
                    int ld_src){
    int i,j,m;
    m=0;
    for(j=0;j<numCol;j++){
        for(i=0;i<numRow;i++){
            if(i >= j){
                dest[m] = src[i+ld_src*j];
                m=m+1;
            }
        }
    }
    return m;
}

int lower_tri_unpack(double* dest,
                     int ld_dest,
                     int numRow,
                     int numCol,
                     double* src){
    int i,j,m;
    m=0;
    for(j=0;j<numCol;j++){
        for(i=0;i<numRow;i++){
            if(i >= j){
                dest[i+ld_dest*j] = src[m];
                m=m+1;
            }else{
                dest[i+ld_dest*j] = 0.0;
            }
        }
    }
    return m;
}
/*
 * Function Implementations:
 */

void tree_balanced_init(tree_node **treePtr,
                        int  *numNodesPtr,
                        int   numLeaves){
    //Find the number of layers, p
    long int pow_p = 1;
    int p = 0, 
        i = 0,
        startOfLastLayer, 
        leftChild, 
        rightChild, 
        parent, 
        lastLayerIdx = 0,
        numNodes = 0;
    int *leafTracking;
    tree_node *tree;
    //Especially simple case for a tree with 1 parent and no children
    if(numLeaves == 1){
        tree = (tree_node*) malloc(sizeof(*tree));
        tree[0].t_id = 0;
        tree[0].p_id = 0;
        tree[0].l_id = -1;
        tree[0].r_id = -1;
        *numNodesPtr = 1;
        *treePtr = tree;
        return;
    }
    //Especially simple case for a tree with 1 parent and 2 children:
    if(numLeaves == 2){
        tree = (tree_node*) malloc(sizeof(*tree)*3);
        //set root
        tree[0].t_id = 0;
        tree[0].p_id = 0;
        tree[0].l_id = 1;
        tree[0].r_id = 2;
        //set left child
        tree[1].t_id = 1;
        tree[1].p_id = 0;
        tree[1].l_id = -1;
        tree[1].r_id = -1;
        //set right child
		tree[2].t_id = 2;
        tree[2].p_id = 0;
        tree[2].l_id = -1;
        tree[2].r_id = -1;
        *numNodesPtr = 3;
        *treePtr = tree;
        return;
    }
    while(pow_p < numLeaves){
        p++;
        numNodes += pow_p;
        pow_p *= 2;
    }
    numNodes += 2*numLeaves - pow_p;
    //Now we set the start index of last layer
    startOfLastLayer =  numNodes - 2*numLeaves + pow_p ;
    //The number of nodes in the tree is 2^(p-1) - 1 + numLeaves
    //since pow_p = 2^p, we have nodes = pow_p/2 -1 + numLeaves
    tree = (tree_node*) malloc(sizeof(*tree)*numNodes);
    leafTracking = (int*) malloc(sizeof(*leafTracking)*numNodes);
    numNodesPtr[0] = numNodes;
    //set the parent id and child id numbers of the root
	tree[0].t_id = 0;// = DisTreeNode(placeHolder, dec_index, 0, 0, 1, 2);
    tree[0].p_id = 0;
    tree[0].l_id = 1;
    tree[0].r_id = 2;
    leafTracking[0] = numLeaves;
    //Now we can use the index i in the following loop to determine the index
    //arrays of every element in the tree
    //We iterate over all but the last layer of the tree.
    //In the second to last layer, we also make the last layer nodes
    lastLayerIdx = startOfLastLayer;
    for(i = 1; i < startOfLastLayer; i++){
        //set proposed left and right child indexes, and parent
        leftChild = i*2 + 1;
        rightChild = i*2 + 2;
        parent = (i-1)/2;
        if(i % 2){//if i is a left child
            //resize tracking index to half of parent, rounded up
            //This says that if our tree is not perfectly balanced,
            //then give one extra node to the left side of the tree.
            leafTracking[i] = leafTracking[tree[parent].t_id] -
                                (leafTracking[tree[parent].t_id] / 2);
		}else{//i is a right child
            //resize tracking index to half of parent, rounded down
            //This gives fewer branches to right side of tree overall.
            leafTracking[i] = (leafTracking[tree[parent].t_id] / 2);
        }
		if(leftChild >= startOfLastLayer){//children would be in last layer
			//if tree index i is a singleton node
			if(leafTracking[i] == 1){
                tree[i].t_id = i;
                tree[i].p_id = (i-1)/2;
                tree[i].l_id = -1;
                tree[i].r_id = -1;
			}else{//otherwise it has two singleton children
                tree[i].t_id = i;
                tree[i].p_id = (i-1)/2;
                //Set the left and right children 
                tree[i].l_id = lastLayerIdx;
                tree[i].r_id = lastLayerIdx+1;
                leftChild = tree[i].l_id;
                rightChild = tree[i].r_id;
				//now initialize its children
				//for this special case, set left and right child nodes
				//differently
                //Left node:
                tree[leftChild].t_id = leftChild;
                tree[leftChild].p_id = i;
                tree[leftChild].l_id = -1;
                tree[leftChild].r_id = -1;
                //right node
                tree[rightChild].t_id = rightChild;
                tree[rightChild].p_id = i;
                tree[rightChild].l_id = -1;
                tree[rightChild].r_id = -1;
                //Iterate the last layer index counter.
                lastLayerIdx += 2;
			}
		}else{//tree node is a node on the interior of the tree
            tree[i].t_id = i;
            tree[i].p_id = (i-1)/2;
            tree[i].l_id = leftChild;
            tree[i].r_id = rightChild;
		}
	}
    *treePtr = tree;
    free(leafTracking);
}

tsqr_tree* alloc_tsqr_tree(MPI_Comm comm){
    int i=0,j=0;
    int addrSize=0,offSet=0;
    int numLeaves;
    tsqr_tree* retn_tree;
    tree_node* tree;
    retn_tree = (tsqr_tree*) malloc(sizeof(*retn_tree));
    MPI_Comm_dup(comm, &(retn_tree->comm));
	MPI_Comm_size(retn_tree->comm, &numLeaves);
    tree_balanced_init(&(retn_tree->n),&(retn_tree->numNodes),numLeaves);
    retn_tree->leafIdx = (int*) malloc(sizeof(*retn_tree->leafIdx)*numLeaves);
    retn_tree->addr    = malloc(
                            sizeof(*(retn_tree->addr))*retn_tree->numNodes);
    retn_tree->Q  = (double**) malloc(
                            sizeof(*(retn_tree->Q))*retn_tree->numNodes);
    retn_tree->Q_memSize  = (int*) malloc(sizeof(*(retn_tree->Q_memSize))
                                                        *retn_tree->numNodes);
    retn_tree->R  = (double**) malloc(
                            sizeof(*(retn_tree->R))*retn_tree->numNodes);
    retn_tree->R_memSize  = (int*) malloc(sizeof(*(retn_tree->R_memSize))
                                                        *retn_tree->numNodes);
    retn_tree->tau = (double**) malloc(
                            sizeof(*(retn_tree->tau))*retn_tree->numNodes);
    retn_tree->tau_numel= (int*) malloc(
                          sizeof(*(retn_tree->tau_numel))*retn_tree->numNodes);
    retn_tree->q_numRow = (int*) malloc(
                          sizeof(*(retn_tree->q_numRow))*retn_tree->numNodes);
    retn_tree->q_numCol = (int*) malloc(
                          sizeof(*(retn_tree->q_numCol))*retn_tree->numNodes);
    retn_tree->r_numCol = (int*) malloc(
                          sizeof(*(retn_tree->r_numCol))*retn_tree->numNodes);
    retn_tree->flt_wrk = (double*) malloc(sizeof(double));
    retn_tree->wrk_sz  = 1;
    tree = retn_tree->n;
    for(i=0;i<retn_tree->numNodes;i++){
        retn_tree->Q[i] = NULL;
        retn_tree->Q_memSize[i] = 0;
        retn_tree->R[i] = NULL;
        retn_tree->R_memSize[i] = 0;
        retn_tree->tau[i] = NULL;
        retn_tree->tau_numel[i] = 0;
        retn_tree->q_numRow[i] = 0;
        retn_tree->q_numCol[i] = 0;
        retn_tree->r_numCol[i] = 0;
        if(isRoot(tree,i)){
            alloc_int_list(&(retn_tree->addr[i]),numLeaves);
            for(j=0;j<retn_tree->addr[i].len;j++){
                retn_tree->addr[i].arr[j] = j;
            }
        }else{
            addrSize = retn_tree->addr[tree[i].p_id].len;
            if(isLchild(tree,i)){
                offSet = 0;
                if(addrSize % 2 == 1){
                    addrSize = ((addrSize - 1)/2) + 1;
                }else{
                    addrSize = addrSize/2;
                }
            }else if(isRchild(tree,i)){
                if(addrSize % 2 == 1){
                    offSet = ((addrSize - 1)/2) + 1;
                    addrSize = (addrSize - 1)/2;
                }else{
                    offSet = addrSize/2;
                    addrSize = addrSize/2;
                }
            }
            alloc_int_list(&(retn_tree->addr[i]),addrSize);
            for(j=0;j<retn_tree->addr[i].len;j++){
                retn_tree->addr[i].arr[j] = j+
                              retn_tree->addr[tree[i].p_id].arr[offSet];
            }
        }
        if(isLeaf(tree,i)){
            retn_tree->leafIdx[retn_tree->addr[i].arr[0]] = i;
        }
    }
    return retn_tree;
}

void free_tsqr_tree(tsqr_tree* qrTree){
    int i;
    for(i=0;i<qrTree->numNodes;i++){
        free(qrTree->addr[i].arr);
        if(qrTree->Q_memSize[i] > 0){
            free(qrTree->Q[i]);
        }
        if(qrTree->R_memSize[i] > 0){
            free(qrTree->R[i]);
        }
        if(qrTree->tau_numel[i] > 0){
            free(qrTree->tau[i]);
        }
    }
    if(qrTree->wrk_sz > 0){
        free(qrTree->flt_wrk);
    }
    free(qrTree->tau_numel);
    free(qrTree->tau);
    free(qrTree->Q_memSize);
    free(qrTree->q_numCol);
    free(qrTree->q_numRow);
    free(qrTree->R_memSize);
    free(qrTree->r_numCol);
    free(qrTree->Q);
    free(qrTree->R);
    free(qrTree->addr);
    free(qrTree->leafIdx);
    free(qrTree->n);
    MPI_Comm_free(&(qrTree->comm));
    free(qrTree);
}

void fill_tsqr_tree(tsqr_tree* qrTree,
                    int_list numRow_distrib,
                    int numCol){
    int i;
    int numLeaves = qrTree->addr[0].len;
    int sumOfKids = 0;
    int* leafIdx = qrTree->leafIdx;
    for(i=0;i<numLeaves;i++){
        qrTree->q_numRow[leafIdx[i]] = numRow_distrib.arr[i];
        qrTree->q_numCol[leafIdx[i]] = (numRow_distrib.arr[i] < numCol) ?
                                        numRow_distrib.arr[i] : numCol;
        qrTree->r_numCol[leafIdx[i]] = numCol;
    }
    for(i=qrTree->numNodes-1;i>=0;i--){
        if(isLeaf(qrTree->n,i) == 0){
            sumOfKids = qrTree->q_numCol[qrTree->n[i].l_id] +
                        qrTree->q_numCol[qrTree->n[i].r_id];
            qrTree->q_numRow[i] = sumOfKids;
            qrTree->q_numCol[i] = (sumOfKids < numCol) ? sumOfKids : numCol;
            qrTree->r_numCol[i] = numCol;
        }
    }
}

void tsqr_factorize(tsqr_tree* qrTree, double* A_mpi_rnk){
    int mpi_rk; //go grab mpi rank.
    int n_idx, left_idx, right_idx, prev_idx;
    int right_mpi_rk, left_mpi_rk, min_rowcol;
    int send_size, recv_size, tri_size;
    int negative_one = -1;
    int lapack_info = 0;
    int lapack_wrk_sz = 0;
    //int j;
    MPI_Request send_req, recv_req;
    MPI_Status stat_recv, stat_send;
    prev_idx = -1;//initialize prev_idx as illegal indexing value.
	MPI_Comm_rank(qrTree->comm, &mpi_rk);
    for( n_idx  = qrTree->leafIdx[mpi_rk]; //starting at a leaf,
         n_idx != prev_idx;   //Root sets itself as parent, so go until repeat.
         n_idx  = qrTree->n[n_idx].p_id){ //go up to your parent.
        prev_idx = n_idx;
        if(isLeaf(qrTree->n,n_idx) == 0){
            left_idx  = qrTree->n[n_idx].l_id;
            right_idx = qrTree->n[n_idx].r_id;
            //get the child MPI ranks for communication.
            left_mpi_rk  = qrTree->addr[left_idx].arr[0];
            right_mpi_rk = qrTree->addr[right_idx].arr[0];
        }

#if defined(TSQR_DEBUG)
        for(int j=0;j<qrTree->addr[n_idx].len;j++){
            printf("%d",qrTree->addr[n_idx].arr[j]);
        }
        printf("\n");
getc(stdin);
#endif
        //If this node is going to do a QR factor
        if(mpi_rk == qrTree->addr[n_idx].arr[0]){
            realloc_to_max(&(qrTree->Q[n_idx]),
                            &(qrTree->Q_memSize[n_idx]),
                            sizeof(*(qrTree->Q[n_idx])),
                            qrTree->q_numRow[n_idx]*qrTree->r_numCol[n_idx]);
            tri_size = get_tri_pack_size(qrTree->q_numRow[n_idx],
                                      qrTree->r_numCol[n_idx], 'U');
            realloc_to_max(&(qrTree->R[n_idx]),
                            &(qrTree->R_memSize[n_idx]),
                            sizeof(*(qrTree->R[n_idx])),
                            tri_size);
            min_rowcol = (qrTree->r_numCol[n_idx] < qrTree->q_numRow[n_idx])?
                         (qrTree->r_numCol[n_idx]):(qrTree->q_numRow[n_idx]);
            realloc_to_max(&(qrTree->tau[n_idx]),
                            &(qrTree->tau_numel[n_idx]),
                            sizeof(*(qrTree->tau[n_idx])),
                            min_rowcol);
            if(isLeaf(qrTree->n, n_idx) == 1){
                //Then copy the input matrix A into the Q array.
                memcpy(qrTree->Q[n_idx],
                       A_mpi_rnk, 
                       sizeof(A_mpi_rnk[0])*
                       qrTree->q_numRow[n_idx]*
                       qrTree->r_numCol[n_idx]);
            }else{
                //Recv the right child R factor.
                recv_size = get_tri_pack_size(qrTree->q_numRow[right_idx],
                                      qrTree->r_numCol[right_idx], 'U');
                realloc_to_max(&(qrTree->R[right_idx]),
                               &(qrTree->R_memSize[right_idx]),
                                sizeof(*(qrTree->R[right_idx])),
                                recv_size);
                /*
                        EXECUTING  MPI RECV FROM right_mpi_rk
                */

#if defined(TSQR_DEBUG)
                printf("\n*********************************************\n"
                        "   PROC %d RECVING DE PROC %d. Utri Size: %d"
                        "\n*********************************************\n",
                        mpi_rk, right_mpi_rk, recv_size);
getc(stdin);
#endif
                MPI_Irecv(qrTree->R[right_idx], recv_size, MPI_DOUBLE,
                            right_mpi_rk, 0, qrTree->comm, &recv_req);
                //Copy the left child R factor and the right child
                //R factor into a vertically concatenated block array.
                //This array is located at qrTree->Q[n_idx]
                upper_tri_unpack(qrTree->Q[n_idx],//double* dest,
                                 qrTree->q_numRow[n_idx],
                                 qrTree->q_numCol[left_idx],
                                 qrTree->r_numCol[n_idx],
                                 qrTree->R[left_idx]);
                /*
                    EXECUTING MPI WAIT FROM RECV EXECUTED ABOVE.
                */
                MPI_Wait(&recv_req, &stat_recv);
                upper_tri_unpack(&(
                                  qrTree->Q[n_idx][qrTree->q_numCol[left_idx]]
                                 ),
                                 qrTree->q_numRow[n_idx],
                                 qrTree->q_numCol[right_idx],
                                 qrTree->r_numCol[n_idx],
                                 qrTree->R[right_idx]);
                
            }
            //Do a LAPACK call on the array qrTree->Q[n_idx]
            //storing the result in the n_idx Q and R arrays.
            dgeqrf_(&(qrTree->q_numRow[n_idx]),
                    &(qrTree->r_numCol[n_idx]),
                    qrTree->Q[n_idx],
                    &(qrTree->q_numRow[n_idx]),
                    qrTree->tau[n_idx],
                    qrTree->flt_wrk,
                    &negative_one,
                    &lapack_info);
            lapack_wrk_sz = (int) qrTree->flt_wrk[0];
            realloc_to_max( &(qrTree->flt_wrk),
                        &(qrTree->wrk_sz),
                        sizeof(*(qrTree->flt_wrk)),
                        lapack_wrk_sz);
            dgeqrf_(&(qrTree->q_numRow[n_idx]),
                    &(qrTree->r_numCol[n_idx]),
                    qrTree->Q[n_idx],
                    &(qrTree->q_numRow[n_idx]),
                    qrTree->tau[n_idx],
                    qrTree->flt_wrk,
                    &lapack_wrk_sz,
                    &lapack_info);
#if defined(TSQR_DEBUG)
            printf("\nTripack size = %d\n",
#endif
            upper_tri_pack(qrTree->R[n_idx],
                             qrTree->q_numRow[n_idx],
                             qrTree->r_numCol[n_idx],
                             qrTree->Q[n_idx],
                             qrTree->q_numRow[n_idx])
#if defined(TSQR_DEBUG)
            )
#endif
                                                     ;
        }else if(mpi_rk == right_mpi_rk){//if we are the same proc as right kid
            //Then we send the R factor to the processor which holds
            //the left child.
            send_size = get_tri_pack_size(qrTree->q_numRow[right_idx],
                                      qrTree->r_numCol[right_idx], 'U');
            /*
                    EXECUTING MPI SEND TO left_mpi_rk
            */
#if defined(TSQR_DEBUG)
            printf("\n*********************************************\n"
                   "   PROC %d SENDING TO PROC %d. Utri Size: %d"
                   "\n*********************************************\n",
                    mpi_rk, left_mpi_rk, send_size);
getc(stdin);
#endif
            if(right_idx >= 0){
                MPI_Isend(qrTree->R[right_idx], send_size, MPI_DOUBLE,
                            left_mpi_rk, 0, qrTree->comm, &send_req);
                MPI_Wait(&send_req, &stat_send);
            }
        }
    }
    send_size = get_tri_pack_size(qrTree->q_numRow[0],
                                  qrTree->r_numCol[0],'U');
    if(mpi_rk != qrTree->addr[0].arr[0]){
        realloc_to_max( &(qrTree->R[0]),
                        &(qrTree->R_memSize[0]),
                        sizeof(*(qrTree->R[0])),
                        send_size);
    }
    MPI_Bcast(qrTree->R[0], send_size, MPI_DOUBLE,
                qrTree->addr[0].arr[0], qrTree->comm);

#if defined(TSQR_DEBUG)    
    printf("\n*********************************************\n"
           "             PROC %d DONE WITH TSQR."
           "\n*********************************************\n",mpi_rk);
getc(stdin);
#endif
}

void tsqr_make_q(tsqr_tree* qrTree, double* Q_mpi_rnk){
    int mpi_rk; //go grab mpi rank.
    int n_idx, left_idx, right_idx,ldr,ldw,offSet,root_idx;
    int right_mpi_rk, left_sibling_rk;
    int i, j;
    int send_size, recv_size;
    int negative_one = -1;
    int lapack_info = 0;
    int lapack_wrk_sz = 0;
    char side_left = 'L';
    char no_trans  = 'N';
    MPI_Request send_req, recv_req;
    MPI_Status stat_recv, stat_send;
	MPI_Comm_rank(qrTree->comm, &mpi_rk);
	n_idx = 0;//initialize to root index.
	root_idx = 0;
    left_idx  = qrTree->n[n_idx].l_id;
    right_idx = qrTree->n[n_idx].r_id;
    if(right_idx >= 0){
        right_mpi_rk = qrTree->addr[right_idx].arr[0];
    }
    
#if defined(TSQR_DEBUG)
    printf("\n*********************************************\n"
           "             PROC %d STARTING MAKE Q."
           "\n*********************************************\n",mpi_rk);
getc(stdin);
#endif
	if(mpi_rk == qrTree->addr[n_idx].arr[0]){
        dorgqr_(&(qrTree->q_numRow[n_idx]),
                &(qrTree->q_numCol[n_idx]),
                &(qrTree->q_numCol[n_idx]),
                  qrTree->Q[n_idx],
                &(qrTree->q_numRow[n_idx]),
                  qrTree->tau[n_idx],
                  qrTree->flt_wrk,
                &negative_one,
                &lapack_info);
        lapack_wrk_sz = (int) qrTree->flt_wrk[0];
        realloc_to_max( &(qrTree->flt_wrk),
                        &(qrTree->wrk_sz),
                        sizeof(*(qrTree->flt_wrk)),
                        qrTree->q_numRow[n_idx]*qrTree->q_numCol[n_idx]
                        + lapack_wrk_sz);
        memcpy(qrTree->flt_wrk,qrTree->Q[n_idx], 
                       sizeof(*(qrTree->flt_wrk))*
                       qrTree->q_numRow[n_idx]*
                       qrTree->q_numCol[n_idx]);
        /*
            Do a lapack call to set up in-place Q construction in the
            flt_wrk array.
        */
        dorgqr_(&(qrTree->q_numRow[n_idx]),
                &(qrTree->q_numCol[n_idx]),
                &(qrTree->q_numCol[n_idx]),
                  qrTree->flt_wrk,
                &(qrTree->q_numRow[n_idx]),
                  qrTree->tau[n_idx],
                &(qrTree->flt_wrk[qrTree->q_numRow[n_idx]*
                                  qrTree->q_numCol[n_idx]]),
                &lapack_wrk_sz,
                &lapack_info);
                
        if(right_idx >= 0){
            /*
                Pack the bottom half of that product into a rectangular matrix.
            */
            send_size = qrTree->q_numCol[right_idx]*qrTree->q_numCol[root_idx];
            realloc_to_max( &(qrTree->R[right_idx]),
                            &(qrTree->R_memSize[right_idx]),
                            sizeof(*(qrTree->R[right_idx])),
                            send_size);
            offSet = qrTree->q_numCol[left_idx];
            ldr = qrTree->q_numCol[right_idx];
            ldw = qrTree->q_numRow[n_idx];
            for(j=0;j<qrTree->q_numCol[root_idx];j++){
                for(i=0;i<qrTree->q_numCol[right_idx];i++){
                    qrTree->R[right_idx][i+ldr*j] = qrTree->flt_wrk[i+offSet+ldw*j];
                }
            }
        /*
            MPI Send the bottom half to the right child.
        */
#if defined(TSQR_DEBUG)
        printf("\n*********************************************\n"
                  "   PROC %d SENDING TO PROC %d. Send Size: %d"
               "\n*********************************************\n",
                    mpi_rk, right_mpi_rk, send_size);
getc(stdin);
#endif
            MPI_Isend(qrTree->R[right_idx], send_size, MPI_DOUBLE,
                        right_mpi_rk, 0, qrTree->comm, &send_req);
            
            /*
                Pack the top half of that product into a rectangular matrix.
            */
            send_size = qrTree->q_numRow[left_idx]*qrTree->q_numCol[root_idx];
            realloc_to_max( &(qrTree->R[left_idx]),
                            &(qrTree->R_memSize[left_idx]),
                            sizeof(*(qrTree->R[left_idx])),
                            send_size);
            ldr = qrTree->q_numRow[left_idx];
            ldw = qrTree->q_numRow[n_idx];
            for(j=0;j<qrTree->q_numCol[root_idx];j++){
                for(i=0;i<qrTree->q_numRow[left_idx];i++){
                    if(i < qrTree->q_numCol[left_idx]){
                        qrTree->R[left_idx][i+ldr*j] = qrTree->flt_wrk[i+ldw*j];
                    }else{
                        qrTree->R[left_idx][i+ldr*j] = 0.0;
                    }
                }
            }
#if defined(TSQR_DEBUG)
        printf("\nleft_idx = %d\n",left_idx);
getc(stdin);
#endif
            /*
                Wait for MPI Send to finish.
            */
            MPI_Wait(&send_req, &stat_send);
        }
	}
	while(qrTree->numNodes > 1){//Begin looping
        //Start with iterator increment step.
        if(mpi_rk <= qrTree->addr[left_idx].arr[qrTree->addr[left_idx].len-1]){
            n_idx = left_idx;
        }else{
            n_idx = right_idx;
        }
#if defined(TSQR_DEBUG)
        printf("\n*********************************************\n"
                 "   PROC %d AT NODE %d"
               "\n*********************************************\n",
               mpi_rk,n_idx);
        for(j=0;j<qrTree->addr[n_idx].len;j++){
            printf("%d",qrTree->addr[n_idx].arr[j]);
        }
        printf("\n");
getc(stdin);
#endif
	    //this the index value of a leaf's left and right child.
        left_idx  = qrTree->n[n_idx].l_id;
        right_idx = qrTree->n[n_idx].r_id;
        if(isLeaf(qrTree->n,n_idx) == 0){//When n_idx is not a leaf
            right_mpi_rk = qrTree->addr[right_idx].arr[0];
        }else{
            right_mpi_rk = -1;
        }
        left_sibling_rk = qrTree->addr[qrTree->n[n_idx].p_id].arr[0];
        if(qrTree->addr[n_idx].arr[0] == mpi_rk){//If this proc is constructing Q
#if defined(TSQR_DEBUG)
            printf("\n*********************************************\n"
                   "   PROC %d MATCHES ADDRESS[0]."
                   "\n*********************************************\n",
                        mpi_rk);
getc(stdin);
#endif
            /*
                MPI Recv the right matrix of upcoming product 
            */
            if(isRchild(qrTree->n, n_idx) == 1){
                recv_size = qrTree->q_numCol[n_idx] * 
                            qrTree->q_numCol[root_idx];
                realloc_to_max( &(qrTree->flt_wrk),
                                &(qrTree->wrk_sz),
                                sizeof(*(qrTree->flt_wrk)),
                                recv_size);
#if defined(TSQR_DEBUG)
                printf("\n*********************************************\n"
                         "   PROC %d RECVING DE PROC %d. Recv Size: %d"
                       "\n*********************************************\n",
                        mpi_rk, left_sibling_rk, recv_size);
getc(stdin);
#endif
                MPI_Irecv(qrTree->flt_wrk, recv_size, MPI_DOUBLE,
                            left_sibling_rk, 0, qrTree->comm, &recv_req);
                MPI_Wait(&recv_req, &stat_recv);
                recv_size = qrTree->q_numRow[n_idx] * 
                            qrTree->q_numCol[root_idx];
                realloc_to_max( &(qrTree->R[n_idx]),
                                &(qrTree->R_memSize[n_idx]),
                                sizeof(*(qrTree->R[n_idx])),
                                recv_size);
                ldw = qrTree->q_numCol[n_idx];
                ldr = qrTree->q_numRow[n_idx];
                for(j=0;j<qrTree->q_numCol[root_idx];j++){
                    for(i=0;i<qrTree->q_numRow[n_idx];i++){
                        if(i < qrTree->q_numCol[n_idx]){
                            qrTree->R[n_idx][i+ldr*j] = qrTree->flt_wrk[i+ldw*j];
                        }else{
                            qrTree->R[n_idx][i+ldr*j] = 0.0;
                        }
                    }
                }
            }
#if defined(TSQR_DEBUG)
            printf("\n*********************************************\n"
                   "   PROC %d STARTING DORMQR."
                   "\n*********************************************\n",
                        mpi_rk);
getc(stdin);
#endif
            /*
                Multiply the packed HH Q factor into the recv'd factor.
            */
            dormqr_(&side_left,
                    &no_trans,
                    &(qrTree->q_numRow[n_idx]),
                    &(qrTree->q_numCol[root_idx]),
                    &(qrTree->q_numCol[n_idx]),
                      qrTree->Q[n_idx],
                    &(qrTree->q_numRow[n_idx]),
                      qrTree->tau[n_idx],
                      qrTree->R[n_idx],
                    &(qrTree->q_numRow[n_idx]),
                      qrTree->flt_wrk,
                    &negative_one,
                    &lapack_info);
            lapack_wrk_sz = (int) qrTree->flt_wrk[0];
            realloc_to_max( &(qrTree->flt_wrk),
                            &(qrTree->wrk_sz),
                            sizeof(*(qrTree->flt_wrk)),
                            lapack_wrk_sz);
            dormqr_(&side_left,
                    &no_trans,
                    &(qrTree->q_numRow[n_idx]),
                    &(qrTree->q_numCol[root_idx]),
                    &(qrTree->q_numCol[n_idx]),
                      qrTree->Q[n_idx],
                    &(qrTree->q_numRow[n_idx]),
                      qrTree->tau[n_idx],
                      qrTree->R[n_idx],
                    &(qrTree->q_numRow[n_idx]),
                      qrTree->flt_wrk,
                    &lapack_wrk_sz,
                    &lapack_info);
            if(isLeaf(qrTree->n,n_idx) == 0){
                /*
                    Pack the bottom half of that product into a rectangular
                    matrix.
                */
                send_size = qrTree->q_numCol[right_idx]*
                            qrTree->q_numCol[root_idx];
                realloc_to_max( &(qrTree->R[right_idx]),
                                &(qrTree->R_memSize[right_idx]),
                                sizeof(*(qrTree->R[right_idx])),
                                send_size);
                offSet = qrTree->q_numCol[left_idx];
                ldr = qrTree->q_numCol[right_idx];
                ldw = qrTree->q_numRow[n_idx];
                for(j=0;j<qrTree->q_numCol[root_idx];j++){
                    for(i=0;i<qrTree->q_numCol[right_idx];i++){
                        qrTree->R[right_idx][i+ldr*j] = 
                                               qrTree->R[n_idx][i+offSet+ldw*j];
                    }
                }
                /*
                    MPI Send the bottom half to the right child if this
                    node is not a leaf.
                */
#if defined(TSQR_DEBUG)
                printf("\n*********************************************\n"
                         "   PROC %d SENDING TO PROC %d. Send Size: %d"
                       "\n*********************************************\n",
                    mpi_rk, right_mpi_rk, send_size);
getc(stdin);
#endif
                if(right_idx >= 0){
                    MPI_Isend(qrTree->R[right_idx], send_size, MPI_DOUBLE,
                                right_mpi_rk, 0, qrTree->comm, &send_req);
                
                    /*
                        Pack the top half of that product into a rectangular matrix.
                    */
                    send_size = qrTree->q_numRow[left_idx]*
                                qrTree->q_numCol[root_idx];
                    realloc_to_max( &(qrTree->R[left_idx]),
                                    &(qrTree->R_memSize[left_idx]),
                                    sizeof(*(qrTree->R[left_idx])),
                                    send_size);
                    ldr = qrTree->q_numRow[left_idx];
                    ldw = qrTree->q_numRow[n_idx];
                    for(j=0;j<qrTree->q_numCol[root_idx];j++){
                        for(i=0;i<qrTree->q_numRow[left_idx];i++){
                            if(i < qrTree->q_numCol[left_idx]){
                                qrTree->R[left_idx][i+ldr*j] =
                                                  qrTree->R[n_idx][i+ldw*j];
                            }else{
                                qrTree->R[left_idx][i+ldr*j] = 0.0;
                            }
                        }
                    }
                    /*
                        Wait for MPI Send to finish.
                    */
                    MPI_Wait(&send_req, &stat_send);
                }
            }
        }
        if(isLeaf(qrTree->n,n_idx) == 1){//Break when n_idx is a leaf
            break;
        }
    }
#if defined(TSQR_DEBUG)
    printf("\n*********************************************\n"
             "   PROC %d OUT OF TREE LOOP"
           "\n*********************************************\n",
           mpi_rk);
    /*
        Multiply the packed HH Q factor into the recv'd factor R
    */
    printf("\n*********************************************\n"
           "   PROC %d STARTING DORMQR."
           "\n*********************************************\n",
                mpi_rk);
getc(stdin);
#endif
    //finish off the Q by saving  Q_mpi_rnk with the final matrix.
    if(qrTree->numNodes == 1){
        memcpy(Q_mpi_rnk,qrTree->flt_wrk, 
               sizeof(*Q_mpi_rnk)*
               qrTree->q_numRow[root_idx]*
               qrTree->q_numCol[root_idx]);
    }else{
        memcpy(Q_mpi_rnk,qrTree->R[n_idx], 
                   sizeof(*Q_mpi_rnk)*
                   qrTree->q_numRow[n_idx]*
                   qrTree->q_numCol[root_idx]);
    }
}


void wflq_factorize(tsqr_tree* qrTree, double* A_mpi_rnk){
    int mpi_rk; //go grab mpi rank.
    int n_idx, left_idx, right_idx, prev_idx;
    int right_mpi_rk, left_mpi_rk, min_rowcol;
    int send_size, recv_size, tri_size;
    int negative_one = -1;
    int lapack_info = 0;
    int lapack_wrk_sz = 0;
    MPI_Request send_req, recv_req;
    MPI_Status stat_recv, stat_send;
    prev_idx = -1;//initialize prev_idx as illegal indexing value.
	MPI_Comm_rank(qrTree->comm, &mpi_rk);
    for( n_idx  = qrTree->leafIdx[mpi_rk]; //starting at a leaf,
         n_idx != prev_idx;   //Root sets itself as parent, so go until repeat.
         n_idx  = qrTree->n[n_idx].p_id){ //go up to your parent.
        prev_idx = n_idx;
        if(isLeaf(qrTree->n,n_idx) == 0){
            left_idx  = qrTree->n[n_idx].l_id;
            right_idx = qrTree->n[n_idx].r_id;
            //get the child MPI ranks for communication.
            left_mpi_rk  = qrTree->addr[left_idx].arr[0];
            right_mpi_rk = qrTree->addr[right_idx].arr[0];
        }

#if defined(TSQR_DEBUG)
        for(int j=0;j<qrTree->addr[n_idx].len;j++){
            printf("%d",qrTree->addr[n_idx].arr[j]);
        }
        printf("\n");
        getc(stdin);
#endif
        //If this node is going to do a QR factor
        if(mpi_rk == qrTree->addr[n_idx].arr[0]){
            realloc_to_max(&(qrTree->Q[n_idx]),
                            &(qrTree->Q_memSize[n_idx]),
                            sizeof(*(qrTree->Q[n_idx])),
                            qrTree->q_numRow[n_idx]*qrTree->r_numCol[n_idx]);
            tri_size = get_tri_pack_size(qrTree->r_numCol[n_idx],
                                        qrTree->q_numRow[n_idx], 'L');
            realloc_to_max(&(qrTree->R[n_idx]),
                            &(qrTree->R_memSize[n_idx]),
                            sizeof(*(qrTree->R[n_idx])),
                            tri_size);
            min_rowcol = (qrTree->r_numCol[n_idx] < qrTree->q_numRow[n_idx])?
                         (qrTree->r_numCol[n_idx]):(qrTree->q_numRow[n_idx]);
            realloc_to_max(&(qrTree->tau[n_idx]),
                            &(qrTree->tau_numel[n_idx]),
                            sizeof(*(qrTree->tau[n_idx])),
                            min_rowcol);
            if(isLeaf(qrTree->n, n_idx) == 1){
                //Then copy the input matrix A into the Q array.
                memcpy(qrTree->Q[n_idx],
                       A_mpi_rnk, 
                       sizeof(A_mpi_rnk[0])*
                       qrTree->q_numRow[n_idx]*
                       qrTree->r_numCol[n_idx]);
            }else{
                //Recv the right child L factor.
                recv_size = get_tri_pack_size(qrTree->r_numCol[right_idx],
                                        qrTree->q_numRow[right_idx], 'L');
                realloc_to_max(&(qrTree->R[right_idx]),
                               &(qrTree->R_memSize[right_idx]),
                                sizeof(*(qrTree->R[right_idx])),
                                recv_size);
                /*
                        EXECUTING  MPI RECV FROM right_mpi_rk
                */

#if defined(TSQR_DEBUG)
                printf("\n*********************************************\n"
                        "   PROC %d RECVING DE PROC %d. Utri Size: %d"
                        "\n*********************************************\n",
                        mpi_rk, right_mpi_rk, recv_size);
                getc(stdin);
#endif
                MPI_Irecv(qrTree->R[right_idx], recv_size, MPI_DOUBLE,
                            right_mpi_rk, 0, qrTree->comm, &recv_req);
                //Copy the left child R factor and the right child
                //R factor into a vertically concatenated block array.
                //This array is located at qrTree->Q[n_idx]
                lower_tri_unpack(qrTree->Q[n_idx],//double* dest,
                                 qrTree->r_numCol[n_idx],
                                 qrTree->r_numCol[n_idx],
                                 qrTree->q_numCol[left_idx],
                                 qrTree->R[left_idx]);
                /*
                    EXECUTING MPI WAIT FROM RECV EXECUTED ABOVE.
                */
                MPI_Wait(&recv_req, &stat_recv);
                lower_tri_unpack(&(
                                  qrTree->Q[n_idx][
                                            qrTree->r_numCol[n_idx]*
                                            qrTree->q_numCol[left_idx]]
                                 ),
                                 qrTree->r_numCol[n_idx],
                                 qrTree->r_numCol[n_idx],
                                 qrTree->q_numCol[right_idx],
                                 qrTree->R[right_idx]);
                
            }
            //Do a LAPACK call on the array qrTree->Q[n_idx]
            //storing the result in the n_idx Q and R arrays.
            dgelqf_(&(qrTree->r_numCol[n_idx]),
                    &(qrTree->q_numRow[n_idx]),
                      qrTree->Q[n_idx],
                    &(qrTree->r_numCol[n_idx]),
                      qrTree->tau[n_idx],
                      qrTree->flt_wrk,
                    &negative_one,
                    &lapack_info);
            lapack_wrk_sz = (int) qrTree->flt_wrk[0];
            realloc_to_max( &(qrTree->flt_wrk),
                            &(qrTree->wrk_sz),
                            sizeof(*(qrTree->flt_wrk)),
                            lapack_wrk_sz);
            dgelqf_(&(qrTree->r_numCol[n_idx]),
                    &(qrTree->q_numRow[n_idx]),
                      qrTree->Q[n_idx],
                    &(qrTree->r_numCol[n_idx]),
                      qrTree->tau[n_idx],
                      qrTree->flt_wrk,
                    &lapack_wrk_sz,
                    &lapack_info);
#if defined(TSQR_DEBUG)
            printf("\nTripack size = %d\n",
#endif
            lower_tri_pack(qrTree->R[n_idx],
                             qrTree->r_numCol[n_idx],
                             qrTree->q_numRow[n_idx],
                             qrTree->Q[n_idx],
                             qrTree->r_numCol[n_idx])
#if defined(TSQR_DEBUG)
            )
#endif
                                                     ;
        }else if(mpi_rk == right_mpi_rk){//if we are the same proc as right kid
            //Then we send the R factor to the processor which holds
            //the left child.
            send_size = get_tri_pack_size(qrTree->r_numCol[right_idx],
                                        qrTree->q_numRow[right_idx], 'L');
            /*
                    EXECUTING MPI SEND TO left_mpi_rk
            */
#if defined(TSQR_DEBUG)
            printf("\n*********************************************\n"
                   "   PROC %d SENDING TO PROC %d. Utri Size: %d"
                   "\n*********************************************\n",
                    mpi_rk, left_mpi_rk, send_size);
            getc(stdin);
#endif
            if(right_idx >= 0){
                MPI_Isend(qrTree->R[right_idx], send_size, MPI_DOUBLE,
                            left_mpi_rk, 0, qrTree->comm, &send_req);
                MPI_Wait(&send_req, &stat_send);
            }
        }
    }
    send_size = get_tri_pack_size(qrTree->r_numCol[0],
                                  qrTree->q_numRow[0],'L');
    if(mpi_rk != qrTree->addr[0].arr[0]){
        realloc_to_max( &(qrTree->R[0]),
                        &(qrTree->R_memSize[0]),
                        sizeof(*(qrTree->R[0])),
                        send_size);
    }
    MPI_Bcast(qrTree->R[0], send_size, MPI_DOUBLE,
                qrTree->addr[0].arr[0], qrTree->comm);

#if defined(TSQR_DEBUG)    
    printf("\n*********************************************\n"
           "             PROC %d DONE WITH WFLQ."
           "\n*********************************************\n",mpi_rk);
    getc(stdin);
#endif
}


void wflq_make_q(tsqr_tree* qrTree, double* Q_mpi_rnk){
    int mpi_rk; //go grab mpi rank.
    int n_idx, left_idx, right_idx,ldr,ldw,offSet,root_idx;
    int right_mpi_rk, left_sibling_rk;
    int i, j;
    int send_size, recv_size;
    int negative_one = -1;
    int lapack_info = 0;
    int lapack_wrk_sz = 0;
    char right_side = 'R';
    char no_trans = 'N';
    MPI_Request send_req, recv_req;
    MPI_Status stat_recv, stat_send;
	MPI_Comm_rank(qrTree->comm, &mpi_rk);
	n_idx = 0;//initialize to root index.
	root_idx = 0;
    left_idx  = qrTree->n[n_idx].l_id;
    right_idx = qrTree->n[n_idx].r_id;
    if(right_idx >= 0){
        right_mpi_rk = qrTree->addr[right_idx].arr[0];
    }
    
#if defined(TSQR_DEBUG)
    printf("\n*********************************************\n"
           "             PROC %d STARTING MAKE Q."
           "\n*********************************************\n",mpi_rk);
getc(stdin);
#endif
	if(mpi_rk == qrTree->addr[n_idx].arr[0]){
	    dorglq_(&(qrTree->q_numCol[n_idx]),
                &(qrTree->q_numRow[n_idx]),
                &(qrTree->q_numCol[n_idx]),
                  qrTree->Q[n_idx],
                &(qrTree->r_numCol[n_idx]),
                  qrTree->tau[n_idx],
                  qrTree->flt_wrk,
                &negative_one,
                &lapack_info);
        lapack_wrk_sz = (int) qrTree->flt_wrk[0];
        realloc_to_max( &(qrTree->flt_wrk),
                        &(qrTree->wrk_sz),
                        sizeof(*(qrTree->flt_wrk)),
                        qrTree->q_numRow[n_idx]*qrTree->r_numCol[n_idx]
                        + lapack_wrk_sz);
        memcpy(qrTree->flt_wrk,qrTree->Q[n_idx], 
                       sizeof(*(qrTree->flt_wrk))*
                       qrTree->q_numRow[n_idx]*
                       qrTree->r_numCol[n_idx]);
        /*
            Do a lapack call to set up in-place Q construction in the
            flt_wrk array.
        */
        dorglq_(&(qrTree->q_numCol[n_idx]),
                &(qrTree->q_numRow[n_idx]),
                &(qrTree->q_numCol[n_idx]),
                  qrTree->flt_wrk,
                &(qrTree->r_numCol[n_idx]),
                  qrTree->tau[n_idx],
                &(qrTree->flt_wrk[qrTree->q_numRow[n_idx]*
                                  qrTree->r_numCol[n_idx]]),
                &lapack_wrk_sz,
                &lapack_info);
        /*
            Pack the right half of that product into a rectangular matrix.
        */
        if(right_idx >= 0){
            send_size = qrTree->q_numCol[right_idx]*qrTree->q_numCol[root_idx];
            realloc_to_max( &(qrTree->R[right_idx]),
                            &(qrTree->R_memSize[right_idx]),
                            sizeof(*(qrTree->R[right_idx])),
                            send_size);
            offSet = qrTree->q_numCol[left_idx];
            ldr = qrTree->q_numCol[root_idx];
            ldw = qrTree->r_numCol[root_idx];
            for(i=0;i<qrTree->q_numCol[right_idx];i++){
                for(j=0;j<qrTree->q_numCol[root_idx];j++){
                    qrTree->R[right_idx][j+ldr*i] = 
                                            qrTree->flt_wrk[j+ldw*(i+offSet)];
                }
            }
            /*
                MPI Send the bottom half to the right child.
            */
#if defined(TSQR_DEBUG)
        printf("\n*********************************************\n"
                  "   PROC %d SENDING TO PROC %d. Send Size: %d"
               "\n*********************************************\n",
                    mpi_rk, right_mpi_rk, send_size);
getc(stdin);
#endif
            MPI_Isend(qrTree->R[right_idx], send_size, MPI_DOUBLE,
                        right_mpi_rk, 0, qrTree->comm, &send_req);
            /*
                Pack the top half of that product into a rectangular matrix.
            */
            send_size = qrTree->q_numRow[left_idx]*qrTree->q_numCol[root_idx];
            realloc_to_max( &(qrTree->R[left_idx]),
                            &(qrTree->R_memSize[left_idx]),
                            sizeof(*(qrTree->R[left_idx])),
                            send_size);
            ldr = qrTree->q_numCol[root_idx];
            ldw = qrTree->r_numCol[root_idx];
            for(i=0;i<qrTree->q_numRow[left_idx];i++){
                for(j=0;j<qrTree->q_numCol[root_idx];j++){
                    if(i < qrTree->q_numCol[left_idx]){
                        qrTree->R[left_idx][j+ldr*i] = qrTree->flt_wrk[j+ldw*i];
                    }else{
                        qrTree->R[left_idx][j+ldr*i] = 0.0;
                    }
                }
            }
#if defined(TSQR_DEBUG)
        printf("\nleft_idx = %d\n",left_idx);
getc(stdin);
#endif
            /*
                Wait for MPI Send to finish.
            */
            MPI_Wait(&send_req, &stat_send);
        }
	}
	while(qrTree->numNodes > 1){//Begin looping
        //Start with iterator increment step.
        if(mpi_rk <= qrTree->addr[left_idx].arr[qrTree->addr[left_idx].len-1]){
            n_idx = left_idx;
        }else{
            n_idx = right_idx;
        }
#if defined(TSQR_DEBUG)
        printf("\n*********************************************\n"
                 "   PROC %d AT NODE %d"
               "\n*********************************************\n",
               mpi_rk,n_idx);
        for(j=0;j<qrTree->addr[n_idx].len;j++){
            printf("%d",qrTree->addr[n_idx].arr[j]);
        }
        printf("\n");
getc(stdin);
#endif
	    //this the index value of a leaf's left and right child.
        left_idx  = qrTree->n[n_idx].l_id;
        right_idx = qrTree->n[n_idx].r_id;
        if(isLeaf(qrTree->n,n_idx) == 0){//When n_idx is not a leaf
            right_mpi_rk = qrTree->addr[right_idx].arr[0];
        }else{
            right_mpi_rk = -1;
        }
        left_sibling_rk = qrTree->addr[qrTree->n[n_idx].p_id].arr[0];
        if(qrTree->addr[n_idx].arr[0] == mpi_rk){//If this proc is constructing Q
#if defined(TSQR_DEBUG)
            printf("\n*********************************************\n"
                   "   PROC %d MATCHES ADDRESS[0]."
                   "\n*********************************************\n",
                        mpi_rk);
getc(stdin);
#endif
            /*
                MPI Recv the right matrix of upcoming product 
            */
            if(isRchild(qrTree->n, n_idx) == 1){
                recv_size = qrTree->q_numCol[n_idx] * 
                            qrTree->q_numCol[root_idx];
                realloc_to_max( &(qrTree->flt_wrk),
                                &(qrTree->wrk_sz),
                                sizeof(*(qrTree->flt_wrk)),
                                recv_size);
#if defined(TSQR_DEBUG)
                printf("\n*********************************************\n"
                         "   PROC %d RECVING DE PROC %d. Recv Size: %d"
                       "\n*********************************************\n",
                        mpi_rk, left_sibling_rk, recv_size);
getc(stdin);
#endif
                MPI_Irecv(qrTree->flt_wrk, recv_size, MPI_DOUBLE,
                            left_sibling_rk, 0, qrTree->comm, &recv_req);
                MPI_Wait(&recv_req, &stat_recv);
                recv_size = qrTree->q_numRow[n_idx] * 
                            qrTree->q_numCol[root_idx];
                realloc_to_max( &(qrTree->R[n_idx]),
                                &(qrTree->R_memSize[n_idx]),
                                sizeof(*(qrTree->R[n_idx])),
                                recv_size);
                ldw = qrTree->q_numCol[root_idx];
                ldr = qrTree->q_numCol[root_idx];
                for(i=0;i<qrTree->q_numRow[n_idx];i++){
                    for(j=0;j<qrTree->q_numCol[root_idx];j++){
                        if(i < qrTree->q_numCol[n_idx]){
                            qrTree->R[n_idx][j+ldr*i] = qrTree->flt_wrk[j+ldw*i];
                        }else{
                            qrTree->R[n_idx][j+ldr*i] = 0.0;
                        }
                    }
                }
            }
#if defined(TSQR_DEBUG)
            printf("\n*********************************************\n"
                   "   PROC %d STARTING DORMQR."
                   "\n*********************************************\n",
                        mpi_rk);
getc(stdin);
#endif
            /*
                Multiply the packed HH Q factor into the recv'd factor.
            */
            dormlq_(&right_side,
                    &no_trans,
                    &(qrTree->q_numCol[root_idx]),
                    &(qrTree->q_numRow[n_idx]),
                    &(qrTree->q_numCol[n_idx]),
                      qrTree->Q[n_idx],
                    &(qrTree->r_numCol[n_idx]),
                      qrTree->tau[n_idx],
                      qrTree->R[n_idx],
                    &(qrTree->q_numCol[root_idx]),
                      qrTree->flt_wrk,
                    &negative_one,
                    &lapack_info);
            lapack_wrk_sz = (int) qrTree->flt_wrk[0];
            realloc_to_max(&(qrTree->flt_wrk),
                           &(qrTree->wrk_sz),
                           sizeof(*(qrTree->flt_wrk)),
                           lapack_wrk_sz);
            dormlq_(&right_side,
                    &no_trans,
                    &(qrTree->q_numCol[root_idx]),
                    &(qrTree->q_numRow[n_idx]),
                    &(qrTree->q_numCol[n_idx]),
                      qrTree->Q[n_idx],
                    &(qrTree->r_numCol[n_idx]),
                      qrTree->tau[n_idx],
                      qrTree->R[n_idx],
                    &(qrTree->q_numCol[root_idx]),
                      qrTree->flt_wrk,
                    &lapack_wrk_sz,
                    &lapack_info);
            if(isLeaf(qrTree->n,n_idx) == 0){
                /*
                    Pack the right half of that product into a rectangular
                    matrix.
                */
                if(right_idx >= 0){
                    send_size = qrTree->q_numCol[right_idx]*
                                qrTree->q_numCol[root_idx];
                    realloc_to_max( &(qrTree->R[right_idx]),
                                    &(qrTree->R_memSize[right_idx]),
                                    sizeof(*(qrTree->R[right_idx])),
                                    send_size);
                    offSet = qrTree->q_numCol[left_idx];
                    ldr = qrTree->q_numCol[root_idx];
                    ldw = qrTree->q_numCol[root_idx];
                    for(i=0;i<qrTree->q_numCol[right_idx];i++){
                        for(j=0;j<qrTree->q_numCol[root_idx];j++){
                            qrTree->R[right_idx][j+ldr*i] = 
                                                qrTree->R[n_idx][j+(i+offSet)*ldw];
                        }
                    }
                    /*
                        MPI Send the bottom half to the right child if this
                        node is not a leaf.
                    */
#if defined(TSQR_DEBUG)
                printf("\n*********************************************\n"
                         "   PROC %d SENDING TO PROC %d. Send Size: %d"
                       "\n*********************************************\n",
                    mpi_rk, right_mpi_rk, send_size);
getc(stdin);
#endif
                    MPI_Isend(qrTree->R[right_idx], send_size, MPI_DOUBLE,
                                right_mpi_rk, 0, qrTree->comm, &send_req);
                    /*
                        Pack the top half of that product into a rectangular matrix.
                    */
                    send_size = qrTree->q_numRow[left_idx]*
                                qrTree->q_numCol[root_idx];
                    realloc_to_max( &(qrTree->R[left_idx]),
                                    &(qrTree->R_memSize[left_idx]),
                                    sizeof(*(qrTree->R[left_idx])),
                                    send_size);
                    ldr = qrTree->q_numCol[root_idx];
                    ldw = qrTree->q_numCol[root_idx];
                    for(i=0;i<qrTree->q_numRow[left_idx];i++){
                        for(j=0;j<qrTree->q_numCol[root_idx];j++){
                            if(i < qrTree->q_numCol[left_idx]){
                                qrTree->R[left_idx][j+ldr*i] =
                                                  qrTree->R[n_idx][j+ldw*i];
                            }else{
                                qrTree->R[left_idx][j+ldr*i] = 0.0;
                            }
                        }
                    }
                    /*
                        Wait for MPI Send to finish.
                    */
                    MPI_Wait(&send_req, &stat_send);
                }
            }
        }
        if(isLeaf(qrTree->n,n_idx) == 1){//Break when n_idx is a leaf
            break;
        }
    }
#if defined(TSQR_DEBUG)
    printf("\n*********************************************\n"
             "   PROC %d OUT OF TREE LOOP"
           "\n*********************************************\n",
           mpi_rk);
    /*
        Multiply the packed HH Q factor into the recv'd factor R
    */
    printf("\n*********************************************\n"
           "   PROC %d STARTING DORMQR."
           "\n*********************************************\n",
                mpi_rk);
getc(stdin);
#endif
    //finish off the Q by saving  Q_mpi_rnk with the final matrix.
    if(qrTree->numNodes == 1){
        for(i=0;i<qrTree->q_numRow[root_idx];i++){
            memcpy(Q_mpi_rnk+i*qrTree->q_numCol[root_idx],
                   qrTree->flt_wrk+i*qrTree->r_numCol[n_idx], 
                   sizeof(*Q_mpi_rnk)*
                   qrTree->q_numCol[root_idx]);
        }
    }else{
        memcpy(Q_mpi_rnk,qrTree->R[n_idx], 
                   sizeof(*Q_mpi_rnk)*
                   qrTree->q_numRow[n_idx]*
                   qrTree->q_numCol[root_idx]);
    }
}

