/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*    This file is part of treeQP.                                                                  *
*                                                                                                  *
*    treeQP -- A toolbox of tree-sparse Quadratic Programming solvers.                             *
*    Copyright (C) 2017 by Dimitris Kouzoupis.                                                     *
*    Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.            *
*    All rights reserved.                                                                          *
*                                                                                                  *
*    treeQP is free software; you can redistribute it and/or                                       *
*    modify it under the terms of the GNU Lesser General Public                                    *
*    License as published by the Free Software Foundation; either                                  *
*    version 3 of the License, or (at your option) any later version.                              *
*                                                                                                  *
*    treeQP is distributed in the hope that it will be useful,                                     *
*    but WITHOUT ANY WARRANTY; without even the implied warranty of                                *
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU                             *
*    Lesser General Public License for more details.                                               *
*                                                                                                  *
*    You should have received a copy of the GNU Lesser General Public                              *
*    License along with treeQP; if not, write to the Free Software Foundation,                     *
*    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                            *
*                                                                                                  *
*    Author: Dimitris Kouzoupis, dimitris.kouzoupis (at) imtek.uni-freiburg.de                     *
*                                                                                                  *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


#include <stdlib.h>
#include <assert.h>

#include "treeqp/utils/blasfeo_utils.h"
#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_v_aux_ext_dep.h"

// NOTE(dimitris): uncomment line below to use dynamic memory allocation for debugging purposes
// #define __DEBUG__

void convert_strvecs_to_single_vec(int_t n, struct d_strvec sv[], real_t *v) {
    int_t ind = 0;
    int_t i;
    for (i = 0; i < n; i++) {
        d_cvt_strvec2vec(sv[i].m, &sv[i], 0, &v[ind]);
        ind += sv[i].m;
    }
}


void convert_strmats_to_single_vec(int_t n, struct d_strmat sMat[], real_t *mat) {
    int_t ind = 0;
    int_t i;
    for (i = 0; i < n; i++) {
        d_cvt_strmat2mat(sMat[i].m, sMat[i].n, &sMat[i], 0, 0, &mat[ind], sMat[i].m);
        ind += sMat[i].m*sMat[i].n;
    }
}


void convert_strmats_tran_to_single_vec(int_t n, struct d_strmat sMat[], real_t *mat) {
    int_t ind = 0;
    int_t i;
    for (i = 0; i < n; i++) {
        d_cvt_tran_strmat2mat(sMat[i].m, sMat[i].n, &sMat[i], 0, 0, &mat[ind], sMat[i].n);
        ind += sMat[i].m*sMat[i].n;
    }
}


void init_blasfeo_memory(int_t memorySize, char **ptr) {
    c_zeros_align(ptr, memorySize);
}


void clean_blasfeo_memory(char **ptr) {
    c_free(*ptr);
}


// wrappers to blasfeo create or allocate functions
static void create_strvec(int_t rows, struct d_strvec *sV, char **ptr) {
#ifdef __DEBUG__
    d_allocate_strvec(rows, sV);
    printf(" -- using dynamically allocated memory for debugging --\n");
#else
    d_create_strvec(rows, sV, *ptr);
    *ptr += sV->memory_size;
#endif
}


static void create_strmat(int_t rows, int_t cols, struct d_strmat *sA, char **ptr) {
    #ifdef __DEBUG__
        d_allocate_strmat(rows, cols, sA);
        printf(" -- using dynamically allocated memory for debugging --\n");
    #else
        d_create_strmat(rows, cols, sA, *ptr);
        *ptr += sA->memory_size;
    #endif
}


// create and initialize to input data
void wrapper_vec_to_strvec(int_t rows, real_t *V, struct d_strvec *sV, char **ptr) {
    create_strvec(rows, sV, ptr);
    d_cvt_vec2strvec(rows, V, sV, 0);
}


void wrapper_mat_to_strmat(int_t rows, int_t cols, real_t *A, struct d_strmat *sA, char **ptr) {
    create_strmat(rows, cols, sA, ptr);
    d_cvt_mat2strmat(rows, cols, A, rows, sA, 0, 0);
}


// create and initialize to zero
void init_strvec(int_t rows, struct d_strvec *sV, char **ptr) {
    create_strvec(rows, sV, ptr);
    dvecse_libstr(rows, 0.0, sV, 0);
}


void init_strmat(int_t rows, int_t cols, struct d_strmat *sA, char **ptr) {
    create_strmat(rows, cols, sA, ptr);
    dgese_libstr(rows, cols, 0.0, sA, 0, 0);
}


// allocate and free double pointers
void malloc_double_ptr_strmat(struct d_strmat ***arr, int_t m, int_t n) {
    int_t ii;

    *arr = malloc(m * sizeof(struct d_strmat*));

    for (ii = 0; ii < m; ii++)
        (*arr)[ii] = malloc(n * sizeof(struct d_strmat));
}


// TODO(dimitris): not used yet
void malloc_double_ptr_strvec(struct d_strvec ***arr, int_t m, int_t n) {
    int_t ii;

    *arr = malloc(m * sizeof(struct d_strvec*));

    for (ii = 0; ii < m; ii++)
        (*arr)[ii]= malloc(n * sizeof(struct d_strvec));
}


// TODO(dimitris): Check with valgrind
void free_double_ptr_strmat(struct d_strmat **arr, int_t m) {
    int_t ii;

    for (ii = 0; ii < m; ii++)
        free(arr[ii]);

    free(arr);
}


void free_double_ptr_strvec(struct d_strvec **arr, int_t m) {
    int_t ii;

    for (ii = 0; ii < m; ii++)
        free(arr[ii]);

    free(arr);
}


void create_double_ptr_strmat(struct d_strmat ***arr, int_t m, int_t n, char **ptr) {
    int_t ii;

    *arr = (struct d_strmat **) *ptr;
    *ptr += m*sizeof(struct d_strmat*);

    for (ii = 0; ii < m; ii++) {
        (*arr)[ii] = (struct d_strmat *) *ptr;
        *ptr += n*sizeof(struct d_strmat);
    }
}


void create_double_ptr_strvec(struct d_strvec ***arr, int_t m, int_t n, char **ptr) {
    int_t ii;

    *arr = (struct d_strvec **) *ptr;
    *ptr += m*sizeof(struct d_strvec*);

    for (ii = 0; ii < m; ii++) {
        (*arr)[ii] = (struct d_strvec *) *ptr;
        *ptr += n*sizeof(struct d_strvec);
    }
}


// TODO(dimitris): this function is not blasfeo related, rename to e.g. memory_utils?
void create_double_ptr_int(int_t ***arr, int_t m, int_t n, char **ptr) {
    *arr = (int_t **) *ptr;
    *ptr += m*sizeof(int_t*);

    for (int_t ii = 0; ii < m; ii++) {
        (*arr)[ii] = (int_t *) *ptr;
        *ptr += n*sizeof(int_t);
        // initialize to zero
        for (int_t jj = 0; jj < n; jj++) {
            (*arr)[ii][jj] = 0;
        }
    }
}


real_t check_error_strmat(struct d_strmat *M1, struct d_strmat *M2) {
    int_t ii, jj;
    real_t err = 0;

    if ((M1->m != M2->m) || (M1->n != M2->n)) {
        printf("[TREEQP]: Error! Matrices do not have the same dimensions ");
        printf("(%d x %d) vs (%d x %d)\n", M1->m, M1->n, M2->m, M2->n);
        exit(1);
    }
    for (ii = 0; ii < M1->m; ii++) {
        for (jj = 0; jj < M1->n; jj++) {
            err = MAX(ABS(DMATEL_LIBSTR(M1, ii, jj) - DMATEL_LIBSTR(M2, ii, jj)), err);
        }
    }
    if (err > 0) {
        printf("[TREEQP]: Error! Matrices are different (error = %2.2e)\n", err);
        exit(1);
    }
    return err;
}


real_t check_error_strvec(struct d_strvec *V1, struct d_strvec *V2) {
    int_t ii;
    real_t err = 0;

    if (V1->m != V2->m) {
        printf("[TREEQP]: Error! Vectors do not have the same dimensions ");
        printf("(%d x 1) vs (%d x 1)\n", V1->m, V2->m);
        exit(1);
    }
    for (ii = 0; ii < V1->m; ii++) {
        err = MAX(ABS(DVECEL_LIBSTR(V1, ii) - DVECEL_LIBSTR(V2, ii)), err);
    }
    if (err > 0) {
        printf("[TREEQP]: Error! Vectors are different (error = %2.2e)\n", err);
        d_print_tran_strvec(V1->m, V1, 0);
        d_print_tran_strvec(V2->m, V2, 0);
        exit(1);
    }
    return err;
}


answer_t is_strmat_diagonal(struct d_strmat *M) {
    answer_t ans = YES;
    assert(M->m == M->n);
    for (int_t ii = 0; ii < M->m; ii++) {
        for (int_t jj = 0; jj < M->n; jj++) {
            if (ii != jj) {
                if (DMATEL_LIBSTR(M, ii, jj) != 0) {
                    ans = NO;
                }
            }
        }
    }
    return ans;
}


answer_t is_strmat_zero(struct d_strmat *M) {
    answer_t ans = YES;
    for (int_t ii = 0; ii < M->m; ii++) {
        for (int_t jj = 0; jj < M->n; jj++) {
            if (DMATEL_LIBSTR(M, ii, jj) != 0) {
                ans = NO;
            }
        }
    }
    return ans;
}

void print_blasfeo_target() {
    printf("\n");
    #if defined(LA_HIGH_PERFORMANCE)
    printf("blasfeo compiled with LA = HIGH_PERFORMANCE\n");
    #elif defined(LA_REFERENCE)
    printf("blasfeo compiled with LA = REFERENCE\n");
    #elif defined(LA_BLAS)
    printf("blasfeo compiled with LA = BLAS\n");
    #endif
    printf("\n");
}