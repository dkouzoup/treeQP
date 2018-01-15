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

void make_int_multiple_of(int num, int *size)
{
    *size = (*size + num - 1) / num * num;
}



int align_char_to(int num, char **c_ptr)
{
    size_t s_ptr = (size_t)*c_ptr;
    s_ptr = (s_ptr + num - 1) / num * num;
    int offset = num - (int)(s_ptr - (size_t)(*c_ptr));
    *c_ptr = (char *)s_ptr;
    return offset;
}



void convert_strvecs_to_single_vec(int n, struct blasfeo_dvec sv[], double *v)
{
    int ind = 0;
    for (int i = 0; i < n; i++) {
        blasfeo_unpack_dvec(sv[i].m, &sv[i], 0, &v[ind]);
        ind += sv[i].m;
    }
}



void convert_strmats_to_single_vec(int n, struct blasfeo_dmat sMat[], double *mat)
{
    int ind = 0;
    for (int i = 0; i < n; i++)
    {
        blasfeo_unpack_dmat(sMat[i].m, sMat[i].n, &sMat[i], 0, 0, &mat[ind], sMat[i].m);
        ind += sMat[i].m*sMat[i].n;
    }
}



void convert_strmats_tran_to_single_vec(int n, struct blasfeo_dmat sMat[], double *mat) {
    int ind = 0;
    for (int i = 0; i < n; i++) {
        blasfeo_unpack_tran_dmat(sMat[i].m, sMat[i].n, &sMat[i], 0, 0, &mat[ind], sMat[i].n);
        ind += sMat[i].m*sMat[i].n;
    }
}


void init_blasfeo_memory(int memorySize, char **ptr) {
    c_zeros_align(ptr, memorySize);
}


void clean_blasfeo_memory(char **ptr) {
    c_free(*ptr);
}


// wrappers to blasfeo create or allocate functions
static void create_strvec(int rows, struct blasfeo_dvec *sV, char **ptr) {
#ifdef __DEBUG__
    blasfeo_allocate_dvec(rows, sV);
    printf(" -- using dynamically allocated memory for debugging --\n");
#else
    blasfeo_create_dvec(rows, sV, *ptr);
    *ptr += sV->memsize;
#endif
}


static void create_strmat(int rows, int cols, struct blasfeo_dmat *sA, char **ptr) {
    #ifdef __DEBUG__
        blasfeo_allocate_dmat(rows, cols, sA);
        printf(" -- using dynamically allocated memory for debugging --\n");
    #else
        blasfeo_create_dmat(rows, cols, sA, *ptr);
        *ptr += sA->memsize;
    #endif
}


// create and initialize to input data
void wrapper_vec_to_strvec(int rows, double *V, struct blasfeo_dvec *sV, char **ptr) {
    create_strvec(rows, sV, ptr);
    blasfeo_pack_dvec(rows, V, sV, 0);
}


void wrapper_mat_to_strmat(int rows, int cols, double *A, struct blasfeo_dmat *sA, char **ptr) {
    create_strmat(rows, cols, sA, ptr);
    blasfeo_pack_dmat(rows, cols, A, rows, sA, 0, 0);
}


// create and initialize to zero
void init_strvec(int rows, struct blasfeo_dvec *sV, char **ptr) {
    create_strvec(rows, sV, ptr);
    blasfeo_dvecse(rows, 0.0, sV, 0);
}


void init_strmat(int rows, int cols, struct blasfeo_dmat *sA, char **ptr) {
    create_strmat(rows, cols, sA, ptr);
    blasfeo_dgese(rows, cols, 0.0, sA, 0, 0);
}


// allocate and free double pointers
void malloc_double_ptr_strmat(struct blasfeo_dmat ***arr, int m, int n) {
    int ii;

    *arr = malloc(m * sizeof(struct blasfeo_dmat*));

    for (ii = 0; ii < m; ii++)
        (*arr)[ii] = malloc(n * sizeof(struct blasfeo_dmat));
}


// TODO(dimitris): not used yet
void malloc_double_ptr_strvec(struct blasfeo_dvec ***arr, int m, int n) {
    int ii;

    *arr = malloc(m * sizeof(struct blasfeo_dvec*));

    for (ii = 0; ii < m; ii++)
        (*arr)[ii]= malloc(n * sizeof(struct blasfeo_dvec));
}


// TODO(dimitris): Check with valgrind
void free_double_ptr_strmat(struct blasfeo_dmat **arr, int m) {
    int ii;

    for (ii = 0; ii < m; ii++)
        free(arr[ii]);

    free(arr);
}


void free_double_ptr_strvec(struct blasfeo_dvec **arr, int m) {
    int ii;

    for (ii = 0; ii < m; ii++)
        free(arr[ii]);

    free(arr);
}


void create_double_ptr_strmat(struct blasfeo_dmat ***arr, int m, int n, char **ptr) {
    int ii;

    *arr = (struct blasfeo_dmat **) *ptr;
    *ptr += m*sizeof(struct blasfeo_dmat*);

    for (ii = 0; ii < m; ii++) {
        (*arr)[ii] = (struct blasfeo_dmat *) *ptr;
        *ptr += n*sizeof(struct blasfeo_dmat);
    }
}


void create_double_ptr_strvec(struct blasfeo_dvec ***arr, int m, int n, char **ptr) {
    int ii;

    *arr = (struct blasfeo_dvec **) *ptr;
    *ptr += m*sizeof(struct blasfeo_dvec*);

    for (ii = 0; ii < m; ii++) {
        (*arr)[ii] = (struct blasfeo_dvec *) *ptr;
        *ptr += n*sizeof(struct blasfeo_dvec);
    }
}


// TODO(dimitris): this function is not blasfeo related, rename to e.g. memory_utils?
void create_double_ptr_int(int ***arr, int m, int n, char **ptr) {
    *arr = (int **) *ptr;
    *ptr += m*sizeof(int*);

    for (int ii = 0; ii < m; ii++) {
        (*arr)[ii] = (int *) *ptr;
        *ptr += n*sizeof(int);
        // initialize to zero
        for (int jj = 0; jj < n; jj++) {
            (*arr)[ii][jj] = 0;
        }
    }
}


double check_error_strmat(struct blasfeo_dmat *M1, struct blasfeo_dmat *M2) {
    int ii, jj;
    double err = 0;

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


double check_error_strvec(struct blasfeo_dvec *V1, struct blasfeo_dvec *V2) {
    int ii;
    double err = 0;

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
        blasfeo_print_tran_dvec(V1->m, V1, 0);
        blasfeo_print_tran_dvec(V2->m, V2, 0);
        exit(1);
    }
    return err;
}


answer_t iblasfeo_smat_diagonal(struct blasfeo_dmat *M) {
    answer_t ans = YES;
    assert(M->m == M->n);
    for (int ii = 0; ii < M->m; ii++) {
        for (int jj = 0; jj < M->n; jj++) {
            if (ii != jj) {
                if (DMATEL_LIBSTR(M, ii, jj) != 0) {
                    ans = NO;
                }
            }
        }
    }
    return ans;
}


answer_t iblasfeo_smat_zero(struct blasfeo_dmat *M) {
    answer_t ans = YES;
    for (int ii = 0; ii < M->m; ii++) {
        for (int jj = 0; jj < M->n; jj++) {
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