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

#include "treeqp/utils/memory.h"
#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_v_aux_ext_dep.h"

// NOTE(dimitris): uncomment line below to use dynamic memory allocation for debugging purposes
// #define _USE_VALGRIND_

#ifdef _USE_VALGRIND_
// print warning when by-passing pointer and allocating new memory (for debugging)
static void print_warning ()
{
    printf(" -- using dynamically allocated memory for debugging --\n");
}
#endif


// make an integer multiple of num (for calculate_size routines)
void make_int_multiple_of(int num, int *size)
{
    *size = (*size + num - 1) / num * num;
}



// move char pointer to next multiple of num (for memory alignment)
int align_char_to(int num, char **c_ptr)
{
    size_t s_ptr = (size_t)*c_ptr;
    s_ptr = (s_ptr + num - 1) / num * num;
    int offset = num - (int)(s_ptr - (size_t)(*c_ptr));
    *c_ptr = (char *)s_ptr;
    return offset;
}



// wrappers to blasfeo create or allocate functions
static void create_strvec(int rows, struct blasfeo_dvec *sV, char **ptr)
{
    assert((size_t)*ptr % 8 == 0 && "strvec not 8-byte aligned!");

#ifdef _USE_VALGRIND_
    blasfeo_allocate_dvec(rows, sV);
    print_warning();
#else
    blasfeo_create_dvec(rows, sV, *ptr);
    *ptr += sV->memsize;
#endif
}



static void create_strmat(int rows, int cols, struct blasfeo_dmat *sA, char **ptr)
{
#ifdef LA_HIGH_PERFORMANCE
    assert((size_t)*ptr % 64 == 0 && "strmat not 64-byte aligned!");
#else
    assert((size_t)*ptr % 8 == 0 && "strmat not 8-byte aligned!");
#endif

#ifdef _USE_VALGRIND_
    blasfeo_allocate_dmat(rows, cols, sA);
    print_warning();
#else
    blasfeo_create_dmat(rows, cols, sA, *ptr);
    *ptr += sA->memsize;
#endif
}



// create and initialize to input data
void wrapper_vec_to_strvec(int rows, double *V, struct blasfeo_dvec *sV, char **ptr)
{
    create_strvec(rows, sV, ptr);
    blasfeo_pack_dvec(rows, V, sV, 0);
}



void wrapper_mat_to_strmat(int rows, int cols, double *A, struct blasfeo_dmat *sA, char **ptr)
{
    create_strmat(rows, cols, sA, ptr);
    blasfeo_pack_dmat(rows, cols, A, rows, sA, 0, 0);
}



// create and initialize to zero
void init_strvec(int rows, struct blasfeo_dvec *sV, char **ptr)
{
    create_strvec(rows, sV, ptr);
    blasfeo_dvecse(rows, 0.0, sV, 0);
}



void init_strmat(int rows, int cols, struct blasfeo_dmat *sA, char **ptr)
{
    create_strmat(rows, cols, sA, ptr);
    blasfeo_dgese(rows, cols, 0.0, sA, 0, 0);
}



// allocate and free double pointers
void malloc_double_ptr_strmat(struct blasfeo_dmat ***arr, int m, int n)
{
    *arr = malloc(m * sizeof(struct blasfeo_dmat*));

    for (int ii = 0; ii < m; ii++)
    {
        (*arr)[ii] = malloc(n * sizeof(struct blasfeo_dmat));
    }
}



// TODO(dimitris): not used yet
void malloc_double_ptr_strvec(struct blasfeo_dvec ***arr, int m, int n)
{
    *arr = malloc(m * sizeof(struct blasfeo_dvec*));

    for (int ii = 0; ii < m; ii++)
    {
        (*arr)[ii]= malloc(n * sizeof(struct blasfeo_dvec));
    }
}



// TODO(dimitris): Check with valgrind
void free_double_ptr_strmat(struct blasfeo_dmat **arr, int m)
{
    for (int ii = 0; ii < m; ii++)
    {
        free(arr[ii]);
    }
    free(arr);
}



void free_double_ptr_strvec(struct blasfeo_dvec **arr, int m)
{
    for (int ii = 0; ii < m; ii++)
    {
        free(arr[ii]);
    }
    free(arr);
}



void create_double_ptr_strmat(struct blasfeo_dmat ***arr, int m, int n, char **ptr)
{
    *arr = (struct blasfeo_dmat **) *ptr;
    *ptr += m*sizeof(struct blasfeo_dmat*);

    for (int ii = 0; ii < m; ii++)
    {
        (*arr)[ii] = (struct blasfeo_dmat *) *ptr;
        *ptr += n*sizeof(struct blasfeo_dmat);
    }
}



void create_double_ptr_strvec(struct blasfeo_dvec ***arr, int m, int n, char **ptr)
{
    *arr = (struct blasfeo_dvec **) *ptr;
    *ptr += m*sizeof(struct blasfeo_dvec*);

    for (int ii = 0; ii < m; ii++)
    {
        (*arr)[ii] = (struct blasfeo_dvec *) *ptr;
        *ptr += n*sizeof(struct blasfeo_dvec);
    }
}



void create_double_ptr_int(int ***arr, int m, int n, char **ptr)
{
    *arr = (int **) *ptr;
    *ptr += m*sizeof(int*);

    for (int ii = 0; ii < m; ii++)
    {
        (*arr)[ii] = (int *) *ptr;
        *ptr += n*sizeof(int);
        // initialize to zero
        for (int jj = 0; jj < n; jj++)
        {
            (*arr)[ii][jj] = 0;
        }
    }
}
