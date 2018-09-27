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

#include <blasfeo_d_aux.h>
#include <blasfeo_d_aux_ext_dep.h>
#include <blasfeo_v_aux_ext_dep.h>

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



void create_int(int m, int **v, char **ptr)
{
#ifdef _USE_VALGRIND_
    *v = (int *)calloc(m, sizeof(int));
    print_warning();
#else
    *v = (int *)*ptr;
    *ptr += sizeof(int) * m;
#endif
}



void create_double(int m, double **v, char **ptr)
{
    assert((size_t)*ptr % 8 == 0 && "double not 8-byte aligned!");

#ifdef _USE_VALGRIND_
    *v = (double *)calloc(m, sizeof(double));
    print_warning();
#else
    *v = (double *)*ptr;
    *ptr += sizeof(double) * m;
#endif
}



// wrappers to blasfeo create or allocate functions
void create_strvec(int m, struct blasfeo_dvec *sv, char **ptr)
{
    assert((size_t)*ptr % 8 == 0 && "strvec not 8-byte aligned!");

#ifdef _USE_VALGRIND_
    blasfeo_allocate_dvec(m, sv);
    print_warning();
#else
    blasfeo_create_dvec(m, sv, *ptr);
    *ptr += sv->memsize;
#endif
}



void create_strmat(int m, int n, struct blasfeo_dmat *sM, char **ptr)
{
#ifdef LA_HIGH_PERFORMANCE
    assert((size_t)*ptr % 64 == 0 && "strmat not 64-byte aligned!");
#else
    assert((size_t)*ptr % 8 == 0 && "strmat not 8-byte aligned!");
#endif

#ifdef _USE_VALGRIND_
    blasfeo_allocate_dmat(m, n, sM);
    print_warning();
#else
    blasfeo_create_dmat(m, n, sM, *ptr);
    *ptr += sM->memsize;
#endif
}



void create_double_ptr_int(int m, int n, int ***arr, char **ptr)
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



void create_double_ptr_strvec(int m, int n, struct blasfeo_dvec ***arr, char **ptr)
{
#ifdef _USE_VALGRIND_
    *arr = malloc(m * sizeof(struct blasfeo_dvec*));
    print_warning();
#else
    *arr = (struct blasfeo_dvec **) *ptr;
    *ptr += m*sizeof(struct blasfeo_dvec*);
#endif

    for (int ii = 0; ii < m; ii++)
    {
#ifdef _USE_VALGRIND_
        (*arr)[ii]= malloc(n * sizeof(struct blasfeo_dvec));
#else
        (*arr)[ii] = (struct blasfeo_dvec *) *ptr;
        *ptr += n*sizeof(struct blasfeo_dvec);
#endif
    }
}



void create_double_ptr_strmat(int m, int n, struct blasfeo_dmat ***arr, char **ptr)
{
#ifdef _USE_VALGRIND_
    *arr = calloc(m * sizeof(struct blasfeo_dmat*));
    print_warning();
#else
    *arr = (struct blasfeo_dmat **) *ptr;
    *ptr += m*sizeof(struct blasfeo_dmat*);
#endif

    for (int ii = 0; ii < m; ii++)
    {
#ifdef _USE_VALGRIND_
        (*arr)[ii] = calloc(n * sizeof(struct blasfeo_dmat));
#else
        (*arr)[ii] = (struct blasfeo_dmat *) *ptr;
        *ptr += n*sizeof(struct blasfeo_dmat);
#endif
    }
}



void wrapper_vec_to_strvec(int m, const double *v, struct blasfeo_dvec *sv, char **ptr)
{
    create_strvec(m, sv, ptr);
    blasfeo_pack_dvec(m, (double *)v, sv, 0);
}



void wrapper_mat_to_strmat(int m, int n, const double *M, struct blasfeo_dmat *sM, char **ptr)
{
    create_strmat(m, n, sM, ptr);
    blasfeo_pack_dmat(m, n, (double *)M, m, sM, 0, 0);
}



void init_strvec(int m, struct blasfeo_dvec *sv, char **ptr)
{
    create_strvec(m, sv, ptr);
    blasfeo_dvecse(m, 0.0, sv, 0);
}



void init_strmat(int m, int n, struct blasfeo_dmat *sM, char **ptr)
{
    create_strmat(m, n, sM, ptr);
    blasfeo_dgese(m, n, 0.0, sM, 0, 0);
}
