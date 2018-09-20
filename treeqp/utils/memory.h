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


#ifndef TREEQP_UTILS_MEMORY_H_
#define TREEQP_UTILS_MEMORY_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/utils/types.h"
#include "treeqp/utils/utils.h"

#include <blasfeo_target.h>
#include <blasfeo_common.h>


// memory alignment
void make_int_multiple_of(int num, int *size);

int align_char_to(int num, char **c_ptr);


// memory allocation
void create_int(int m, int **v, char **ptr);

void create_double(int m, double **v, char **ptr);

void create_strvec(int m, struct blasfeo_dvec *sv, char **ptr);

void create_strmat(int m, int n, struct blasfeo_dmat *sM, char **ptr);

void create_double_ptr_int(int m, int n, int ***arr, char **ptr);

void create_double_ptr_strvec(int m, int n, struct blasfeo_dvec ***arr, char **ptr);

void create_double_ptr_strmat(int m, int n, struct blasfeo_dmat ***arr, char **ptr);


// create and initialize blasfeo structures
void wrapper_vec_to_strvec(int m, const double *v, struct blasfeo_dvec *sv, char **ptr);

void wrapper_mat_to_strmat(int m, int n, const double *M, struct blasfeo_dmat *sM, char **ptr);

void init_strvec(int m, struct blasfeo_dvec *sv, char **ptr);

void init_strmat(int m, int n, struct blasfeo_dmat *sM, char **ptr);


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /*  TREEQP_UTILS_MEMORY_H_ */
