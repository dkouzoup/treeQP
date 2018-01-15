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


#ifndef TREEQP_UTILS_BLASFEO_UTILS_H_
#define TREEQP_UTILS_BLASFEO_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/utils/types.h"
#include "treeqp/utils/utils.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"

void make_int_multiple_of(int num, int *size);
int align_char_to(int num, char **c_ptr);

void convert_strvecs_to_single_vec(int n, struct blasfeo_dvec sv[], double *v);
void convert_strmats_to_single_vec(int n, struct blasfeo_dmat sMat[], double *mat);
void convert_strmats_tran_to_single_vec(int n, struct blasfeo_dmat sMat[], double *mat);

void wrapper_mat_to_strmat(int rows, int cols, double *A, struct blasfeo_dmat *sA, char **ptr);
void wrapper_vec_to_strvec(int rows, double *V, struct blasfeo_dvec *sV, char **ptr);

void init_strvec(int rows, struct blasfeo_dvec *sV, char **ptr);
void init_strmat(int rows, int cols, struct blasfeo_dmat *sA, char **ptr);

void malloc_double_ptr_strvec(struct blasfeo_dvec ***arr, int m, int n);
void malloc_double_ptr_strmat(struct blasfeo_dmat ***arr, int m, int n);
void free_double_ptr_strmat(struct blasfeo_dmat **arr, int m);
void free_double_ptr_strvec(struct blasfeo_dvec **arr, int m);

void create_double_ptr_strmat(struct blasfeo_dmat ***arr, int m, int n, char **ptr);
void create_double_ptr_strvec(struct blasfeo_dvec ***arr, int m, int n, char **ptr);
void create_double_ptr_int(int ***arr, int m, int n, char **ptr);

double check_error_strmat(struct blasfeo_dmat *M1, struct blasfeo_dmat *M2);
double check_error_strvec(struct blasfeo_dvec *V1, struct blasfeo_dvec *V2);

answer_t is_strmat_diagonal(struct blasfeo_dmat *M);
answer_t is_strmat_zero(struct blasfeo_dmat *M);

void print_blasfeo_target();

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /*  TREEQP_UTILS_BLASFEO_UTILS_H_ */
