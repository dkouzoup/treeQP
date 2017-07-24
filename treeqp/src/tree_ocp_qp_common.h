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


#ifndef TREEQP_SRC_TREE_OCP_QP_COMMON_H_
#define TREEQP_SRC_TREE_OCP_QP_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"

// info returned by solver
typedef struct {
    int_t iter;
} treeqp_info_t;


// input to solver
typedef struct {
    int_t N;
    const int_t *nx;
    const int_t *nu;
    const struct d_strmat *A;
    const struct d_strmat *B;
    const struct d_strvec *b;
    const struct d_strvec *Q;  // NOTE(dimitris): currently only supporting diag. weights
    const struct d_strvec *R;
    const struct d_strvec *q;
    const struct d_strvec *r;
    const struct d_strvec *Qinv;
    const struct d_strvec *Rinv;
    const struct d_strvec *xmin;
    const struct d_strvec *xmax;
    const struct d_strvec *umin;
    const struct d_strvec *umax;
    const struct node *tree;
} tree_ocp_qp_in;


// output of solver
typedef struct {
    treeqp_info_t info;
    struct d_strvec *x;
    struct d_strvec *u;
} tree_ocp_qp_out;

int_t tree_ocp_qp_out_workspace_size(tree_ocp_qp_in *qp_in);
void tree_ocp_qp_out_create_workspace(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, void *ptr);

void print_tree_ocp_qp_in(tree_ocp_qp_in *qp_in);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  // TREEQP_SRC_TREE_OCP_QP_COMMON_H_
