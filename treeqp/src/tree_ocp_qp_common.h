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
    int_t N;  // TODO(dimitris): think again about convention of N and N+1
    const int_t *nx;
    const int_t *nu;
    const struct d_strmat *A;
    const struct d_strmat *B;
    const struct d_strvec *b;
    const struct d_strmat *Q;
    const struct d_strmat *R;
    const struct d_strmat *S;
    const struct d_strvec *q;
    const struct d_strvec *r;
    const struct d_strvec *xmin;
    const struct d_strvec *xmax;
    const struct d_strvec *umin;
    const struct d_strvec *umax;
    const struct node *tree;
} tree_ocp_qp_in;

int_t tree_ocp_qp_in_calculate_size(int_t Nn, int_t *nx, int_t *nu, struct node *tree);

void create_tree_ocp_qp_in(int_t Nn, int_t *nx, int_t *nu, struct node *tree,
    tree_ocp_qp_in *qp_in, void *ptr);

// output of solver
typedef struct {
    treeqp_info_t info;
    struct d_strvec *x;
    struct d_strvec *u;
    struct d_strvec *lam;  // multipliers of equality constraints
    struct d_strvec *mu_x;  // multipliers of state bounds (+: upper bound active, -: lower bound)
    struct d_strvec *mu_u;  // multipliers of input bounds
} tree_ocp_qp_out;

int_t tree_ocp_qp_out_calculate_size(int_t Nn, int_t *nx, int_t *nu);

void create_tree_ocp_qp_out(int_t Nn, int_t *nx, int_t *nu, tree_ocp_qp_out *qp_out, void *ptr);

real_t maximum_error_in_dynamic_constraints(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out);
real_t *calculate_KKT_residuals(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out);

int_t number_of_states(tree_ocp_qp_in *qp_in);
int_t number_of_controls(tree_ocp_qp_in *qp_in);
int_t number_of_primal_variables(tree_ocp_qp_in *qp_in);

void print_tree_ocp_qp_in(tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_fill_lti_data_diag_weights(double *A, double *B, double *b,
    double *Q, double *q, double *P, double *p, double *R, double *r,
    double *xmin, double *xmax, double *umin, double *umax, double *x0, tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_read_dynamics_colmajor(real_t *A, real_t *B, real_t *b, tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_read_objective_diag(real_t *Qd, real_t *Rd, real_t *q, real_t *r,
    tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_set_inf_bounds(tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_set_constant_bounds(real_t *xmin, real_t *xmax, real_t *umin, real_t *umax,
    tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_set_x0_bounds(tree_ocp_qp_in *qp_in, real_t *x0);

void write_qp_out_to_txt(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, const char *fpath);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  // TREEQP_SRC_TREE_OCP_QP_COMMON_H_
