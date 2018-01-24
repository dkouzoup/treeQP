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
typedef struct
{
    int iter;
    double solver_time;
    double interface_time;
} treeqp_info_t;


// input to solver
typedef struct
{
    int N;  // TODO(dimitris): think again about convention of N and N+1
    int *nx;
    int *nu;
    struct blasfeo_dmat *A;
    struct blasfeo_dmat *B;
    struct blasfeo_dvec *b;
    struct blasfeo_dmat *Q;
    struct blasfeo_dmat *R;
    struct blasfeo_dmat *S;
    struct blasfeo_dvec *q;
    struct blasfeo_dvec *r;
    struct blasfeo_dvec *xmin;
    struct blasfeo_dvec *xmax;
    struct blasfeo_dvec *umin;
    struct blasfeo_dvec *umax;
    // TODO(dimitris): add general constraints
    // TODO(dimitris): decide on compatibility with HPIPM QP format
    struct node *tree;
} tree_ocp_qp_in;

int tree_ocp_qp_in_calculate_size(int Nn, int *nx, int *nu, struct node *tree);

void create_tree_ocp_qp_in(int Nn, int *nx, int *nu, struct node *tree,
    tree_ocp_qp_in *qp_in, void *ptr);

// output of solver
typedef struct
{
    treeqp_info_t info;
    struct blasfeo_dvec *x;
    struct blasfeo_dvec *u;
    struct blasfeo_dvec *lam;  // multipliers of equality constraints
    struct blasfeo_dvec *mu_x;  // multipliers of state bounds (+: upper bound active, -: lower bound)
    struct blasfeo_dvec *mu_u;  // multipliers of input bounds
} tree_ocp_qp_out;

int tree_ocp_qp_out_calculate_size(int Nn, int *nx, int *nu);

void create_tree_ocp_qp_out(int Nn, int *nx, int *nu, tree_ocp_qp_out *qp_out, void *ptr);

void calculate_KKT_residuals(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, double *res);
double max_KKT_residual(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out);

int number_of_states(tree_ocp_qp_in *qp_in);
int max_number_of_states(tree_ocp_qp_in *qp_in);
int number_of_controls(tree_ocp_qp_in *qp_in);
int max_number_of_controls(tree_ocp_qp_in *qp_in);
int number_of_primal_variables(tree_ocp_qp_in *qp_in);
int number_of_dynamic_constraints(tree_ocp_qp_in *qp_in);

void print_tree_ocp_qp_in(tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_fill_lti_data_diag_weights(double *A, double *B, double *b,
    double *Q, double *q, double *P, double *p, double *R, double *r,
    double *xmin, double *xmax, double *umin, double *umax, double *x0, tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_read_dynamics_colmajor(double *A, double *B, double *b, tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_read_objective_diag(double *Qd, double *Rd, double *q, double *r, tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_read_objective_colmajor(double *Q, double *R, double *S, double *q, double *r, tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_set_inf_bounds(tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_set_constant_bounds(double *xmin, double *xmax, double *umin, double *umax,
    tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_set_x0_bounds(tree_ocp_qp_in *qp_in, double *x0);

void write_qp_out_to_txt(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, const char *fpath);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREEQP_SRC_TREE_OCP_QP_COMMON_H_ */
