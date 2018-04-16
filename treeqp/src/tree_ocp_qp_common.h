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
    // TODO(dimitris): TOTAL TIME?!
} treeqp_info_t;



// input to solver
typedef struct
{
    int N;                      // Number of nodes (= N+1 in nominal MPC!) - TODO(dimitris): rename to Nn
    int *nx;                    // number of states per node
    int *nu;                    // number of inputs per node
    int *nc;                    // number of general constraints per node (C: nc x nx, D: nc x nu)
    struct blasfeo_dmat *A;     // x[n] = A[n-1]*x[p(n)] + B[n-1]*u[p(n)] + b[n-1]
    struct blasfeo_dmat *B;
    struct blasfeo_dvec *b;
    struct blasfeo_dmat *Q;     // [x' u']*[Q S'; S R]*[x; u] + [x 'u']*[q; r]
    struct blasfeo_dmat *R;
    struct blasfeo_dmat *S;
    struct blasfeo_dvec *q;
    struct blasfeo_dvec *r;
    struct blasfeo_dvec *xmin;  // xmin <= x <= xmax
    struct blasfeo_dvec *xmax;
    struct blasfeo_dvec *umin;  // umin <= u <= umax
    struct blasfeo_dvec *umax;
    struct blasfeo_dmat *C;     // dmin <= Cx + Du <= dmax
    struct blasfeo_dmat *D;
    struct blasfeo_dvec *dmin;
    struct blasfeo_dvec *dmax;
    struct node *tree;
    // TODO(dimitris): decide on compatibility with HPIPM QP format
} tree_ocp_qp_in;



// output of solver
typedef struct
{
    treeqp_info_t info;         // struct with information (niter, cpu time, etc)
    struct blasfeo_dvec *x;     // optimal states per node
    struct blasfeo_dvec *u;     // optimal inputs per node
    struct blasfeo_dvec *lam;   // multipliers of equality constraints per edge
    struct blasfeo_dvec *mu_x;  // multipliers of state bounds (+: upper bound active, -: lower bound)
    struct blasfeo_dvec *mu_u;  // multipliers of input bounds
    struct blasfeo_dvec *mu_d;  // multipliers of general constraints
} tree_ocp_qp_out;


// TODO(dimitris): follow same order in .c file

// TODO(dimitris): rename to total_
int number_of_states(tree_ocp_qp_in *qp_in);

int max_number_of_states(tree_ocp_qp_in *qp_in);

int number_of_controls(tree_ocp_qp_in *qp_in);

int max_number_of_controls(tree_ocp_qp_in *qp_in);

int number_of_primal_variables(tree_ocp_qp_in *qp_in);

int number_of_dynamic_constraints(tree_ocp_qp_in *qp_in);

int number_of_general_constraints(tree_ocp_qp_in *qp_in);

int max_number_of_general_constraints(tree_ocp_qp_in *qp_in);

int tree_ocp_qp_in_calculate_size(int Nn, int *nx, int *nu, int *nc, struct node *tree);

void tree_ocp_qp_in_create(int Nn, int *nx, int *nu, int *nc, struct node *tree, tree_ocp_qp_in *qp_in, void *ptr);

int tree_ocp_qp_out_calculate_size(int Nn, int *nx, int *nu, int *nc);

void tree_ocp_qp_out_create(int Nn, int *nx, int *nu, int *nc, tree_ocp_qp_out *qp_out, void *ptr);



void calculate_KKT_residuals(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, double *res);

double max_KKT_residual(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out);


// TODO(dimitris): move to utils
void tree_ocp_qp_in_print_dims(tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_print(tree_ocp_qp_in *qp_in);

void tree_ocp_qp_out_print(int Nn, tree_ocp_qp_out *qp_out);

void tree_ocp_qp_out_write_to_txt(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, const char *fpath);



// TODO(dimitris): move to C interface

void tree_ocp_qp_in_set_ltv_dynamics_colmajor(double *A, double *B, double *b, tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_set_ltv_objective_diag(double *Qd, double *Rd, double *q, double *r, tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_set_ltv_objective_colmajor(double *Q, double *R, double *S, double *q, double *r, tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_set_const_bounds(double *xmin, double *xmax, double *umin, double *umax, tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_set_inf_bounds(tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_set_x0_bounds(tree_ocp_qp_in *qp_in, double *x0);

// TODO(dimitris): split to small functions and clean up
void tree_ocp_qp_in_fill_lti_data_diag_weights(double *A, double *B, double *b,
    double *Q, double *q, double *P, double *p, double *R, double *r,
    double *xmin, double *xmax, double *umin, double *umax, double *x0,
    double *C, double *CN, double *D, double *dmin, double *dmax, tree_ocp_qp_in *qp_in);


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREEQP_SRC_TREE_OCP_QP_COMMON_H_ */
