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
typedef struct treeqp_info_t_
{
    int iter;
    double solver_time;
    double interface_time;
    // TODO(dimitris): add total time
} treeqp_info_t;



// internal memory to eliminate x0 from QP
typedef struct qp_internal_t_
{
    int *is_initialized;        // flag to denote whether (A0, b0) pair is initialized

    struct blasfeo_dvec x0;     // space to pack x0

    struct blasfeo_dmat *A0;    // matrices A of all children of root
    struct blasfeo_dvec *b0;    // vectors b of all children of root

    struct blasfeo_dmat C0;     // TODO: is_initialized for this too
    struct blasfeo_dvec dmax0;
    struct blasfeo_dvec dmin0;

    // TODO(dimitris): contribution from S term missing!

} qp_internal_t;



// input to solver
typedef struct tree_ocp_qp_in_
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

    qp_internal_t internal_memory;

} tree_ocp_qp_in;



// output of solver
typedef struct tree_ocp_qp_out_
{
    treeqp_info_t info;         // struct with information (niter, cpu time, etc)

    struct blasfeo_dvec *x;     // optimal states per node
    struct blasfeo_dvec *u;     // optimal inputs per node

    struct blasfeo_dvec *lam;   // multipliers of equality constraints per edge
    struct blasfeo_dvec *mu_x;  // multipliers of state bounds (+: upper bound active, -: lower bound)
    struct blasfeo_dvec *mu_u;  // multipliers of input bounds
    struct blasfeo_dvec *mu_d;  // multipliers of general constraints

} tree_ocp_qp_out;



// TODO(dimitris): follow same order of functions in .c file

int total_number_of_states(tree_ocp_qp_in *qp_in);

int max_number_of_states(tree_ocp_qp_in *qp_in);

int total_number_of_controls(tree_ocp_qp_in *qp_in);

int max_number_of_controls(tree_ocp_qp_in *qp_in);

int total_number_of_general_constraints(tree_ocp_qp_in *qp_in);

int max_number_of_general_constraints(tree_ocp_qp_in *qp_in);

int total_number_of_primal_variables(tree_ocp_qp_in *qp_in);

int total_number_of_dynamic_constraints(tree_ocp_qp_in *qp_in);



int tree_ocp_qp_in_calculate_size(int Nn, int *nx, int *nu, int *nc, struct node *tree);

void tree_ocp_qp_in_create(int Nn, int *nx, int *nu, int *nc, struct node *tree, tree_ocp_qp_in *qp_in, void *ptr);

int tree_ocp_qp_out_calculate_size(int Nn, int *nx, int *nu, int *nc);

void tree_ocp_qp_out_create(int Nn, int *nx, int *nu, int *nc, tree_ocp_qp_out *qp_out, void *ptr);


void tree_ocp_qp_in_eliminate_x0(tree_ocp_qp_in *qp_in);

void tree_ocp_qp_out_eliminate_x0(tree_ocp_qp_out *qp_out);

void tree_ocp_qp_out_calculate_KKT_res(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, double *res);

double tree_ocp_qp_out_max_KKT_res(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out);


// TODO(dimitris): move to utils
void tree_ocp_qp_in_print_dims(tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_print(tree_ocp_qp_in *qp_in);

void tree_ocp_qp_out_print(int Nn, tree_ocp_qp_out *qp_out);

void tree_ocp_qp_out_write_to_txt(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, const char *fpath);



// set the dynamics of the edge connecting nodes [indx+1] and [p(indx+1)]
void tree_ocp_qp_in_set_edge_dynamics_colmajor(double *A, double *B, double *b, tree_ocp_qp_in *qp_in, int indx);

void tree_ocp_qp_in_set_node_objective_colmajor(double *Q, double *R, double *S, double *q, double *r, tree_ocp_qp_in *qp_in, int indx);

void tree_ocp_qp_in_set_node_objective_diag(double *Qd, double *Rd, double *q, double *r, tree_ocp_qp_in *qp_in, int indx);

void tree_ocp_qp_in_set_node_bounds(double *xmin, double *xmax, double *umin, double *umax, tree_ocp_qp_in *qp_in, int indx);

void tree_ocp_qp_in_set_node_general_constraints(double *C, double *D, double *dmin, double *dmax, tree_ocp_qp_in *qp_in, int indx);


// A, B, b contain all matrices/vectors of appropriate dimensions concatenated in one vector
void tree_ocp_qp_in_set_ltv_dynamics_colmajor(double *A, double *B, double *b, tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_set_ltv_objective_colmajor(double *Q, double *R, double *S, double *q, double *r, tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_set_ltv_objective_diag(double *Qd, double *Rd, double *q, double *r, tree_ocp_qp_in *qp_in);


void tree_ocp_qp_in_set_const_bounds(double *xmin, double *xmax, double *umin, double *umax, tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_set_inf_bounds(tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_set_x0_strvec(tree_ocp_qp_in *qp_in, struct blasfeo_dvec *sx0);

void tree_ocp_qp_in_set_x0_colmaj(tree_ocp_qp_in *qp_in, double *x0);


// TODO(dimitris): clean up
void tree_ocp_qp_in_fill_lti_data_diag_weights(double *A, double *B, double *b,
    double *Q, double *q, double *P, double *p, double *R, double *r,
    double *xmin, double *xmax, double *umin, double *umax, double *x0,
    double *C, double *CN, double *D, double *dmin, double *dmax, tree_ocp_qp_in *qp_in);

void tree_ocp_qp_in_fill_lti_data_diag_weights_OLD(double *A, double *B, double *b,
    double *Q, double *q, double *P, double *p, double *R, double *r,
    double *xmin, double *xmax, double *umin, double *umax, double *x0,
    double *C, double *CN, double *D, double *dmin, double *dmax, tree_ocp_qp_in *qp_in);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREEQP_SRC_TREE_OCP_QP_COMMON_H_ */
