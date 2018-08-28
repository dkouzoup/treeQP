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


#include <blasfeo_target.h>
#include <blasfeo_common.h>

#include "treeqp/utils/types.h"


// info returned by solver
typedef struct treeqp_info_t_
{
    int Nn;
    int iter;
    double solver_time;
    double interface_time;
    // TODO(dimitris): add total time and iteration logs
} treeqp_info_t;



// internal memory to eliminate x0 from QP
typedef struct qp_internal_t_
{
    int *is_A_initialized;      // flag to denote whether A0 are initialized (tree[0].nkids matrices in total)
    int *is_b_initialized;      // flag to denote whether b0 are initialized
    int is_C_initialized;       // flag to denote whether C0 is initialized
    int is_dmin_initialized;    // flag to denote whether dmin is initialized
    int is_dmax_initialized;    // flag to denote whether dmax is initialized
    int is_S_initialized;       // flag to denote whether S0 is initialized
    int is_r_initialized;       // flat to denote whether r0 is initialized

    struct blasfeo_dvec x0;     // memory to pack x0 from column major

    struct blasfeo_dmat *A0;    // matrices A of all children of root
    struct blasfeo_dvec *b0;    // vectors b of all children of root

    struct blasfeo_dmat C0;
    struct blasfeo_dvec dmax0;
    struct blasfeo_dvec dmin0;

    struct blasfeo_dmat S0;
    struct blasfeo_dvec r0;

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

int total_number_of_states(const tree_ocp_qp_in * const qp_in);

int max_number_of_states(const tree_ocp_qp_in * const qp_in);

int total_number_of_controls(const tree_ocp_qp_in * const qp_in);

int max_number_of_controls(const tree_ocp_qp_in * const qp_in);

int total_number_of_general_constraints(const tree_ocp_qp_in * const qp_in);

int max_number_of_general_constraints(const tree_ocp_qp_in * const qp_in);

int total_number_of_primal_variables(const tree_ocp_qp_in * const qp_in);

int total_number_of_dynamic_constraints(const tree_ocp_qp_in * const qp_in);



int tree_ocp_qp_in_calculate_size(const int Nn, const int * const nx, const int * const nu, const int * const nc, const struct node * const tree);

void tree_ocp_qp_in_create(const int Nn, const int * const nx, const int * const nu, const int * const nc, struct node * const tree, tree_ocp_qp_in * const qp_in, void *ptr);

int tree_ocp_qp_out_calculate_size(const int Nn, const int * const nx, const int * const nu, const int * const nc);

void tree_ocp_qp_out_create(const int Nn, const int * const nx, const int * const nu, const int * const nc, tree_ocp_qp_out * const qp_out, void *ptr);



void tree_ocp_qp_in_eliminate_x0(tree_ocp_qp_in * const qp_in);

void tree_ocp_qp_out_eliminate_x0(tree_ocp_qp_out * const qp_out);

void tree_ocp_qp_out_calculate_KKT_res(const tree_ocp_qp_in * const qp_in, const tree_ocp_qp_out * const qp_out, double *res);

double tree_ocp_qp_out_max_KKT_res(const tree_ocp_qp_in * const qp_in, const tree_ocp_qp_out * const qp_out);



// --------------- SETTERS / GETTERS ---------------

// set/get dynamics of edge connecting nodes [indx+1] and [p(indx+1)]
void tree_ocp_qp_in_set_edge_A_colmajor(const double * const A, const int lda, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_edge_A_colmajor(double * const A, const int lda, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_edge_B_colmajor(const double * const B, const int lda, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_edge_B_colmajor(double * const B, const int lda, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_edge_b(const double * const b, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_edge_b(double * const b, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_edge_dynamics_colmajor(const double * const A, const double * const B, const double * const b, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_edge_dynamics_colmajor(double * const A, double * const B, double * const b, const tree_ocp_qp_in * const qp_in, const int indx);



// set/get objective of node [indx]
void tree_ocp_qp_in_set_node_Q_colmajor(const double * const Q, const int lda, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_Q_colmajor(double * const Q, const int lda, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_node_R_colmajor(const double * const R, const int lda, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_R_colmajor(double * const R, const int lda, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_node_S_colmajor(const double * const S, const int lda, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_S_colmajor(double * const S, const int lda, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_node_q(const double * const q, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_q(double * const q, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_node_r(const double * const r, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_r(double * const r, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_node_objective_colmajor(const double * const Q, const double * const R, const double * const S, const double * const q, const double * const r, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_objective_colmajor(double * const Q, double * const R, double * const S, double * const q, double * const r, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_node_objective_diag(const double * const Qd, const double * const Rd, const double * const q, const double * const r, tree_ocp_qp_in * const qp_in, const int indx);


// set/get bounds of node [indx]
void tree_ocp_qp_in_set_node_xmin(const double * const xmin, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_xmin(double * const xmin, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_node_xmax(const double * const xmax, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_xmax(double * const xmax, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_node_umin(const double * const umin, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_umin(double * const umin, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_node_umax(const double * const umax, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_umax(double * const umax, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_node_bounds(const double * const xmin, const double * const xmax, const double * const umin, const double * const umax, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_bounds(double * const xmin, double * const xmax, double * const umin, double * const umax, const tree_ocp_qp_in * const qp_in, const int indx);


// set/get general constraints of node [indx]
void tree_ocp_qp_in_set_node_C_colmajor(const double * const C, const int lda, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_C_colmajor(double * const C, const int lda, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_node_D_colmajor(const double * const D, const int lda, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_D_colmajor(double * const D, const int lda, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_node_dmin(const double * const dmin, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_dmin(double * const dmin, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_node_dmax(const double * const dmax, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_dmax(double * const dmax, const tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_set_node_general_constraints(const double * const C, const double * const D, const double * const dmin, const double * const dmax, tree_ocp_qp_in * const qp_in, const int indx);

void tree_ocp_qp_in_get_node_general_constraints(double * const C, double * const D, double * const dmin, double * const dmax, const tree_ocp_qp_in * const qp_in, const int indx);

// TODO(dimitris): use setters to replace dual_initialization function

// set/get primal solution of node [indx]
void tree_ocp_qp_out_set_node_x(const double * const x, tree_ocp_qp_out * const qp_out, const int indx);

void tree_ocp_qp_out_get_node_x(double * const x, const tree_ocp_qp_out * const qp_out, const int indx);

void tree_ocp_qp_out_set_node_u(const double * const u, tree_ocp_qp_out * const qp_out, const int indx);

void tree_ocp_qp_out_get_node_u(double * const u, const tree_ocp_qp_out * const qp_out, const int indx);

// set/get dual solution of edge connecting nodes [indx+1] and [p(indx+1)]
void tree_ocp_qp_out_set_edge_lam(const double * const lam, tree_ocp_qp_out * const qp_out, const int indx);

void tree_ocp_qp_out_get_edge_lam(double * const lam, const tree_ocp_qp_out * const qp_out, const int indx);

// set/get dual solution of node [indx]
void tree_ocp_qp_out_set_node_mu_x(const double * const mu_x, tree_ocp_qp_out * const qp_out, const int indx);

void tree_ocp_qp_out_get_node_mu_x(double * const mu_x, const tree_ocp_qp_out * const qp_out, const int indx);

void tree_ocp_qp_out_set_node_mu_u(const double * const mu_x, tree_ocp_qp_out * const qp_out, const int indx);

void tree_ocp_qp_out_get_node_mu_u(double * const mu_x, const tree_ocp_qp_out * const qp_out, const int indx);

void tree_ocp_qp_out_set_node_mu_d(const double * const mu_x, tree_ocp_qp_out * const qp_out, const int indx);

void tree_ocp_qp_out_get_node_mu_d(double * const mu_x, const tree_ocp_qp_out * const qp_out, const int indx);



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
