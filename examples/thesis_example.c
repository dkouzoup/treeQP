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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "treeqp/src/tree_qp_common.h"
#include "treeqp/src/dual_Newton_tree.h"
#include "treeqp/src/hpmpc_tree.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/tree.h"
#include "treeqp/utils/profiling.h"
#include "treeqp/utils/print.h"
#include "treeqp/utils/blasfeo.h"

#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_v_aux_ext_dep.h>
#include <blasfeo_d_aux_ext_dep.h>
#include <blasfeo_d_blas.h>

// #define USE_HPMPC

int main() {

    int nk[] = {2, 2, 1, 0, 0, 0};
    int nx[] = {2, 2, 2, 2, 2, 2};
    int nu[] = {1, 1, 1, 0, 0, 0};

    tree_qp_in qp_in;

    int in_size = tree_qp_in_calculate_size(6, nx, nu, NULL, nk);
    void *in_mem = malloc(in_size);
    tree_qp_in_create(6, nx, nu, NULL, nk, &qp_in, in_mem);

    double A1[] = {1.1, 3.3, 2.2, 4.4};
    double A2[] = {5.5, 7.7, 6.6, 8.8};

    double B1[] = {1.0, 2.0};
    double B2[] = {3.0, 4.0};

    double b1[] = {0.0, 0.0};
    double b2[] = {1.0, 1.0};

    tree_qp_in_set_edge_dynamics_colmajor(A1, B1, b1, &qp_in, 0);
    tree_qp_in_set_edge_dynamics_colmajor(A1, B1, b1, &qp_in, 2);
    tree_qp_in_set_edge_dynamics_colmajor(A2, B2, b2, &qp_in, 1);
    tree_qp_in_set_edge_dynamics_colmajor(A2, B2, b2, &qp_in, 3);
    tree_qp_in_set_edge_dynamics_colmajor(A2, B2, b2, &qp_in, 4);

    double Qd[] = {2.0, 2.0};
    double Rd[] = {1.0};
    double q[] = {0.0, 0.0};
    double r[] = {0.0};

    for (int ii = 0; ii < 6; ii++)
        tree_qp_in_set_node_objective_diag(Qd, Rd, q, r, &qp_in, ii);

    double x0[] = {2.1, 2.1};
    double umin[] = {-1};
    double umax[] = {1};

    tree_qp_in_set_node_xmin(x0, &qp_in, 0);
    tree_qp_in_set_node_xmax(x0, &qp_in, 0);

    for (int ii = 0; ii < 3; ii++)
    {
        tree_qp_in_set_node_umin(umin, &qp_in, ii);
        tree_qp_in_set_node_umax(umax, &qp_in, ii);
    }

    tree_qp_in_print(&qp_in);

    tree_qp_out qp_out;

    int out_size = tree_qp_out_calculate_size(6, nx, nu, NULL);
    void *out_mem = malloc(out_size);
    tree_qp_out_create(6, nx, nu, NULL, &qp_out, out_mem);

    #ifndef USE_HPMPC
    treeqp_tdunes_opts_t opts;
    int opts_size = treeqp_tdunes_opts_calculate_size(6);
    void *opts_mem = malloc(opts_size);
    treeqp_tdunes_opts_create(6, &opts, opts_mem);
    treeqp_tdunes_opts_set_default(6, &opts);

    for (int ii = 0; ii < 6; ii++)
        opts.qp_solver[ii] = TREEQP_CLIPPING_SOLVER;

    opts.maxIter = 100;

    treeqp_tdunes_workspace work;

    int alg_size = treeqp_tdunes_calculate_size(&qp_in, &opts);
    void *alg_mem = malloc(alg_size);
    treeqp_tdunes_create(&qp_in, &opts, &work, alg_mem);

    int status = treeqp_tdunes_solve(&qp_in, &qp_out, &opts, &work);

    #else

    treeqp_hpmpc_opts_t opts;
    int opts_size = treeqp_hpmpc_opts_calculate_size(6);
    void *opts_mem = malloc(opts_size);
    treeqp_hpmpc_opts_create(6, &opts, opts_mem);
    treeqp_hpmpc_opts_set_default(6, &opts);

    treeqp_hpmpc_workspace work;

    int alg_size = treeqp_hpmpc_calculate_size(&qp_in, &opts);
    void *alg_mem = malloc(alg_size);
    treeqp_hpmpc_create(&qp_in, &opts, &work, alg_mem);

    int status = treeqp_hpmpc_solve(&qp_in, &qp_out, &opts, &work);

    #endif

    #if PROFILE > 0
    #ifndef USE_HPMPC
    timers_print(&work.timings);
    #endif
    #endif

    tree_qp_out_print(6, &qp_out);

    printf("\nSolver status: %d\n", status);
    printf("\nNumber of iterations: %d\n", qp_out.info.iter);
    printf("\nMaximum error in KKT residuals: %2.2e\n\n",
        tree_qp_out_max_KKT_res(&qp_in, &qp_out));

    free(alg_mem);
    free(opts_mem);
    free(out_mem);
    free(in_mem);

    return 0;
}
