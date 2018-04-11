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

#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/src/dual_Newton_tree.h"
#include "treeqp/src/hpmpc_tree.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/tree.h"
#include "treeqp/utils/profiling.h"
#include "treeqp/utils/blasfeo.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_v_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blas.h"

#include "examples/random_qp_utils/data.c"

// #define USE_HPMPC

int main() {
    // build a small, asymemtric tree
    //
    //         3
    //       /
    //      /
    //     1 - 4
    //   /
    // 0 - 2 - 5

    struct node *tree = malloc(Nn*sizeof(struct node));
    setup_tree(Nn, ns, tree);
    // for (int ii = 0; ii < Nn; ii++) print_node(&tree[ii]);

    // set up QP data
    tree_ocp_qp_in qp_in;

    int qp_in_size = tree_ocp_qp_in_calculate_size(Nn, nx, nu, NULL, tree);
    void *qp_in_memory = malloc(qp_in_size);
    tree_ocp_qp_in_create(Nn, nx, nu, NULL, tree, &qp_in, qp_in_memory);

    tree_ocp_qp_in_read_dynamics_colmajor(A, B, b, &qp_in);
    #ifdef CLIPPING
    tree_ocp_qp_in_read_objective_diag(Qd, Rd, q, r, &qp_in);
    #else
    tree_ocp_qp_in_read_objective_colmajor(Q, R, S, q, r, &qp_in);
    #endif
    tree_ocp_qp_in_set_inf_bounds(&qp_in);

    print_tree_ocp_qp_in(&qp_in);

    // set up QP solver
#ifndef USE_HPMPC
    treeqp_tdunes_options_t opts = treeqp_tdunes_default_options(Nn);

    opts.maxIter = 10;
    opts.stationarityTolerance = 1.0e-10;
    opts.regType  = TREEQP_NO_REGULARIZATION;

#ifdef CLIPPING
    for (int ii = 0; ii < Nn; ii++) opts.qp_solver[ii] = TREEQP_CLIPPING_SOLVER;
#else
    for (int ii = 0; ii < Nn; ii++) opts.qp_solver[ii] = TREEQP_QPOASES_SOLVER;
#endif

    treeqp_tdunes_workspace work;

    int treeqp_size = treeqp_tdunes_calculate_size(&qp_in, &opts);
    void *qp_solver_memory = malloc(treeqp_size);
    create_treeqp_tdunes(&qp_in, &opts, &work, qp_solver_memory);

#else
    treeqp_hpmpc_options_t opts = treeqp_hpmpc_default_options(Nn);
    treeqp_hpmpc_workspace work;

    int treeqp_size = treeqp_hpmpc_calculate_size(&qp_in, &opts);
    void *qp_solver_memory = malloc(treeqp_size);
    create_treeqp_hpmpc(&qp_in, &opts, &work, qp_solver_memory);
#endif

    // set up QP solution
    tree_ocp_qp_out qp_out;

    int qp_out_size = tree_ocp_qp_out_calculate_size(Nn, nx, nu, NULL);
    void *qp_out_memory = malloc(qp_out_size);
    tree_ocp_qp_out_create(Nn, nx, nu, NULL, &qp_out, qp_out_memory);

    // solve QP
#if PROFILE > 0
    initialize_timers( );
#endif
#ifndef USE_HPMPC
    treeqp_tdunes_solve(&qp_in, &qp_out, &opts, &work);
#else
    treeqp_hpmpc_solve(&qp_in, &qp_out, &opts, &work);
#endif
#if PROFILE > 0
    update_min_timers(0);
#endif

    #if PROFILE > 0 && PRINT_LEVEL > 0
    print_timers(qp_out.info.iter);
    #endif

    // TODO(dimitris): print_ocp_qp_out function
    int indx = 0;
    int indu = 0;
    for (int ii = 0; ii < qp_in.N; ii++)
    {
        printf("--------\n");
        printf(" Node %d\n", ii);
        printf("--------\n");
        printf("x = \n");
        blasfeo_print_exp_tran_dvec(qp_in.nx[ii], &qp_out.x[ii], 0);
        printf("xopt = \n");
        d_print_e_mat(1, qp_in.nx[ii], &xopt[indx], 1);
        indx += qp_in.nx[ii];

        printf("u=\n");
        blasfeo_print_exp_tran_dvec(qp_in.nu[ii], &qp_out.u[ii], 0);
        printf("uopt = \n");
        d_print_e_mat(1, qp_in.nu[ii], &uopt[indu], 1);
        indu += qp_in.nu[ii];
    }
    printf("ITERS = %d\n", qp_out.info.iter);

    double kkt_err = max_KKT_residual(&qp_in, &qp_out);
    printf("\nMaximum error in KKT residuals:\t%2.2e\n\n", kkt_err);

    free(qp_solver_memory);
    free(qp_out_memory);
    free(qp_in_memory);

    free_tree(Nn, tree);
    free(tree);

    print_blasfeo_target();

    assert(qp_out.info.iter == 1 && "Unconstrained QP did not converge in out iteration!");
    assert(kkt_err < 1e-12 && "maximum KKT residual too high!");

    return 0;
}