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


#include <stdio.h>
#include <stdlib.h>

#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/src/dual_Newton_tree.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/tree_utils.h"
#include "treeqp/utils/profiling_utils.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_v_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blas.h"

#include "examples/data_random_qp/data.c"

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
    setup_tree(Nn, nc, tree);
    // for (int_t ii = 0; ii < Nn; ii++) {
    //     print_node(&tree[ii]);
    // }

    // set up QP data
    tree_ocp_qp_in qp_in;

    int_t qp_in_size = tree_ocp_qp_in_calculate_size(Nn, nx, nu, tree);
    void *qp_in_memory = malloc(qp_in_size);
    create_tree_ocp_qp_in(Nn, nx, nu, tree, &qp_in, qp_in_memory);

    tree_ocp_qp_in_read_dynamics_colmajor(A, B, b, &qp_in);
    tree_ocp_qp_in_read_objective_diag_colmajor(Qd, Rd, q, r, &qp_in);
    tree_ocp_qp_in_set_inf_bounds(&qp_in);

    print_tree_ocp_qp_in(&qp_in);

    // set up QP solver
    treeqp_tdunes_options_t opts;

    // TODO(dimitris): move to function in solver
    opts.maxIter = 10;
    opts.termCondition = TREEQP_INFNORM;
    opts.stationarityTolerance = 1.0e-12;
    opts.lineSearchMaxIter = 100;
    opts.lineSearchGamma = 0.1;
    opts.lineSearchBeta = 0.8;
    opts.regType  = TREEQP_ALWAYS_LEVENBERG_MARQUARDT;
    opts.regValue = 1.0e-8;

    treeqp_tdunes_workspace work;

    int_t treeqp_size = treeqp_tdunes_calculate_size(&qp_in);
    void *qp_solver_memory = malloc(treeqp_size);
    create_treeqp_tdunes(&qp_in, &opts, &work, qp_solver_memory);

    // set up QP solution
    tree_ocp_qp_out qp_out;

    int_t qp_out_size = tree_ocp_qp_out_calculate_size(Nn, nx, nu);
    void *qp_out_memory = malloc(qp_out_size);
    create_tree_ocp_qp_out(Nn, nx, nu, &qp_out, qp_out_memory);

    // solve QP
    initialize_timers( );
    treeqp_tdunes_solve(&qp_in, &qp_out, &opts, &work);
    update_min_timers(0);


    #if PROFILE > 0 && PRINT_LEVEL > 0
    print_timers(qp_out.info.iter);
    #endif

    // TODO(dimitris): print_ocp_qp_out function
    int_t indx = 0;
    int_t indu = 0;
    for (int_t ii = 0; ii < qp_in.N; ii++) {
        printf("--------\n");
        printf(" Node %d\n", ii);
        printf("--------\n");
        printf("x = \n");
        d_print_tran_strvec(qp_in.nx[ii], &qp_out.x[ii], 0);
        printf("xopt = \n");
        d_print_mat(1, qp_in.nx[ii], &xopt[indx], 1);
        indx += qp_in.nx[ii];

        printf("u=\n");
        d_print_tran_strvec(qp_in.nu[ii], &qp_out.u[ii], 0);
        printf("uopt = \n");
        d_print_mat(1, qp_in.nu[ii], &uopt[indu], 1);
        indu += qp_in.nu[ii];
    }
    printf("ITERS = %d\n", qp_out.info.iter);

    free(qp_solver_memory);
    free(qp_out_memory);
    free(qp_in_memory);

    free_tree(Nn, tree);
    free(tree);

    return 0;
}