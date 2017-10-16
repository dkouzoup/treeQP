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

#include "treeqp/src/dual_Newton_tree.h"
#include "treeqp/src/hpmpc_tree.h"
#include "treeqp/src/tree_ocp_qp_common.h"

// TODO(dimitris): remove flags.h
#include "treeqp/flags.h"

#include "treeqp/utils/types.h"
#include "treeqp/utils/blasfeo_utils.h"
#include "treeqp/utils/profiling_utils.h"
#include "treeqp/utils/tree_utils.h"
#include "treeqp/utils/utils.h"
#include "treeqp/utils/timing.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_v_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blas.h"

// TODO(dimitris): clean this up (and add matlab/python gen. script in utils)
#include "examples/spring_mass_utils/data.c"

int main( ) {
    return_t status;

    int_t Nn = calculate_number_of_nodes(md, Nr, Nh);
    int_t Np = Nn - ipow(md, Nr);

    // read initial point from txt file
    int_t nl = Nn*NX;
    real_t *lambda = malloc(nl*sizeof(real_t));
    status = read_double_vector_from_txt(lambda, nl, "examples/spring_mass_utils/lambda0_tree.txt");
    if (status != TREEQP_OK) return -1;

    // read constraint on x0 from txt file
    real_t x0[NX];
    status = read_double_vector_from_txt(x0, NX, "examples/spring_mass_utils/x0.txt");
    if (status != TREEQP_OK) return status;

    // setup scenario tree
    struct node *tree = malloc(Nn*sizeof(struct node));
    setup_multistage_tree(md, Nr, Nh, Nn, tree);

    // setup QP
    tree_ocp_qp_in qp_in;

    int_t *nx = malloc(Nn*sizeof(int_t));
    int_t *nu = malloc(Nn*sizeof(int_t));

    for (int_t ii = 0; ii < Nn; ii++) {
        // state and input dimensions on each node (only different at root/leaves)
        if (ii > 0) {
            nx[ii] = NX;
        } else {
            nx[ii] = NX;
        }

        if (tree[ii].nkids > 0) {  // not a leaf
            nu[ii] = NU;
        } else {
            nu[ii] = 0;
        }
    }

    int_t qp_in_size = tree_ocp_qp_in_calculate_size(Nn, nx, nu, tree);
    void *qp_in_memory = malloc(qp_in_size);
    create_tree_ocp_qp_in(Nn, nx, nu, tree, &qp_in, qp_in_memory);

    // NOTE(dimitris): skipping first dynamics that represent the nominal ones
    tree_ocp_qp_in_fill_lti_data_diag_weights(&A[NX*NX], &B[NX*NU], &b[NX], dQ, q, dP, p, dR, r,
        xmin, xmax, umin, umax, x0, &qp_in);

    // set up tree-sparse dual Newton solver
    treeqp_tdunes_options_t tdunes_opts = treeqp_tdunes_default_options();

    treeqp_tdunes_workspace tdunes_work;
    void *tdunes_memory = malloc(treeqp_tdunes_calculate_size(&qp_in));
    create_treeqp_tdunes(&qp_in, &tdunes_opts, &tdunes_work, tdunes_memory);

    // set up HPMPC solver
	treeqp_hpmpc_options_t hpmpc_opts = treeqp_hpmpc_default_options();

    treeqp_hpmpc_workspace hpmpc_work;
    void *hpmpc_memory = malloc(treeqp_hpmpc_calculate_size(&qp_in, &hpmpc_opts));
    create_treeqp_hpmpc(&qp_in, &hpmpc_opts, &hpmpc_work, hpmpc_memory);

    // setup QP solution
    tree_ocp_qp_out qp_out;

    int_t qp_out_size = tree_ocp_qp_out_calculate_size(Nn, nx, nu);
    void *qp_out_memory = malloc(qp_out_size);
    create_tree_ocp_qp_out(Nn, nx, nu, &qp_out, qp_out_memory);

    // solve with tree-sparse dual Newton strategy
    real_t overhead;
    real_t max_overhead = 0;
    for (int_t jj = 0; jj < NRUNS; jj++) {
        treeqp_tdunes_set_dual_initialization(lambda, &tdunes_work);
        treeqp_tdunes_solve(&qp_in, &qp_out, &tdunes_opts, &tdunes_work);
        printf("tdunes run # %d (%d iterations)\n", jj, qp_out.info.iter);
        printf("solver time:\t %5.3f ms\n", qp_out.info.solver_time*1e3);
        printf("interface time:\t %5.3f ms\n", qp_out.info.interface_time*1e3);
        overhead = 100*qp_out.info.interface_time/qp_out.info.solver_time;
        printf("overhead:\t %5.2f %% \n", overhead);
        if (overhead > max_overhead) max_overhead = overhead;
    }

    real_t err = maximum_error_in_dynamic_constraints(&qp_in, &qp_out);
    printf("\nMaximum violation of dynamic constraints (tdunes):\t %2.2e\n\n", err);

    real_t kkt_err = max_KKT_residual(&qp_in, &qp_out);
    printf("Maximum error in KKT residuals (tdunes):\t\t %2.2e\n\n", kkt_err);

    printf("Maximum overhead of treeQP interface (tdunes):\t\t %4.2f%%\n\n", max_overhead);

    for (int_t ii = 0; ii < 5; ii++) {
        d_print_tran_strvec(qp_in.nx[ii], &qp_out.x[ii], 0);
    }

    // solve with tree-sparse HPMPC
    max_overhead = 0;
    for (int_t jj = 0; jj < NRUNS; jj++) {
        treeqp_hpmpc_solve(&qp_in, &qp_out, &hpmpc_opts, &hpmpc_work);
        printf("hpmpc run # %d (%d iterations)\n", jj, qp_out.info.iter);
        printf("solver time:\t %5.2f ms\n", qp_out.info.solver_time*1e3);
        printf("interface time:\t %5.2f ms\n", qp_out.info.interface_time*1e3);
        overhead = 100*qp_out.info.interface_time/qp_out.info.solver_time;
        printf("overhead:\t %5.2f %% \n", overhead);
        if (overhead > max_overhead) max_overhead = overhead;
    }

    err = maximum_error_in_dynamic_constraints(&qp_in, &qp_out);
    printf("\nMaximum violation of dynamic constraints (hpmpc):\t %2.2e\n\n", err);

    kkt_err = max_KKT_residual(&qp_in, &qp_out);
    printf("Maximum error in KKT residuals (hpmpc):\t\t\t %2.2e\n\n", kkt_err);

    printf("Maximum overhead of treeQP interface (hpmpc):\t\t %4.2f%%\n\n", max_overhead);

    for (int_t ii = 0; ii < 5; ii++) {
        d_print_tran_strvec(qp_in.nx[ii], &qp_out.x[ii], 0);
    }

    // Free memory
    free(nx);
    free(nu);

    free(qp_in_memory);
    free(qp_out_memory);

    free(tdunes_memory);
    free(hpmpc_memory);

    free_tree(Nn, tree);
    free(tree);

    free(lambda);

    return 0;
}
