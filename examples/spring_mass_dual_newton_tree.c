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

// NOTE(dimitris): Current limitations
// TODO(dimitris): write limitations here

#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/src/dual_Newton_tree.h"
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

#include "examples/spring_mass_utils/data.c"

// TODO(dimitris): check that all these options are supported for tree version
// TODO(dimitris): put this function in common file
treeqp_tdunes_options_t set_default_options(void) {
    treeqp_tdunes_options_t opts;
    termination_t cond = TREEQP_INFNORM;

    #ifdef READ_OPTIONS_FROM_C_FILE
    opts.maxIter = iterNEWTON;
    opts.termCondition = cond;
    opts.stationarityTolerance = termNEWTON;

    opts.lineSearchMaxIter = maxIterLS;
    opts.lineSearchGamma = gammaLS;
    opts.lineSearchBeta = betaLS;

    opts.regType  = typeREG;
    // opts.regTol   = tolREG;
    opts.regValue = valueREG;
    #else
    opts.maxIter = 100;
    opts.termCondition = cond;
    opts.stationarityTolerance = 1.0e-12;

    opts.lineSearchMaxIter = 50;
    opts.lineSearchGamma = 0.1;
    opts.lineSearchBeta = 0.6;

    // TODO(dimitris): implement on the fly regularization
    opts.regType  = TREEQP_ALWAYS_LEVENBERG_MARQUARDT;
    // opts.regTol   = 1.0e-12;
    opts.regValue = 1.0e-8;
    #endif

    return opts;
}


int main( ) {
    return_t status;

    int Nn = calculate_number_of_nodes(md, Nr, Nh);
    int Np = Nn - ipow(md, Nr);

    treeqp_tdunes_options_t opts = set_default_options();

    // read initial point from txt file
    int nl = Nn*NX;
    double *lambda = malloc(nl*sizeof(double));
    status = read_double_vector_from_txt(lambda, nl, "examples/spring_mass_utils/lambda0_tree.txt");
    if (status != TREEQP_OK) return -1;

    // read constraint on x0 from txt file
    double x0[NX];
    status = read_double_vector_from_txt(x0, NX, "examples/spring_mass_utils/x0.txt");
    if (status != TREEQP_OK) return status;

    // setup scenario tree
    struct node *tree = malloc(Nn*sizeof(struct node));
    setup_multistage_tree(md, Nr, Nh, Nn, tree);

    // setup QP
    tree_ocp_qp_in qp_in;

    int *nx = malloc(Nn*sizeof(int));
    int *nu = malloc(Nn*sizeof(int));

    for (int ii = 0; ii < Nn; ii++) {
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

    int qp_in_size = tree_ocp_qp_in_calculate_size(Nn, nx, nu, tree);
    void *qp_in_memory = malloc(qp_in_size);
    create_tree_ocp_qp_in(Nn, nx, nu, tree, &qp_in, qp_in_memory);

    // NOTE(dimitris): skipping first dynamics that represent the nominal ones
    tree_ocp_qp_in_fill_lti_data_diag_weights(&A[NX*NX], &B[NX*NU], &b[NX], dQ, q, dP, p, dR, r,
        xmin, xmax, umin, umax, x0, &qp_in);

    // qp_in.N = 10;
    // print_tree_ocp_qp_in(&qp_in);
    // exit(1);

    // setup QP solver
    treeqp_tdunes_workspace work;

    int treeqp_size = treeqp_tdunes_calculate_size(&qp_in);
    void *qp_solver_memory = malloc(treeqp_size);
    create_treeqp_tdunes(&qp_in, &opts, &work, qp_solver_memory);

    // setup QP solution
    tree_ocp_qp_out qp_out;

    int qp_out_size = tree_ocp_qp_out_calculate_size(Nn, nx, nu);
    void *qp_out_memory = malloc(qp_out_size);
    create_tree_ocp_qp_out(Nn, nx, nu, &qp_out, qp_out_memory);

    #if PRINT_LEVEL > 0
    printf("\n-------- treeQP workspace requires %d bytes \n", treeqp_size);
    #endif

    #if PROFILE > 0
    initialize_timers( );
    #endif

    for (int jj = 0; jj < NRUNS; jj++) {
        treeqp_tdunes_set_dual_initialization(lambda, &work);

        #if PROFILE > 0
        treeqp_tic(&tot_tmr);
        #endif

        treeqp_tdunes_solve(&qp_in, &qp_out, &opts, &work);
        // exit(1);
        #if PROFILE > 0
        total_time = treeqp_toc(&tot_tmr);
        update_min_timers(jj);
        #endif
    }

    write_solution_to_txt(&qp_in, Np, qp_out.info.iter, tree, &work);

    double err = maximum_error_in_dynamic_constraints(&qp_in, &qp_out);
    printf("\nMaximum violation of dynamic constraints: %2.2e\n", err);

    #if PROFILE > 0 && PRINT_LEVEL > 0
    print_timers(qp_out.info.iter);
    #endif

    for (int ii = 0; ii < 5; ii++) {
        blasfeo_print_tran_dvec(qp_in.nx[ii], &qp_out.x[ii], 0);
    }

    // Free memory
    free(nx);
    free(nu);

    free(qp_in_memory);
    free(qp_solver_memory);
    free(qp_out_memory);

    free_tree(Nn, tree);
    free(tree);

    free(lambda);

    return 0;
}
