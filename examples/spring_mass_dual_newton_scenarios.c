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
// - simple bounds, diagonal weights
// - x0 eliminated (no MHE)
// - not varying nx, nu
// - no arbitrary trees

#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/src/dual_Newton_scenarios.h"
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

#include "examples/data_spring_mass/data.c"

treeqp_dune_options_t set_default_options(void) {
    treeqp_dune_options_t opts;
    termination_t cond = TREEQP_INFNORM;

    #ifdef READ_OPTIONS_FROM_C_FILE
    opts.maxIter = iterNEWTON;
    opts.termCondition = cond;
    opts.stationarityTolerance = termNEWTON;

    opts.lineSearchMaxIter = maxIterLS;
    opts.lineSearchGamma = gammaLS;
    opts.lineSearchBeta = betaLS;

    opts.regType  = typeREG;
    opts.regTol   = tolREG;
    opts.regValue = valueREG;
    #else
    opts.maxIter = 100;
    opts.termCondition = cond;
    opts.stationarityTolerance = 1.0e-12;

    opts.lineSearchMaxIter = 50;
    opts.lineSearchGamma = 0.1;
    opts.lineSearchBeta = 0.6;

    opts.regType  = TREEQP_ALWAYS_LEVENBERG_MARQUARDT;
    opts.regTol   = 1.0e-12;
    opts.regValue = 1.0e-8;
    #endif

    return opts;
}


int main() {
    return_t status;

    int_t nl = calculate_dimension_of_lambda(Nr, md, NU);
    int_t Nn = calculate_number_of_nodes(md, Nr, Nh);
    int_t Ns = ipow(md, Nr);

    treeqp_dune_options_t opts = set_default_options();

    check_compiler_flags();

    // read initial point from txt file
    real_t *mu = malloc(Ns*Nh*NX*sizeof(real_t));
    real_t *lambda = malloc(nl*sizeof(real_t));
    status = read_double_vector_from_txt(mu, Ns*Nh*NX, "examples/data_spring_mass/mu0_scen.txt");
    if (status != 0) return -1;
    status = read_double_vector_from_txt(lambda, nl, "examples/data_spring_mass/lambda0_scen.txt");
    if (status != 0) return -1;

    // read constraint on x0 from txt file
    real_t x0[NX];
    status = read_double_vector_from_txt(x0, NX, "examples/data_spring_mass/x0.txt");
    if (status != 0) return -1;

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
            nx[ii] = 0;  // NOTE(dimitris): x0 variable is eliminated
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

    // print_tree_ocp_qp_in(&qp_in);
    // exit(1);

    // setup QP solver
    treeqp_sdunes_workspace work;

    int_t treeqp_size = treeqp_dune_scenarios_calculate_size(&qp_in);
    void *qp_solver_memory = malloc(treeqp_size);
    create_treeqp_dune_scenarios(&qp_in, &opts, &work, qp_solver_memory);

    // setup QP solution
    tree_ocp_qp_out qp_out;

    int_t qp_out_size = tree_ocp_qp_out_calculate_size(Nn, nx, nu);
    void *qp_out_memory = malloc(qp_out_size);
    create_tree_ocp_qp_out(Nn, nx, nu, &qp_out, qp_out_memory);

    #if PRINT_LEVEL > 0
    printf("\n-------- treeQP workspace requires %d bytes \n", treeqp_size);
    #endif

    #if PROFILE > 0
    initialize_timers();
    #endif

    for (int_t jj = 0; jj < NRUNS; jj++) {
        treeqp_sdunes_set_dual_initialization(lambda, mu, &work);

        #if PROFILE > 0
        treeqp_tic(&tot_tmr);
        #endif

        status = treeqp_dune_scenarios_solve(&qp_in, &qp_out, &opts, &work);

        // printf("QP solver status at run %d: %d\n", jj, status);

        #if PROFILE > 0
        total_time = treeqp_toc(&tot_tmr);
        update_min_timers(jj);
        #endif
    }  // end NRUNS

    write_scenarios_solution_to_txt(Ns, Nh, Nr, md, NX, NU, qp_out.info.iter, &work);

    real_t err = maximum_error_in_dynamic_constraints(&qp_in, &qp_out);
    printf("\nMaximum violation of dynamic constraints: %2.2e\n", err);

    #if PROFILE > 0 && PRINT_LEVEL > 0
    print_timers(qp_out.info.iter);
    #endif

    for (int_t ii = 0; ii < 5; ii++) {
        d_print_tran_strvec(qp_in.nx[ii], &qp_out.x[ii], 0);
    }

    // Free allocated memory
    free(nx);
    free(nu);

    free(qp_in_memory);
    free(qp_solver_memory);
    free(qp_out_memory);

    free_tree(Nn, tree);
    free(tree);

    free(mu);
    free(lambda);

    return 0;
}
