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

// NOTE(dimitris): Current limitations
// TODO(dimitris): write limitations here

#include "treeqp/src/tree_qp_common.h"
#include "treeqp/src/dual_Newton_tree.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/memory.h"
#include "treeqp/utils/profiling.h"
#include "treeqp/utils/tree.h"
#include "treeqp/utils/utils.h"
#include "treeqp/utils/timing.h"

#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_v_aux_ext_dep.h>
#include <blasfeo_d_aux_ext_dep.h>
#include <blasfeo_d_blas.h>

#include "examples/spring_mass_utils/data.c"

int main( ) {
    return_t status;

    int Nn = calculate_number_of_nodes(md, Nr, Nh);
    int Np = Nn - ipow(md, Nr);

    treeqp_tdunes_opts_t opts;
    int tdunes_opts_size = treeqp_tdunes_opts_calculate_size(Nn);
    void *tdunes_opts_mem = malloc(tdunes_opts_size);
    treeqp_tdunes_opts_create(Nn, &opts, tdunes_opts_mem);
    treeqp_tdunes_opts_set_default(Nn, &opts);

    #ifdef READ_TREE_OPTIONS_FROM_C_FILE
    treeqp_tdunes_matlab_options(Nn, &opts);
    #endif

    // read initial point from txt file
    int nl = (Nn-1)*NX;
    double *lambda = malloc(nl*sizeof(double));
    status = read_double_vector_from_txt(lambda, nl, "examples/spring_mass_utils/lambda0_tree.txt");
    if (status != TREEQP_OK) return -1;

    // read constraint on x0 from txt file
    double x0[NX];
    status = read_double_vector_from_txt(x0, NX, "examples/spring_mass_utils/x0.txt");
    if (status != TREEQP_OK) return status;

    // setup QP
    tree_qp_in qp_in;

    int *nx = malloc(Nn*sizeof(int));
    int *nu = malloc(Nn*sizeof(int));
    int *nk = malloc(Nn*sizeof(int));
    setup_multistage_tree(md, Nr, Nh, nk);

    for (int ii = 0; ii < Nn; ii++)
    {
        // state and input dimensions on each node (only different at root/leaves)
        if (ii > 0)
        {
            nx[ii] = NX;
        } else {
            nx[ii] = NX;
        }

        if (nk[ii] > 0)  // not a leaf
        {
            nu[ii] = NU;
        } else {
            nu[ii] = 0;
        }
    }

    int qp_in_size = tree_qp_in_calculate_size(Nn, nx, nu, NULL, nk);
    void *qp_in_memory = malloc(qp_in_size);
    tree_qp_in_create(Nn, nx, nu, NULL, nk, &qp_in, qp_in_memory);

    // NOTE(dimitris): skipping first dynamics that represent the nominal ones
    tree_qp_in_fill_lti_data_diag_weights(&A[NX*NX], &B[NX*NU], &b[NX], dQ, q, dP, p, dR, r,
        xmin, xmax, umin, umax, x0, NULL, NULL, NULL, NULL, NULL, &qp_in);

    // qp_in.N = 10;
    // tree_qp_in_print(&qp_in);
    // exit(1);

    // setup QP solver
    treeqp_tdunes_workspace work;

    int treeqp_size = treeqp_tdunes_calculate_size(&qp_in, &opts);
    void *qp_solver_memory = malloc(treeqp_size);
    treeqp_tdunes_create(&qp_in, &opts, &work, qp_solver_memory);

    // setup QP solution
    tree_qp_out qp_out;

    int qp_out_size = tree_qp_out_calculate_size(Nn, nx, nu, NULL);
    void *qp_out_memory = malloc(qp_out_size);
    tree_qp_out_create(Nn, nx, nu, NULL, &qp_out, qp_out_memory);

    #if PRINT_LEVEL > 0
    printf("\n-------- treeQP workspace requires %d bytes \n", treeqp_size);
    #endif

    #if PROFILE > 0
    initialize_timers( );
    #endif

    for (int jj = 0; jj < NREP; jj++)
    {
        // TODO(dimitris): set dual init. in qp_out
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

    write_solution_to_txt(&qp_in, Np, qp_out.info.iter, qp_in.tree, &work);

    #if PROFILE > 0 && PRINT_LEVEL > 0
    print_timers(qp_out.info.iter);
    #endif

    #if PRINT_LEVEL > 0
    for (int ii = 0; ii < 5; ii++) {
        blasfeo_print_tran_dvec(qp_in.nx[ii], &qp_out.x[ii], 0);
    }
    #endif

    double kkt_err = tree_qp_out_max_KKT_res(&qp_in, &qp_out);
    #if PRINT_LEVEL > 0
    printf("Maximum error in KKT residuals (tdunes):\t\t %2.2e\n\n", kkt_err);
    assert(kkt_err < 1e-8 && "KKT tolerance in spring_mass_dual_newton_tree.c too high!");
    #endif

    // Free memory
    free(nx);
    free(nu);
    free(nk);

    free(qp_in_memory);
    free(tdunes_opts_mem);
    free(qp_solver_memory);
    free(qp_out_memory);

    free(lambda);

    return 0;
}
