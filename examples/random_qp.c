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

#if DATA == 0
#include "examples/random_qp_utils/data00.c"
#elif DATA == 1
#include "examples/random_qp_utils/data01.c"
#elif DATA == 2
#include "examples/random_qp_utils/data02.c"
#elif DATA == 3
#include "examples/random_qp_utils/data03.c"
#elif DATA == 4
#include "examples/random_qp_utils/data04.c"
#elif DATA == 5
#include "examples/random_qp_utils/data05.c"
#else
#include "examples/random_qp_utils/data.c"
#endif

// #define USE_HPMPC

int main()
{
    // build a small, asymemtric tree
    //
    //         3
    //       /
    //      /
    //     1 - 4
    //   /
    // 0 - 2 - 5

    // set up QP data
    tree_qp_in qp_in;

    int qp_in_size = tree_qp_in_calculate_size(Nn, nx, nu, NULL, nk);
    void *qp_in_memory = malloc(qp_in_size);
    tree_qp_in_create(Nn, nx, nu, NULL, nk, &qp_in, qp_in_memory);

    tree_qp_in_set_ltv_dynamics_colmajor(A, B, b, &qp_in);
#ifdef CLIPPING
    tree_qp_in_set_ltv_objective_diag(Qd, Rd, q, r, &qp_in);
#else
    tree_qp_in_set_ltv_objective_colmajor(Q, R, S, q, r, &qp_in);
#endif
    tree_qp_in_set_inf_bounds(&qp_in);

#if 0
    double x0[] = {1., 1.,};
    tree_qp_in_set_x0_colmaj(&qp_in, x0);
#endif

#ifndef DATA
    tree_qp_in_print(&qp_in);
#endif

    // set up QP solution
    tree_qp_out qp_out;

    int qp_out_size = tree_qp_out_calculate_size(Nn, nx, nu, NULL);
    void *qp_out_memory = malloc(qp_out_size);
    tree_qp_out_create(Nn, nx, nu, NULL, &qp_out, qp_out_memory);

#if 0
    // eliminate x0 variable
    tree_qp_in_eliminate_x0(&qp_in);
    tree_qp_out_eliminate_x0(&qp_out);
#endif

    // set up QP solver
#ifndef USE_HPMPC

    treeqp_tdunes_opts_t opts;
    int tdunes_opts_size = treeqp_tdunes_opts_calculate_size(Nn);
    void *opts_memory = malloc(tdunes_opts_size);
    treeqp_tdunes_opts_create(Nn, &opts, opts_memory);
    treeqp_tdunes_opts_set_default(Nn, &opts);

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
    treeqp_tdunes_create(&qp_in, &opts, &work, qp_solver_memory);

#else
    treeqp_hpmpc_opts_t opts;
    int hpmpc_opts_size = treeqp_hpmpc_opts_calculate_size(Nn);
    void *opts_memory = malloc(hpmpc_opts_size);
    treeqp_hpmpc_opts_create(Nn, &opts, opts_memory);
    treeqp_hpmpc_opts_set_default(Nn, &opts);

    treeqp_hpmpc_workspace work;

    int treeqp_size = treeqp_hpmpc_calculate_size(&qp_in, &opts);
    void *qp_solver_memory = malloc(treeqp_size);
    treeqp_hpmpc_create(&qp_in, &opts, &work, qp_solver_memory);
#endif  // USE_HPMPC

    // solve QP
#ifndef USE_HPMPC
    treeqp_tdunes_solve(&qp_in, &qp_out, &opts, &work);
#else
    treeqp_hpmpc_solve(&qp_in, &qp_out, &opts, &work);
#endif

#ifndef DATA
#if PROFILE > 0 && PRINT_LEVEL > 0
    timers_print(&work.timings);
#endif
    tree_qp_out_print(Nn, &qp_out);
    print_blasfeo_target();
#endif

    int indx = 0;
    int indu = 0;
    double err;
    double max_err = 0;

    for (int ii = 0; ii < qp_in.N; ii++)
    {
        // printf("x = \n");
        // blasfeo_print_exp_tran_dvec(qp_in.nx[ii], &qp_out.x[ii], 0);
        // printf("xopt = \n");
        // d_print_exp_mat(1, qp_in.nx[ii], &xopt[indx], 1);

        err = check_error_strvec_double(&qp_out.x[ii], &xopt[indx]);
        max_err = MAX(err, max_err);
        indx += qp_in.nx[ii];

        // printf("error = %e\n\n", err);

        // printf("u=\n");
        // blasfeo_print_exp_tran_dvec(qp_in.nu[ii], &qp_out.u[ii], 0);
        // printf("uopt = \n");
        // d_print_exp_mat(1, qp_in.nu[ii], &uopt[indu], 1);

        err = check_error_strvec_double(&qp_out.u[ii], &uopt[indu]);
        max_err = MAX(err, max_err);
        indu += qp_in.nu[ii];

        // printf("error = %e\n\n", err);
    }

#ifndef USE_HPMPC
    printf("SOLVER:\ttdunes\n");
#else
    printf("SOLVER:\thpmpc\n");
#endif

    printf("ITERS:\t%d\n", qp_out.info.iter);

    double kkt_err = tree_qp_out_max_KKT_res(&qp_in, &qp_out);
    printf("KKT:\t%2.2e\n", kkt_err);
    printf("ERROR:\t%e\n\n", max_err);

    free(qp_solver_memory);
    free(opts_memory);
    free(qp_out_memory);
    free(qp_in_memory);

    assert(kkt_err < 1e-12 && "maximum KKT residual too high!");
    assert(max_err < 1e-12 && "deviation from given solution too high!");
    assert(qp_out.info.iter == 1 || qp_out.info.iter == 0 && "Unconstrained QP did not converge in one iteration!");

    return 0;
}
