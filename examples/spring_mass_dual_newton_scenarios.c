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

// #include <xmmintrin.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// NOTE(dimitris): Current limitations
// - simple bounds, diagonal weights
// - x0 eliminated (no MHE)
// - not varying nx, nu
// - no arbitrary trees

#include "treeqp/src/tree_qp_common.h"
#include "treeqp/src/dual_Newton_scenarios.h"
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


int main() {
    // NOTE(dimitris): to detect NaNs on a mac (together with xmmintrin.h)
    // _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);

    return_t status;

    int nl = treeqp_sdunes_calculate_dual_dimension(Nr, md, NU);
    int Nn = calculate_number_of_nodes(md, Nr, Nh);
    int Ns = ipow(md, Nr);

    treeqp_sdunes_opts_t opts;
    treeqp_sdunes_opts_set_default(Nn, &opts);

    #ifdef READ_SCENARIOS_OPTIONS_FROM_C_FILE
    treeqp_sdunes_matlab_options(&opts);
    #endif

    // read initial point from txt file
    double *mu = malloc(Ns*Nh*NX*sizeof(double));
    double *lambda = malloc(nl*sizeof(double));
    status = read_double_vector_from_txt(mu, Ns*Nh*NX, "examples/spring_mass_utils/mu0_scen.txt");
    if (status != TREEQP_OK) return -1;
    status = read_double_vector_from_txt(lambda, nl, "examples/spring_mass_utils/lambda0_scen.txt");
    if (status != TREEQP_OK) return -1;

    // read constraint on x0 from txt file
    double x0[NX];
    status = read_double_vector_from_txt(x0, NX, "examples/spring_mass_utils/x0.txt");
    if (status != TREEQP_OK) return -1;

    // setup QP
    tree_qp_in qp_in;

    int *nx = malloc(Nn*sizeof(int));
    int *nu = malloc(Nn*sizeof(int));
    int *nk = malloc(Nn*sizeof(int));
    setup_multistage_tree(md, Nr, Nh, nk);

    for (int ii = 0; ii < Nn; ii++)
    {
        nx[ii] = NX;

        if (nk[ii] > 0)  // not a leaf
        {
            nu[ii] = NU;
        }
        else
        {
            nu[ii] = 0;
        }
    }

    int qp_in_size = tree_qp_in_calculate_size(Nn, nx, nu, NULL, nk);
    void *qp_in_memory = malloc(qp_in_size);
    tree_qp_in_create(Nn, nx, nu, NULL, nk, &qp_in, qp_in_memory);

    // NOTE(dimitris): skipping first dynamics that represent the nominal ones
    tree_qp_in_fill_lti_data_diag_weights(&A[NX*NX], &B[NX*NU], &b[NX], dQ, q, dP, p, dR, r,
        xmin, xmax, umin, umax, x0, NULL, NULL, NULL, NULL, NULL, &qp_in);

    // tree_qp_in_print(&qp_in);
    // exit(1);

    // setup QP solution
    tree_qp_out qp_out;

    int qp_out_size = tree_qp_out_calculate_size(Nn, nx, nu, NULL);
    void *qp_out_memory = malloc(qp_out_size);
    tree_qp_out_create(Nn, nx, nu, NULL, &qp_out, qp_out_memory);

    // eliminate x0 from QP
    tree_qp_in_eliminate_x0(&qp_in);
    tree_qp_out_eliminate_x0(&qp_out);

    // setup QP solver
    treeqp_sdunes_workspace work;

    int treeqp_size = treeqp_sdunes_calculate_size(&qp_in, &opts);
    void *qp_solver_memory = malloc(treeqp_size);
    treeqp_sdunes_create(&qp_in, &opts, &work, qp_solver_memory);

    #if PRINT_LEVEL > 0
    printf("\n-------- treeQP workspace requires %d bytes \n", treeqp_size);
    #endif

    for (int jj = 0; jj < NREP; jj++)
    {
        treeqp_sdunes_set_dual_initialization(lambda, mu, &work);
        status = treeqp_sdunes_solve(&qp_in, &qp_out, &opts, &work);
        // printf("QP solver status at run %d: %d\n", jj, status);
    }
    write_scenarios_solution_to_txt(Ns, Nh, Nr, md, NX, NU, qp_out.info.iter, &work);

    #if PROFILE > 0 && PRINT_LEVEL > 0
    timers_print(&work.timings);
    #endif

    #if PRINT_LEVEL > 0
    for (int ii = 0; ii < 5; ii++) {
        blasfeo_print_tran_dvec(qp_in.nx[ii], &qp_out.x[ii], 0);
    }
    #endif

    double kkt_err = tree_qp_out_max_KKT_res(&qp_in, &qp_out);

    #if PRINT_LEVEL > 0
    printf("Maximum error in KKT residuals (sdunes):\t\t %2.2e\n\n", kkt_err);
    assert(kkt_err < 1e-8 && "KKT tolerance in spring_mass_dual_newton_scenarios example too high!");
    #endif

    // Free allocated memory
    free(nx);
    free(nu);
    free(nk);

    free(qp_in_memory);
    free(qp_solver_memory);
    free(qp_out_memory);

    free(mu);
    free(lambda);

    return 0;
}
