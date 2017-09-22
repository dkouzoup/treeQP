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
#include <assert.h>
#include <unistd.h>  // NOTE(dimitris): to read current directory
#include <string.h>

#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/src/dual_Newton_tree.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/tree_utils.h"
#include "treeqp/utils/utils.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_v_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blas.h"

#include "examples/fault_tolerance_utils/markov_tree.c"

int main() {

    int_t MPCsteps = 10;

    real_t *stateTrajectory = malloc(nx[0]*(MPCsteps+1)*sizeof(real_t));
    real_t *inputTrajectory = malloc(nu[0]*MPCsteps*sizeof(real_t));

    for (int_t jj = 0; jj < nx[0]; jj++) {
        stateTrajectory[jj] = x0[jj];
    }

    struct node *tree = malloc(Nn*sizeof(struct node));
    setup_tree(Nn, nc, tree);

    // set up QP data
    tree_ocp_qp_in qp_in;

    int_t qp_in_size = tree_ocp_qp_in_calculate_size(Nn, nx, nu, tree);
    void *qp_in_memory = malloc(qp_in_size);
    create_tree_ocp_qp_in(Nn, nx, nu, tree, &qp_in, qp_in_memory);

    tree_ocp_qp_in_read_dynamics_colmajor(A, B, b, &qp_in);
    tree_ocp_qp_in_read_objective_diag_colmajor(Qd, Rd, q, r, &qp_in);
    tree_ocp_qp_in_set_constant_bounds(xmin, xmax, umin, umax, &qp_in);
    tree_ocp_qp_in_set_x0_bounds(&qp_in, x0);

    // print_tree_ocp_qp_in(&qp_in);

    // set up QP solver
    treeqp_tdunes_options_t opts;

    opts.maxIter = 100;
    opts.termCondition = TREEQP_INFNORM;
    opts.stationarityTolerance = 1.0e-12;
    opts.lineSearchMaxIter = 50;
    opts.lineSearchGamma = 0.1;
    opts.lineSearchBeta = 0.8;
    opts.regType  = TREEQP_NO_REGULARIZATION;
    opts.regValue = 0.0;

    treeqp_tdunes_workspace work;

    int_t treeqp_size = treeqp_tdunes_calculate_size(&qp_in);
    void *qp_solver_memory = malloc(treeqp_size);
    create_treeqp_tdunes(&qp_in, &opts, &work, qp_solver_memory);

    // set up QP solution
    tree_ocp_qp_out qp_out;

    int_t qp_out_size = tree_ocp_qp_out_calculate_size(Nn, nx, nu);
    void *qp_out_memory = malloc(qp_out_size);
    create_tree_ocp_qp_out(Nn, nx, nu, &qp_out, qp_out_memory);

    real_t err;

    // MPC loop
    for (int_t tt = 0; tt < MPCsteps; tt++) {

        // solve QP
        treeqp_tdunes_solve(&qp_in, &qp_out, &opts, &work);

        // check that x0 constraint is satisfied
        for (int_t jj = 0; jj < nx[0]; jj++) {
            assert(DVECEL_LIBSTR(&qp_out.x[0], jj) == x0[jj]);
        }

        // simulate system
        for (int_t ii = 0; ii < nx[0]; ii++) {
            x0[ii] = b[ii];
            for (int_t jj = 0; jj < nx[0]; jj++) {
                x0[ii] += A[ii + jj * nx[0]] * DVECEL_LIBSTR(&qp_out.x[0], jj);
            }
            for (int_t jj = 0; jj < nu[0]; jj++)
                x0[ii] += B[ii + jj * nx[0]] * DVECEL_LIBSTR(&qp_out.u[0], jj);
        }

        // print iteration results
        err = maximum_error_in_dynamic_constraints(&qp_in, &qp_out);

        printf("----------------------------------------------------\n");
        printf("\n MPC iteration #%d converged in %d iterations\n\n", tt+1, qp_out.info.iter);
        printf(" Max. violation of dynamic constraints: %2.2e\n", err);
        printf("----------------------------------------------------\n");
        printf("x = \n");
        d_print_e_tran_strvec(nx[0], &qp_out.x[0], 0);
        printf("u = \n");
        d_print_e_tran_strvec(nu[0], &qp_out.u[0], 0);

        if (qp_out.info.iter == opts.maxIter) {
            printf("Maximum number of iterations reached!\n");
            exit(1);
        }
        if (err > 1e-10) {
            printf("Violation of dynamic constraints too high!\n");
            exit(1);
        }

        // save state and input trajectories
        for (int_t jj = 0; jj < nx[0]; jj++) {
            stateTrajectory[jj + (tt+1)*nx[0]] = x0[jj];
        }
        for (int_t jj = 0; jj < nu[0]; jj++) {
            inputTrajectory[jj + tt*nu[0]] = DVECEL_LIBSTR(&qp_out.u[0], jj);
        }

        // update bound on x0
        tree_ocp_qp_in_set_x0_bounds(&qp_in, x0);

    }

    // store results in txt files
    char cwd[1024];
    getcwd(cwd, sizeof(cwd));
    fprintf(stdout, "\nCurrent working dir: %s\n\n", cwd);

    // TODO(dimitris): do this in a more general way
    if(strstr(cwd, "examples") != NULL) {
        write_qp_out_to_txt(&qp_in, &qp_out, "fault_tolerance_utils");
        write_double_vector_to_txt(stateTrajectory, nx[0]*(MPCsteps+1), "fault_tolerance_utils/xMPC.txt");
        write_double_vector_to_txt(inputTrajectory, nu[0]*MPCsteps, "fault_tolerance_utils/uMPC.txt");
    } else {
        write_qp_out_to_txt(&qp_in, &qp_out, "examples/fault_tolerance_utils");
        write_double_vector_to_txt(stateTrajectory, nx[0]*(MPCsteps+1), "examples/fault_tolerance_utils/xMPC.txt");
        write_double_vector_to_txt(inputTrajectory, nu[0]*MPCsteps, "examples/fault_tolerance_utils/uMPC.txt");
    }

    free(qp_solver_memory);
    free(qp_out_memory);
    free(qp_in_memory);

    free_tree(Nn, tree);
    free(tree);

    free(stateTrajectory);
    free(inputTrajectory);

    return 0;
}