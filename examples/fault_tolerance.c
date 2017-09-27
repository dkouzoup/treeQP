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
#include "treeqp/utils/timing.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_v_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blas.h"

#include "examples/fault_tolerance_utils/load_data.h"

real_t random_real( ) {
    return (real_t) rand() / (real_t) RAND_MAX;
}


int_t sample_from_markov_chain(real_t *transition_matrix, int_t curr_state, int_t n_realizations) {

    real_t *matrix_row = &transition_matrix[curr_state*n_realizations];

    real_t u = random_real( );
    real_t accsum = 0;
    int_t next_state;

    // printf("row=\n");
    // for (int_t ii = 0; ii < n_realizations; ii++) {
    //     printf("%2.5e\t", matrix_row[ii]);
    // }
    // printf("\n");

    for (int_t ii = 0; ii < n_realizations; ii++) {
        accsum += matrix_row[ii];
        // printf("i = %d, accsum = %2.2e, number = %2.2e, accsum >= u = %d\n", ii, accsum, u, accsum >= u);
        if (accsum >= u) {
            next_state = ii;
            break;
        }
    }
    return next_state;
}


int_t main() {

    // define simulation length and number of considered trees
    int_t MPCsteps = 100;

    // read code generated data
    sim_data *sim = load_sim_data();

    // NOTE(dimitris): NOMINAL_MPC macro defined in load_data.h
    #ifdef NOMINAL_MPC
        input_data *data = load_nominal_data();
    #else
        input_data *data = load_data();
    #endif

    int_t nx = get_nx();
    int_t nu = get_nu();
    int_t n_masses = nx/2;
    int_t n_realizations = get_number_of_realizations();
    real_t *transition_matrix = get_ptr_transition_matrix( );

    // set up bounds and initial condition for closed loop simulation
    real_t *x0 = calloc(nx, sizeof(real_t));
    real_t *xmin = malloc(nx*sizeof(real_t));
    real_t *xmax = malloc(nx*sizeof(real_t));
    real_t *umin = malloc(nx*sizeof(real_t));
    real_t *umax = malloc(nx*sizeof(real_t));

    real_t Pmin = -5;
    real_t Pmax = 5;
    real_t Vmin = -10;
    real_t Vmax = 10;
    real_t Fmin = -1;
    real_t Fmax = 1;

    for (int_t ii = 0; ii < nx; ii++) {
        if (ii < nx/2) {
            xmin[ii] = Pmin;
            xmax[ii] = Pmax;
        } else {
            xmin[ii] = Vmin;
            xmax[ii] = Vmax;
        }
        x0[ii] = 1;
    }
    x0[1] = -1;

    // x0[0] = 1;
    // x0[1] = -1;
    // x0[2] = 3;
    // x0[3] = -5;

    for (int_t ii = 0; ii < nu; ii++) {
        umin[ii] = Fmin;
        umax[ii] = Fmax;
    }

    treeqp_timer timer;

    real_t *stateTrajectory = malloc(nx*(MPCsteps+1)*sizeof(real_t));
    real_t *inputTrajectory = malloc(nu*MPCsteps*sizeof(real_t));
    real_t *cpuTimes = malloc(MPCsteps*sizeof(real_t));

    for (int_t jj = 0; jj < nx; jj++) {
        stateTrajectory[jj] = x0[jj];
    }

    // set up QP solver options
    treeqp_tdunes_options_t opts;

    opts.maxIter = 200;
    opts.termCondition = TREEQP_INFNORM;
    opts.stationarityTolerance = 1.0e-8;
    opts.lineSearchMaxIter = 100;
    opts.lineSearchGamma = 0.1;
    opts.lineSearchBeta = 0.8;
    opts.regType  = TREEQP_ALWAYS_LEVENBERG_MARQUARDT;
    opts.regValue = 1e-10;

    // set up problem data
    struct node **forest = malloc(n_realizations*sizeof(struct node*));
    tree_ocp_qp_in *qp_ins = malloc(n_realizations*sizeof(tree_ocp_qp_in));
    void **qp_in_memories = malloc(n_realizations*sizeof(void*));
    treeqp_tdunes_workspace *works = malloc(n_realizations*sizeof(treeqp_tdunes_workspace));
    void **solver_memories = malloc(n_realizations*sizeof(void*));
    tree_ocp_qp_out *qp_outs = malloc(n_realizations*sizeof(tree_ocp_qp_out));
    void **qp_out_memories = malloc(n_realizations*sizeof(void*));

    int_t size;

    for (int_t ii = 0; ii < n_realizations; ii++) {
        // create solver only if tree has been generated for this configuration
        if (data[ii].Nn != -1) {
            //set up tree
            forest[ii] = malloc(data[ii].Nn*sizeof(struct node));
            setup_tree(data[ii].Nn, data[ii].nc, forest[ii]);

            // set up QP data
            size = tree_ocp_qp_in_calculate_size(data[ii].Nn, data[ii].nx, data[ii].nu, forest[ii]);
            qp_in_memories[ii] = malloc(size);
            create_tree_ocp_qp_in(data[ii].Nn, data[ii].nx, data[ii].nu, forest[ii], &qp_ins[ii], qp_in_memories[ii]);
            tree_ocp_qp_in_read_dynamics_colmajor(data[ii].A, data[ii].B, data[ii].b, &qp_ins[ii]);
            tree_ocp_qp_in_read_objective_diag_colmajor(data[ii].Qd, data[ii].Rd, data[ii].q, data[ii].r, &qp_ins[ii]);
            tree_ocp_qp_in_set_constant_bounds(xmin, xmax, umin, umax, &qp_ins[ii]);
            tree_ocp_qp_in_set_x0_bounds(&qp_ins[ii], x0);

            // set up QP solver
            size = treeqp_tdunes_calculate_size(&qp_ins[ii]);
            solver_memories[ii] = malloc(size);
            create_treeqp_tdunes(&qp_ins[ii], &opts, &works[ii], solver_memories[ii]);

            // set up QP solution
            size = tree_ocp_qp_out_calculate_size(data[ii].Nn, data[ii].nx, data[ii].nu);
            qp_out_memories[ii] = malloc(size);
            create_tree_ocp_qp_out(data[ii].Nn, data[ii].nx, data[ii].nu, &qp_outs[ii], qp_out_memories[ii]);
        }
    }

    real_t err;
    real_t *A, *B, *b;

    int_t mpc_config = n_realizations-1;
    int_t sim_config = n_realizations-1;

    // NOTE(dimitris): get rid of first random number which gives too low probability
    random_real( );

    // MPC loop
    for (int_t tt = 0; tt < MPCsteps; tt++) {

        // solve QP
        treeqp_tic(&timer);
        treeqp_tdunes_solve(&qp_ins[mpc_config], &qp_outs[mpc_config], &opts, &works[mpc_config]);
        cpuTimes[tt] = treeqp_toc(&timer);

        // run some sanity checks
        for (int_t jj = 0; jj < nx; jj++) {
            assert(DVECEL_LIBSTR(&qp_outs[mpc_config].x[0], jj) == x0[jj]);
        }
        assert(qp_outs[mpc_config].info.iter < opts.maxIter && "maximum number of iterations reached");

        err = maximum_error_in_dynamic_constraints(&qp_ins[mpc_config], &qp_outs[mpc_config]);
        assert(err <= opts.stationarityTolerance && "violation of dynamic constraints too high");

        // simulate system
        A = sim[sim_config].A;
        B = sim[sim_config].B;
        b = sim[sim_config].b;
        for (int_t ii = 0; ii < nx; ii++) {
            x0[ii] = b[ii];
            for (int_t jj = 0; jj < nx; jj++) {
                x0[ii] += A[ii + jj * nx] * DVECEL_LIBSTR(&qp_outs[mpc_config].x[0], jj);
            }
            for (int_t jj = 0; jj < nu; jj++)
                x0[ii] += B[ii + jj * nx] * DVECEL_LIBSTR(&qp_outs[mpc_config].u[0], jj);
        }

        // print iteration results
        printf("-------------------------------------------------------------------------------\n");
        printf("\n > MPC iteration #%d converged in %d iterations\n\n", tt+1, qp_outs[mpc_config].info.iter);
        printf("\tproblem solved in %f ms\n\n", cpuTimes[tt]*1e3);
        printf("\tmax. violation of dynamic constraints: %2.2e\n\n", err);
        printf("\tcurrent spring configuration index: %d\n\n", sim_config);
        printf("\tx = ");
        d_print_e_tran_strvec(nx, &qp_outs[mpc_config].x[0], 0);
        printf("\tu = ");
        d_print_e_tran_strvec(nu, &qp_outs[mpc_config].u[0], 0);


        // save state and input trajectories
        for (int_t jj = 0; jj < nx; jj++) {
            stateTrajectory[jj + (tt+1)*nx] = x0[jj];
        }
        for (int_t jj = 0; jj < nu; jj++) {
            inputTrajectory[jj + tt*nu] = DVECEL_LIBSTR(&qp_outs[mpc_config].u[0], jj);
        }

        // update bound on x0
        for (int_t ii = 0; ii < n_realizations; ii++) {
            if (data[ii].Nn != -1) {
                tree_ocp_qp_in_set_x0_bounds(&qp_ins[ii], x0);
            }
        }

        sim_config = sample_from_markov_chain(transition_matrix, sim_config, n_realizations);
        // NOTE(dimitris): take care that mpc_config is generated (line below segfaults if not)
        // mpc_config = sim_config;
    }

    // store results in txt files
    char cwd[1024];
    getcwd(cwd, sizeof(cwd));
    fprintf(stdout, "\nCurrent working dir: %s\n\n", cwd);

    // TODO(dimitris): do this in a more general way
    if(strstr(cwd, "examples") != NULL) {
        // write_qp_out_to_txt(&qp_in, &qp_out, "fault_tolerance_utils");
        write_double_vector_to_txt(stateTrajectory, nx*(MPCsteps+1), "fault_tolerance_utils/xMPC.txt");
        write_double_vector_to_txt(inputTrajectory, nu*MPCsteps, "fault_tolerance_utils/uMPC.txt");
        write_double_vector_to_txt(cpuTimes, MPCsteps, "fault_tolerance_utils/cpuTimes.txt");
        write_int_vector_to_txt(&n_masses, 1, "fault_tolerance_utils/n_masses.txt");
    } else {
        // write_qp_out_to_txt(&qp_in, &qp_out, "examples/fault_tolerance_utils");
        write_double_vector_to_txt(stateTrajectory, nx*(MPCsteps+1), "examples/fault_tolerance_utils/xMPC.txt");
        write_double_vector_to_txt(inputTrajectory, nu*MPCsteps, "examples/fault_tolerance_utils/uMPC.txt");
        write_double_vector_to_txt(cpuTimes, MPCsteps, "examples/fault_tolerance_utils/cpuTimes.txt");
        write_int_vector_to_txt(&n_masses, 1, "examples/fault_tolerance_utils/n_masses.txt");
    }

    // free allocated memory
    for (int_t ii = 0; ii < n_realizations; ii++) {
        if (data[ii].Nn != -1) {
            free_tree(data[ii].Nn, forest[ii]);
            free(forest[ii]);
            free(qp_in_memories[ii]);
            free(solver_memories[ii]);
            free(qp_out_memories[ii]);
        }
    }
    free(forest);
    free(qp_in_memories);
    free(qp_ins);
    free(solver_memories);
    free(works);
    free(qp_out_memories);
    free(qp_outs);

    free(stateTrajectory);
    free(inputTrajectory);
    free(cpuTimes);

    free(x0);
    free(xmin);
    free(xmax);
    free(umin);
    free(umax);
    free(data);
    free(sim);

    return 0;
}