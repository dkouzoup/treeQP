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
#include <stdbool.h>

#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/src/dual_Newton_tree.h"
#include "treeqp/src/hpmpc_tree.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/tree.h"
#include "treeqp/utils/utils.h"
#include "treeqp/utils/timing.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_v_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blas.h"

#include "examples/fault_tolerance_utils/load_data.h"


typedef enum
{
    NOMINAL_CONTROLLER = 0,  // nominal MPC
    PRUNED_TREE_CONTROLLER,  // robust MPC using pruned tree structure
    MULTI_STAGE_CONTROLLER,  // robust MPC using multi-stage tree structure
} controller_t;



typedef enum
{
    TREEQP_TDUNES = 0,  // dual Newton strategy on tree
    TREEQP_HPMPC,  // HPMPC (interior point method)
} solver_t;



double random_real( )
{
    return (double) rand() / (double) RAND_MAX;
}



int sample_from_markov_chain(double *transition_matrix, int curr_state, int n_realizations)
{
    double *matrix_row = &transition_matrix[curr_state*n_realizations];

    double u = random_real( );
    double accsum = 0;
    int next_state;

    for (int ii = 0; ii < n_realizations; ii++)
    {
        accsum += matrix_row[ii];
        // printf("i = %d, accsum = %2.2e, number = %2.2e, accsum >= u = %d\n", ii, accsum, u, accsum >= u);
        if (accsum >= u)
        {
            next_state = ii;
            break;
        }
    }
    return next_state;
}



double calculate_closed_loop_objective(int MPCsteps, int nx, int nu, double *Q, double *q,
    double *R, double *r, double *states, double *controls)
{
    double obj = 0;
    double xj, uj;

    // NOTE(dimitris): Q, R are assumed constant and diagonal. x0 does not contribute to cost.
    for (int ii = 0; ii < MPCsteps; ii++)
    {
        for (int jj = 0; jj < nx; jj++)
        {
            xj = states[(ii+1)*nx + jj];
            obj += xj*Q[jj]*xj + xj*q[jj];
        }
        for (int jj = 0; jj < nu; jj++)
        {
            uj = controls[ii*nu + jj];
            obj += uj*R[jj]*xj + uj*r[jj];
        }
    }
    return obj;
}



// TODO(dimitris): make it possible to run executable from _any_ directory
void get_utils_rel_path(int max_str_length, char *path)
{
    char cwd[1024];
    getcwd(cwd, sizeof(cwd));

    if(strstr(cwd, "examples") != NULL)
    {
        snprintf(path, max_str_length, "fault_tolerance_utils/");
    }
    else
    {
        snprintf(path, max_str_length, "examples/fault_tolerance_utils/");
    }
}


int main()
{
    char lib_string[256];
    char utils_rel_path[1024];

    /************************************************
    * problem specification
    ************************************************/

    solver_t solver = TREEQP_HPMPC;
    controller_t controller = PRUNED_TREE_CONTROLLER;

    bool controller_with_varying_spring_configuration = true;

    int MPCsteps = 100;

    // common model data
    int ncontrols = 2;
    int npackets = 4;
    int nsprings = 3;

    // common controller data
    int nhorizon = 10;

    // // common tree controller data
    // TODO(dimitris): decide whether to put this in filename or not and under what format
    // double pfault = 0.001;

    // multi-stage controller data
    int nreal = 3;
    int nrobust = 4;

    // pruned controller data
    double pcov = 0.99;
    int nscenmax = 40;
    // int adaptive = 0;

    /************************************************
    * load data from dynamic libraries
    ************************************************/

    get_utils_rel_path(sizeof(utils_rel_path), utils_rel_path);

    snprintf(lib_string, sizeof(lib_string), "%slib_sim_npackets%02d_nsprings%02d_ncontrols%02d.so",
        utils_rel_path, npackets, nsprings, ncontrols);

    int n_realizations, nx, nu;
    load_dimensions(&n_realizations, &nx, &nu, lib_string);
    assert(nx == 2*npackets-2);
    assert(nu == ncontrols);

    // read code generated integrators for simulation
    sim_data *sim = load_sim_data(n_realizations, lib_string);

    input_data *data;
    switch (controller)
    {
        case NOMINAL_CONTROLLER:
            snprintf(lib_string, sizeof(lib_string),
                "%slib_nominal_npackets%02d_nsprings%02d_ncontrols%02d_nhorizon%03d.so",
                utils_rel_path, npackets, nsprings, ncontrols, nhorizon);
            data = load_nominal_data(n_realizations, lib_string);
            break;
        case PRUNED_TREE_CONTROLLER:
            snprintf(lib_string, sizeof(lib_string),
                "%slib_pruned_npackets%02d_nsprings%02d_ncontrols%02d_nhorizon%03d_pfault1e-02_pcov%2.2f_nscenmax%06d.so",
                utils_rel_path, npackets, nsprings, ncontrols, nhorizon, 100*pcov, nscenmax);
            data = load_tree_data(n_realizations, lib_string);
            break;
        case MULTI_STAGE_CONTROLLER:
            snprintf(lib_string, sizeof(lib_string),
                "%slib_multistage_npackets%02d_nsprings%02d_ncontrols%02d_nhorizon%03d_pfault1e-02_nrobust%02d_nreal_%02d.so",
                utils_rel_path, npackets, nsprings, ncontrols, nhorizon, nrobust, nreal);
            data = load_tree_data(n_realizations, lib_string);
            break;
        default:
            printf("Unknown specified controller, exiting . . .\n");
            exit(1);
    }

    int n_masses = nx/2;
    double *transition_matrix = get_ptr_transition_matrix( );

    // set up bounds and initial condition for closed loop simulation
    double *x0 = calloc(nx, sizeof(double));
    double *xmin = malloc(nx*sizeof(double));
    double *xmax = malloc(nx*sizeof(double));
    double *umin = malloc(nx*sizeof(double));
    double *umax = malloc(nx*sizeof(double));

    double Pmin = -3;
    double Pmax = 2.5;
    double Vmin = -8;
    double Vmax = 8;
    double Fmin = -10;
    double Fmax = 10;

    for (int ii = 0; ii < nx; ii++)
    {
        if (ii < nx/2)
        {
            xmin[ii] = Pmin;
            xmax[ii] = Pmax;
        }
        else
        {
            xmin[ii] = Vmin;
            xmax[ii] = Vmax;
        }
    }

    for (int ii = 0; ii < n_masses; ii++)
    {
        x0[ii] = 0.0;
        x0[n_masses+ii] = 0.0;
    }

    for (int ii = 0; ii < nu; ii++)
    {
        umin[ii] = Fmin;
        umax[ii] = Fmax;
    }

    treeqp_timer timer;

    double *stateTrajectory = malloc(nx*(MPCsteps+1)*sizeof(double));
    double *inputTrajectory = malloc(nu*MPCsteps*sizeof(double));
    double *cpuTimes = malloc(MPCsteps*sizeof(double));
    double *spring_configs = malloc((MPCsteps+1)*sizeof(double));

    for (int jj = 0; jj < nx; jj++)
    {
        stateTrajectory[jj] = x0[jj];
    }

    // set up QP solver options
    treeqp_tdunes_options_t tdunes_opts;
    treeqp_hpmpc_options_t hpmpc_opts;

    int max_Nn = data[0].Nn;
    for (int ii = 1; ii < n_realizations; ii++)
    {
        if (data[ii].Nn > max_Nn) max_Nn = data[ii].Nn;
    }

    int maxIter = 200;
    double tol = 1e-8;

    switch (solver)
    {
        case TREEQP_TDUNES:
            tdunes_opts = treeqp_tdunes_default_options(max_Nn);

            tdunes_opts.maxIter = maxIter;
            tdunes_opts.termCondition = TREEQP_INFNORM;
            tdunes_opts.stationarityTolerance = tol;
            tdunes_opts.lineSearchMaxIter = 100;
            tdunes_opts.lineSearchGamma = 0.1;
            tdunes_opts.lineSearchBeta = 0.8;
            tdunes_opts.regType  = TREEQP_ALWAYS_LEVENBERG_MARQUARDT;
            tdunes_opts.regValue = 1e-10;

            for (int ii = 0; ii < max_Nn; ii++) tdunes_opts.qp_solver[ii] = TREEQP_CLIPPING_SOLVER;
            break;
        case TREEQP_HPMPC:
            hpmpc_opts = treeqp_hpmpc_default_options(max_Nn);
            // TODO(dimitris): change maxIter and tol
            break;
        default:
            printf("Unknown specified solver. Exiting . . .\n");
            exit(1);
    }

    // set up problem data
    treeqp_tdunes_workspace *works_tdunes;
    treeqp_hpmpc_workspace *works_hpmpc;

    struct node **forest = malloc(n_realizations*sizeof(struct node*));
    tree_ocp_qp_in *qp_ins = malloc(n_realizations*sizeof(tree_ocp_qp_in));
    void **qp_in_memories = malloc(n_realizations*sizeof(void*));
    works_tdunes = malloc(n_realizations*sizeof(treeqp_tdunes_workspace));
    works_hpmpc = malloc(n_realizations*sizeof(treeqp_hpmpc_workspace));
    void **solver_memories = malloc(n_realizations*sizeof(void*));
    tree_ocp_qp_out *qp_outs = malloc(n_realizations*sizeof(tree_ocp_qp_out));
    void **qp_out_memories = malloc(n_realizations*sizeof(void*));

    int size;

    for (int ii = 0; ii < n_realizations; ii++)
    {
        // create solver only if tree has been generated for this configuration
        if (data[ii].Nn != -1)
        {
            //set up tree
            forest[ii] = malloc(data[ii].Nn*sizeof(struct node));
            setup_tree(data[ii].Nn, data[ii].nc, forest[ii]);

            // set up QP data
            size = tree_ocp_qp_in_calculate_size(data[ii].Nn, data[ii].nx, data[ii].nu, forest[ii]);
            qp_in_memories[ii] = malloc(size);
            create_tree_ocp_qp_in(data[ii].Nn, data[ii].nx, data[ii].nu, forest[ii], &qp_ins[ii], qp_in_memories[ii]);
            tree_ocp_qp_in_read_dynamics_colmajor(data[ii].A, data[ii].B, data[ii].b, &qp_ins[ii]);
            tree_ocp_qp_in_read_objective_diag(data[ii].Qd, data[ii].Rd, data[ii].q, data[ii].r, &qp_ins[ii]);
            tree_ocp_qp_in_set_constant_bounds(xmin, xmax, umin, umax, &qp_ins[ii]);
            tree_ocp_qp_in_set_x0_bounds(&qp_ins[ii], x0);

            // set up QP solver
            switch (solver)
            {
                case TREEQP_TDUNES:
                    size = treeqp_tdunes_calculate_size(&qp_ins[ii], &tdunes_opts);
                    solver_memories[ii] = malloc(size);
                    create_treeqp_tdunes(&qp_ins[ii], &tdunes_opts, &works_tdunes[ii], solver_memories[ii]);
                    break;
                case TREEQP_HPMPC:
                    size = treeqp_hpmpc_calculate_size(&qp_ins[ii], &hpmpc_opts);
                    solver_memories[ii] = malloc(size);
                    create_treeqp_hpmpc(&qp_ins[ii], &hpmpc_opts, &works_hpmpc[ii], solver_memories[ii]);
                    break;
            }

            // set up QP solution
            size = tree_ocp_qp_out_calculate_size(data[ii].Nn, data[ii].nx, data[ii].nu);
            qp_out_memories[ii] = malloc(size);
            create_tree_ocp_qp_out(data[ii].Nn, data[ii].nx, data[ii].nu, &qp_outs[ii], qp_out_memories[ii]);
        }
    }

    double kkt_err;
    double *A, *B, *b;

    int mpc_config = n_realizations-1;
    int sim_config = n_realizations-1;

    // NOTE(dimitris): get rid of first random number which gives too low probability
    random_real( );

    spring_configs[0] = sim_config;

    // MPC loop
    for (int tt = 0; tt < MPCsteps; tt++)
    {
        // solve QP
        treeqp_tic(&timer);
        switch (solver)
        {
            case TREEQP_TDUNES:
                treeqp_tdunes_solve(&qp_ins[mpc_config], &qp_outs[mpc_config], &tdunes_opts, &works_tdunes[mpc_config]);
                break;
            case TREEQP_HPMPC:
                treeqp_hpmpc_solve(&qp_ins[mpc_config], &qp_outs[mpc_config], &hpmpc_opts, &works_hpmpc[mpc_config]);
                break;
        }
        cpuTimes[tt] = treeqp_toc(&timer);
        // print_tree_ocp_qp_out(qp_ins[mpc_config].N, &qp_outs[mpc_config]);

        // run some sanity checks
        for (int jj = 0; jj < nx; jj++)
        {
            assert(ABS(DVECEL_LIBSTR(&qp_outs[mpc_config].x[0], jj) - x0[jj]) < 1e-10);
        }
        assert(qp_outs[mpc_config].info.iter < maxIter && "maximum number of iterations reached");

        kkt_err = max_KKT_residual(&qp_ins[mpc_config], &qp_outs[mpc_config]);
        // printf("KKT = %f\n", kkt_err);
        assert(kkt_err <= tol && "violation of KKT conditions too high");

        // apply disturbance
        if (tt % 10 == 0)
            blasfeo_dvecse(nu, Fmax, &qp_outs[mpc_config].u[0], 0);

        // simulate system
        A = sim[sim_config].A;
        B = sim[sim_config].B;
        b = sim[sim_config].b;
        for (int ii = 0; ii < nx; ii++)
        {
            x0[ii] = b[ii];
            for (int jj = 0; jj < nx; jj++)
            {
                x0[ii] += A[ii + jj * nx] * DVECEL_LIBSTR(&qp_outs[mpc_config].x[0], jj);
            }
            for (int jj = 0; jj < nu; jj++)
            {
                x0[ii] += B[ii + jj * nx] * DVECEL_LIBSTR(&qp_outs[mpc_config].u[0], jj);
            }
        }

        // print iteration results
        printf("-------------------------------------------------------------------------------\n");
        printf("\n > MPC iteration #%d converged in %d iterations\n\n", tt+1, qp_outs[mpc_config].info.iter);
        printf("\tproblem solved in %f ms\n\n", cpuTimes[tt]*1e3);
        printf("\tmax. violation of KKT conditions: %2.2e\n\n", kkt_err);
        printf("\tcurrent spring configuration index: %d\n\n", sim_config);
        printf("\tx = ");
        blasfeo_print_exp_tran_dvec(nx, &qp_outs[mpc_config].x[0], 0);
        printf("\tu = ");
        blasfeo_print_exp_tran_dvec(nu, &qp_outs[mpc_config].u[0], 0);


        // save state and input trajectories
        for (int jj = 0; jj < nx; jj++)
        {
            stateTrajectory[jj + (tt+1)*nx] = x0[jj];
        }
        for (int jj = 0; jj < nu; jj++)
        {
            inputTrajectory[jj + tt*nu] = DVECEL_LIBSTR(&qp_outs[mpc_config].u[0], jj);
        }

        // update bound on x0
        for (int ii = 0; ii < n_realizations; ii++)
        {
            if (data[ii].Nn != -1)
            {
                tree_ocp_qp_in_set_x0_bounds(&qp_ins[ii], x0);
            }
        }

        printf("> current spring configuration: %d, current controller configuration %d \n", sim_config, mpc_config);

        sim_config = sample_from_markov_chain(transition_matrix, sim_config, n_realizations);

        if (controller_with_varying_spring_configuration)
        {
            if (data[sim_config].A != NULL)  // configuration is exported
            {
                mpc_config = sim_config;
            }
        }

        spring_configs[tt+1] = sim_config;
    }

    // print some results
    double *Q = data[mpc_config].Qd;
    double *q = data[mpc_config].q;
    double *R = data[mpc_config].Rd;
    double *r = data[mpc_config].r;
    double obj = calculate_closed_loop_objective(MPCsteps, nx, nu, Q, q, R, r,
        stateTrajectory, inputTrajectory);

    printf("\nClosed loop objective: %f\n\n", obj);

    // store results in txt files
    char var_string[1024];
    snprintf(var_string, sizeof(var_string), "%sxMPC.txt", utils_rel_path);
    write_double_vector_to_txt(stateTrajectory, nx*(MPCsteps+1), var_string);
    snprintf(var_string, sizeof(var_string), "%suMPC.txt", utils_rel_path);
    write_double_vector_to_txt(inputTrajectory, nu*MPCsteps, var_string);
    snprintf(var_string, sizeof(var_string), "%scpuTimes.txt", utils_rel_path);
    write_double_vector_to_txt(cpuTimes, MPCsteps, var_string);
    snprintf(var_string, sizeof(var_string), "%sn_masses.txt", utils_rel_path);
    write_int_vector_to_txt(&n_masses, 1, var_string);

    // free allocated memory
    for (int ii = 0; ii < n_realizations; ii++)
    {
        if (data[ii].Nn != -1)
        {
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
    free(works_tdunes);
    free(works_hpmpc);
    free(qp_out_memories);
    free(qp_outs);

    free(stateTrajectory);
    free(inputTrajectory);
    free(cpuTimes);
    free(spring_configs);

    free(x0);
    free(xmin);
    free(xmax);
    free(umin);
    free(umax);
    free(data);
    free(sim);

    return 0;
}