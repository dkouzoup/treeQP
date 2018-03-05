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

typedef struct
{
    int nrobust;
    int nreal;
} multi_stage_params;


typedef struct
{
    double pcov;
    int nscenmax;
} pruned_params;


typedef struct
{
    int MPCsteps;
    int npackets;
    int nsprings;
    int ncontrols;
    int nhorizon;
    char *solver_name;
    char *sim_type;
    bool varying_config;
    multi_stage_params ms_controller;
    pruned_params pruned_controller;
} params;


typedef struct
{
    double *cpu_times;
    double *input_trajectory;
    double *state_trajectory;
} results;

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



// NOTE(dimitris): needed for python interface
void malloc_double_ptr(double **ptr, int n)
{
    *ptr = malloc(n*sizeof(double));
}

void free_ptr(double *ptr)
{
    free(ptr);
}



sim_data *load_sim_data_from_lib(char *treeQP_abs_path, params *sim_params, int *n_realizations_ptr)
{
    char lib_string[256];

    snprintf(lib_string, sizeof(lib_string),
        "%s/examples/fault_tolerance_utils/lib_sim_npackets%02d_nsprings%02d_ncontrols%02d.so",
        treeQP_abs_path, sim_params->npackets, sim_params->nsprings, sim_params->ncontrols);

    printf("\n...loading simulation model from:\n\n%s\n\n",lib_string);

    int nx, nu;  // for sanity checks

    load_dimensions(n_realizations_ptr, &nx, &nu, lib_string);
    assert(nx == 2*sim_params->npackets-2);
    assert(nu == sim_params->ncontrols);

    // read code generated integrators for simulation
    sim_data *sim = load_sim_data(*n_realizations_ptr, lib_string);

    printf("...done.\n\n");
    return sim;
}



input_data *load_multi_stage_data_from_lib(char *treeQP_abs_path, int nreal, params *sim_params)
{
    char lib_string[256];

    snprintf(lib_string, sizeof(lib_string),
        "%s/examples/fault_tolerance_utils/lib_multistage_npackets%02d_nsprings%02d_ncontrols%02d_nhorizon%03d_pfault1e-02_nrobust%02d_nreal_%02d.so",
        treeQP_abs_path, sim_params->npackets, sim_params->nsprings, sim_params->ncontrols, sim_params->nhorizon,
        sim_params->ms_controller.nrobust, sim_params->ms_controller.nreal);

    // read code generated multi-stage controller
    input_data *data = load_tree_data(nreal, lib_string);

    return data;
}



input_data *load_pruned_data_from_lib(char *treeQP_abs_path, int nreal, params *sim_params)
{
    char lib_string[256];

    snprintf(lib_string, sizeof(lib_string),
        "%s/examples/fault_tolerance_utils/lib_pruned_npackets%02d_nsprings%02d_ncontrols%02d_nhorizon%03d_pfault1e-02_pcov%2.2f_nscenmax%06d.so",
        treeQP_abs_path, sim_params->npackets, sim_params->nsprings, sim_params->ncontrols, sim_params->nhorizon,
        100*sim_params->pruned_controller.pcov, sim_params->pruned_controller.nscenmax);

    // read code generated pruned controller
    input_data *data = load_tree_data(nreal, lib_string);

    return data;
}



input_data *load_nominal_data_from_lib(char *treeQP_abs_path, int nreal, params *sim_params)
{
    char lib_string[256];

    snprintf(lib_string, sizeof(lib_string),
        "%s/examples/fault_tolerance_utils/lib_nominal_npackets%02d_nsprings%02d_ncontrols%02d_nhorizon%03d.so",
        treeQP_abs_path, sim_params->npackets, sim_params->nsprings, sim_params->ncontrols, sim_params->nhorizon);

    // read code generated nominal controller
    input_data *data = load_nominal_data(nreal, lib_string);

    return data;
}



int run_closed_loop_simulation(char *treeQP_abs_path, params *sim_params, results *res)
{
    /************************************************
    * initialize controller
    ************************************************/

    // define controller type
    controller_t controller_type;

    if (strstr(sim_params->sim_type, "nominal") != NULL)
        controller_type = NOMINAL_CONTROLLER;
    else if (strstr(sim_params->sim_type, "pruned") != NULL)
       controller_type = PRUNED_TREE_CONTROLLER;
    else if (strstr(sim_params->sim_type, "multi-stage") != NULL)
        controller_type = MULTI_STAGE_CONTROLLER;
    else
    {
        printf("\nUNKNOWN CONTROLLER TYPE: EXITING ...\n\n");
        exit(-1);
    }

    // define solver
    solver_t solver;

    if (strstr(sim_params->solver_name, "HPMPC") != NULL)
        solver = TREEQP_HPMPC;
    else if (strstr(sim_params->solver_name, "DUNES") != NULL)
        solver = TREEQP_TDUNES;
    else
    {
        printf("\nUNKNOWN SOLVER: EXITING ...\n\n");
        exit(-1);
    }

    bool controller_with_varying_spring_configuration = sim_params->varying_config;


    /************************************************
    * print info
    ************************************************/

    if (controller_type == MULTI_STAGE_CONTROLLER)
    {
        printf("\n*** SIMULATION WITH MULTI-STAGE CONTROLLER ***\n\n");
    }
    else if (controller_type == PRUNED_TREE_CONTROLLER)
    {
        printf("\n***** SIMULATION WITH PRUNED CONTROLLER ******\n\n");
    } else if (controller_type == NOMINAL_CONTROLLER)
    {
        printf("\n***** SIMULATION WITH NOMINAL CONTROLLER *****\n\n");
    }
    printf("number of MPC steps = %d\n", sim_params->MPCsteps);
    printf("prediction horizon = %d\n", sim_params->nhorizon);
    printf("number of packets = %d\n", sim_params->npackets);
    printf("number of springs per packet = %d\n", sim_params->nsprings);
    printf("number of controlled packets = %d\n", sim_params->ncontrols);
    printf("\n**********************************************\n");

    int MPCsteps = sim_params->MPCsteps;
    int nx = 2*sim_params->npackets-2;
    int nu = sim_params->ncontrols;
    int n_realizations = -1;  // TODO(dimitris): calculate it from params
    int n_masses = nx/2;

    /************************************************
    * load data from dynamic libraries
    ************************************************/

    sim_data *sim = load_sim_data_from_lib(treeQP_abs_path, sim_params, &n_realizations);

    input_data *data;
    if (controller_type == MULTI_STAGE_CONTROLLER)
        data = load_multi_stage_data_from_lib(treeQP_abs_path, n_realizations, sim_params);
    else if (controller_type == PRUNED_TREE_CONTROLLER)
        data = load_pruned_data_from_lib(treeQP_abs_path, n_realizations, sim_params);
    else if (controller_type == NOMINAL_CONTROLLER)
        data = load_nominal_data_from_lib(treeQP_abs_path, n_realizations, sim_params);

    // TODO(dimitris): get realizations from python
    double *transition_matrix = get_ptr_transition_matrix( );

    /************************************************
    * ........
    ************************************************/

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

    // TODO(dimitris): TEMP!
    double *spring_configs = malloc((MPCsteps+1)*sizeof(double));

    for (int jj = 0; jj < nx; jj++)
    {
        res->state_trajectory[jj] = x0[jj];
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

    // TODO(dimitris): generate in python instead
    // NOTE(dimitris): get rid of first random number which gives too low probability
    random_real( );

    spring_configs[0] = sim_config;


    /************************************************
    * run closed loop simulation
    ************************************************/

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
        res->cpu_times[tt] = treeqp_toc(&timer);
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
        printf("\tproblem solved in %f ms\n\n", res->cpu_times[tt]*1e3);
        printf("\tmax. violation of KKT conditions: %2.2e\n\n", kkt_err);
        printf("\tcurrent spring configuration index: %d\n\n", sim_config);
        printf("\tx = ");
        blasfeo_print_exp_tran_dvec(nx, &qp_outs[mpc_config].x[0], 0);
        printf("\tu = ");
        blasfeo_print_exp_tran_dvec(nu, &qp_outs[mpc_config].u[0], 0);


        // save state and input trajectories
        for (int jj = 0; jj < nx; jj++)
        {
            res->state_trajectory[jj + (tt+1)*nx] = x0[jj];
        }
        for (int jj = 0; jj < nu; jj++)
        {
            res->input_trajectory[jj + tt*nu] = DVECEL_LIBSTR(&qp_outs[mpc_config].u[0], jj);
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
        res->state_trajectory, res->input_trajectory);

    printf("\nClosed loop objective: %f\n\n", obj);

    /************************************************
    * free memory
    ************************************************/

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

    free(spring_configs);

    free(x0);
    free(xmin);
    free(xmax);
    free(umin);
    free(umax);

    free(sim);
    free(data);
    return 1;
}



int main()
{
    /************************************************
    * problem specification
    ************************************************/

    params sim_params;

    sim_params.MPCsteps = 100;

    sim_params.npackets = 4;
    sim_params.nsprings = 3;
    sim_params.ncontrols = 2;
    sim_params.nhorizon = 10;

    sim_params.solver_name =  "HPMPC";  // "HPMPC", "DUNES"
    sim_params.sim_type = "pruned";
    sim_params.varying_config = false;

    sim_params.ms_controller.nrobust = 4;
    sim_params.ms_controller.nreal = 3;

    sim_params.pruned_controller.pcov = 0.99;
    sim_params.pruned_controller.nscenmax = 40;

    results res;
    int nx = 2*(sim_params.npackets-1);
    int nu = sim_params.ncontrols;

    res.cpu_times = malloc(sim_params.MPCsteps*sizeof(double));
    res.input_trajectory = malloc(sim_params.MPCsteps*nu*sizeof(double));
    res.state_trajectory = malloc((sim_params.MPCsteps+1)*nx*sizeof(double));

    // TODO(dimitris): currently only works if executed from treeQP root dir
    char treeQP_abs_path[1024];
    getcwd(treeQP_abs_path, sizeof(treeQP_abs_path));

    /************************************************
    * closed loop simulation
    ************************************************/

    run_closed_loop_simulation(treeQP_abs_path, &sim_params, &res);


    /************************************************
    * free memory
    ************************************************/

    free(res.cpu_times);
    free(res.input_trajectory);
    free(res.state_trajectory);

    return 0;
}