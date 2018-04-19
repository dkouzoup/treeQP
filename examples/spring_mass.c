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

#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/src/hpmpc_tree.h"
#include "treeqp/src/hpipm_tree.h"
#include "treeqp/src/dual_Newton_tree.h"

#include "treeqp/utils/types.h"
#include "treeqp/utils/memory.h"
#include "treeqp/utils/profiling.h"
#include "treeqp/utils/tree.h"
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

    int Nn = calculate_number_of_nodes(md, Nr, Nh);
    int Np = Nn - ipow(md, Nr);

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
    int *nc = malloc(Nn*sizeof(int));

    #ifdef TEST_GENERAL_CONSTRAINTS
    int NC = 1;  // chose between 1 (either state or inpute constraint converted) and 2 (both converted)
    int make_u_constr_general = 1;  // if 1, choose which bound to convert to general constraint

    int NCn;  // number of general constraints at last stage
    if (NC == 2)
    {
        NCn = 1;
    }
    else
    {
        if (make_u_constr_general == 0)
        {
            NCn = 1;
        }
        else
        {
            NCn = 0;
        }
    }
    #endif

    // NOTE(dimitris): changing xmax to have some active state  constraints at solution
    xmax[1] = .2;

    for (int ii = 0; ii < Nn; ii++)
    {
        // state and input dimensions on each node (only different at root/leaves)
        if (ii > 0)
        {
            nx[ii] = NX;
        }
        else
        {
            nx[ii] = NX;
        }

        if (tree[ii].nkids > 0)  // not a leaf
        {
            nu[ii] = NU;
            nc[ii] = 0;
            #ifdef TEST_GENERAL_CONSTRAINTS
            nc[ii] = NC;
            #endif
        }
        else
        {
            nu[ii] = 0;
            nc[ii] = 0;
            #ifdef TEST_GENERAL_CONSTRAINTS
            nc[ii] = NCn;
            #endif
        }
    }

    int qp_in_size = tree_ocp_qp_in_calculate_size(Nn, nx, nu, nc, tree);
    void *qp_in_memory = malloc(qp_in_size);
    tree_ocp_qp_in_create(Nn, nx, nu, nc, tree, &qp_in, qp_in_memory);

    #ifdef TEST_GENERAL_CONSTRAINTS
    // set C, D, dmin, dmax equivalent to bounds
    double *C = calloc(NC*NX, sizeof(double));
    double *CN = calloc(NCn*NX, sizeof(double));
    double *D = calloc(NC*NU, sizeof(double));
    double *dmin = calloc(NC, sizeof(double));
    double *dmax = calloc(NC, sizeof(double));

    if (NC == 2)
    {
        C[0+1*NC] = 1.0;
        D[1+0*NC] = 1.0;
        CN[0+1*NCn] = 1.0;

        dmin[0] = xmin[1];
        dmin[1] = umin[0];
        dmax[0] = xmax[1];
        dmax[1] = umax[0];
    }
    else if (NC == 1)
    {
        if (make_u_constr_general == 1)
        {
            D[0] = 1;

            dmin[0] = umin[0];
            dmax[0] = umax[0];
        }
        else
        {
            C[1] = 1;
            CN[1] = 1;
            dmin[0] = xmin[1];
            dmax[0] = xmax[1];
        }
    }

    tree_ocp_qp_in_fill_lti_data_diag_weights(&A[NX*NX], &B[NX*NU], &b[NX], dQ, q, dP, p, dR, r,
        xmin, xmax, umin, umax, x0, C, CN, D, dmin, dmax, &qp_in);

    // remove bounds
    tree_ocp_qp_in_set_inf_bounds(&qp_in);

    if (NC == 1 && make_u_constr_general == 1)
    {
        // restore bounds on x
        for (int ii = 0; ii < Nn; ii++)
        {
            blasfeo_pack_dvec(nx[ii], xmin, &qp_in.xmin[ii], 0);
            blasfeo_pack_dvec(nx[ii], xmax, &qp_in.xmax[ii], 0);
        }
    }
    else if (NC == 1 && make_u_constr_general == 0)
    {
        // restore bounds on u
        for (int ii = 0; ii < Nn; ii++)
        {
            blasfeo_pack_dvec(nu[ii], umin, &qp_in.umin[ii], 0);
            blasfeo_pack_dvec(nu[ii], umax, &qp_in.umax[ii], 0);
        }
    }
    // restore bound on x0
    tree_ocp_qp_in_set_x0_bounds(&qp_in, x0);

    #else
    // NOTE(dimitris): skipping first dynamics that represent the nominal ones
    tree_ocp_qp_in_fill_lti_data_diag_weights(&A[NX*NX], &B[NX*NU], &b[NX], dQ, q, dP, p, dR, r,
        xmin, xmax, umin, umax, x0, NULL, NULL, NULL, NULL, NULL, &qp_in);
    #endif

    // set up tree-sparse dual Newton solver
    #ifdef SOLVE_WITH_TDUNES
    treeqp_tdunes_options_t tdunes_opts = treeqp_tdunes_default_options(Nn);
    for (int ii = 0; ii < Nn; ii++)
    {
        #ifdef TEST_GENERAL_CONSTRAINTS
        tdunes_opts.qp_solver[ii] = TREEQP_QPOASES_SOLVER;
        #else
        tdunes_opts.qp_solver[ii] = TREEQP_CLIPPING_SOLVER;
        #endif
    }
    tdunes_opts.maxIter = 100;  // set to 1 for debugging

    treeqp_tdunes_workspace tdunes_work;
    void *tdunes_memory = malloc(treeqp_tdunes_calculate_size(&qp_in, &tdunes_opts));
    create_treeqp_tdunes(&qp_in, &tdunes_opts, &tdunes_work, tdunes_memory);
    #endif

    // set up HPMPC solver
    #ifdef SOLVE_WITH_HPMPC
    treeqp_hpmpc_opts_t hpmpc_opts;
    int hpmpc_opts_size = treeqp_hpmpc_opts_calculate_size(Nn);
    void *hpmpc_opts_mem = malloc(hpmpc_opts_size);
    treeqp_hpmpc_opts_create(Nn, &hpmpc_opts, hpmpc_opts_mem);
    treeqp_hpmpc_opts_set_default(&hpmpc_opts);

    treeqp_hpmpc_workspace hpmpc_work;
    void *hpmpc_memory = malloc(treeqp_hpmpc_calculate_size(&qp_in, &hpmpc_opts));
    treeqp_hpmpc_create(&qp_in, &hpmpc_opts, &hpmpc_work, hpmpc_memory);
    #endif

    // set up HPIPM solver
    #ifdef SOLVE_WITH_HPIPM
    treeqp_hpipm_opts_t hpipm_opts;
    int hpipm_opts_size = treeqp_hpipm_opts_calculate_size(Nn);
    void *hpipm_opts_mem = malloc(hpipm_opts_size);
    treeqp_hpipm_opts_create(Nn, &hpipm_opts, hpipm_opts_mem);
    treeqp_hpipm_opts_set_default(&hpipm_opts);

    treeqp_hpipm_workspace hpipm_work;
    void *hpipm_memory = malloc(treeqp_hpipm_calculate_size(&qp_in, &hpipm_opts));
    treeqp_hpipm_create(&qp_in, &hpipm_opts, &hpipm_work, hpipm_memory);
    #endif

    // setup QP solution
    tree_ocp_qp_out qp_out;

    int qp_out_size = tree_ocp_qp_out_calculate_size(Nn, nx, nu, nc);
    void *qp_out_memory = malloc(qp_out_size);
    tree_ocp_qp_out_create(Nn, nx, nu, nc, &qp_out, qp_out_memory);

    double overhead;
    double max_overhead;
    double kkt_err;

    // solve with tree-sparse dual Newton strategy
    #ifdef SOLVE_WITH_TDUNES
    max_overhead = 0;
    for (int jj = 0; jj < NREP; jj++)
    {
        treeqp_tdunes_set_dual_initialization(lambda, &tdunes_work);

        if (tdunes_opts.maxIter == 1)  // run algorithm one iteration at a time for debugging
        {
            for (int ll = 0; ll < 50; ll++)
            {
                treeqp_tdunes_solve(&qp_in, &qp_out, &tdunes_opts, &tdunes_work);

                for (int kk = 1; kk < Nn; kk++)
                {   // pass multipliers to next iteration
                    blasfeo_dveccp(nx[kk], &qp_out.lam[kk], 0, &tdunes_work.slambda[tree[kk].dad], tdunes_work.idxpos[kk]);
                }
                printf("Newton iteration #%d\n", ll);
            }
        }
        else
        {
            treeqp_tdunes_solve(&qp_in, &qp_out, &tdunes_opts, &tdunes_work);
        }

        printf("tdunes run # %d (%d iterations)\n", jj, qp_out.info.iter);
        printf("solver time:\t %5.3f ms\n", qp_out.info.solver_time*1e3);
        printf("interface time:\t %5.3f ms\n", qp_out.info.interface_time*1e3);
        overhead = 100*qp_out.info.interface_time/qp_out.info.solver_time;
        printf("overhead:\t %5.2f %% \n", overhead);
        if (overhead > max_overhead) max_overhead = overhead;
    }

    kkt_err = max_KKT_residual(&qp_in, &qp_out);
    printf("Maximum error in KKT residuals (tdunes):\t\t %2.2e\n\n", kkt_err);
    assert(kkt_err < 1e-10 && "KKT tolerance of tree dual Newton in spring_mass.c too high!");

    printf("Maximum overhead of treeQP interface (tdunes):\t\t %4.2f%%\n\n", max_overhead);

    for (int ii = 0; ii < 5; ii++)
    {
        blasfeo_print_tran_dvec(qp_in.nx[ii], &qp_out.x[ii], 0);
    }
    #endif

    // solve with tree-sparse HPMPC
    #ifdef SOLVE_WITH_HPMPC
    int hpmpc_status = -1;

    max_overhead = 0;
    for (int jj = 0; jj < NREP; jj++)
    {
        hpmpc_status = treeqp_hpmpc_solve(&qp_in, &qp_out, &hpmpc_opts, &hpmpc_work);
        if (hpmpc_status != 0)
        {
            printf("HPMPC failed with status %d! <--------------------------------------------------\n", hpmpc_status);
            // exit(-1);
        }
        printf("hpmpc run # %d (%d iterations)\n", jj, qp_out.info.iter);
        printf("solver time:\t %5.2f ms\n", qp_out.info.solver_time*1e3);
        printf("interface time:\t %5.2f ms\n", qp_out.info.interface_time*1e3);
        overhead = 100*qp_out.info.interface_time/qp_out.info.solver_time;
        printf("overhead:\t %5.2f %% \n", overhead);
        if (overhead > max_overhead) max_overhead = overhead;
    }

    kkt_err = max_KKT_residual(&qp_in, &qp_out);
    printf("Maximum error in KKT residuals (hpmpc):\t\t\t %2.2e\n\n", kkt_err);
    assert(kkt_err < 1e-10 && "KKT tolerance of tree hpmpc in spring_mass.c too high!");

    printf("Maximum overhead of treeQP interface (hpmpc):\t\t %4.2f%%\n\n", max_overhead);

    for (int ii = 0; ii < 5; ii++)
    {
        blasfeo_print_tran_dvec(qp_in.nx[ii], &qp_out.x[ii], 0);
    }
    // printf("HPMPC\n:");
    // tree_ocp_qp_out_print(Nn, &qp_out);
    #endif


    // solve with tree-sparse HPIPM
    #ifdef SOLVE_WITH_HPIPM
    int hpipm_status = -1;

    max_overhead = 0;
    for (int jj = 0; jj < NREP; jj++)
    {
        hpipm_status = treeqp_hpipm_solve(&qp_in, &qp_out, &hpipm_opts, &hpipm_work);
        if (hpipm_status != 0)
        {
            printf("HPIPM failed with status %d! <--------------------------------------------------\n", hpipm_status);
            // exit(-1);
        }
        printf("hpipm run # %d (%d iterations)\n", jj, qp_out.info.iter);
        printf("solver time:\t %5.2f ms\n", qp_out.info.solver_time*1e3);
        printf("interface time:\t %5.2f ms\n", qp_out.info.interface_time*1e3);
        overhead = 100*qp_out.info.interface_time/qp_out.info.solver_time;
        printf("overhead:\t %5.2f %% \n", overhead);
        if (overhead > max_overhead) max_overhead = overhead;
    }

    kkt_err = max_KKT_residual(&qp_in, &qp_out);
    printf("Maximum error in KKT residuals (hpipm):\t\t\t %2.2e\n\n", kkt_err);
    // assert(kkt_err < 1e-10 && "KKT tolerance of tree hpipm in spring_mass.c too high!");

    printf("Maximum overhead of treeQP interface (hpipm):\t\t %4.2f%%\n\n", max_overhead);

    for (int ii = 0; ii < 5; ii++)
    {
        blasfeo_print_tran_dvec(qp_in.nx[ii], &qp_out.x[ii], 0);
    }
    // printf("HPIPM\n:");
    // tree_ocp_qp_out_print(Nn, &qp_out);
    #endif

    // Free memory
    free(nx);
    free(nu);
    free(nc);

    #ifdef TEST_GENERAL_CONSTRAINTS
    free(C);
    free(CN);
    free(D);
    free(dmin);
    free(dmax);
    #endif

    free(qp_in_memory);
    free(qp_out_memory);

    #ifdef SOLVE_WITH_TDUNES
    free(tdunes_memory);
    #endif
    #ifdef SOLVE_WITH_HPMPC
    free(hpmpc_memory);
    #endif
    #ifdef SOLVE_WITH_HPIPM
    free(hpipm_memory);
    #endif

    free_tree(Nn, tree);
    free(tree);

    free(lambda);

    if (xmax[1] == .2) printf("[TREEQP] Warning! Bound on state has been overwritten with tighter one!\n\n");

    return 0;
}
