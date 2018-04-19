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

#include "treeqp/src/hpmpc_tree.h"
#include "treeqp/src/hpipm_tree.h"
#include "treeqp/src/dual_Newton_tree.h"
#include "treeqp/src/tree_ocp_qp_common.h"

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

#include "examples/spring_mass_utils/data.c"

int main( ) {
    return_t status;

    Nh = 3;  // use shorter horizon to debug

    int Nn = calculate_number_of_nodes(md, Nr, Nh);
    int Np = Nn - ipow(md, Nr);

    // read constraint on x0 from txt file
    double x0[NX];
    for (int ii = 0; ii < NX; ii++) x0[ii] = 0.0;

    // setup scenario tree
    struct node *tree = malloc(Nn*sizeof(struct node));
    setup_multistage_tree(md, Nr, Nh, Nn, tree);

    // setup QP

    int *nx = malloc(Nn*sizeof(int));
    int *nu = malloc(Nn*sizeof(int));
    int *nc = malloc(Nn*sizeof(int));

    for (int ii = 0; ii < Nn; ii++)
    {
        // state and input dimensions on each node (only different at root/leaves)
        nx[ii] = NX;

        if (tree[ii].nkids > 0)  // not a leaf
            nu[ii] = NU;
        else
            nu[ii] = 0;

        nc[ii] = 0;
    }

    tree_ocp_qp_in qp_in;
    int qp_in_size = tree_ocp_qp_in_calculate_size(Nn, nx, nu, nc, tree);
    void *qp_in_memory = malloc(qp_in_size);
    tree_ocp_qp_in_create(Nn, nx, nu, nc, tree, &qp_in, qp_in_memory);

    // NOTE(dimitris): skipping first dynamics that represent the nominal ones
    tree_ocp_qp_in_fill_lti_data_diag_weights(&A[NX*NX], &B[NX*NU], &b[NX], dQ, q, dP, p, dR, r,
        xmin, xmax, umin, umax, x0, NULL, NULL, NULL, NULL, NULL, &qp_in);

    // set up HPMPC solver
	treeqp_hpmpc_opts_t hpmpc_opts = treeqp_hpmpc_opts_set_default();

    treeqp_hpmpc_workspace hpmpc_work;
    void *hpmpc_memory = malloc(treeqp_hpmpc_calculate_size(&qp_in, &hpmpc_opts));
    treeqp_hpmpc_create(&qp_in, &hpmpc_opts, &hpmpc_work, hpmpc_memory);

    // set up HPIPM solver
	treeqp_hpipm_options_t hpipm_opts = treeqp_hpipm_default_options();
    treeqp_hpipm_workspace hpipm_work;
    void *hpipm_memory = malloc(treeqp_hpipm_calculate_size(&qp_in, &hpipm_opts));
    create_treeqp_hpipm(&qp_in, &hpipm_opts, &hpipm_work, hpipm_memory);

    // setup QP solution
    tree_ocp_qp_out qp_out;

    int qp_out_size = tree_ocp_qp_out_calculate_size(Nn, nx, nu, nc);
    void *qp_out_memory = malloc(qp_out_size);
    tree_ocp_qp_out_create(Nn, nx, nu, nc, &qp_out, qp_out_memory);


    int hpipm_status = treeqp_hpipm_solve(&qp_in, &qp_out, &hpipm_opts, &hpipm_work);

	printf("\nipm residuals max: res_g = %e, res_b = %e, res_d = %e, res_m = %e\n", hpipm_work.hpipm_memory.qp_res[0], hpipm_work.hpipm_memory.qp_res[1], hpipm_work.hpipm_memory.qp_res[2], hpipm_work.hpipm_memory.qp_res[3]);

	printf("\nipm iter = %d\n", hpipm_work.hpipm_memory.iter);
	printf("\nalpha_aff\tmu_aff\t\tsigma\t\talpha\t\tmu\n");
	d_print_e_tran_mat(5, hpipm_work.hpipm_memory.iter, hpipm_work.hpipm_memory.stat, 5);

    printf("SOL HPIPM\n");
    for (int ii = 0; ii < 5; ii++)
        blasfeo_print_tran_dvec(qp_in.nx[ii], &qp_out.x[ii], 0);

    int hpmpc_status = treeqp_hpmpc_solve(&qp_in, &qp_out, &hpmpc_opts, &hpmpc_work);

    printf("SOL HPMPC\n");
    for (int ii = 0; ii < 5; ii++)
        blasfeo_print_tran_dvec(qp_in.nx[ii], &qp_out.x[ii], 0);

    return 0;
}
