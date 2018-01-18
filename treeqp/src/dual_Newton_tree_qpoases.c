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

#include "treeqp/src/dual_Newton_tree.h"
#include "treeqp/src/dual_Newton_tree_qpoases.h"
#include "treeqp/src/tree_ocp_qp_common.h"
// #include "treeqp/utils/blasfeo.h"
#include "treeqp/utils/memory.h"
#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"

#include <qpOASES_e.h>

answer_t stage_qp_qpoases_is_applicable(tree_ocp_qp_in *qp_in, int node_index)
{
    return YES;
}



int stage_qp_qpoases_calculate_size(int nx, int nu)
{
    int bytes  = 0;

    bytes += sizeof(treeqp_tdunes_qpoases_data);

    int nvd = nx + nu;
    int ngd = 0;  // TODO(dimitris): support general constraints

    bytes += 1 * nvd * nvd * sizeof(double);  // H
    bytes += 1 * nvd * ngd * sizeof(double);  // C
    bytes += 3 * nvd * sizeof(double);  // g, lb, ub
    bytes += 2 * ngd * sizeof(double);  // lc, uc
    bytes += 1 * nvd * sizeof(double);  // prim_sol
    bytes += (nvd+ngd) * sizeof(double);  // dual_sol

    if (ngd > 0)
    {   // QProblem
        bytes += QProblem_calculateMemorySize(nvd, ngd);
    }
    else
    {   // QProblemB
        bytes += QProblemB_calculateMemorySize(nvd);
    }
    make_int_multiple_of(8, &bytes);

    return bytes;
}



void stage_qp_qpoases_assign_structs(void **stage_qp_data, char **c_double_ptr)
{
    treeqp_tdunes_qpoases_data *qpoases_solver_data;

    qpoases_solver_data = (treeqp_tdunes_qpoases_data *)*c_double_ptr;
    *c_double_ptr += sizeof(treeqp_tdunes_qpoases_data);

    *stage_qp_data = (void *) qpoases_solver_data;
}



// NOTE(dimitris): structs and data are assigned separately due to alignment requirements
void stage_qp_qpoases_assign_data(int nx, int nu, void *stage_qp_data, char **c_double_ptr)
{
    treeqp_tdunes_qpoases_data *qpoases_solver_data;
    qpoases_solver_data = (treeqp_tdunes_qpoases_data *)stage_qp_data;

    int nvd = nx + nu;
    int ngd = 0;  // TODO(dimitris): support general constraints

    create_double(nvd*nvd, &qpoases_solver_data->H, c_double_ptr);
    create_double(nvd*ngd, &qpoases_solver_data->C, c_double_ptr);
    create_double(nvd, &qpoases_solver_data->g, c_double_ptr);
    create_double(nvd, &qpoases_solver_data->lb, c_double_ptr);
    create_double(nvd, &qpoases_solver_data->ub, c_double_ptr);
    create_double(ngd, &qpoases_solver_data->lc, c_double_ptr);
    create_double(ngd, &qpoases_solver_data->uc, c_double_ptr);
    create_double(nvd, &qpoases_solver_data->prim_sol, c_double_ptr);
    create_double(nvd+ngd, &qpoases_solver_data->dual_sol, c_double_ptr);

    assert((size_t)*c_double_ptr % 8 == 0 && "double not 8-byte aligned!");

    if (ngd > 0)
    {   // QProblem
        QProblem_assignMemory(nvd, ngd, (QProblem **) &(qpoases_solver_data->QP), *c_double_ptr);
        *c_double_ptr += QProblem_calculateMemorySize(nvd, ngd);
    }
    else
    {   // QProblemB
        QProblemB_assignMemory(nvd, (QProblemB **) &(qpoases_solver_data->QPB), *c_double_ptr);
        *c_double_ptr += QProblemB_calculateMemorySize(nvd);
    }
}



void stage_qp_qpoases_init(tree_ocp_qp_in *qp_in, int node_index, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_qpoases_data *qpoases_solver_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[node_index];

    int nx = qp_in->nx[node_index];
    int nu = qp_in->nu[node_index];

    QProblemB *QPB = qpoases_solver_data->QPB;
    // TODO(dimitris): handle general constraints
    QProblem *QP = qpoases_solver_data->QP;

    // convert data

    blasfeo_unpack_tran_dmat(nx, nx, &qp_in->Q[node_index], 0, 0, &qpoases_solver_data->H[0], nx+nu);
    blasfeo_unpack_tran_dmat(nu, nu, &qp_in->R[node_index], 0, 0, &qpoases_solver_data->H[nx*(nx+nu)+nx], nx+nu);
    blasfeo_unpack_dmat(nu, nx, &qp_in->S[node_index], 0, 0, &qpoases_solver_data->H[nx], nx+nu);
    blasfeo_unpack_tran_dmat(nu, nx, &qp_in->S[node_index], 0, 0, &qpoases_solver_data->H[nx*(nx+nu)], nx+nu);

    blasfeo_unpack_dvec(nx, &qp_in->q[node_index], 0, &qpoases_solver_data->g[0]);
    blasfeo_unpack_dvec(nu, &qp_in->r[node_index], 0, &qpoases_solver_data->g[nx]);

    blasfeo_unpack_dvec(nx, &qp_in->xmin[node_index], 0, &qpoases_solver_data->lb[0]);
    blasfeo_unpack_dvec(nu, &qp_in->umin[node_index], 0, &qpoases_solver_data->lb[nx]);
    blasfeo_unpack_dvec(nx, &qp_in->xmax[node_index], 0, &qpoases_solver_data->ub[0]);
    blasfeo_unpack_dvec(nu, &qp_in->umax[node_index], 0, &qpoases_solver_data->ub[nx]);

    // solve first QP instance

	int nWSR = 10;      // TODO(dimitris): move those max values to options
    double cputime = 1000;

    QProblemBCON(QPB, nx+nu, HST_POSDEF);
    QProblemB_setPrintLevel(QPB, PL_MEDIUM);  // TODO(dimitris): other options?
    QProblemB_printProperties(QPB);  // TODO(dimitris): what is this for?

	QProblemB_init(QPB, qpoases_solver_data->H, qpoases_solver_data->g,
        qpoases_solver_data->lb, qpoases_solver_data->ub, &nWSR, &cputime);

    // QProblemB_getPrimalSolution(QPB, qpoases_solver_data->prim_sol);
    // printf("primal sol:\n");
    // for (int ii = 0; ii < nx+nu; ii++)
    // {
    //     printf("%f\t", qpoases_solver_data->prim_sol[ii]);
    // }
    // printf("\n\n");
    // exit(1);

    // printf("Q:\n");
    // blasfeo_print_dmat(nx, nx, &qp_in->Q[node_index], 0, 0);
    // printf("R:\n");
    // blasfeo_print_dmat(nu, nu, &qp_in->R[node_index], 0, 0);
    // printf("S:\n");
    // blasfeo_print_dmat(nu, nx, &qp_in->S[node_index], 0, 0);
    // printf("\n\n");
    // printf("matrix H (row major):\n");
    // for (int ii = 0; ii < nx+nu; ii++)
    // {
    //     for (int jj = 0; jj < nx+nu; jj++)
    //     {
    //         printf("%f\t", qpoases_solver_data->H[ii*(nx+nu)+jj]);
    //     }
    //     printf("\n");
    // }
    // exit(1);

    // solve first QP instance
}
