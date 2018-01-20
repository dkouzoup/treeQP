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
#include "blasfeo/include/blasfeo_d_blas.h"

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

    bytes += 3 * sizeof(struct blasfeo_dmat);  // sCholZTHZ, sZ, sP
    bytes += 3 * blasfeo_memsize_dmat(nvd, nvd);

    if (ngd > 0)
    {   // QProblem
        bytes += QProblem_calculateMemorySize(nvd, ngd);
    }
    else
    {   // QProblemB
        bytes += QProblemB_calculateMemorySize(nvd);
    }

    return bytes;
}



void stage_qp_qpoases_assign_structs(void **stage_qp_data, char **c_double_ptr)
{
    treeqp_tdunes_qpoases_data *qpoases_solver_data;

    qpoases_solver_data = (treeqp_tdunes_qpoases_data *)*c_double_ptr;
    *c_double_ptr += sizeof(treeqp_tdunes_qpoases_data);

    qpoases_solver_data->sCholZTHZ = (struct blasfeo_dmat *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dmat);

    qpoases_solver_data->sZ = (struct blasfeo_dmat *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dmat);

    qpoases_solver_data->sP = (struct blasfeo_dmat *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dmat);

    *stage_qp_data = (void *) qpoases_solver_data;
}



void stage_qp_qpoases_assign_blasfeo_data(int nx, int nu, void *stage_qp_data, char **c_double_ptr)
{
    treeqp_tdunes_qpoases_data *qpoases_solver_data;
    qpoases_solver_data = (treeqp_tdunes_qpoases_data *)stage_qp_data;

    init_strmat(nx+nu, nx+nu, qpoases_solver_data->sCholZTHZ, c_double_ptr);
    init_strmat(nx+nu, nx+nu, qpoases_solver_data->sZ, c_double_ptr);
    init_strmat(nx+nu, nx+nu, qpoases_solver_data->sP, c_double_ptr);
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



static void QProblemB_build_elimination_matrix(QProblemB *QPB, int node_index,
    treeqp_tdunes_workspace *work)
{
    treeqp_tdunes_qpoases_data *qpoases_solver_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[node_index];

    int nvd = QProblemB_getNV(QPB);
    int nzd = QProblemB_getNZ(QPB);  // nx + nu - n_act

    // extract Cholesky factor
    blasfeo_pack_tran_dmat(nzd, nzd, QPB->R, nvd, qpoases_solver_data->sCholZTHZ, 0, 0);

    // TODO(dimitris): use this for QP (not used in QPB)
    // blasfeo_pack_dmat(nvd, nvd, QPB->flipper->Q, nvd, qpoases_solver_data->sZ, 0, 0);

    // build Z
    int pos;
    blasfeo_dgese(nvd, nvd, 0.0, qpoases_solver_data->sZ, 0, 0);
    for (int ii = 0; ii < nzd; ii++)
    {
        pos = QPB->bounds->freee->number[ii];
        DMATEL_LIBSTR(qpoases_solver_data->sZ, pos, ii) = 1.0;
    }

    // calculate P (matrix substitution + symmetric matrix matrix multiplication)

    // D <= alpha * B * A^{-T} , with A lower triangular employing explicit inverse of diagonal
    blasfeo_dtrsm_rltn(nvd, nzd, 1.0, qpoases_solver_data->sCholZTHZ, 0, 0,
        qpoases_solver_data->sZ, 0, 0, qpoases_solver_data->sZ, 0, 0);

    // D <= beta * C + alpha * A * B^T
    // TODO(dimitris): replace with dsyrk!
    // TODO(dimitris): are m, n, k correct?
    blasfeo_dgemm_nt(nvd, nvd, nzd, 1.0, qpoases_solver_data->sZ, 0, 0, qpoases_solver_data->sZ,
        0, 0, 0.0, qpoases_solver_data->sP, 0, 0, qpoases_solver_data->sP, 0, 0);

    // printf("P (strmat):\n");
    // blasfeo_print_dmat(nvd, nvd, qpoases_solver_data->sP, 0, 0);
}



void stage_qp_qpoases_init(tree_ocp_qp_in *qp_in, int node_index, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_qpoases_data *qpoases_solver_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[node_index];

    int nx = qp_in->nx[node_index];
    int nu = qp_in->nu[node_index];

    // TODO(dimitris): figure out why this is needed (weird results for NREP > 1 otherwise)
    // *probably because it's used as scrap space in eval_dual, fix it
    blasfeo_dvecse(nx, 0.0, &work->sxas[node_index], 0);
    blasfeo_dvecse(nu, 0.0, &work->suas[node_index], 0);

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

    if (node_index > 0)
    {
        blasfeo_dgecp(nx, qp_in->A[node_index-1].n, &qp_in->A[node_index-1], 0, 0, &work->sAB[node_index-1], 0, 0);
        blasfeo_dgecp(nx, qp_in->B[node_index-1].n, &qp_in->B[node_index-1], 0, 0, &work->sAB[node_index-1], 0, qp_in->A[node_index-1].n);
    }

    // solve first QP instance

	int nWSR = 10;  // TODO(dimitris): move those max values to options
    double cputime = 1000;

    QProblemBCON(QPB, nx+nu, HST_POSDEF);
    QProblemB_setPrintLevel(QPB, PL_MEDIUM);  // TODO(dimitris): other options?
    QProblemB_printProperties(QPB);  // TODO(dimitris): what is this for?

	return_t status = QProblemB_init(QPB, qpoases_solver_data->H, qpoases_solver_data->g,
        qpoases_solver_data->lb, qpoases_solver_data->ub, &nWSR, &cputime);

    assert(status == 0 && "initialization of qpOASES failed!");
}



static void QProblemB_solve(tree_ocp_qp_in *qp_in, int node_index,  treeqp_tdunes_workspace *work)
{
    treeqp_tdunes_qpoases_data *qpoases_solver_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[node_index];

    int nx = qp_in->nx[node_index];
    int nu = qp_in->nu[node_index];

    QProblemB *QPB = qpoases_solver_data->QPB;

    // TODO(dimitris): TEMP!
    int nWSR = 1000;
    double cputime = 1000;

    // g_new = - [xmod; umod]
    blasfeo_unpack_dvec(nx, &work->sqmod[node_index], 0, &qpoases_solver_data->g[0]);
    blasfeo_unpack_dvec(nu, &work->srmod[node_index], 0, &qpoases_solver_data->g[nx]);
    for (int ii = 0; ii < nx+nu; ii++) qpoases_solver_data->g[ii] = -qpoases_solver_data->g[ii];

	QProblemB_hotstart(QPB, qpoases_solver_data->g, qpoases_solver_data->lb, qpoases_solver_data->ub, &nWSR, &cputime);

    blasfeo_pack_dvec(nx, &QPB->x[0], &work->sx[node_index], 0);
    blasfeo_pack_dvec(nu, &QPB->x[nx], &work->su[node_index], 0);
}



void stage_qp_qpoases_solve_extended(tree_ocp_qp_in *qp_in, int node_index, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_qpoases_data *qpoases_solver_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[node_index];

    QProblemB *QPB = qpoases_solver_data->QPB;

    QProblemB_solve(qp_in, node_index, work);
    QProblemB_build_elimination_matrix(QPB, node_index, work);
}



void stage_qp_qpoases_solve(tree_ocp_qp_in *qp_in, int node_index, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_qpoases_data *qpoases_solver_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[node_index];

    QProblemB *QPB = qpoases_solver_data->QPB;

    QProblemB_solve(qp_in, node_index, work);
}



void stage_qp_qpoases_set_CmPnCmT(tree_ocp_qp_in *qp_in, int node_index, int dad_index, int offset,
    void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;

    struct blasfeo_dmat *sP_dad =
        ((treeqp_tdunes_qpoases_data *)work->stage_qp_data[dad_index])->sP;

    struct blasfeo_dmat *sC = &work->sAB[node_index-1];
    struct blasfeo_dmat *sM = &work->sM[node_index];
    struct blasfeo_dmat *sW_dad = &work->sW[dad_index];

    int nx = qp_in->nx[node_index];
    int nu = qp_in->nu[node_index];
    int nxdad = qp_in->nx[dad_index];
    int nudad = qp_in->nu[dad_index];

    // TODO(dimitris): check with @giaf that this is more efficient than blasfeo_dgemm_nn
    // M = C[k] * P[idxdad] (used both for Ut and W)
    blasfeo_dgemm_nt(nx, nxdad+nudad, nxdad+nudad, 1.0, sC, 0, 0, sP_dad, 0, 0, 0.0,
        sM, 0, 0, sM, 0, 0);

    // W[idxdad]+offset = C[k]*M' = C[k]*P[idxdad]*C[k]'
    blasfeo_dsyrk_ln(nx, nxdad+nudad, 1.0, sC, 0, 0, sM, 0, 0, 0.0, sW_dad, offset, offset,
        sW_dad, offset, offset);
}



void stage_qp_qpoases_add_EPmE(tree_ocp_qp_in *qp_in, int node_index, int dad_index, int offset,
    void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;

    struct blasfeo_dmat *sP =
        ((treeqp_tdunes_qpoases_data *)work->stage_qp_data[node_index])->sP;

    struct blasfeo_dmat *sW_dad = &work->sW[dad_index];

    int nx = qp_in->nx[node_index];

    // W[idxdad]+offset += E'*P[k]*E
    blasfeo_dgead(nx, nx, 1.0, sP, 0, 0, sW_dad, offset, offset);
}



void stage_qp_qpoases_add_CmPnCkT(tree_ocp_qp_in *qp_in, int node_index, int sib_index, int dad_index,
    int row_offset, int col_offset, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;

    int nx = qp_in->nx[node_index];
    int nxsib = qp_in->nx[sib_index];
    int nxdad = qp_in->nx[dad_index];
    int nudad = qp_in->nu[dad_index];

    struct blasfeo_dmat *sP_dad =
        ((treeqp_tdunes_qpoases_data *)work->stage_qp_data[dad_index])->sP;

    struct blasfeo_dmat *sC = &work->sAB[node_index-1];
    struct blasfeo_dmat *sC_sib = &work->sAB[sib_index-1];

    struct blasfeo_dmat *sM = &work->sM[node_index];
    struct blasfeo_dmat *sW_dad = &work->sW[dad_index];

    // TODO(dimitris): check with @giaf that this is more efficient than blasfeo_dgemm_nn
    // M = C[idxsib] * P[idxdad]
    blasfeo_dgemm_nt(nxsib, nxdad+nudad, nxdad+nudad, 1.0, sC_sib, 0, 0, sP_dad, 0, 0, 0.0,
        sM, 0, 0, sM, 0, 0);

    // W[idxdad]+offset += C[k]*M' = C[k]*P[idxdad]*C[idxsib]'
    blasfeo_dgemm_nt(nx, nxsib, nxdad+nudad, 1.0, sC, 0, 0, sM, 0, 0, 0.0,
        sW_dad, row_offset, col_offset, sW_dad, row_offset, col_offset);
}



void stage_qp_qpoases_eval_dual_term(tree_ocp_qp_in *qp_in, int node_index, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_qpoases_data *qpoases_solver_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[node_index];

    int nx = qp_in->nx[node_index];
    int nu = qp_in->nu[node_index];

    double *fval = &work->fval[node_index];
    double cmod = work->cmod[node_index];

    // feval = - (1/2)x[k]' * Q[k] * x[k] + x[k]' * qmod[k] - cmod[k]
    // NOTE: qmod[k] has already a minus sign
    // NOTE: xas used as workspace

    // z <= beta * y + alpha * A * x, A symmetric and only lower triangular part of A is accessed
    blasfeo_dsymv_l(nx, nx, 1.0, &qp_in->Q[node_index], 0, 0, &work->sx[node_index], 0, 0.0,
        &work->sxas[node_index], 0, &work->sxas[node_index], 0);
    *fval = -0.5*blasfeo_ddot(nx, &work->sxas[node_index], 0, &work->sx[node_index], 0) - cmod;
    *fval += blasfeo_ddot(nx, &work->sqmod[node_index], 0, &work->sx[node_index], 0);

    // feval -= (1/2)u[k]' * R[k] * u[k] - u[k]' * rmod[k]
    blasfeo_dsymv_l(nu, nu, 1.0, &qp_in->R[node_index], 0, 0, &work->su[node_index], 0, 0.0,
        &work->suas[node_index], 0, &work->suas[node_index], 0);
    *fval -= 0.5*blasfeo_ddot(nu, &work->suas[node_index], 0, &work->su[node_index], 0);
    *fval += blasfeo_ddot(nu, &work->srmod[node_index], 0, &work->su[node_index], 0);
}



void stage_qp_qpoases_export_mu(tree_ocp_qp_out *qp_out, int node_index, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_qpoases_data *qpoases_solver_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[node_index];

    QProblemB *QPB = qpoases_solver_data->QPB;

    int nx = qp_out->x[node_index].m;
    int nu = qp_out->u[node_index].m;

    blasfeo_pack_dvec(nx, &QPB->y[0], &qp_out->mu_x[node_index], 0);
    blasfeo_pack_dvec(nu, &QPB->y[nx], &qp_out->mu_u[node_index], 0);

    // TODO(dimitris): have same convention as qpOASES instead of flipping sign here
    for (int ii = 0; ii < nx; ii++)
    {
        DVECEL_LIBSTR(&qp_out->mu_x[node_index], ii) = - DVECEL_LIBSTR(&qp_out->mu_x[node_index], ii);
    }
    for (int ii = 0; ii < nu; ii++)
    {
        DVECEL_LIBSTR(&qp_out->mu_u[node_index], ii) = - DVECEL_LIBSTR(&qp_out->mu_u[node_index], ii);
    }
}
