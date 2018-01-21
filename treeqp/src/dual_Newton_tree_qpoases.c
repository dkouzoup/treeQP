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

answer_t stage_qp_qpoases_is_applicable(tree_ocp_qp_in *qp_in, int idx)
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
    treeqp_tdunes_qpoases_data *qpoases_data;

    qpoases_data = (treeqp_tdunes_qpoases_data *)*c_double_ptr;
    *c_double_ptr += sizeof(treeqp_tdunes_qpoases_data);

    qpoases_data->sCholZTHZ = (struct blasfeo_dmat *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dmat);

    qpoases_data->sZ = (struct blasfeo_dmat *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dmat);

    qpoases_data->sP = (struct blasfeo_dmat *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dmat);

    *stage_qp_data = (void *) qpoases_data;
}



void stage_qp_qpoases_assign_blasfeo_data(int nx, int nu, void *stage_qp_data, char **c_double_ptr)
{
    treeqp_tdunes_qpoases_data *qpoases_data;
    qpoases_data = (treeqp_tdunes_qpoases_data *)stage_qp_data;

    init_strmat(nx+nu, nx+nu, qpoases_data->sCholZTHZ, c_double_ptr);
    init_strmat(nx+nu, nx+nu, qpoases_data->sZ, c_double_ptr);
    init_strmat(nx+nu, nx+nu, qpoases_data->sP, c_double_ptr);
}



void stage_qp_qpoases_assign_data(int nx, int nu, void *stage_qp_data, char **c_double_ptr)
{
    treeqp_tdunes_qpoases_data *qpoases_data;
    qpoases_data = (treeqp_tdunes_qpoases_data *)stage_qp_data;

    int nvd = nx + nu;
    int ngd = 0;  // TODO(dimitris): support general constraints

    create_double(nvd*nvd, &qpoases_data->H, c_double_ptr);
    create_double(nvd*ngd, &qpoases_data->C, c_double_ptr);
    create_double(nvd, &qpoases_data->g, c_double_ptr);
    create_double(nvd, &qpoases_data->lb, c_double_ptr);
    create_double(nvd, &qpoases_data->ub, c_double_ptr);
    create_double(ngd, &qpoases_data->lc, c_double_ptr);
    create_double(ngd, &qpoases_data->uc, c_double_ptr);

    if (ngd > 0)
    {   // QProblem
        QProblem_assignMemory(nvd, ngd, (QProblem **) &(qpoases_data->QP), *c_double_ptr);
        *c_double_ptr += QProblem_calculateMemorySize(nvd, ngd);
    }
    else
    {   // QProblemB
        QProblemB_assignMemory(nvd, (QProblemB **) &(qpoases_data->QPB), *c_double_ptr);
        *c_double_ptr += QProblemB_calculateMemorySize(nvd);
    }
}



static void QProblemB_build_elimination_matrix(QProblemB *QPB, int idx,
    treeqp_tdunes_workspace *work)
{
    treeqp_tdunes_qpoases_data *qpoases_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[idx];

    int nvd = QProblemB_getNV(QPB);
    int nzd = QProblemB_getNZ(QPB);  // nx + nu - n_act

    // extract Cholesky factor
    blasfeo_pack_tran_dmat(nzd, nzd, QPB->R, nvd, qpoases_data->sCholZTHZ, 0, 0);

    // TODO(dimitris): use this for QP (not used in QPB)
    // blasfeo_pack_dmat(nvd, nvd, QPB->flipper->Q, nvd, qpoases_data->sZ, 0, 0);

    // build Z
    int pos;
    blasfeo_dgese(nvd, nvd, 0.0, qpoases_data->sZ, 0, 0);
    for (int ii = 0; ii < nzd; ii++)
    {
        pos = QPB->bounds->freee->number[ii];
        DMATEL_LIBSTR(qpoases_data->sZ, pos, ii) = 1.0;
    }

    // calculate P (matrix substitution + symmetric matrix matrix multiplication)

    // D <= alpha * B * A^{-T} , with A lower triangular employing explicit inverse of diagonal
    blasfeo_dtrsm_rltn(nvd, nzd, 1.0, qpoases_data->sCholZTHZ, 0, 0,
        qpoases_data->sZ, 0, 0, qpoases_data->sZ, 0, 0);

    // D <= beta * C + alpha * A * B^T
    // TODO(dimitris): replace with dsyrk!
    // TODO(dimitris): are m, n, k correct?
    blasfeo_dgemm_nt(nvd, nvd, nzd, 1.0, qpoases_data->sZ, 0, 0, qpoases_data->sZ, 0, 0, 0.0,
        qpoases_data->sP, 0, 0, qpoases_data->sP, 0, 0);

    // printf("P (strmat):\n");
    // blasfeo_print_dmat(nvd, nvd, qpoases_data->sP, 0, 0);
}



void stage_qp_qpoases_init(tree_ocp_qp_in *qp_in, int idx, stage_qp_t solver_dad, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_qpoases_data *qpoases_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[idx];

    int nx = qp_in->nx[idx];
    int nu = qp_in->nu[idx];

    // TODO(dimitris): figure out why this is needed (weird results for NREP > 1 otherwise)
    // *probably because it's used as scrap space in eval_dual, fix it
    blasfeo_dvecse(nx, 0.0, &work->sxas[idx], 0);
    blasfeo_dvecse(nu, 0.0, &work->suas[idx], 0);

    QProblemB *QPB = (QProblemB *)qpoases_data->QPB;
    // TODO(dimitris): handle general constraints
    QProblem *QP = (QProblem *)qpoases_data->QP;

    // convert data
    blasfeo_unpack_tran_dmat(nx, nx, &qp_in->Q[idx], 0, 0, &qpoases_data->H[0], nx+nu);
    blasfeo_unpack_tran_dmat(nu, nu, &qp_in->R[idx], 0, 0, &qpoases_data->H[nx*(nx+nu)+nx], nx+nu);
    blasfeo_unpack_dmat(nu, nx, &qp_in->S[idx], 0, 0, &qpoases_data->H[nx], nx+nu);
    blasfeo_unpack_tran_dmat(nu, nx, &qp_in->S[idx], 0, 0, &qpoases_data->H[nx*(nx+nu)], nx+nu);

    blasfeo_unpack_dvec(nx, &qp_in->q[idx], 0, &qpoases_data->g[0]);
    blasfeo_unpack_dvec(nu, &qp_in->r[idx], 0, &qpoases_data->g[nx]);

    blasfeo_unpack_dvec(nx, &qp_in->xmin[idx], 0, &qpoases_data->lb[0]);
    blasfeo_unpack_dvec(nu, &qp_in->umin[idx], 0, &qpoases_data->lb[nx]);
    blasfeo_unpack_dvec(nx, &qp_in->xmax[idx], 0, &qpoases_data->ub[0]);
    blasfeo_unpack_dvec(nu, &qp_in->umax[idx], 0, &qpoases_data->ub[nx]);

    struct blasfeo_dmat *sA = &qp_in->A[idx-1];
    struct blasfeo_dmat *sB = &qp_in->B[idx-1];
    struct blasfeo_dmat *sAB = &work->sAB[idx-1];

    if (idx > 0)
    {
        blasfeo_dgecp(nx, sA->n, sA, 0, 0, sAB, 0, 0);
        blasfeo_dgecp(nx, sB->n, sB, 0, 0, sAB, 0, sA->n);
    }

    // solve first QP instance

	int nWSR = 10;  // TODO(dimitris): move those max values to options
    double cputime = 1000;

    QProblemBCON(QPB, nx+nu, HST_POSDEF);
    QProblemB_setPrintLevel(QPB, PL_MEDIUM);  // TODO(dimitris): other options?
    QProblemB_printProperties(QPB);  // TODO(dimitris): what is this for?

	return_t status = QProblemB_init(QPB, qpoases_data->H, qpoases_data->g,
        qpoases_data->lb, qpoases_data->ub, &nWSR, &cputime);

    assert(status == 0 && "initialization of qpOASES failed!");
}



static void QProblemB_solve(tree_ocp_qp_in *qp_in, int idx,  treeqp_tdunes_workspace *work)
{
    treeqp_tdunes_qpoases_data *qpoases_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[idx];

    int nx = qp_in->nx[idx];
    int nu = qp_in->nu[idx];

    QProblemB *QPB = (QProblemB *)qpoases_data->QPB;

    // TODO(dimitris): TEMP!
    int nWSR = 1000;
    double cputime = 1000;

    // g_new = - [xmod; umod]
    blasfeo_unpack_dvec(nx, &work->sqmod[idx], 0, &qpoases_data->g[0]);
    blasfeo_unpack_dvec(nu, &work->srmod[idx], 0, &qpoases_data->g[nx]);
    for (int ii = 0; ii < nx+nu; ii++) qpoases_data->g[ii] = -qpoases_data->g[ii];

	QProblemB_hotstart(QPB, qpoases_data->g, qpoases_data->lb, qpoases_data->ub, &nWSR, &cputime);

    blasfeo_pack_dvec(nx, &QPB->x[0], &work->sx[idx], 0);
    blasfeo_pack_dvec(nu, &QPB->x[nx], &work->su[idx], 0);
}



void stage_qp_qpoases_solve_extended(tree_ocp_qp_in *qp_in, int idx, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_qpoases_data *qpoases_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[idx];

    QProblemB *QPB = (QProblemB *)qpoases_data->QPB;

    QProblemB_solve(qp_in, idx, work);
    QProblemB_build_elimination_matrix(QPB, idx, work);
}



void stage_qp_qpoases_solve(tree_ocp_qp_in *qp_in, int idx, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_qpoases_data *qpoases_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[idx];

    QProblemB *QPB = (QProblemB *)qpoases_data->QPB;

    QProblemB_solve(qp_in, idx, work);
}



void stage_qp_qpoases_set_CmPnCmT(tree_ocp_qp_in *qp_in, int idx, int idxdad, int offset,
    void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;

    struct blasfeo_dmat *sP_dad =
        ((treeqp_tdunes_qpoases_data *)work->stage_qp_data[idxdad])->sP;

    struct blasfeo_dmat *sC = &work->sAB[idx-1];
    struct blasfeo_dmat *sM = &work->sM[idx];
    struct blasfeo_dmat *sW_dad = &work->sW[idxdad];

    int nx = qp_in->nx[idx];
    int nu = qp_in->nu[idx];
    int nxdad = qp_in->nx[idxdad];
    int nudad = qp_in->nu[idxdad];

    // TODO(dimitris): check with @giaf that this is more efficient than blasfeo_dgemm_nn
    // M = C[k] * P[idxdad] (used both for Ut and W)
    blasfeo_dgemm_nt(nx, nxdad+nudad, nxdad+nudad, 1.0, sC, 0, 0, sP_dad, 0, 0, 0.0,
        sM, 0, 0, sM, 0, 0);

    // W[idxdad]+offset = C[k]*M' = C[k]*P[idxdad]*C[k]'
    blasfeo_dsyrk_ln(nx, nxdad+nudad, 1.0, sC, 0, 0, sM, 0, 0, 0.0,
        sW_dad, offset, offset, sW_dad, offset, offset);
}



void stage_qp_qpoases_add_EPmE(tree_ocp_qp_in *qp_in, int idx, int idxdad, int offset,
    void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;

    struct blasfeo_dmat *sP =
        ((treeqp_tdunes_qpoases_data *)work->stage_qp_data[idx])->sP;

    struct blasfeo_dmat *sW_dad = &work->sW[idxdad];

    int nx = qp_in->nx[idx];

    // W[idxdad]+offset += E'*P[k]*E
    blasfeo_dgead(nx, nx, 1.0, sP, 0, 0, sW_dad, offset, offset);
}



void stage_qp_qpoases_add_CmPnCkT(tree_ocp_qp_in *qp_in, int idx, int idxsib, int idxdad,
    int row_offset, int col_offset, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;

    int nx = qp_in->nx[idx];
    int nxsib = qp_in->nx[idxsib];
    int nxdad = qp_in->nx[idxdad];
    int nudad = qp_in->nu[idxdad];

    struct blasfeo_dmat *sP_dad = ((treeqp_tdunes_qpoases_data *)work->stage_qp_data[idxdad])->sP;

    struct blasfeo_dmat *sC = &work->sAB[idx-1];
    struct blasfeo_dmat *sC_sib = &work->sAB[idxsib-1];

    struct blasfeo_dmat *sM = &work->sM[idx];
    struct blasfeo_dmat *sW_dad = &work->sW[idxdad];

    // TODO(dimitris): check with @giaf that this is more efficient than blasfeo_dgemm_nn
    // M = C[idxsib] * P[idxdad]
    blasfeo_dgemm_nt(nxsib, nxdad+nudad, nxdad+nudad, 1.0, sC_sib, 0, 0, sP_dad, 0, 0, 0.0,
        sM, 0, 0, sM, 0, 0);

    // W[idxdad]+offset += C[k]*M' = C[k]*P[idxdad]*C[idxsib]'
    blasfeo_dgemm_nt(nx, nxsib, nxdad+nudad, 1.0, sC, 0, 0, sM, 0, 0, 0.0,
        sW_dad, row_offset, col_offset, sW_dad, row_offset, col_offset);
}



void stage_qp_qpoases_eval_dual_term(tree_ocp_qp_in *qp_in, int idx, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_qpoases_data *qpoases_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[idx];

    int nx = qp_in->nx[idx];
    int nu = qp_in->nu[idx];

    double *fval = &work->fval[idx];
    double cmod = work->cmod[idx];

    // feval = - (1/2)x[k]' * Q[k] * x[k] + x[k]' * qmod[k] - cmod[k]
    // NOTE: qmod[k] has already a minus sign
    // NOTE: xas used as workspace

    // z <= beta * y + alpha * A * x, A symmetric and only lower triangular part of A is accessed
    blasfeo_dsymv_l(nx, nx, 1.0, &qp_in->Q[idx], 0, 0, &work->sx[idx], 0, 0.0,
        &work->sxas[idx], 0, &work->sxas[idx], 0);
    *fval = -0.5*blasfeo_ddot(nx, &work->sxas[idx], 0, &work->sx[idx], 0) - cmod;
    *fval += blasfeo_ddot(nx, &work->sqmod[idx], 0, &work->sx[idx], 0);

    // feval -= (1/2)u[k]' * R[k] * u[k] - u[k]' * rmod[k]
    blasfeo_dsymv_l(nu, nu, 1.0, &qp_in->R[idx], 0, 0, &work->su[idx], 0, 0.0,
        &work->suas[idx], 0, &work->suas[idx], 0);
    *fval -= 0.5*blasfeo_ddot(nu, &work->suas[idx], 0, &work->su[idx], 0);
    *fval += blasfeo_ddot(nu, &work->srmod[idx], 0, &work->su[idx], 0);
}



void stage_qp_qpoases_export_mu(tree_ocp_qp_out *qp_out, int idx, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_qpoases_data *qpoases_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[idx];

    QProblemB *QPB = (QProblemB *)qpoases_data->QPB;

    int nx = qp_out->x[idx].m;
    int nu = qp_out->u[idx].m;

    blasfeo_pack_dvec(nx, &QPB->y[0], &qp_out->mu_x[idx], 0);
    blasfeo_pack_dvec(nu, &QPB->y[nx], &qp_out->mu_u[idx], 0);

    // TODO(dimitris): have same convention as qpOASES instead of flipping sign here
    for (int ii = 0; ii < nx; ii++)
    {
        DVECEL_LIBSTR(&qp_out->mu_x[idx], ii) = - DVECEL_LIBSTR(&qp_out->mu_x[idx], ii);
    }
    for (int ii = 0; ii < nu; ii++)
    {
        DVECEL_LIBSTR(&qp_out->mu_u[idx], ii) = - DVECEL_LIBSTR(&qp_out->mu_u[idx], ii);
    }
}
