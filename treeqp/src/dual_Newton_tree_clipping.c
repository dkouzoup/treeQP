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

#include "treeqp/src/dual_Newton_tree.h"
#include "treeqp/src/dual_Newton_tree_clipping.h"
#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/utils/blasfeo.h"
#include "treeqp/utils/memory.h"
#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_blas.h"



answer_t stage_qp_clipping_is_applicable(tree_ocp_qp_in *qp_in, int idx)
{
    answer_t ans = YES;

    if (is_strmat_diagonal(&qp_in->Q[idx]) == NO)
    {
        ans = NO;
    }

    if (is_strmat_diagonal(&qp_in->R[idx]) == NO)
    {
        ans = NO;
    }

    if (is_strmat_zero(&qp_in->S[idx]) == NO)
    {
        ans = NO;
    }

    if (qp_in->nc[idx] > 0)
    {
        ans = NO;
    }

    // NOTE(dimitris): currently throwing an error instead of returning answer
    if (ans == NO)
    {
        printf("[TREEQP]: Error! Specified stage QP solver (clipping) not applicable.\n");
        exit(1);
    }

    return ans;
}



int stage_qp_clipping_calculate_size(int nx, int nu, int nc)
{
    int bytes  = 0;

    bytes += sizeof(treeqp_tdunes_clipping_data);
    bytes += 6*sizeof(struct blasfeo_dvec);  // Q, R, Qinv, Rinv, QinvCal, RinvCal
    bytes += 3*blasfeo_memsize_dvec(nx);  // Q, Qinv, QinvCal
    bytes += 3*blasfeo_memsize_dvec(nu);  // R, Rinv, RinvCal

    assert (nc == 0);

    return bytes;
}



void stage_qp_clipping_assign_structs(void **stage_qp_data, char **c_double_ptr)
{
    treeqp_tdunes_clipping_data *clipping_data;

    clipping_data = (treeqp_tdunes_clipping_data *)*c_double_ptr;
    *c_double_ptr += sizeof(treeqp_tdunes_clipping_data);

    clipping_data->sQ = (struct blasfeo_dvec *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dvec);

    clipping_data->sR = (struct blasfeo_dvec *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dvec);

    clipping_data->sQinv = (struct blasfeo_dvec *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dvec);

    clipping_data->sRinv = (struct blasfeo_dvec *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dvec);

    clipping_data->sQinvCal = (struct blasfeo_dvec *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dvec);

    clipping_data->sRinvCal = (struct blasfeo_dvec *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dvec);

    *stage_qp_data = (void *) clipping_data;
}



void stage_qp_clipping_assign_blasfeo_data(int nx, int nu, void *stage_qp_data, char **c_double_ptr)
{
    treeqp_tdunes_clipping_data *clipping_data;
    clipping_data = (treeqp_tdunes_clipping_data *)stage_qp_data;

    init_strvec(nx, clipping_data->sQ, c_double_ptr);
    init_strvec(nu, clipping_data->sR, c_double_ptr);
    init_strvec(nx, clipping_data->sQinv, c_double_ptr);
    init_strvec(nu, clipping_data->sRinv, c_double_ptr);
    init_strvec(nx, clipping_data->sQinvCal, c_double_ptr);
    init_strvec(nu, clipping_data->sRinvCal, c_double_ptr);
}



void stage_qp_clipping_assign_data(int nx, int nu, int nc, void *stage_qp_data, char **c_double_ptr)
{
    // NOTE(dimitris): dummy function, all data are in blasfeo format.
}



void stage_qp_clipping_init(tree_ocp_qp_in *qp_in, int idx, stage_qp_t solver_dad, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_clipping_data *clipping_data =
        (treeqp_tdunes_clipping_data *)work->stage_qp_data[idx];

    int nx = qp_in->nx[idx];
    int nu = qp_in->nu[idx];

    // extract diagonal from weight matrices
    blasfeo_ddiaex(nx, 1.0, &qp_in->Q[idx], 0, 0, clipping_data->sQ, 0);
    blasfeo_ddiaex(nu, 1.0, &qp_in->R[idx], 0, 0, clipping_data->sR, 0);

    // build inverse of weight matrices
    for (int nn = 0; nn < nx; nn++)
    {
        BLASFEO_DVECEL(clipping_data->sQinv, nn) = 1.0/BLASFEO_DVECEL(clipping_data->sQ, nn);
    }
    for (int nn = 0; nn < nu; nn++)
    {
        BLASFEO_DVECEL(clipping_data->sRinv, nn) = 1.0/BLASFEO_DVECEL(clipping_data->sR, nn);
    }

    // NOTE(dimitris): AB matrices needed if we mix clipping and qpoases for the stage QPs
    struct blasfeo_dmat *sA = &qp_in->A[idx-1];
    struct blasfeo_dmat *sB = &qp_in->B[idx-1];
    struct blasfeo_dmat *sAB = &work->sAB[idx-1];

    if ((idx > 0) && (solver_dad == TREEQP_QPOASES_SOLVER))
    {
        blasfeo_dgecp(nx, sA->n, sA, 0, 0, sAB, 0, 0);
        blasfeo_dgecp(nx, sB->n, sB, 0, 0, sAB, 0, sA->n);
    }
}



void stage_qp_clipping_solve_extended(tree_ocp_qp_in *qp_in, int idx, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_clipping_data *clipping_data =
        (treeqp_tdunes_clipping_data *)work->stage_qp_data[idx];

    int nx = qp_in->nx[idx];
    int nu = qp_in->nu[idx];

    struct blasfeo_dvec *sxmin = &qp_in->xmin[idx];
    struct blasfeo_dvec *sxmax = &qp_in->xmax[idx];
    struct blasfeo_dvec *sumin = &qp_in->umin[idx];
    struct blasfeo_dvec *sumax = &qp_in->umax[idx];

    struct blasfeo_dvec *sxUnc = &work->sxUnc[idx];
    struct blasfeo_dvec *suUnc = &work->suUnc[idx];
    struct blasfeo_dvec *sxas = &work->sxas[idx];
    struct blasfeo_dvec *suas = &work->suas[idx];

    // x[k] = Q[k]^-1 .* qmod[k]
    // NOTE(dimitris): minus sign already in mod. gradient
    blasfeo_dvecmuldot(nx, clipping_data->sQinv, 0, &work->sqmod[idx], 0, sxUnc, 0);

    // x[k] = median(xmin, x[k], xmax), xas[k] = active set
    blasfeo_dveccl_mask(nx, sxmin, 0, sxUnc, 0, sxmax, 0, &work->sx[idx], 0, sxas, 0);

    // u[k] = R[k]^-1 .* rmod[k]
    blasfeo_dvecmuldot(nu, clipping_data->sRinv, 0, &work->srmod[idx], 0, suUnc, 0);

    // u[k] = median(umin, u[k], umax), uas[k] = active set
    blasfeo_dveccl_mask(nu, sumin, 0, suUnc, 0, sumax, 0, &work->su[idx], 0, suas, 0);

    // QinvCal[kk] = Qinv[kk] .* (1 - abs(xas[kk])), aka elimination matrix
    blasfeo_dvecze(nx, sxas, 0, clipping_data->sQinv, 0, clipping_data->sQinvCal, 0);

    // RinvCal[kk] = Rinv[kk] .* (1 - abs(uas[kk]))
    blasfeo_dvecze(nu, suas, 0, clipping_data->sRinv, 0, clipping_data->sRinvCal, 0);
}



void stage_qp_clipping_solve(tree_ocp_qp_in *qp_in, int idx, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_clipping_data *clipping_data =
        (treeqp_tdunes_clipping_data *)work->stage_qp_data[idx];

    int nx = qp_in->nx[idx];
    int nu = qp_in->nu[idx];

    struct blasfeo_dvec *sxmin = &qp_in->xmin[idx];
    struct blasfeo_dvec *sxmax = &qp_in->xmax[idx];
    struct blasfeo_dvec *sumin = &qp_in->umin[idx];
    struct blasfeo_dvec *sumax = &qp_in->umax[idx];

    struct blasfeo_dvec *sqmod = &work->sqmod[idx];
    struct blasfeo_dvec *srmod = &work->srmod[idx];

    // x[k] = Q[k]^-1 .* qmod[k]
    // NOTE(dimitris): minus sign already in mod. gradient
    blasfeo_dvecmuldot(nx, clipping_data->sQinv, 0, sqmod, 0, &work->sx[idx], 0);
    // x[k] = median(xmin, x[k], xmax)
    blasfeo_dveccl(nx, sxmin, 0, &work->sx[idx], 0, sxmax, 0, &work->sx[idx], 0);

    // u[k] = R[k]^-1 .* rmod[k]
    blasfeo_dvecmuldot(nu, clipping_data->sRinv, 0, srmod, 0, &work->su[idx], 0);
    // u[k] = median(umin, u[k], umax)
    blasfeo_dveccl(nu, sumin, 0, &work->su[idx], 0, sumax, 0, &work->su[idx], 0);
}



void stage_qp_clipping_set_CmPnCmT(tree_ocp_qp_in *qp_in, int idx, int idxdad, int offset,
    void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;

    struct blasfeo_dvec *sQinvCal_dad =
        ((treeqp_tdunes_clipping_data *)work->stage_qp_data[idxdad])->sQinvCal;
    struct blasfeo_dvec *sRinvCal_dad =
        ((treeqp_tdunes_clipping_data *)work->stage_qp_data[idxdad])->sRinvCal;

    int nx = qp_in->nx[idx];
    int nu = qp_in->nu[idx];
    int nxdad = qp_in->nx[idxdad];
    int nudad = qp_in->nu[idxdad];

    struct blasfeo_dmat *sA = &qp_in->A[idx-1];
    struct blasfeo_dmat *sB = &qp_in->B[idx-1];
    struct blasfeo_dmat *sM = &work->sM[idx];
    struct blasfeo_dmat *sW_dad = &work->sW[idxdad];

    // M = A[k] * Qinvcal[idxdad] (used both for Ut and W)
    blasfeo_dgemm_nd(nx, nxdad, 1.0, sA, 0, 0, sQinvCal_dad, 0, 0.0, sM, 0, 0, sM, 0, 0);

    // W[idxdad]+offset = A[k]*M' = A[k]*Qinvcal[idxdad]*A[k]'
    blasfeo_dsyrk_ln(nx, nxdad, 1.0, sA, 0, 0, sM, 0, 0, 0.0,
        sW_dad, offset, offset, sW_dad, offset, offset);

    // M = B[k]*Rinvcal[idxdad]
    blasfeo_dgemm_nd(nx, nudad, 1.0,  sB, 0, 0, sRinvCal_dad, 0, 0.0, sM, 0, nxdad, sM, 0, nxdad);

    // W[idxdad]+offset += B[k]*M' = B[k]*Rinvcal[idxdad]*B[k]'
    blasfeo_dsyrk_ln(nx, nudad, 1.0, sB, 0, 0, sM, 0, nxdad, 1.0,
        sW_dad, offset, offset, sW_dad, offset, offset);
}



void stage_qp_clipping_add_EPmE(tree_ocp_qp_in *qp_in, int idx, int idxdad, int offset, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;

    struct blasfeo_dvec *sQinvCal =
        ((treeqp_tdunes_clipping_data *)work->stage_qp_data[idx])->sQinvCal;

    struct blasfeo_dmat *sW_dad = &work->sW[idxdad];

    int nx = qp_in->nx[idx];

    // W[idxdad]+offset += Qinvcal[k]
    blasfeo_ddiaad(nx, 1.0, sQinvCal, 0, sW_dad, offset, offset);
}



void stage_qp_clipping_add_CmPnCkT(tree_ocp_qp_in *qp_in, int idx, int idxsib, int idxdad,
    int row_offset, int col_offset, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;

    int nx = qp_in->nx[idx];
    int nxsib = qp_in->nx[idxsib];
    int nxdad = qp_in->nx[idxdad];
    int nudad = qp_in->nu[idxdad];

    struct blasfeo_dvec *sQinvCal_dad =
        ((treeqp_tdunes_clipping_data *)work->stage_qp_data[idxdad])->sQinvCal;
    struct blasfeo_dvec *sRinvCal_dad =
        ((treeqp_tdunes_clipping_data *)work->stage_qp_data[idxdad])->sRinvCal;

    struct blasfeo_dmat *sA = &qp_in->A[idx-1];
    struct blasfeo_dmat *sB = &qp_in->B[idx-1];
    struct blasfeo_dmat *sA_sib = &qp_in->A[idxsib-1];
    struct blasfeo_dmat *sB_sib = &qp_in->B[idxsib-1];

    struct blasfeo_dmat *sM = &work->sM[idx];
    struct blasfeo_dmat *sW_dad = &work->sW[idxdad];

    // M = A[idxsib] * Qinvcal[idxdad]
    blasfeo_dgemm_nd(nxsib, nxdad, 1.0,  sA_sib, 0, 0, sQinvCal_dad, 0, 0.0, sM, 0, 0, sM, 0, 0);

    // W[idxdad]+offset = A[k]*M' = A[k]*Qinvcal[idxdad]*A[idxsib]'
    blasfeo_dgemm_nt(nx, nxsib, nxdad, 1.0, sA, 0, 0, sM, 0, 0, 0.0,
        sW_dad, row_offset, col_offset, sW_dad, row_offset, col_offset);

    // M = B[idxsib]*Rinvcal[idxdad]
    blasfeo_dgemm_nd(nxsib, nudad, 1.0, sB_sib, 0, 0, sRinvCal_dad, 0, 0.0,
        sM, 0, nxdad, sM, 0, nxdad);

    // W[idxdad]+offset += B[k]*M' = B[k]*Rinvcal[idxdad]*B[idxsib]'
    blasfeo_dgemm_nt(nx, nxsib, nudad, 1.0, sB, 0, 0, sM, 0, nxdad, 1.0,
        sW_dad, row_offset, col_offset, sW_dad, row_offset, col_offset);
}



void stage_qp_clipping_eval_dual_term(tree_ocp_qp_in *qp_in, int idx, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_clipping_data *clipping_data =
        (treeqp_tdunes_clipping_data *)work->stage_qp_data[idx];

    int nx = qp_in->nx[idx];
    int nu = qp_in->nu[idx];

    double *fval = &work->fval[idx];
    double cmod = work->cmod[idx];

    // feval = - (1/2)x[k]' * Q[k] * x[k] + x[k]' * qmod[k] - cmod[k]
    // NOTE: qmod[k] has already a minus sign
    // NOTE: xas used as workspace
    blasfeo_dvecmuldot(nx, clipping_data->sQ, 0, &work->sx[idx], 0, &work->sxas[idx], 0);
    *fval = -0.5*blasfeo_ddot(nx, &work->sxas[idx], 0, &work->sx[idx], 0) - cmod;
    *fval += blasfeo_ddot(nx, &work->sqmod[idx], 0, &work->sx[idx], 0);

    // feval -= (1/2)u[k]' * R[k] * u[k] - u[k]' * rmod[k]
    blasfeo_dvecmuldot(nu, clipping_data->sR, 0, &work->su[idx], 0, &work->suas[idx], 0);
    *fval -= 0.5*blasfeo_ddot(nu, &work->suas[idx], 0, &work->su[idx], 0);
    *fval += blasfeo_ddot(nu, &work->srmod[idx], 0, &work->su[idx], 0);
}



void stage_qp_clipping_export_mu(tree_ocp_qp_out *qp_out, int idx, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_clipping_data *clipping_data =
        (treeqp_tdunes_clipping_data *)work->stage_qp_data[idx];

    int nx = qp_out->x[idx].m;
    int nu = qp_out->u[idx].m;

    blasfeo_daxpy(nx, -1., &qp_out->x[idx], 0, &work->sxUnc[idx], 0, &qp_out->mu_x[idx], 0);
    blasfeo_daxpy(nu, -1., &qp_out->u[idx], 0, &work->suUnc[idx], 0, &qp_out->mu_u[idx], 0);
    blasfeo_dvecmuldot(nx, clipping_data->sQ, 0, &qp_out->mu_x[idx], 0, &qp_out->mu_x[idx], 0);
    blasfeo_dvecmuldot(nu, clipping_data->sR, 0, &qp_out->mu_u[idx], 0, &qp_out->mu_u[idx], 0);
}
