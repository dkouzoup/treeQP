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
#include "treeqp/src/dual_Newton_tree_clipping.h"
#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/utils/blasfeo.h"
#include "treeqp/utils/memory.h"
#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_blas.h"



answer_t stage_qp_clipping_is_applicable(tree_ocp_qp_in *qp_in, int node_index)
{
    answer_t ans = YES;

    if (is_strmat_diagonal(&qp_in->Q[node_index]) == NO)
    {
        ans = NO;
    }

    if (is_strmat_diagonal(&qp_in->R[node_index]) == NO)
    {
        ans = NO;
    }

    if (is_strmat_zero(&qp_in->S[node_index]) == NO)
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



int stage_qp_clipping_calculate_size(int nx, int nu)
{
    int bytes  = 0;

    bytes += sizeof(treeqp_tdunes_clipping_data);
    bytes += 6*sizeof(struct blasfeo_dvec);  // Q, R, Qinv, Rinv, QinvCal, RinvCal
    bytes += 3*blasfeo_memsize_dvec(nx);  // Q, Qinv, QinvCal
    bytes += 3*blasfeo_memsize_dvec(nu);  // R, Rinv, RinvCal

    return bytes;
}



void stage_qp_clipping_assign_structs(void **stage_qp_data, char **c_double_ptr)
{
    treeqp_tdunes_clipping_data *clipping_solver_data;

    clipping_solver_data = (treeqp_tdunes_clipping_data *)*c_double_ptr;
    *c_double_ptr += sizeof(treeqp_tdunes_clipping_data);

    clipping_solver_data->sQ = (struct blasfeo_dvec *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dvec);

    clipping_solver_data->sR = (struct blasfeo_dvec *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dvec);

    clipping_solver_data->sQinv = (struct blasfeo_dvec *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dvec);

    clipping_solver_data->sRinv = (struct blasfeo_dvec *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dvec);

    clipping_solver_data->sQinvCal = (struct blasfeo_dvec *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dvec);

    clipping_solver_data->sRinvCal = (struct blasfeo_dvec *)*c_double_ptr;
    *c_double_ptr += sizeof(struct blasfeo_dvec);

    *stage_qp_data = (void *) clipping_solver_data;
}



// NOTE(dimitris): structs and data are assigned separately due to alignment requirements
void stage_qp_clipping_assign_data(int nx, int nu, void *stage_qp_data, char **c_double_ptr)
{
    treeqp_tdunes_clipping_data *clipping_solver_data;
    clipping_solver_data = (treeqp_tdunes_clipping_data *)stage_qp_data;

    init_strvec(nx, clipping_solver_data->sQ, c_double_ptr);
    init_strvec(nu, clipping_solver_data->sR, c_double_ptr);
    init_strvec(nx, clipping_solver_data->sQinv, c_double_ptr);
    init_strvec(nu, clipping_solver_data->sRinv, c_double_ptr);
    init_strvec(nx, clipping_solver_data->sQinvCal, c_double_ptr);
    init_strvec(nu, clipping_solver_data->sRinvCal, c_double_ptr);
}



void stage_qp_clipping_init(tree_ocp_qp_in *qp_in, int node_index, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_clipping_data *clipping_solver_data =
        (treeqp_tdunes_clipping_data *)work->stage_qp_data[node_index];

    int nx = qp_in->nx[node_index];
    int nu = qp_in->nu[node_index];

    blasfeo_ddiaex(nx, 1.0, &qp_in->Q[node_index], 0, 0, clipping_solver_data->sQ, 0);
    blasfeo_ddiaex(nu, 1.0, &qp_in->R[node_index], 0, 0, clipping_solver_data->sR, 0);

    for (int nn = 0; nn < nx; nn++)
    {
        DVECEL_LIBSTR(clipping_solver_data->sQinv, nn) =
            1.0/DVECEL_LIBSTR(clipping_solver_data->sQ, nn);
    }
    for (int nn = 0; nn < nu; nn++)
    {
        DVECEL_LIBSTR(clipping_solver_data->sRinv, nn) =
            1.0/DVECEL_LIBSTR(clipping_solver_data->sR, nn);
    }
}



void stage_qp_clipping_solve_extended(tree_ocp_qp_in *qp_in, int node_index, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_clipping_data *clipping_solver_data =
        (treeqp_tdunes_clipping_data *)work->stage_qp_data[node_index];

    int nx = qp_in->nx[node_index];
    int nu = qp_in->nu[node_index];

    // x[k] = Q[k]^-1 .* qmod[k]
    // NOTE(dimitris): minus sign already in mod. gradient
    blasfeo_dvecmuldot(nx, clipping_solver_data->sQinv, 0,
        &work->sqmod[node_index], 0, &work->sxUnc[node_index], 0);

    // x[k] = median(xmin, x[k], xmax), xas[k] = active set
    blasfeo_dveccl_mask(nx, &qp_in->xmin[node_index], 0, &work->sxUnc[node_index], 0,
        &qp_in->xmax[node_index], 0, &work->sx[node_index], 0, &work->sxas[node_index], 0);

    // u[k] = R[k]^-1 .* rmod[k]
    blasfeo_dvecmuldot(nu, clipping_solver_data->sRinv, 0, &work->srmod[node_index], 0,
        &work->suUnc[node_index], 0);

    // u[k] = median(umin, u[k], umax), uas[k] = active set
    blasfeo_dveccl_mask(nu, &qp_in->umin[node_index], 0, &work->suUnc[node_index], 0,
        &qp_in->umax[node_index], 0, &work->su[node_index], 0, &work->suas[node_index], 0);

    // QinvCal[kk] = Qinv[kk] .* (1 - abs(xas[kk])), aka elimination matrix
    blasfeo_dvecze(nx, &work->sxas[node_index], 0, clipping_solver_data->sQinv, 0,
        clipping_solver_data->sQinvCal, 0);

    // RinvCal[kk] = Rinv[kk] .* (1 - abs(uas[kk]))
    blasfeo_dvecze(nu, &work->suas[node_index], 0, clipping_solver_data->sRinv, 0,
        clipping_solver_data->sRinvCal, 0);
}



void stage_qp_clipping_solve(tree_ocp_qp_in *qp_in, int node_index, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_clipping_data *clipping_solver_data =
        (treeqp_tdunes_clipping_data *)work->stage_qp_data[node_index];

    int nx = qp_in->nx[node_index];
    int nu = qp_in->nu[node_index];

    // x[k] = Q[k]^-1 .* qmod[k]
    // NOTE(dimitris): minus sign already in mod. gradient
    blasfeo_dvecmuldot(nx, clipping_solver_data->sQinv, 0, &work->sqmod[node_index], 0, &work->sx[node_index], 0);

    // x[k] = median(xmin, x[k], xmax)
    blasfeo_dveccl(nx, &qp_in->xmin[node_index], 0, &work->sx[node_index], 0, &qp_in->xmax[node_index], 0, &work->sx[node_index], 0);

    // u[k] = R[k]^-1 .* rmod[k]
    blasfeo_dvecmuldot(nu, clipping_solver_data->sRinv, 0, &work->srmod[node_index], 0, &work->su[node_index], 0);
    // u[k] = median(umin, u[k], umax)
    blasfeo_dveccl(nu, &qp_in->umin[node_index], 0, &work->su[node_index], 0, &qp_in->umax[node_index], 0, &work->su[node_index], 0);
}



void stage_qp_clipping_eval_dual_term(tree_ocp_qp_in *qp_in, int node_index, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_clipping_data *clipping_solver_data =
        (treeqp_tdunes_clipping_data *)work->stage_qp_data[node_index];

    int nx = qp_in->nx[node_index];
    int nu = qp_in->nu[node_index];

    double *fval = &work->fval[node_index];
    double cmod = work->cmod[node_index];

    // feval = - (1/2)x[k]' * Q[k] * x[k] + x[k]' * qmod[k] - cmod[k]
    // NOTE: qmod[k] has already a minus sign
    // NOTE: xas used as workspace
    blasfeo_dvecmuldot(nx, clipping_solver_data->sQ, 0, &work->sx[node_index], 0, &work->sxas[node_index], 0);
    *fval = -0.5*blasfeo_ddot(nx, &work->sxas[node_index], 0, &work->sx[node_index], 0) - cmod;
    *fval += blasfeo_ddot(nx, &work->sqmod[node_index], 0, &work->sx[node_index], 0);

    // feval -= (1/2)u[k]' * R[k] * u[k] - u[k]' * rmod[k]
    blasfeo_dvecmuldot(nu, clipping_solver_data->sR, 0, &work->su[node_index], 0, &work->suas[node_index], 0);
    *fval -= 0.5*blasfeo_ddot(nu, &work->suas[node_index], 0, &work->su[node_index], 0);
    *fval += blasfeo_ddot(nu, &work->srmod[node_index], 0, &work->su[node_index], 0);
}



// TODO(dimitris): think how to improve inputs of all these functions
void stage_qp_clipping_export_mu(tree_ocp_qp_out *qp_out, int node_index, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_clipping_data *clipping_solver_data =
        (treeqp_tdunes_clipping_data *)work->stage_qp_data[node_index];

    int nx = qp_out->x[node_index].m;
    int nu = qp_out->u[node_index].m;

    blasfeo_daxpy(nx, -1., &qp_out->x[node_index], 0, &work->sxUnc[node_index], 0, &qp_out->mu_x[node_index], 0);
    blasfeo_daxpy(nu, -1., &qp_out->u[node_index], 0, &work->suUnc[node_index], 0, &qp_out->mu_u[node_index], 0);
    blasfeo_dvecmuldot(nx, clipping_solver_data->sQ, 0, &qp_out->mu_x[node_index], 0, &qp_out->mu_x[node_index], 0);
    blasfeo_dvecmuldot(nu, clipping_solver_data->sR, 0, &qp_out->mu_u[node_index], 0, &qp_out->mu_u[node_index], 0);
}
