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
