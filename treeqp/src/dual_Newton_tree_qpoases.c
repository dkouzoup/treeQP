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

// #include "blasfeo/include/blasfeo_target.h"
// #include "blasfeo/include/blasfeo_common.h"
// #include "blasfeo/include/blasfeo_d_aux.h"

#include <qpOASES_e.h>

answer_t stage_qp_qpoases_is_applicable(tree_ocp_qp_in *qp_in, int node_index)
{
    return YES;
}



int stage_qp_qpoases_calculate_size(int nx, int nu)
{
    int bytes  = 0;

    bytes += sizeof(treeqp_tdunes_qpoases_data);
    // TODO(dimitris): add calculate_size routines of qpOASES
    return bytes;
}



void stage_qp_qpoases_assign_structs(void **stage_qp_data, char **c_double_ptr)
{
    treeqp_tdunes_qpoases_data *qpoases_solver_data;

    qpoases_solver_data = (treeqp_tdunes_qpoases_data *)*c_double_ptr;
    *c_double_ptr += sizeof(treeqp_tdunes_qpoases_data);

    // TODO(dimitris): anything else?

    *stage_qp_data = (void *) qpoases_solver_data;
}



// NOTE(dimitris): structs and data are assigned separately due to alignment requirements
void stage_qp_qpoases_assign_data(int nx, int nu, void *stage_qp_data, char **c_double_ptr)
{
    treeqp_tdunes_qpoases_data *qpoases_solver_data;
    qpoases_solver_data = (treeqp_tdunes_qpoases_data *)stage_qp_data;
    // TODO(dimitris): add assign routines of qpOASES
}



void stage_qp_qpoases_init(tree_ocp_qp_in *qp_in, int node_index, void *work_)
{
    treeqp_tdunes_workspace *work = (treeqp_tdunes_workspace *) work_;
    treeqp_tdunes_qpoases_data *qpoases_solver_data =
        (treeqp_tdunes_qpoases_data *)work->stage_qp_data[node_index];

    int nx = qp_in->nx[node_index];
    int nu = qp_in->nu[node_index];
}
