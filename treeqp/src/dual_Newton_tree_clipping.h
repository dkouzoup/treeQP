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

#include "treeqp/src/tree_qp_common.h"
#include "treeqp/utils/types.h"

#include <blasfeo_target.h>
#include <blasfeo_common.h>

typedef struct treeqp_clipping_data_
{
    struct blasfeo_dvec *sQ;
    struct blasfeo_dvec *sR;
    struct blasfeo_dvec *sQinv;
    struct blasfeo_dvec *sRinv;
    struct blasfeo_dvec *sQinvCal;
    struct blasfeo_dvec *sRinvCal;
} treeqp_tdunes_clipping_data;



answer_t stage_qp_clipping_is_applicable(const tree_qp_in *qp_in, int idx);

int stage_qp_clipping_calculate_size(int nx, int nu, int nc);

void stage_qp_clipping_assign_structs(void **stage_qp_data, char **c_double_ptr);

void stage_qp_clipping_assign_blasfeo_data(int nx, int nu, void *stage_qp_data, char **c_double_ptr);

void stage_qp_clipping_assign_data(int nx, int nu, int nc, void *stage_qp_data, char **c_double_ptr);

return_t stage_qp_clipping_init(const tree_qp_in *qp_in, int idx, stage_qp_t solver_dad, void *work_);

return_t stage_qp_clipping_solve_extended(const tree_qp_in *qp_in, int idx, void *work_);

return_t stage_qp_clipping_solve(const tree_qp_in *qp_in, int idx, void *work_);

void stage_qp_clipping_set_CmPnCmT(const tree_qp_in *qp_in, int idx, int idxdad, int offset, void *work_);

void stage_qp_clipping_add_EPmE(const tree_qp_in *qp_in, int idx, int idxdad, int offset, void *work_);

void stage_qp_clipping_add_CmPnCkT(const tree_qp_in *qp_in, int idx, int idxsib, int idxdad, int row_offset, int col_offset, void *work_);

void stage_qp_clipping_eval_dual_term(const tree_qp_in *qp_in, int idx, void *work_);

void stage_qp_clipping_export_mu(tree_qp_out *qp_out, int idx, void *work_);