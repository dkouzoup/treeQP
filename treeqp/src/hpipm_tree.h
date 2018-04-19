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


#ifndef TREEQP_SRC_HPIPM_TREE_H_
#define TREEQP_SRC_HPIPM_TREE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"

#include "hpipm/include/hpipm_tree.h"
#include "hpipm/include/hpipm_d_tree_ocp_qp_dim.h"
#include "hpipm/include/hpipm_d_tree_ocp_qp.h"
#include "hpipm/include/hpipm_d_tree_ocp_qp_sol.h"
#include "hpipm/include/hpipm_d_tree_ocp_qp_ipm.h"

typedef struct
{
	int maxIter;
	double mu0;
	double tol;  // tolerance for res_g_max, res_b_max, res_d_max, res_m_max (can also tune individually in hpipm...)
	double alpha_min;
	int warm_start;
} treeqp_hpipm_options_t;

treeqp_hpipm_options_t treeqp_hpipm_default_options();

typedef struct treeqp_hpipm_workspace_
{
    int *nkids;
	struct tree hpipm_tree;  // NOTE(dimitris): no extra memory for this struct

	struct d_tree_ocp_qp_dim hpipm_qp_dim;
	struct d_tree_ocp_qp hpipm_qp_in;
	struct d_tree_ocp_qp_ipm_arg arg;
	struct d_tree_ocp_qp_sol hpipm_qp_out;
    struct d_tree_ocp_qp_ipm_workspace hpipm_memory;

} treeqp_hpipm_workspace;

int treeqp_hpipm_calculate_size(tree_ocp_qp_in *qp_in, treeqp_hpipm_options_t *opts);

void create_treeqp_hpipm(tree_ocp_qp_in *qp_in, treeqp_hpipm_options_t *opts,
    treeqp_hpipm_workspace *work, void *ptr);

int treeqp_hpipm_solve(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, treeqp_hpipm_options_t *opts,
    treeqp_hpipm_workspace *work);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREEQP_SRC_HPIPM_TREE_H_ */
