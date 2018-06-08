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

#include <blasfeo_target.h>
#include <blasfeo_common.h>

#include "hpipm/include/hpipm_tree.h"
#include "hpipm/include/hpipm_d_tree_ocp_qp_dim.h"
#include "hpipm/include/hpipm_d_tree_ocp_qp.h"
#include "hpipm/include/hpipm_d_tree_ocp_qp_sol.h"
#include "hpipm/include/hpipm_d_tree_ocp_qp_ipm.h"



// memory to store temporary calculations when calculating solver size
typedef struct scrap_memory_hpipm_t_
{
    int *nkids;
    int *nb;
    int *nbx;
    int *nbu;
    int *ns;
} scrap_memory_hpipm_t;



typedef struct treeqp_hpipm_opts_t_
{
	int maxIter;			 	 // maximum number of IP iterations (status = 1 if reached)
	double mu0;			 		 // max element value in cost function
	double tol; 	  			 // tolerance for res_g_max, res_b_max, res_d_max, res_m_max (can also tune individually in hpipm...)
	double alpha_min;			 // minimum step size (status = 2 if reached)
	int warm_start;				 // read initial guess from x and u
	scrap_memory_hpipm_t scrap;  // scrap memory needed in treeqp_hpmpc_calculate_size
} treeqp_hpipm_opts_t;



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



int treeqp_hpipm_opts_calculate_size(int Nn);

void treeqp_hpipm_opts_create(int Nn, treeqp_hpipm_opts_t *opts, void *ptr);

void treeqp_hpipm_opts_set_default(treeqp_hpipm_opts_t *opts);



int treeqp_hpipm_calculate_size(tree_ocp_qp_in *qp_in, treeqp_hpipm_opts_t *opts);

void treeqp_hpipm_create(tree_ocp_qp_in *qp_in, treeqp_hpipm_opts_t *opts, treeqp_hpipm_workspace *work, void *ptr);

int treeqp_hpipm_solve(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, treeqp_hpipm_opts_t *opts, treeqp_hpipm_workspace *work);



#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREEQP_SRC_HPIPM_TREE_H_ */
