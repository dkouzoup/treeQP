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


#ifndef TREEQP_SRC_HPMPC_TREE_H_
#define TREEQP_SRC_HPMPC_TREE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"

typedef struct
{
	int maxIter;  // k_max
	int mu0;  // max element value in cost function
	double mu_tol;
	double alpha_min;
	int warm_start;  // read initial guess from x and u
	int compute_mult;

} treeqp_hpmpc_options_t;

treeqp_hpmpc_options_t treeqp_hpmpc_default_options();

typedef struct treeqp_hpmpc_workspace_
{
    int *nb;
    int **idxb;

    struct blasfeo_dvec *sux;
    struct blasfeo_dvec *slam;
    struct blasfeo_dvec *sst;

    struct blasfeo_dmat *sRSQrq;
    struct blasfeo_dmat *sBAbt;
    struct blasfeo_dmat *sDCt;
    struct blasfeo_dvec *sd;

    double *status;
    void *internal;

} treeqp_hpmpc_workspace;

int treeqp_hpmpc_calculate_size(tree_ocp_qp_in *qp_in, treeqp_hpmpc_options_t *opts);

void create_treeqp_hpmpc(tree_ocp_qp_in *qp_in, treeqp_hpmpc_options_t *opts,
    treeqp_hpmpc_workspace *work, void *ptr);

int treeqp_hpmpc_solve(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, treeqp_hpmpc_options_t *opts,
    treeqp_hpmpc_workspace *work);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREEQP_SRC_HPMPC_TREE_H_ */
