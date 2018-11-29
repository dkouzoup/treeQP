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

#include "treeqp/src/tree_qp_common.h"
#include "treeqp/utils/types.h"

#include <blasfeo_target.h>
#include <blasfeo_common.h>


// memory to store temporary calculations when calculating solver size
typedef struct scrap_memory_hpmpc_t_
{
    int *nb;
} scrap_memory_hpmpc_t;



typedef struct treeqp_hpmpc_opts_t_
{
	int maxIter;                 // maximum number of IP iterations (status = 1 if reached)
	double mu0;                  // max element value in cost function
	double mu_tol;               // tolerance for termination condition (status = 0 if reached)
	double alpha_min;            // minimum step size (status = 2 if reached)
	int warm_start;              // read initial guess from x and u
	int compute_mult;            // compute dual variables
    scrap_memory_hpmpc_t scrap;  // scrap memory needed in treeqp_hpmpc_calculate_size
} treeqp_hpmpc_opts_t;



typedef struct treeqp_hpmpc_workspace_
{
    int *nb;
    int **idxb;

    struct blasfeo_dvec *sux;
    struct blasfeo_dvec *slam;
    struct blasfeo_dvec *spi;
    struct blasfeo_dvec *sst;

    struct blasfeo_dmat *sRSQrq;
    struct blasfeo_dmat *sBAbt;
    struct blasfeo_dmat *sDCt;
    struct blasfeo_dvec *sd;

    double *status;
    void *internal;

} treeqp_hpmpc_workspace;



int treeqp_hpmpc_opts_calculate_size(int Nn);

void treeqp_hpmpc_opts_create(int Nn, treeqp_hpmpc_opts_t *opts, void *ptr);

void treeqp_hpmpc_opts_set_default(int Nn, treeqp_hpmpc_opts_t *opts);



int number_of_bounds(const struct blasfeo_dvec *vmin, const struct blasfeo_dvec *vmax);

void setup_nb(const tree_qp_in *qp_in, int *nb);

int get_size_idxb(const tree_qp_in *qp_in);

void setup_nb_idxb(const tree_qp_in *qp_in, int *nb, int **idxb);



int treeqp_hpmpc_calculate_size(const tree_qp_in *qp_in, const treeqp_hpmpc_opts_t *opts);

void treeqp_hpmpc_create(const tree_qp_in *qp_in, const treeqp_hpmpc_opts_t *opts, treeqp_hpmpc_workspace *work, void *ptr);

return_t treeqp_hpmpc_solve(const tree_qp_in *qp_in, tree_qp_out *qp_out, const treeqp_hpmpc_opts_t *opts, treeqp_hpmpc_workspace *work);



#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREEQP_SRC_HPMPC_TREE_H_ */
