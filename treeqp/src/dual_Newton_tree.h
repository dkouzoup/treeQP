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


#ifndef TREEQP_SRC_DUAL_NEWTON_TREE_H_
#define TREEQP_SRC_DUAL_NEWTON_TREE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/flags.h"
#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/utils/types.h"
// #include "treeqp/utils/tree.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"


typedef struct treeqp_tdunes_workspace_
{
    int Nn;
    int Np;

    int *npar;  // 1 x Nh: number of parallel factorizations per stage
    int *idxpos;  // 1 x Nn: position of node inside vector lambda (0 for first child in branch)

    stage_qp_t *qp_solver;  // 1 x Nn: type of stage QP solver

    #ifdef _CHECK_LAST_ACTIVE_SET_
    int *xasChanged;  // 1 x Nn
    int *uasChanged;  // 1 x Nn
    int *blockChanged;  // 1 x Np
    #endif

    double *fval;  // 1 x Nn
    double *cmod;  // 1 x Nn

    struct blasfeo_dvec *sQ;  // 1 x Nn
    struct blasfeo_dvec *sR;  // 1 x Nn
    struct blasfeo_dvec *sQinv;  // 1 x Nn
    struct blasfeo_dvec *sRinv;  // 1 x Nn
    struct blasfeo_dvec *sQinvCal;  // 1 x Nn
    struct blasfeo_dvec *sRinvCal;  // 1 x Nn
    struct blasfeo_dvec *sqmod;  // 1 x Nn
    struct blasfeo_dvec *srmod;  // 1 x Nn

    // NOTE(dimitris):
    // - There are as many Hessian blocks on the diagonal as parent nodes in the tree
    // - Each of those blocks except for the root has a lower diagonal block from its parent
    // - The upper diagonal blocks from the children are neglected due to symmetry
    // - Dimension of block k is (nc[k]*nx) x (nc[k]*nx), where nc[k] = tree[k].nkids
    // - Dimension of parent block is (nc[k]*nx) x nx

    struct blasfeo_dvec *regMat;  // 1 x 1
    struct blasfeo_dmat *sM;  // 1 x Nn: matrix MAX(nx, nu) x MAX(nx, nu) to store intermediate results
    struct blasfeo_dmat *sW;  // 1 x Np
    struct blasfeo_dmat *sCholW;  // 1 x Np
    struct blasfeo_dmat *sUt;  // 1 x (Np-1)
    struct blasfeo_dmat *sCholUt;  // 1 x (Np-1)
    struct blasfeo_dvec *sres;  // 1 x Np
    struct blasfeo_dvec *sresMod;  // 1 x Np
    struct blasfeo_dvec *slambda;  // 1 x Np
    struct blasfeo_dvec *sDeltalambda;  // 1 x Np

    struct blasfeo_dvec *sx;  // 1 x Nn
    struct blasfeo_dvec *su;  // 1 x Nn
    struct blasfeo_dvec *sxas;  // 1 x Nn
    struct blasfeo_dvec *suas;  // 1 x Nn
    struct blasfeo_dvec *sxUnc;  // 1 x Nn
    struct blasfeo_dvec *suUnc;  // 1 x Nn
    #ifdef _CHECK_LAST_ACTIVE_SET_
    struct blasfeo_dvec *sxasPrev;  // 1 x Nn
    struct blasfeo_dvec *suasPrev;  // 1 x Nn
    struct blasfeo_dmat *sWdiag;  // 1 x Nn
    #endif
} treeqp_tdunes_workspace;



typedef struct
{
    // iterations
    int maxIter;
    int lineSearchMaxIter;

    // solution of stage QPs (1 x Nn)
    stage_qp_t *qp_solver;

    // numerical tolerances
    double stationarityTolerance;

    // termination condition options
    termination_t termCondition;

    // regularization options
    // TODO(dimitris): implement on-the-gly regularization option!
    regType_t regType;
    // double regTol;
    double regValue;

    // line search options
    double lineSearchGamma;
    double lineSearchBeta;
} treeqp_tdunes_options_t;



treeqp_tdunes_options_t treeqp_tdunes_default_options(int Nn);

int treeqp_tdunes_calculate_size(tree_ocp_qp_in *qp_in);

void create_treeqp_tdunes(tree_ocp_qp_in *qp_in, treeqp_tdunes_options_t *opts,
    treeqp_tdunes_workspace *work, void *ptr);

void treeqp_tdunes_set_dual_initialization(double *lambda, treeqp_tdunes_workspace *work);

int treeqp_tdunes_solve(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out,
    treeqp_tdunes_options_t *opts, treeqp_tdunes_workspace *work);

void write_solution_to_txt(tree_ocp_qp_in *qp_in, int Np, int iter, struct node *tree,
    treeqp_tdunes_workspace *work);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREEQP_SRC_DUAL_NEWTON_TREE_H_ */
