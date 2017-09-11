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
// #include "treeqp/utils/tree_utils.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"

typedef struct treeqp_tdunes_workspace_ {
    int_t Nn;
    int_t Np;

    int_t *npar;  // 1 x Nh: number of parallel factorizations per stage
    #ifdef _CHECK_LAST_ACTIVE_SET_
    int_t *xasChanged;  // 1 x Nn
    int_t *uasChanged;  // 1 x Nn
    int_t *blockChanged;  // 1 x Np
    #endif

    real_t *fval;  // 1 x Nn
    real_t *cmod;  // 1 x Nn

    struct d_strvec *sQinv;  // 1 x Nn
    struct d_strvec *sRinv;  // 1 x Nn
    struct d_strvec *sQinvCal;  // 1 x Nn
    struct d_strvec *sRinvCal;  // 1 x Nn
    struct d_strvec *sqmod;  // 1 x Nn
    struct d_strvec *srmod;  // 1 x Nn

    // NOTE(dimitris):
    // - There are as many Hessian blocks on the diagonal as parent nodes in the tree
    // - Each of those blocks except for the root has a lower diagonal block from its parent
    // - The upper diagonal blocks from the children are neglected due to symmetry
    // - Dimension of block k is (nc[k]*nx) x (nc[k]*nx), where nc[k] = tree[k].nkids
    // - Dimension of parent block is (nc[k]*nx) x nx

    struct d_strvec *regMat;  // 1 x 1
    struct d_strmat *sM;  // 1 x Nn: matrix MAX(nx, nu) x MAX(nx, nu) to store intermediate results
    struct d_strmat *sW;  // 1 x Np
    struct d_strmat *sCholW;  // 1 x Np
    struct d_strmat *sUt;  // 1 x (Np-1)
    struct d_strmat *sCholUt;  // 1 x (Np-1)
    struct d_strvec *sres;  // 1 x Np
    struct d_strvec *sresMod;  // 1 x Np
    struct d_strvec *slambda;  // 1 x Np
    struct d_strvec *sDeltalambda;  // 1 x Np

    struct d_strvec *sx;  // 1 x Nn
    struct d_strvec *su;  // 1 x Np
    struct d_strvec *sxas;  // 1 x Nn
    struct d_strvec *suas;  // 1 x Np
    #ifdef _CHECK_LAST_ACTIVE_SET_
    struct d_strvec *sxasPrev;  // 1 x Nn
    struct d_strvec *suasPrev;  // 1 x Np
    struct d_strmat *sWdiag;  // 1 x Nn
    #endif
} treeqp_tdunes_workspace;

// Options of QP solver
typedef struct {
    // iterations
    int_t maxIter;
    int_t lineSearchMaxIter;

    // numerical tolerances
    real_t stationarityTolerance;

    // termination condition options
    termination_t termCondition;

    // regularization options
    // TODO(dimitris): implement on-the-gly regularization option!
    regType_t regType;
    // real_t regTol;
    real_t regValue;

    // line search options
    real_t lineSearchGamma;
    real_t lineSearchBeta;
} treeqp_tdunes_options_t;


int_t treeqp_tdunes_calculate_size(tree_ocp_qp_in *qp_in);

void create_treeqp_tdunes(tree_ocp_qp_in *qp_in, treeqp_tdunes_options_t *opts,
    treeqp_tdunes_workspace *work, void *ptr);

void treeqp_tdunes_set_dual_initialization(real_t *lambda, treeqp_tdunes_workspace *work);

int_t treeqp_tdunes_solve(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out,
    treeqp_tdunes_options_t *opts, treeqp_tdunes_workspace *work);

void write_solution_to_txt(tree_ocp_qp_in *qp_in, int_t Np, int_t iter, struct node *tree,
    treeqp_tdunes_workspace *work);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  // TREEQP_SRC_DUAL_NEWTON_TREE_H_
