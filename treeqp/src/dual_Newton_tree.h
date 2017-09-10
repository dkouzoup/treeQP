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

// #include "blasfeo/include/blasfeo_target.h"
// #include "blasfeo/include/blasfeo_common.h"

typedef struct treeqp_tdunes_workspace_ {
    int_t Nn;
    int_t Np;

    int_t *npar;  // 1 x Nh: number of parallel factorizations per stage
    #ifdef _CHECK_LAST_ACTIVE_SET_
    int_t *blockChanged;  // 1 x Np
    #endif

    struct d_strvec *sQinv;  // 1 x Nn
    struct d_strvec *sRinv;  // 1 x Nn

    struct d_strvec *regMat;  // 1 x 1
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


// int_t treeqp_tdunes_calculate_size(tree_ocp_qp_in *qp_in);

// void create_treeqp_tdunes(tree_ocp_qp_in *qp_in, treeqp_tdunes_options_t *opts,
//     treeqp_tdunes_workspace *work, void *ptr);

// int_t treeqp_tdunes_solve(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out,
//     treeqp_tdunes_options_t *opts, treeqp_tdunes_workspace *work);

// TODO
// void check_compiler_flags();

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  // TREEQP_SRC_DUAL_NEWTON_TREE_H_
