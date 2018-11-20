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

#include "treeqp/src/dual_Newton_common.h"
#include "treeqp/src/tree_qp_common.h"
#include "treeqp/utils/types.h"

#if PROFILE > 0
#include "treeqp/utils/profiling.h"
#endif

#include <blasfeo_target.h>
#include <blasfeo_common.h>



typedef struct stage_qp_fcn_ptrs_
{
    answer_t (*is_applicable)(const tree_qp_in *qp_in, int idx);
    int (*calculate_size)(int nx, int nu, int nc);
    void (*assign_structs)(void **data, char **c_double_ptr);
    void (*assign_blasfeo_data)(int nx, int nu, void *data, char **c_double_ptr);
    void (*assign_data)(int nx, int nu, int nc, void *data, char **c_double_ptr);
    return_t (*init)(const tree_qp_in *qp_in, int idx, stage_qp_t solver_dad, void *work);
    return_t (*solve_extended)(const tree_qp_in *qp_in, int idx, void *work);
    return_t (*solve)(const tree_qp_in *qp_in, int idx, void *work);
    void (*set_CmPnCmT)(const tree_qp_in *qp_in, int idx, int idxdad, int offset, void *work_);
    void (*add_EPmE)(const tree_qp_in *qp_in, int idx, int idxdad, int offset, void *work_);
    void (*add_CmPnCkT)(const tree_qp_in *qp_in, int idx, int idxsib, int idxdad, int row_offset, int col_offset, void *work_);
    void (*eval_dual_term)(const tree_qp_in *qp_in, int idx, void *work_);
    void (*export_mu)(tree_qp_out *qp_out, int idx, void *work_);
} stage_qp_fcn_ptrs;



typedef struct treeqp_tdunes_opts_t_
{
    int maxIter;                    // maximum number of dual Newton iterations

    stage_qp_t *qp_solver;          // stage QP solver stage-wise (TREEQP_CLIPPING_SOLVER or TREEQP_QPOASES_SOLVER)

    int checkLastActiveSet;         // save computations per iteration by monitoring active set changes

    double stationarityTolerance;   // tolerance for termination condition on dual gradient
    termination_t termCondition;    // norm of termination condition (TREEQP_SUMSQUAREDERRORS, TREEQP_TWONORM, TREEQP_INFNORM)

    regType_t regType;              // regularization strategy (TREEQP_NO_REGULARIZATION, TREEQP_ALWAYS_LEVENBERG_MARQUARDT, TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT)
    double regTol;                  // tolerance for adding regularization (in TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT only)
    double regValue;                // value of regularization on diagonal (in TREEQP_ALWAYS_LEVENBERG_MARQUARDT and TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT)

    int lineSearchMaxIter;          // maximum number of line search iterations per Newton iteration
    double lineSearchGamma;         // gamma parameter in line search
    double lineSearchBeta;          // beta parameter in line search
    int lineSearchRestartTrigger;  // if > 0, performs a full step when algorithm reached lineSearchMaxIter that many times in a row

} treeqp_tdunes_opts_t;



typedef struct treeqp_tdunes_workspace_
{
    int Nn;
    int Np;

    int lsIter;
    int lineSearchRestartCounter;

    int *npar;  // 1 x Nh: number of parallel factorizations per stage
    int *idxpos;  // 1 x Nn: position of node inside vector lambda (0 for first child in branch)

    int *xasChanged;  // 1 x Nn
    int *uasChanged;  // 1 x Nn
    int *blockChanged;  // 1 x Np

    double *fval;  // 1 x Nn
    double *cmod;  // 1 x Nn

    stage_qp_fcn_ptrs *stage_qp_ptrs;  // 1 x Nn (function pointers for stage qp operations)
    void **stage_qp_data;  // 1 x Nn (double pointers to structs, struct depends on solver)

    struct blasfeo_dvec *sqmod;  // 1 x Nn
    struct blasfeo_dvec *srmod;  // 1 x Nn

    // NOTE(dimitris):
    // - There are as many Hessian blocks on the diagonal as parent nodes in the tree
    // - Each of those blocks except for the root has a lower diagonal block from its parent
    // - The upper diagonal blocks from the children are neglected due to symmetry
    // - Dimension of block k is (nc[k]*nx) x (nc[k]*nx), where nc[k] = tree[k].nkids
    // - Dimension of parent block is (nc[k]*nx) x nx

    struct blasfeo_dmat *sAB;  // 1 x Nn-1: [A B]
    struct blasfeo_dmat *sM;  // 1 x Nn: matrices MAX(nx[k], nx[sib.]) x MAX(nx[dad], nu[dad]) for intermediate results
    struct blasfeo_dmat *sW;  // 1 x Np
    struct blasfeo_dmat *sCholW;  // 1 x Np
    struct blasfeo_dmat *sUt;  // 1 x (Np-1)
    struct blasfeo_dmat *sCholUt;  // 1 x (Np-1)
    struct blasfeo_dvec *sres;  // 1 x Np
    struct blasfeo_dvec *sresMod;  // 1 x Np
    struct blasfeo_dvec *slambda;  // 1 x Np
    struct blasfeo_dvec *sDeltalambda;  // 1 x Np

    // TODO(dimitris): move xUnc, uUnc to clipping data (not used in qpoases)

    struct blasfeo_dvec *sx;  // 1 x Nn
    struct blasfeo_dvec *su;  // 1 x Nn
    struct blasfeo_dvec *sxas;  // 1 x Nn
    struct blasfeo_dvec *suas;  // 1 x Nn
    struct blasfeo_dvec *sxUnc;  // 1 x Nn
    struct blasfeo_dvec *suUnc;  // 1 x Nn

    struct blasfeo_dvec *sxasPrev;  // 1 x Nn
    struct blasfeo_dvec *suasPrev;  // 1 x Nn
    struct blasfeo_dmat *sWdiag;  // 1 x Nn

    #if PROFILE > 0
    treeqp_profiling_t timings;
    #endif

} treeqp_tdunes_workspace;



int treeqp_tdunes_opts_calculate_size(int Nn);

void treeqp_tdunes_opts_create(int Nn, treeqp_tdunes_opts_t *opts, void *ptr);

void treeqp_tdunes_opts_set_default(int Nn, treeqp_tdunes_opts_t *opts);



int treeqp_tdunes_calculate_size(const tree_qp_in *qp_in, const treeqp_tdunes_opts_t *opts);

void treeqp_tdunes_create(const tree_qp_in *qp_in, const treeqp_tdunes_opts_t *opts, treeqp_tdunes_workspace *work, void *ptr);

void treeqp_tdunes_set_dual_initialization(const double *lambda, treeqp_tdunes_workspace *work);

return_t treeqp_tdunes_solve(const tree_qp_in *qp_in, tree_qp_out *qp_out, const treeqp_tdunes_opts_t *opts, treeqp_tdunes_workspace *work);


// TODO(dimitris): move to utils!
void write_solution_to_txt(const tree_qp_in *qp_in, int Np, int iter, struct node *tree, treeqp_tdunes_workspace *work);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREEQP_SRC_DUAL_NEWTON_TREE_H_ */
