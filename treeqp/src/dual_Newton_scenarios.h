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


#ifndef TREEQP_SRC_DUAL_NEWTON_SCENARIOS_H_
#define TREEQP_SRC_DUAL_NEWTON_SCENARIOS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/flags.h"
#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/tree_utils.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"

typedef struct treeqp_dune_scenarios_workspace_ {
    int_t Ns;  // number of scenarios
    int_t Nh;  // prediction horizon
    int_t Nr;  // robust horizon
    int_t md;  // number of realizations

    int_t **nodeIdx;  // tree index of scenario nodes [Ns*(Nh+1)]
    int_t **boundsRemoved;  // flag to check if bounds of subsystem are removed
    #ifdef _CHECK_LAST_ACTIVE_SET_
    int_t **xasChanged;  // binary variables to denote an AS change since last iteration
    int_t **uasChanged;
    #endif

    int_t *commonNodes;  // common between neighboring scenarios [Ns-1]
    real_t *fvals;  // dual function value for each subsystem [Ns]

    struct d_strvec *regMat;

    struct d_strmat *sJayD;
    struct d_strmat *sJayL;
    struct d_strmat *sCholJayD;
    struct d_strmat *sCholJayL;
    struct d_strmat *sUt;
    struct d_strmat *sK;

    struct d_strmat *sTmpMats;
    struct d_strvec *sTmpVecs;

    struct d_strvec *sResNonAnticip;
    struct d_strvec *sRhsNonAnticip;
    struct d_strvec *slambda;
    struct d_strvec *sDeltalambda;

    struct d_strmat **sZbar;  // = B * RinvCal
    struct d_strmat **sLambdaD;  // diagonal block of partial derivative of r[k] w.r.t. mu[k] + F[k]
    struct d_strmat **sLambdaL;  // lower diagonal block of same partial derivative
    struct d_strmat **sCholLambdaD;
    struct d_strmat **sCholLambdaL;

    struct d_strvec *sQ;
    struct d_strvec *sR;
    struct d_strvec *sq;
    struct d_strvec *sr;
    struct d_strvec *sQinv;
    struct d_strvec *sRinv;

    struct d_strvec **sQinvCal;
    struct d_strvec **sRinvCal;

    struct d_strvec **sx;
    struct d_strvec **su;
    struct d_strvec **sxas;
    struct d_strvec **suas;
    struct d_strvec **sxUnc;
    struct d_strvec **suUnc;

    struct d_strvec **sresk;
    struct d_strvec **sreskMod;
    struct d_strvec **smu;
    struct d_strvec **sDeltamu;

    #ifdef _CHECK_LAST_ACTIVE_SET_
    struct d_strvec **sxasPrev;
    struct d_strvec **suasPrev;
    struct d_strmat **sTmpLambdaD;
    #endif
} treeqp_dune_scenarios_workspace;


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
    regType_t regType;
    real_t regTol;  // tolerance for on-the-fly regularization // TODO(dimitris): implement for tree
    real_t regValue;

    // line search options
    real_t lineSearchGamma;
    real_t lineSearchBeta;
} treeqp_dune_options_t;


int_t treeqp_dune_scenarios_calculate_size(tree_ocp_qp_in *qp_in);

void create_treeqp_dune_scenarios(tree_ocp_qp_in *qp_in, treeqp_dune_options_t *opts,
    treeqp_dune_scenarios_workspace *work, void *ptr);

int_t treeqp_dune_scenarios_solve(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out,
    treeqp_dune_options_t *opts, treeqp_dune_scenarios_workspace *work);

int_t calculate_dimension_of_lambda(int_t Nr, int_t md, int_t nu);

void check_compiler_flags();

void write_solution_to_txt(int_t Ns, int_t Nh, int_t Nr, int_t md, int_t nx, int_t nu,
    int_t NewtonIter, treeqp_dune_scenarios_workspace *work);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  // TREEQP_SRC_DUAL_NEWTON_SCENARIOS_H_
