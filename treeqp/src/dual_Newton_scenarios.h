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

typedef struct treeqp_sdunes_workspace_ {
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

    struct blasfeo_dvec *regMat;

    struct blasfeo_dmat *sJayD;
    struct blasfeo_dmat *sJayL;
    struct blasfeo_dmat *sCholJayD;
    struct blasfeo_dmat *sCholJayL;
    struct blasfeo_dmat *sUt;
    struct blasfeo_dmat *sK;

    struct blasfeo_dmat *sTmpMats;
    struct blasfeo_dvec *sTmpVecs;

    struct blasfeo_dvec *sResNonAnticip;
    struct blasfeo_dvec *sRhsNonAnticip;
    struct blasfeo_dvec *slambda;
    struct blasfeo_dvec *sDeltalambda;

    struct blasfeo_dmat **sZbar;  // = B * RinvCal
    struct blasfeo_dmat **sLambdaD;  // diagonal block of partial derivative of r[k] w.r.t. mu[k] + F[k]
    struct blasfeo_dmat **sLambdaL;  // lower diagonal block of same partial derivative
    struct blasfeo_dmat **sCholLambdaD;
    struct blasfeo_dmat **sCholLambdaL;

    struct blasfeo_dvec *sQ;
    struct blasfeo_dvec *sR;
    struct blasfeo_dvec *sq;
    struct blasfeo_dvec *sr;
    struct blasfeo_dvec *sQinv;
    struct blasfeo_dvec *sRinv;

    struct blasfeo_dvec **sQinvCal;
    struct blasfeo_dvec **sRinvCal;

    struct blasfeo_dvec **sx;
    struct blasfeo_dvec **su;
    struct blasfeo_dvec **sxas;
    struct blasfeo_dvec **suas;
    struct blasfeo_dvec **sxUnc;
    struct blasfeo_dvec **suUnc;

    struct blasfeo_dvec **sresk;
    struct blasfeo_dvec **sreskMod;
    struct blasfeo_dvec **smu;
    struct blasfeo_dvec **sDeltamu;

    #ifdef _CHECK_LAST_ACTIVE_SET_
    struct blasfeo_dvec **sxasPrev;
    struct blasfeo_dvec **suasPrev;
    struct blasfeo_dmat **sTmpLambdaD;
    #endif
} treeqp_sdunes_workspace;


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
} treeqp_dune_options_t;  // TODO(dimitris): rename to treeqp_sdunes_options_t


int_t treeqp_dune_scenarios_calculate_size(tree_ocp_qp_in *qp_in);

void create_treeqp_dune_scenarios(tree_ocp_qp_in *qp_in, treeqp_dune_options_t *opts,
    treeqp_sdunes_workspace *work, void *ptr);

int_t treeqp_dune_scenarios_solve(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out,
    treeqp_dune_options_t *opts, treeqp_sdunes_workspace *work);

int_t calculate_dimension_of_lambda(int_t Nr, int_t md, int_t nu);

void treeqp_sdunes_set_dual_initialization(real_t *lam, real_t *mu, treeqp_sdunes_workspace *work);

void check_compiler_flags();

void write_scenarios_solution_to_txt(int_t Ns, int_t Nh, int_t Nr, int_t md, int_t nx, int_t nu,
    int_t NewtonIter, treeqp_sdunes_workspace *work);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  // TREEQP_SRC_DUAL_NEWTON_SCENARIOS_H_
