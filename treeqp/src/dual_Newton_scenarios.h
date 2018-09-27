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

#include "treeqp/src/tree_qp_common.h"
#include "treeqp/src/dual_Newton_common.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/tree.h"

#if PROFILE > 0
#include "treeqp/utils/profiling.h"
#endif

#include <blasfeo_target.h>
#include <blasfeo_common.h>



typedef struct treeqp_sdunes_opts_t_
{
    int maxIter;                    // maximum number of dual Newton iterations

    int checkLastActiveSet;         // save computations per iteration by monitoring active set changes

    double stationarityTolerance;   // tolerance for termination condition on dual gradient
    termination_t termCondition;    // norm of termination condition (TREEQP_SUMSQUAREDERRORS, TREEQP_TWONORM, TREEQP_INFNORM)

    regType_t regType;              // regularization strategy (TREEQP_NO_REGULARIZATION, TREEQP_ALWAYS_LEVENBERG_MARQUARDT, TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT)
    double regTol;                  // tolerance for adding regularization (in TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT only)
    double regValue;                // value of regularization on diagonal (in TREEQP_ALWAYS_LEVENBERG_MARQUARDT and TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT)

    int lineSearchMaxIter;          // maximum number of line search iterations per Newton iteration
    double lineSearchGamma;         // gamma parameter in line search
    double lineSearchBeta;          // beta parameter in line search
} treeqp_sdunes_opts_t;



typedef struct treeqp_sdunes_workspace_
{
    int Ns;  // number of scenarios
    int Nh;  // prediction horizon
    int Nr;  // robust horizon
    int md;  // number of realizations

    int reverseCholesky;  // depends on checkLastActiveSet option (only on/on, off/off implemented)

    int **nodeIdx;  // tree index of scenario nodes [Ns*(Nh+1)]
    int **boundsRemoved;  // flag to check if bounds of subsystem are removed

    int **xasChanged;  // binary variables to denote an AS change since last iteration
    int **uasChanged;

    int *commonNodes;  // common between neighboring scenarios [Ns-1]
    double *fvals;  // dual function value for each subsystem [Ns]

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

    struct blasfeo_dvec **sxasPrev;
    struct blasfeo_dvec **suasPrev;
    struct blasfeo_dmat **sTmpLambdaD;

    #if PROFILE > 0
    treeqp_profiling_t timings;
    #endif

} treeqp_sdunes_workspace;



int treeqp_sdunes_opts_calculate_size(int Nn);

void treeqp_sdunes_opts_create(int Nn, treeqp_sdunes_opts_t *opts, void *ptr);

void treeqp_sdunes_opts_set_default(int Nn, treeqp_sdunes_opts_t *opts);



int treeqp_sdunes_calculate_size(tree_qp_in *qp_in, treeqp_sdunes_opts_t *opts);

void treeqp_sdunes_create(tree_qp_in *qp_in, treeqp_sdunes_opts_t *opts, treeqp_sdunes_workspace *work, void *ptr);

int treeqp_sdunes_calculate_dual_dimension(int Nr, int md, int nu);

void treeqp_sdunes_set_dual_initialization(double *lam, double *mu, treeqp_sdunes_workspace *work);

return_t treeqp_sdunes_solve(tree_qp_in *qp_in, tree_qp_out *qp_out, treeqp_sdunes_opts_t *opts, treeqp_sdunes_workspace *work);


// TODO(dimitris): move out of here
void write_scenarios_solution_to_txt(int Ns, int Nh, int Nr, int md, int nx, int nu, int NewtonIter, treeqp_sdunes_workspace *work);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREEQP_SRC_DUAL_NEWTON_SCENARIOS_H_ */
