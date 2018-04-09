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


#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // for sqrt in 2-norm
#ifdef PARALLEL
#include <omp.h>
#endif
#include <assert.h>

// TODO(dimitris): Check if merging all loops wrt scenarios improves openmp siginficantly

#include "treeqp/src/dual_Newton_common.h"
#include "treeqp/src/dual_Newton_scenarios.h"
#include "treeqp/src/tree_ocp_qp_common.h"

#include "treeqp/utils/blasfeo.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/memory.h"
#include "treeqp/utils/profiling.h"
#include "treeqp/utils/tree.h"
#include "treeqp/utils/utils.h"
#include "treeqp/utils/timing.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_v_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blas.h"

#define REV_CHOL

#define NEW_FVAL
#define SPLIT_NODES



treeqp_sdunes_options_t treeqp_sdunes_default_options()
{
    treeqp_sdunes_options_t opts;
    termination_t cond = TREEQP_INFNORM;

    opts.maxIter = 100;
    opts.termCondition = cond;
    opts.stationarityTolerance = 1.0e-12;

    opts.checkLastActiveSet = 1;

    opts.lineSearchMaxIter = 50;
    opts.lineSearchGamma = 0.1;
    opts.lineSearchBeta = 0.6;

    opts.regType  = TREEQP_ALWAYS_LEVENBERG_MARQUARDT;
    opts.regTol   = 1.0e-12;
    opts.regValue = 1.0e-8;

    return opts;
}



int calculate_dimension_of_lambda(int Nr, int md, int nu) {
    int Ns = ipow(md, Nr);

    if (Ns == 1) {
        return -1;
    } else {
        return (Nr*Ns - (Ns-1)/(md-1))*nu;
    }
}


int get_maximum_vector_dimension(int Ns, int nx, int nu, int *commonNodes) {
    int ii;
    int maxDimension = nx;

    for (ii = 0; ii < Ns-1; ii++) {
        if (maxDimension < nu*commonNodes[ii]) maxDimension = nu*commonNodes[ii];
    }
    return maxDimension;
}


#ifdef SAVE_DATA

int get_size_of_JayD(int Ns, int nu, int *commonNodes) {
    int ii;
    int size = 0;

    for (ii = 0; ii < Ns-1; ii++) {
        size += ipow(nu*commonNodes[ii], 2);
    }
    return size;
}


int get_size_of_JayL(int Ns, int nu, int *commonNodes) {
    int ii;
    int size = 0;

    for (ii = 0; ii < Ns-2; ii++) {
        size += nu*commonNodes[ii+1]*nu*commonNodes[ii];
    }
    return size;
}


void save_stage_problems(int Ns, int Nh, int Nr, int md,
    treeqp_sdunes_workspace *work) {

    int ii, kk;
    int nu = work->su[0][0].m;
    int nx = work->sx[0][0].m;
    int nl = calculate_dimension_of_lambda(Nr, md, nu);
    double residuals_k[Ns*Nh*nx], residual[nl], xit[Ns*Nh*nx], uit[Ns*Nh*nu];
    double QinvCal_k[Ns*Nh*nx], RinvCal_k[Ns*Nh*nu];
    double LambdaD[Ns*Nh*nx*nx], LambdaL[Ns*(Nh-1)*nx*nx];
    int indRes = 0;
    int indResNonAnt = 0;
    int indX = 0;
    int indU = 0;
    int indLambdaD = 0;
    int indLambdaL = 0;
    int indZnk = 0;
    int *commonNodes = work->commonNodes;

    struct blasfeo_dvec *sResNonAnticip = work->sResNonAnticip;

    for (ii = 0; ii < Ns; ii++) {
        for (kk = 0; kk < Nh; kk++) {
            blasfeo_unpack_dvec(work->sresk[ii][kk].m, &work->sresk[ii][kk], 0,
                &residuals_k[indRes]);
            blasfeo_unpack_dvec(nx, &work->sx[ii][kk], 0, &xit[indX]);
            blasfeo_unpack_dvec(nu, &work->su[ii][kk], 0, &uit[indU]);
            blasfeo_unpack_dvec(nx, &work->sQinvCal[ii][kk], 0, &QinvCal_k[indX]);
            blasfeo_unpack_dvec(nu, &work->sRinvCal[ii][kk], 0, &RinvCal_k[indU]);
            blasfeo_unpack_dmat(nx, nx, &work->sLambdaD[ii][kk], 0, 0, &LambdaD[indLambdaD], nx);
            if (kk < Nh-1) {
                blasfeo_unpack_dmat(nx, nx, &work->sLambdaL[ii][kk], 0, 0,
                    &LambdaL[indLambdaL], nx);
                indLambdaL += nx*nx;
            }
            indRes += work->sresk[ii][kk].m;
            indX += work->sx[ii][kk].m;
            indU += work->su[ii][kk].m;
            indLambdaD += nx*nx;
            indZnk += nu*Nr*nx;
        }
        if (ii < Ns-1) {
            blasfeo_unpack_dvec(sResNonAnticip[ii].m, &sResNonAnticip[ii], 0, &residual[indResNonAnt]);
            indResNonAnt += nu*commonNodes[ii];
        }
    }

    write_double_vector_to_txt(residuals_k, Ns*Nh*nx, "examples/spring_mass_utils/resk.txt");
    write_double_vector_to_txt(residual, nl, "examples/spring_mass_utils/res.txt");
    write_double_vector_to_txt(xit, Ns*Nh*nx, "examples/spring_mass_utils/xit.txt");
    write_double_vector_to_txt(uit, Ns*Nh*nu, "examples/spring_mass_utils/uit.txt");
    write_double_vector_to_txt(QinvCal_k, Ns*Nh*nx, "examples/spring_mass_utils/Qit.txt");
    write_double_vector_to_txt(RinvCal_k, Ns*Nh*nu, "examples/spring_mass_utils/Rit.txt");
    write_double_vector_to_txt(LambdaD, Ns*Nh*nx*nx, "examples/spring_mass_utils/LambdaD.txt");
    write_double_vector_to_txt(LambdaL, Ns*(Nh-1)*nx*nx, "examples/spring_mass_utils/LambdaL.txt");
}

#endif


void write_scenarios_solution_to_txt(int Ns, int Nh, int Nr, int md, int nx, int nu,
    int NewtonIter, treeqp_sdunes_workspace *work) {

    int ii, kk;

    struct blasfeo_dvec *slambda = work->slambda;

    int indMu = 0;
    int indx = 0;
    int indu = 0;
    int indLambda = 0;
    int nl = calculate_dimension_of_lambda(Nr, md, nu);
    double *muIter = malloc(Ns*Nh*nx*sizeof(double));
    double *xIter = malloc(Ns*Nh*nx*sizeof(double));
    double *uIter = malloc(Ns*Nh*nu*sizeof(double));
    double *lambdaIter = malloc(nl*sizeof(double));

    for (ii = 0; ii < Ns; ii++) {
        for (kk = 0; kk < Nh; kk++) {
            blasfeo_unpack_dvec(nx, &work->smu[ii][kk], 0, &muIter[indMu]);
            indMu += nx;
            blasfeo_unpack_dvec(nx, &work->sx[ii][kk], 0, &xIter[indx]);
            indx += nx;
            blasfeo_unpack_dvec(nu, &work->su[ii][kk], 0, &uIter[indu]);
            indu += nu;
        }
        if (ii < Ns-1) {
            blasfeo_unpack_dvec(slambda[ii].m, &slambda[ii], 0, &lambdaIter[indLambda]);
            indLambda += slambda[ii].m;
        }
    }
    write_double_vector_to_txt(lambdaIter, nl, "examples/spring_mass_utils/lambdaIter.txt");
    write_double_vector_to_txt(muIter, Ns*Nh*nx, "examples/spring_mass_utils/muIter.txt");
    write_double_vector_to_txt(xIter, Ns*Nh*nx, "examples/spring_mass_utils/xIter.txt");
    write_double_vector_to_txt(uIter, Ns*Nh*nu, "examples/spring_mass_utils/uIter.txt");
    write_int_vector_to_txt(&NewtonIter, 1, "examples/spring_mass_utils/iter.txt");

    #if PROFILE > 0
    write_timers_to_txt();
    #endif

    free(muIter);
    free(xIter);
    free(uIter);
    free(lambdaIter);
}


answer_t node_processed(int node, int *processedNodes, int indx) {
    int ii;

    for (ii = 0; ii < indx; ii++) {
        if (node == processedNodes[ii]) return YES;
    }
    return NO;
}


int get_number_of_common_nodes(int Nn, int Ns, int Nh, int idx1, int idx2,
    struct node *tree) {

    int kk;
    int commonNodes = -1;
    int node1 = tree[Nn-Ns+idx1].idx;
    int node2 = tree[Nn-Ns+idx2].idx;

    for (kk = Nh; kk >= 0; kk--) {
        // printf("at kk = %d: %d-%d\n", kk, node1, node2);
        if (node1 == node2) {
            commonNodes = kk+1;
            break;
        } else {
            node1 = tree[node1].dad;
            node2 = tree[node2].dad;
        }
    }
    return commonNodes;
}


void build_vector_of_common_nodes(int Nn, int Ns, int Nh, struct node *tree,
    int *commonNodes) {
    int ii;

    for (ii = 0; ii < Ns-1; ii++) {
         commonNodes[ii] = get_number_of_common_nodes(Nn, Ns, Nh, ii, ii+1, tree);
    }
}


int compare_with_previous_active_set(int n, struct blasfeo_dvec *asNow, struct blasfeo_dvec *asBefore) {
    int ii;
    int changed = 0;

    for (ii = 0; ii < n; ii++) {
        if (BLASFEO_DVECEL(asNow, ii) != BLASFEO_DVECEL(asBefore, ii)) {
            changed = 1;
            break;
        }
    }
    blasfeo_dveccp(n, asNow, 0, asBefore, 0);
    return changed;
}



// TODO(dimitris): avoid some ifs in the loop maybe?
static void solve_stage_problems(int Ns, int Nh, int NewtonIter, tree_ocp_qp_in *qp_in,
    treeqp_sdunes_workspace *work, treeqp_sdunes_options_t *opts) {

    int ii, kk, idx, idxp1, idxm1;
    int nu = work->su[0][0].m;
    int nx = work->sx[0][0].m;
    int *commonNodes = work->commonNodes;
    struct blasfeo_dmat *sA = (struct blasfeo_dmat *) qp_in->A;
    struct blasfeo_dmat *sB = (struct blasfeo_dmat *) qp_in->B;
    struct blasfeo_dvec *sq = work->sq;
    struct blasfeo_dvec *sr = work->sr;
    struct blasfeo_dvec *sQinv = work->sQinv;
    struct blasfeo_dvec *sRinv = work->sRinv;
    struct blasfeo_dvec *sxmin = (struct blasfeo_dvec *) qp_in->xmin;
    struct blasfeo_dvec *sxmax = (struct blasfeo_dvec *) qp_in->xmax;
    struct blasfeo_dvec *sumin = (struct blasfeo_dvec *) qp_in->umin;
    struct blasfeo_dvec *sumax = (struct blasfeo_dvec *) qp_in->umax;

    struct blasfeo_dvec *slambda = work->slambda;
    int acc = 0;

    #ifdef PARALLEL
    #ifdef SPLIT_NODES
    #pragma omp parallel for
    #else
    #pragma omp parallel for private(kk, idx, idxm1, idxp1)
    #endif
    #endif
    for (ii = 0; ii < Ns; ii++) {
        #ifdef PARALLEL
        #ifdef SPLIT_NODES
        #pragma omp parallel for private(idx, idxm1, idxp1)
        #endif
        #endif
        for (kk = 0; kk < Nh; kk++) {
            // --- calculate x_opt
            idxm1 = work->nodeIdx[ii][kk];
            idx = work->nodeIdx[ii][kk+1];
            if (kk < Nh-1) {
                // x[k+1] = mu[k+1] - A[k+1]' * mu[k+2]
                idxp1 = work->nodeIdx[ii][kk+2];
                blasfeo_dgemv_t(nx, nx, -1.0, &sA[idxp1-1], 0, 0, &work->smu[ii][kk+1],
                    0, 1.0, &work->smu[ii][kk], 0, &work->sx[ii][kk], 0);
            } else {
                // x[Nh] = mu[Nh]
                blasfeo_dveccp(nx, &work->smu[ii][kk], 0, &work->sx[ii][kk], 0);
            }
            // x[k+1] = x[k+1] - q[k+1]
            blasfeo_daxpy(nx, -1.0, &sq[idx], 0, &work->sx[ii][kk], 0, &work->sx[ii][kk], 0);

            // x[k+1] = Q[k+1]^-1 .* x[k+1]
            blasfeo_dvecmuldot(nx, &sQinv[idx], 0, &work->sx[ii][kk], 0, &work->sx[ii][kk], 0);

            if (work->boundsRemoved[ii][kk+1] != 1) {
                blasfeo_dveccl_mask(nx, &sxmin[idx], 0, &work->sx[ii][kk], 0,
                    &sxmax[idx], 0, &work->sx[ii][kk], 0, &work->sxas[ii][kk], 0);

                // NOTE(dimitris): compares with previous AS and updates previous with current
                if (opts->checkLastActiveSet == 1)
                {
                    work->xasChanged[ii][kk+1] = compare_with_previous_active_set(nx,
                        &work->sxas[ii][kk], &work->sxasPrev[ii][kk]);
                }
                if ((opts->checkLastActiveSet == 0 ) || (work->xasChanged[ii][kk+1]))
                {
                    // QinvCal[kk+1] = Qinv[kk+1] .* (1 - abs(xas[kk+1]))
                    blasfeo_dvecze(nx, &work->sxas[ii][kk], 0, &sQinv[idx], 0,
                        &work->sQinvCal[ii][kk], 0);
                }
            }

            // --- calculate u_opt

            // u[k] = -B[k]' * mu[k] - r[k]
            blasfeo_dgemv_t(nx, nu, -1.0, &sB[idx-1], 0, 0, &work->smu[ii][kk], 0, -1.0,
                &sr[idxm1], 0, &work->su[ii][kk], 0);

            // u[k] = u[k] - C[k]' * lambda
            if ((ii < Ns-1) && (kk < commonNodes[ii])) {
                // shared multiplier with next scenario
                blasfeo_daxpy(nu, -1.0, &slambda[ii], kk*nu, &work->su[ii][kk], 0,
                    &work->su[ii][kk], 0);
            }
            if ((ii > 0) && (kk < commonNodes[ii-1])) {
                // shared multiplier with previous scenario
                blasfeo_daxpy(nu, 1.0, &slambda[ii-1], kk*nu, &work->su[ii][kk], 0,
                    &work->su[ii][kk], 0);
            }
            // u[k] = R[k]^-1 .* u[k]
            blasfeo_dvecmuldot(nu, &sRinv[idxm1], 0, &work->su[ii][kk], 0, &work->su[ii][kk], 0);

            if (work->boundsRemoved[ii][kk] != 1) {
                blasfeo_dveccl_mask(nu, &sumin[idxm1], 0, &work->su[ii][kk], 0,
                    &sumax[idxm1], 0, &work->su[ii][kk], 0, &work->suas[ii][kk], 0);
                if (opts->checkLastActiveSet)
                {
                    work->uasChanged[ii][kk] = compare_with_previous_active_set(nu,
                        &work->suas[ii][kk], &work->suasPrev[ii][kk]);
                }
                if ((opts->checkLastActiveSet == 0 ) || (work->uasChanged[ii][kk]))
                {
                    blasfeo_dvecze(nu, &work->suas[ii][kk], 0, &sRinv[idxm1], 0,
                        &work->sRinvCal[ii][kk], 0);
                }
            }

            // --- calculate Zbar

            if ((opts->checkLastActiveSet == 0 ) || (work->uasChanged[ii][kk] || NewtonIter == 0))
            {
                // Zbar[k] = B[k] * RinvCal[k]
                blasfeo_dgemm_nd(nx, nu, 1.0, &sB[idx-1], 0, 0, &work->sRinvCal[ii][kk],
                    0, 0.0, &work->sZbar[ii][kk], 0, 0, &work->sZbar[ii][kk], 0, 0);
            }

            // --- calculate Lambda blocks

            if ((opts->checkLastActiveSet == 0 ) ||
                ((kk == 0 && (work->uasChanged[ii][kk] || work->xasChanged[ii][kk+1])) ||
                (kk > 0 && (work->uasChanged[ii][kk] || work->xasChanged[ii][kk+1] ||
                work->xasChanged[ii][kk])) ||
                (NewtonIter == 0)))
            {
                // LambdaD[k] = Zbar[k] * B[k]'
                blasfeo_dgemm_nt(nx, nx, nu, 1.0, &work->sZbar[ii][kk], 0, 0, &sB[idx-1],
                    0, 0, 0.0, &work->sLambdaD[ii][kk], 0, 0, &work->sLambdaD[ii][kk], 0, 0);

                // LambdaD[k] = LambdaD[k] + QinvCal[k+1]
                blasfeo_ddiaad(nx, 1.0, &work->sQinvCal[ii][kk], 0, &work->sLambdaD[ii][kk], 0, 0);

                if (kk > 0) {
                    #ifdef REV_CHOL
                    // NOTE(dimitris): calculate LambdaL[k]' instead (aka upper triangular block)

                    // LambdaL[k]' = A[k]'
                    blasfeo_dgetr(nx, nx, &sA[idx-1], 0, 0, &work->sLambdaL[ii][kk-1], 0, 0);

                    // LambdaL[k]' = -QinvCal[k]*A[k]'
                    blasfeo_dgemm_dn(nx, nx, -1.0, &work->sQinvCal[ii][kk-1], 0,
                        &work->sLambdaL[ii][kk-1], 0, 0, 0.0, &work->sLambdaL[ii][kk-1], 0, 0,
                        &work->sLambdaL[ii][kk-1], 0, 0);

                    // LambdaD[k] = LambdaD[k] - A[k]*LambdaL[k]' = LambdaD[k] + A[k]*QinvCal[k]*A[k]'
                    blasfeo_dgemm_nn(nx, nx, nx, -1.0, &sA[idx-1], 0, 0,
                        &work->sLambdaL[ii][kk-1], 0, 0, 1.0, &work->sLambdaD[ii][kk], 0, 0,
                        &work->sLambdaD[ii][kk], 0, 0);

                    #else
                    // LambdaL[k] = -A[k] * QinvCal[k]
                    blasfeo_dgemm_nd(nx, nx, -1.0, &sA[idx-1], 0, 0,
                        &work->sQinvCal[ii][kk-1], 0, 0.0, &work->sLambdaL[ii][kk-1], 0, 0,
                        &work->sLambdaL[ii][kk-1], 0, 0);

                    // LambdaD[k] = LambdaD[k] - LambdaL[k] * A[k]'
                    blasfeo_dgemm_nt(nx, nx, nx, -1.0, &work->sLambdaL[ii][kk-1], 0, 0,
                        &sA[idx-1], 0, 0, 1.0, &work->sLambdaD[ii][kk], 0, 0,
                        &work->sLambdaD[ii][kk], 0, 0);
                    #endif
                }
                if (opts->checkLastActiveSet)
                {
                    // save diagonal block that will be overwritten in factorization
                    blasfeo_dgecp(nx, nx, &work->sLambdaD[ii][kk], 0, 0, &work->sTmpLambdaD[ii][kk], 0, 0);
                }
            }
            else
            {
                blasfeo_dgecp(nx, nx, &work->sTmpLambdaD[ii][kk], 0, 0,
                    &work->sLambdaD[ii][kk], 0, 0);
            }
        }
        acc += commonNodes[ii];
    }
}


static void calculate_residuals(int Ns, int Nh, tree_ocp_qp_in *qp_in,
    treeqp_sdunes_workspace *work) {

    int ii, kk, idx0, idxm1;
    int nu = work->su[0][0].m;
    int nx = work->sx[0][0].m;
    struct blasfeo_dmat *sA = (struct blasfeo_dmat *) qp_in->A;
    struct blasfeo_dmat *sB = (struct blasfeo_dmat *) qp_in->B;
    struct blasfeo_dvec *sb = (struct blasfeo_dvec *) qp_in->b;

    #ifdef PARALLEL
    #pragma omp parallel for private(kk, idx0, idxm1)
    #endif
    for (ii = 0; ii < Ns; ii++) {
        // res[1] = -b[0] - B[0] * u[0]
        // NOTE: different sign convention for b than in paper
        idx0 = work->nodeIdx[ii][1];
        blasfeo_dgemv_n(nx, nu, -1.0, &sB[idx0-1], 0, 0, &work->su[ii][0], 0, -1.0, &sb[idx0-1], 0,
            &work->sresk[ii][0], 0);
        // res[1] = res[1] + x[1]
        blasfeo_daxpy(nx, 1.0, &work->sx[ii][0], 0, &work->sresk[ii][0], 0, &work->sresk[ii][0], 0);

        for (kk = 2; kk < Nh+1; kk++) {
            idxm1 = work->nodeIdx[ii][kk];
            // printf("----> calculating residual of stage %d\n", kk);
            // res[k] = x[k] - A[k-1] * x[k-1]
            blasfeo_dgemv_n(nx, nx, -1.0, &sA[idxm1-1], 0, 0, &work->sx[ii][kk-2], 0, 1.0,
                &work->sx[ii][kk-1], 0, &work->sresk[ii][kk-1], 0);
            // res[k] = res[k] - B[k-1] * u[k-1]
            blasfeo_dgemv_n(nx, nu, -1.0, &sB[idxm1-1], 0, 0, &work->su[ii][kk-1], 0, 1.0,
                &work->sresk[ii][kk-1], 0, &work->sresk[ii][kk-1], 0);
            // res[k] = res[k] - b[k-1]
            blasfeo_daxpy(nx, -1.0, &sb[idxm1-1], 0, &work->sresk[ii][kk-1], 0,
                &work->sresk[ii][kk-1], 0);
        }
    }
}


static void calculate_last_residual(int Ns, int Nh, treeqp_sdunes_workspace *work) {
    int ii, kk;
    int acc = 0;
    int nu = work->su[0][0].m;
    int *commonNodes = work->commonNodes;

    struct blasfeo_dvec *sResNonAnticip = work->sResNonAnticip;

    // initialize at zero
    for (ii = 0; ii < Ns-1; ii++) blasfeo_dvecse(sResNonAnticip[ii].m, 0.0, &sResNonAnticip[ii], 0);

    // first scenario
    for (kk = 0; kk < commonNodes[0]; kk++) {
        blasfeo_daxpy(nu, -1.0, &work->su[0][kk], 0, &sResNonAnticip[0], kk*nu,
            &sResNonAnticip[0], kk*nu);
    }
    acc += commonNodes[0];
    for (ii = 1; ii < Ns-1; ii++) {
        // previous scenario
        for (kk = 0; kk < commonNodes[ii-1]; kk++) {
            blasfeo_daxpy(nu, 1.0, &work->su[ii][kk], 0, &sResNonAnticip[ii-1], kk*nu,
                &sResNonAnticip[ii-1], kk*nu);
        }
        // next scenario
        for (kk = 0; kk < commonNodes[ii]; kk++) {
            blasfeo_daxpy(nu, -1.0, &work->su[ii][kk], 0, &sResNonAnticip[ii], kk*nu,
                &sResNonAnticip[ii], kk*nu);
        }
        acc += commonNodes[ii];
    }
    // last scenario
    for (kk = 0; kk < commonNodes[Ns-2]; kk++) {
        blasfeo_daxpy(nu, 1.0, &work->su[Ns-1][kk], 0, &sResNonAnticip[Ns-2], kk*nu,
            &sResNonAnticip[Ns-2], kk*nu);
    }
}



static void find_starting_point_of_factorization(int Ns, int Nh, int *idxStart,
    treeqp_sdunes_workspace *work) {

    int ii, kk;

    for (ii = 0; ii < Ns; ii++) {
        if (work->uasChanged[ii][0] || work->xasChanged[ii][1]) {
            idxStart[ii] = 0;
        } else {
            idxStart[ii] = -1;
        }
        for (kk = Nh-1; kk > 0; kk--) {
            if (work->uasChanged[ii][kk] || work->xasChanged[ii][kk+1] ||
                work->xasChanged[ii][kk]) {
                idxStart[ii] = kk;
                break;
            }
        }
    }
}



static void factorize_Lambda(int Ns, int Nh, treeqp_sdunes_options_t *opts, treeqp_sdunes_workspace *work) {
    int ii, kk;
    int nx = work->sx[0][0].m;

    struct blasfeo_dvec *regMat = work->regMat;

    #ifdef SAVE_DATA
    int indD, indL;
    double CholLambdaD[Ns*Nh*nx*nx], CholLambdaL[Ns*(Nh-1)*nx*nx];
    indD = 0; indL = 0;
    #endif

    int idxStart[Ns];

    if (opts->checkLastActiveSet)
    {
        find_starting_point_of_factorization(Ns, Nh, idxStart, work);
        // for (ii = 0; ii < Ns; ii++)
        //     printf("restarting factorization of scenario %d at block %d\n", ii, idxStart[ii]);
    }

    // Banded Cholesky factorizations to calculate CholLambdaD[i], CholLambdaL[i]
    #ifdef PARALLEL
    #pragma omp parallel for private(kk)
    #endif
    for (ii = 0; ii < Ns; ii++) {
        #ifdef REV_CHOL
        for (kk = Nh-1; kk > 0; kk--) {
            if ((opts->checkLastActiveSet == 0) || (kk <= idxStart[ii]))
            {
                // Cholesky factorization (possibly regularized)
                factorize_with_reg_opts(&work->sLambdaD[ii][kk], &work->sCholLambdaD[ii][kk],
                    regMat, opts->regType, opts->regTol);

                // Substitution
                // NOTE(dimitris): LambdaL is already transposed (aka upper part of Lambda)
                blasfeo_dtrsm_rltn(nx, nx, 1.0, &work->sCholLambdaD[ii][kk], 0, 0,
                    &work->sLambdaL[ii][kk-1], 0, 0, &work->sCholLambdaL[ii][kk-1], 0, 0);
            }

            #ifdef SAVE_DATA
            blasfeo_unpack_dmat(nx, nx, &work->sCholLambdaD[ii][kk], 0, 0, &CholLambdaD[indD], nx);
            // TODO(dimitris): fix debugging in matlab for reverse Cholesky
            blasfeo_unpack_dmat(nx, nx, &work->sCholLambdaL[ii][kk-1], 0, 0, &CholLambdaL[indL], nx);
            indD += nx*nx; indL += nx*nx;
            #endif

            if ((opts->checkLastActiveSet == 0) || (kk <= idxStart[ii]+1))
            {
                // Update (LambdaD[i][k+-1] -= CholLambdaL[i][k] * CholLambdaL[i][k]')
                blasfeo_dsyrk_ln(nx, nx, -1.0, &work->sCholLambdaL[ii][kk-1], 0, 0,
                    &work->sCholLambdaL[ii][kk-1], 0, 0, 1.0, &work->sLambdaD[ii][kk-1], 0, 0,
                    &work->sLambdaD[ii][kk-1], 0, 0);
            }
        }
        #ifdef REV_CHOL
        if (0 <= idxStart[ii]) {
        #endif
        factorize_with_reg_opts(&work->sLambdaD[ii][0], &work->sCholLambdaD[ii][0],
            regMat, opts->regType, opts->regTol);

        #ifdef REV_CHOL
        }
        #endif

        #ifdef SAVE_DATA
        blasfeo_unpack_dmat(nx, nx, &work->sCholLambdaD[ii][0], 0, 0, &CholLambdaD[indD], nx);
        indD += nx*nx;
        #endif

        #else  /* REV_CHOL */
        for (kk = 0; kk < Nh-1 ; kk++) {
            // Cholesky factorization (possibly regularized)
            factorize_with_reg_opts(&work->sLambdaD[ii][kk], &work->sCholLambdaD[ii][kk],
                regMat, opts->regType, opts->regTol);

            // Substitution
            blasfeo_dtrsm_rltn(nx, nx, 1.0, &work->sCholLambdaD[ii][kk], 0, 0,
                &work->sLambdaL[ii][kk], 0, 0, &work->sCholLambdaL[ii][kk], 0, 0);

            #ifdef SAVE_DATA
            blasfeo_unpack_dmat(nx, nx, &work->sCholLambdaD[ii][kk], 0, 0, &CholLambdaD[indD], nx);
            blasfeo_unpack_dmat(nx, nx, &work->sCholLambdaL[ii][kk], 0, 0, &CholLambdaL[indL], nx);
            indD += nx*nx; indL += nx*nx;
            #endif

            // Update (LambdaD[i][k-1] -= CholLambdaL[i][k] * CholLambdaL[i][k]')
            blasfeo_dsyrk_ln(nx, nx, -1.0, &work->sCholLambdaL[ii][kk], 0, 0,
                &work->sCholLambdaL[ii][kk], 0, 0, 1.0, &work->sLambdaD[ii][kk+1], 0, 0,
                &work->sLambdaD[ii][kk+1], 0, 0);
        }
        factorize_with_reg_opts(&work->sLambdaD[ii][Nh-1], &work->sCholLambdaD[ii][Nh-1],
                regMat, opts->regType, opts->regTol);

        #ifdef SAVE_DATA
        blasfeo_unpack_dmat(nx, nx, &work->sCholLambdaD[ii][Nh-1], 0, 0, &CholLambdaD[indD], nx);
        indD += nx*nx;
        #endif

        #endif  /* REV_CHOL */
    }
    #ifdef SAVE_DATA
    write_double_vector_to_txt(CholLambdaD, Ns*Nh*nx*nx, "examples/spring_mass_utils/CholLambdaD.txt");
    write_double_vector_to_txt(CholLambdaL, Ns*(Nh-1)*nx*nx, "examples/spring_mass_utils/CholLambdaL.txt");
    #endif
}


void form_K(int Ns, int Nh, int Nr, treeqp_sdunes_workspace *work) {
    int ii, jj, kk;
    int indZ, indRinvCal;
    int nx = work->sx[0][0].m;
    int nu = work->su[0][0].m;

    struct blasfeo_dmat *sUt = work->sUt;
    struct blasfeo_dmat *sK = work->sK;
    struct blasfeo_dmat *sTmpMats = work->sTmpMats;

    #ifdef SAVE_DATA
    int indK = 0;
    double K[Ns*Nr*nu*Nr*nu];
    #endif

    #ifdef PARALLEL
    #pragma omp parallel for private(jj, kk)
    #endif
    for (ii = 0; ii < Ns; ii++) {
        // ----- form U[i]'
        for (jj = 0; jj < Nr; jj++) {
            // transpose Zbar[k]
            blasfeo_dgetr(nx, nu, &work->sZbar[ii][jj], 0, 0, &sTmpMats[ii], 0, 0);
            #ifdef REV_CHOL
            for (kk = jj; kk >= 0; kk--) {
            #else
            for (kk = jj; kk < Nh; kk++) {
            #endif
                // matrix substitution
                // D <= B * A^{-T} , with A lower triangular employing explicit inverse of diagonal
                #ifdef LA_HIGH_PERFORMANCE
                // NOTE(dimitris): writing directly on sub-block NIY for BLASFEO_HP
                blasfeo_dtrsm_rltn(nu, nx, 1.0, &work->sCholLambdaD[ii][kk], 0, 0,
                    &sTmpMats[ii], 0, 0, &sTmpMats[ii], 0, 0);
                blasfeo_dgecp(nu, nx, &sTmpMats[ii], 0, 0, &sUt[ii], jj*nu, kk*nx);
                #else
                blasfeo_dtrsm_rltn(nu, nx, 1.0, &work->sCholLambdaD[ii][kk], 0, 0,
                    &sTmpMats[ii], 0, 0, &sUt[ii], jj*nu, kk*nx);
                #endif

                // update
                #ifdef REV_CHOL
                if (kk > 0) {
                    blasfeo_dgemm_nt(nu, nx, nx, -1.0, &sUt[ii], jj*nu, kk*nx,
                        &work->sCholLambdaL[ii][kk-1], 0, 0, 0.0, &sTmpMats[ii], 0, 0,
                        &sTmpMats[ii], 0, 0);
                }
                #else
                if (kk < Nh-1) {
                    blasfeo_dgemm_nt(nu, nx, nx, -1.0, &sUt[ii], jj*nu, kk*nx,
                        &work->sCholLambdaL[ii][kk], 0, 0, 0.0, &sTmpMats[ii], 0, 0,
                        &sTmpMats[ii], 0, 0);
                }
                #endif
            }
        }

        // ----- form upper right part of K[i]

        // symmetric matrix multiplication
        // TODO(dimitris): probably doing this with structure exploitation is cheaper if REV_CHOL=1
        blasfeo_dsyrk_ln(sUt[ii].m, sUt[ii].n, -1.0, &sUt[ii], 0, 0, &sUt[ii], 0, 0, 0.0,
            &sK[ii], 0, 0, &sK[ii], 0, 0);

        // mirror result to upper diagonal part (needed to form J properly)
        blasfeo_dtrtr_l(sK[ii].m, &sK[ii], 0, 0, &sK[ii], 0, 0);

        for (kk = 0; kk < Nr; kk++) {
            blasfeo_ddiaad(nu, 1.0, &work->sRinvCal[ii][kk], 0, &sK[ii], kk*nu, kk*nu);
        }

        #ifdef SAVE_DATA
        blasfeo_unpack_dmat(Nr*nu, Nr*nu, &sK[ii], 0, 0, &K[indK], Nr*nu);
        indK += Nr*nu*Nr*nu;
        #endif
    }

    #ifdef SAVE_DATA
    write_double_vector_to_txt(K, Ns*Nr*nu*Nr*nu, "examples/spring_mass_utils/K.txt");
    #endif
}


void form_and_factorize_Jay(int Ns, int nu, treeqp_sdunes_options_t *opts, treeqp_sdunes_workspace *work) {
    int ii, dim, dimNxt;
    int *commonNodes = work->commonNodes;

    struct blasfeo_dvec *regMat = work->regMat;
    struct blasfeo_dmat *sK = work->sK;
    struct blasfeo_dmat *sJayD = work->sJayD;
    struct blasfeo_dmat *sJayL = work->sJayL;
    struct blasfeo_dmat *sCholJayD = work->sCholJayD;
    struct blasfeo_dmat *sCholJayL = work->sCholJayL;



    #ifdef SAVE_DATA
    int indJayD, indJayL;
    int nJayD = get_size_of_JayD(Ns, nu, commonNodes);
    int nJayL = get_size_of_JayL(Ns, nu, commonNodes);
    double JayD[nJayD], JayL[nJayL], CholJayD[nJayD], CholJayL[nJayL];
    indJayD = 0; indJayL = 0;
    #endif

    // Banded Cholesky factorizations to calculate factor of Jay
    // NOTE: Cannot be parallelized
    for (ii = 0; ii < Ns-1; ii++) {
        dim = nu*commonNodes[ii];
        // Form JayD[i] using blocks K[i] and K[i+1]
        blasfeo_dgead(dim, dim, 1.0, &sK[ii], 0, 0, &sJayD[ii], 0, 0);
        blasfeo_dgead(dim, dim, 1.0, &sK[ii+1], 0, 0, &sJayD[ii], 0, 0);

        // Cholesky factorization (possibly regularized)
        // TODO(dimitris): remove regMat and add opts->regValue to diagonal
        factorize_with_reg_opts(&sJayD[ii], &sCholJayD[ii], regMat, opts->regType, opts->regTol);

        #ifdef SAVE_DATA
        if (ii > 0) {  // undo update
            blasfeo_dsyrk_ln(dim, sCholJayL[ii-1].n, 1.0, &sCholJayL[ii-1], 0, 0,
                &sCholJayL[ii-1], 0, 0, 1.0, &sJayD[ii], 0, 0, &sJayD[ii], 0, 0);
        }
        blasfeo_unpack_dmat(dim, dim, &sJayD[ii], 0, 0, &JayD[indJayD], dim);
        if (ii > 0) {  // redo update
            blasfeo_dsyrk_ln(dim, sCholJayL[ii-1].n, -1.0, &sCholJayL[ii-1], 0, 0,
                &sCholJayL[ii-1], 0, 0, 1.0, &sJayD[ii], 0, 0, &sJayD[ii], 0, 0);
        }
        blasfeo_unpack_dmat(dim, dim, &sCholJayD[ii], 0, 0, &CholJayD[indJayD], dim);
        indJayD += ipow(dim, 2);
        #endif

        if (ii < Ns-2) {
            dimNxt = nu*commonNodes[ii+1];
            // Form JayL[i] using block K[i+1]
            blasfeo_dgead(dimNxt, dim, -1.0, &sK[ii+1], 0, 0, &sJayL[ii], 0, 0);

            // Substitution to form CholJayL[i]
            blasfeo_dtrsm_rltn(dimNxt, dim, 1.0, &sCholJayD[ii], 0, 0, &sJayL[ii], 0, 0,
                &sCholJayL[ii], 0, 0);

            #ifdef SAVE_DATA
            blasfeo_unpack_dmat(dimNxt, dim, &sJayL[ii], 0, 0, &JayL[indJayL], dimNxt);
            blasfeo_unpack_dmat(dimNxt, dim, &sCholJayL[ii], 0, 0, &CholJayL[indJayL], dimNxt);
            indJayL += dimNxt*dim;
            #endif

            // Update for next block (NOTE: the update is added here before forming the block)
            blasfeo_dsyrk_ln(dimNxt, dim, -1.0, &sCholJayL[ii], 0, 0, &sCholJayL[ii], 0, 0,
                1.0, &sJayD[ii+1], 0, 0, &sJayD[ii+1], 0, 0);
        }
    }

    #ifdef SAVE_DATA
    write_double_vector_to_txt(JayD, nJayD, "examples/spring_mass_utils/JayD.txt");
    write_double_vector_to_txt(JayL, nJayL, "examples/spring_mass_utils/JayL.txt");
    write_double_vector_to_txt(CholJayD, nJayD, "examples/spring_mass_utils/CholJayD.txt");
    write_double_vector_to_txt(CholJayL, nJayL, "examples/spring_mass_utils/CholJayL.txt");
    #endif
}


void form_RHS_non_anticipaticity(int Ns, int Nh, int Nr, int md, treeqp_sdunes_workspace *work) {
    // NOTE: RHS has the opposite sign (corected when solving for Deltalambda later)

    int ii, jj, kk;
    int nx = work->sx[0][0].m;
    int nu = work->su[0][0].m;
    int *commonNodes = work->commonNodes;

    struct blasfeo_dvec *sTmpVecs = work->sTmpVecs;
    struct blasfeo_dvec *sResNonAnticip = work->sResNonAnticip;
    struct blasfeo_dvec *sRhsNonAnticip = work->sRhsNonAnticip;

    #ifdef SAVE_DATA
    int ind = 0;
    int nl = calculate_dimension_of_lambda(Nr, md, nu);
    double *rhsNonAnticip = malloc(nl*sizeof(double));
    #endif

    #ifdef PARALLEL
    #pragma omp parallel for private(kk)
    #endif
    for (ii = 0; ii < Ns; ii++) {
        #ifdef REV_CHOL
        // tmp = res[Nh]
        blasfeo_dveccp(nx, &work->sresk[ii][Nh-1] , 0, &sTmpVecs[ii], 0);

        // backward substitution
        for (kk = Nh; kk > 1; kk--) {
            // resMod[k] = inv(CholLambdaD[k-1]) * tmp
            blasfeo_dtrsv_lnn(nx, &work->sCholLambdaD[ii][kk-1], 0, 0,
                &sTmpVecs[ii], 0, &work->sreskMod[ii][kk-1], 0);

            // update
            // tmp = res[k-1] - CholLambdaL[k-1] * resMod[k]
            blasfeo_dgemv_n(nx, nx, -1.0, &work->sCholLambdaL[ii][kk-2], 0, 0,
                &work->sreskMod[ii][kk-1], 0, 1.0, &work->sresk[ii][kk-2], 0, &sTmpVecs[ii], 0);
        }
        blasfeo_dtrsv_lnn(nx, &work->sCholLambdaD[ii][0], 0, 0, &sTmpVecs[ii],
            0, &work->sreskMod[ii][0], 0);

        // forward substitution
        for (kk = 0; kk < Nh-1; kk++) {
            // resMod[k+1] = inv(CholLambdaD[k]') * resMod[k+1]
            blasfeo_dtrsv_ltn(nx, &work->sCholLambdaD[ii][kk], 0, 0,
                &work->sreskMod[ii][kk], 0, &work->sreskMod[ii][kk], 0);

            // resMod[k+2] = resMod[k+2] - CholLambdaL[k+1]' * resMod[k+1]
            blasfeo_dgemv_t(nx, nx, -1.0, &work->sCholLambdaL[ii][kk], 0, 0,
                &work->sreskMod[ii][kk], 0, 1.0, &work->sreskMod[ii][kk+1], 0,
                &work->sreskMod[ii][kk+1], 0);
        }
        blasfeo_dtrsv_ltn(nx, &work->sCholLambdaD[ii][Nh-1], 0, 0,
            &work->sreskMod[ii][Nh-1], 0, &work->sreskMod[ii][Nh-1], 0);
        #else
        // tmp = res[1]
        blasfeo_dveccp(nx, &work->sresk[ii][0] , 0, &sTmpVecs[ii], 0);

        // forward substitution
        for (kk = 0; kk < Nh-1; kk++) {
            // resMod[k+1] = inv(CholLambdaD[k]) * tmp
            blasfeo_dtrsv_lnn(nx, &work->sCholLambdaD[ii][kk], 0, 0, &sTmpVecs[ii],
                0, &work->sreskMod[ii][kk], 0);

            // update
            // tmp = res[k+2] - CholLambdaL[k+1] * resMod[k+1]
            blasfeo_dgemv_n(nx, nx, -1.0, &work->sCholLambdaL[ii][kk], 0, 0,
                &work->sreskMod[ii][kk], 0, 1.0, &work->sresk[ii][kk+1], 0, &sTmpVecs[ii], 0);
        }
        blasfeo_dtrsv_lnn(nx, &work->sCholLambdaD[ii][Nh-1], 0, 0, &sTmpVecs[ii],
            0, &work->sreskMod[ii][Nh-1], 0);

        // backward substitution
        for (kk = Nh; kk > 1; kk--) {
            // resMod[k] = inv(CholLambdaD[k-1]') * resMod[k]
            blasfeo_dtrsv_ltn(nx, &work->sCholLambdaD[ii][kk-1], 0, 0,
                &work->sreskMod[ii][kk-1], 0, &work->sreskMod[ii][kk-1], 0);

            // resMod[k-1] = resMod[k-1] - CholLambdaL[k-1]' * resMod[k]
            blasfeo_dgemv_t(nx, nx, -1.0, &work->sCholLambdaL[ii][kk-2], 0, 0,
                &work->sreskMod[ii][kk-1], 0, 1.0, &work->sreskMod[ii][kk-2], 0,
                &work->sreskMod[ii][kk-2], 0);
        }
        blasfeo_dtrsv_ltn(nx, &work->sCholLambdaD[ii][0], 0, 0,
            &work->sreskMod[ii][0], 0, &work->sreskMod[ii][0], 0);
        #endif
    }

    // for ii == 0
    blasfeo_dveccp(sResNonAnticip[0].m, &sResNonAnticip[0], 0, &sRhsNonAnticip[0], 0);
    for (kk = 0; kk < commonNodes[0]; kk++) {
        blasfeo_dgemv_t(nx, nu, -1.0, &work->sZbar[0][kk], 0, 0,
            &work->sreskMod[0][kk], 0, 1.0, &sRhsNonAnticip[0], kk*nu,
            &sRhsNonAnticip[0], kk*nu);
    }

    for (ii = 1; ii < Ns-1; ii++) {
        blasfeo_dveccp(sResNonAnticip[ii].m, &sResNonAnticip[ii], 0, &sRhsNonAnticip[ii], 0);
        for (kk = 0; kk < commonNodes[ii-1]; kk++) {
            blasfeo_dgemv_t(nx, nu, 1.0, &work->sZbar[ii][kk], 0, 0,
                &work->sreskMod[ii][kk], 0, 1.0, &sRhsNonAnticip[ii-1], kk*nu,
                &sRhsNonAnticip[ii-1], kk*nu);
        }
        for (kk = 0; kk < commonNodes[ii]; kk++) {
            blasfeo_dgemv_t(nx, nu, -1.0, &work->sZbar[ii][kk], 0, 0,
                &work->sreskMod[ii][kk], 0, 1.0, &sRhsNonAnticip[ii], kk*nu,
                &sRhsNonAnticip[ii], kk*nu);
        }
    }
    // for ii == Ns-1
    for (kk = 0; kk < commonNodes[ii-1]; kk++) {
    blasfeo_dgemv_t(nx, nu, 1.0, &work->sZbar[Ns-1][kk], 0, 0,
        &work->sreskMod[Ns-1][kk], 0, 1.0, &sRhsNonAnticip[Ns-2], kk*nu,
        &sRhsNonAnticip[Ns-2], kk*nu);
    }

    #ifdef SAVE_DATA
    for (ii = 0; ii < Ns-1; ii++) {
        blasfeo_unpack_dvec(sRhsNonAnticip[ii].m, &sRhsNonAnticip[ii], 0, &rhsNonAnticip[ind]);
        ind += nu*commonNodes[ii];
    }
    write_double_vector_to_txt(rhsNonAnticip, nl, "examples/spring_mass_utils/rhsNonAnticip.txt");
    free(rhsNonAnticip);
    #endif
    // printf("RHS:\n");
    // for (ii = 0; ii < Ns-1; ii++)
    //     blasfeo_print_dvec(commonNodes[ii]*nu, &sRhsNonAnticip[ii], 0);
}



void calculate_delta_lambda(int Ns, int Nr, int md, treeqp_sdunes_workspace *work) {
    int ii, dim, dimNxt;
    int *commonNodes = work->commonNodes;
    int nu = work->su[0][0].m;

    struct blasfeo_dmat *sCholJayD = work->sCholJayD;
    struct blasfeo_dmat *sCholJayL = work->sCholJayL;
    struct blasfeo_dvec *sRhsNonAnticip = work->sRhsNonAnticip;
    struct blasfeo_dvec *sDeltalambda = work->sDeltalambda;

    #ifdef SAVE_DATA
    int ind = 0;
    int nl = calculate_dimension_of_lambda(Nr, md, nu);
    double Deltalambda[nl];
    #endif

    // special case if Ns == 2 as we do not enter any of the two loops
    if (Ns == 2) {
        dimNxt = nu*commonNodes[0];
    }

    // ------ forward substitution
    for (ii = 0; ii < Ns-2; ii++) {
        dim = nu*commonNodes[ii];
        dimNxt = nu*commonNodes[ii+1];

        // flip sign of residual
        blasfeo_dveccpsc(dim, -1.0, &sRhsNonAnticip[ii], 0, &sDeltalambda[ii], 0);

        // substitution
        blasfeo_dtrsv_lnn(dim, &sCholJayD[ii], 0, 0, &sDeltalambda[ii], 0,
            &sRhsNonAnticip[ii], 0);

        // update
        blasfeo_dgemv_n(dimNxt, dim, 1.0, &sCholJayL[ii], 0, 0, &sRhsNonAnticip[ii], 0, 1.0,
            &sRhsNonAnticip[ii+1], 0, &sRhsNonAnticip[ii+1], 0);
    }
    // ii = Ns-2 (last part of the loop without the update)
    blasfeo_dveccpsc(dimNxt, -1.0, &sRhsNonAnticip[ii], 0, &sDeltalambda[ii], 0);

    blasfeo_dtrsv_lnn(dimNxt, &sCholJayD[ii], 0, 0, &sDeltalambda[ii], 0,
        &sRhsNonAnticip[ii], 0);

    // ------ backward substitution
    for (ii = Ns-1; ii > 1; ii--) {
        dim = nu*commonNodes[ii-1];
        dimNxt = nu*commonNodes[ii-2];

        // substitution
        blasfeo_dtrsv_ltn(dim, &sCholJayD[ii-1], 0, 0, &sRhsNonAnticip[ii-1], 0,
            &sDeltalambda[ii-1], 0);

        // update
        blasfeo_dgemv_t(dim, dimNxt, -1.0, &sCholJayL[ii-2], 0, 0, &sDeltalambda[ii-1], 0, 1.0,
            &sRhsNonAnticip[ii-2], 0, &sRhsNonAnticip[ii-2], 0);
    }
    // ii = 1 (last part of the loop without the update)
    blasfeo_dtrsv_ltn(dimNxt, &sCholJayD[ii-1], 0, 0, &sRhsNonAnticip[ii-1], 0,
        &sDeltalambda[ii-1], 0);

    // printf("Delta lambdas:\n");
    // for (ii = 0; ii < Ns-1;ii++)
    //     blasfeo_print_dvec(sDeltalambda[ii].m, &sDeltalambda[ii], 0);
    #ifdef SAVE_DATA
    for (ii = 0; ii < Ns-1; ii++) {
        blasfeo_unpack_dvec(sDeltalambda[ii].m, &sDeltalambda[ii], 0, &Deltalambda[ind]);
        ind += nu*commonNodes[ii];
    }
    write_double_vector_to_txt(Deltalambda, nl, "examples/spring_mass_utils/Deltalambda.txt");
    #endif
}


void calculate_delta_mu(int Ns, int Nh, int Nr, treeqp_sdunes_workspace *work) {
    int ii, kk;
    int nx = work->sx[0][0].m;
    int nu = work->su[0][0].m;
    int *commonNodes = work->commonNodes;

    struct blasfeo_dvec *sDeltalambda = work->sDeltalambda;
    struct blasfeo_dvec *sTmpVecs = work->sTmpVecs;

    #ifdef SAVE_DATA
    int indRes, indMu;
    double Deltamu[Ns*Nh*nx], rhsDynamics[Ns*Nh*nx];
    indRes = 0; indMu = 0;
    #endif

    #ifdef PARALLEL
    #pragma omp parallel for private(kk)
    #endif
    for (ii = 0; ii < Ns; ii++) {
        // resMod[i] = -C'[i] * Deltalambda[i] - res[i]
        for (kk = 0; kk < Nh; kk++) {
            blasfeo_dvecse(work->sreskMod[ii][kk].m, 0.0, &work->sreskMod[ii][kk], 0);
            if ((ii > 0) && (kk < commonNodes[ii-1])) {
                // shared multiplier with previous scenario
                blasfeo_daxpy(nu, 1.0, &sDeltalambda[ii-1], kk*nu, &work->sreskMod[ii][kk], 0,
                    &work->sreskMod[ii][kk], 0);
            }
            if ((ii < Ns-1) && (kk < commonNodes[ii])) {
                // shared multiplier with next scenario
                blasfeo_daxpy(nu, -1.0, &sDeltalambda[ii], kk*nu, &work->sreskMod[ii][kk], 0,
                    &work->sreskMod[ii][kk], 0);
            }
            if (kk < Nr) {
                // NOTE: cannot have resMod also as output!
                blasfeo_dgemv_n(nx, nu, 1.0, &work->sZbar[ii][kk], 0, 0,
                &work->sreskMod[ii][kk], 0, -1.0, &work->sresk[ii][kk], 0,
                    &sTmpVecs[ii], 0);
                blasfeo_dveccp(nx, &sTmpVecs[ii], 0, &work->sreskMod[ii][kk], 0);
            } else {
                blasfeo_daxpy(nx, -1.0, &work->sresk[ii][kk], 0, &work->sreskMod[ii][kk], 0,
                    &work->sreskMod[ii][kk], 0);
            }
        }
        #ifdef SAVE_DATA
        for (kk = 0; kk < Nh; kk++) {
            blasfeo_unpack_dvec(nx, &work->sreskMod[ii][kk], 0, &rhsDynamics[indRes]);
            indRes += nx;
        }
        #endif

        // ------ forward-backward substitution to calculate mu

        #ifdef REV_CHOL
        // backward substitution
        for (kk = Nh; kk > 1; kk--) {
            // Deltamu[k] = inv(CholLambdaD[k-1]) * res[k]
            blasfeo_dtrsv_lnn(nx, &work->sCholLambdaD[ii][kk-1], 0, 0,
                &work->sreskMod[ii][kk-1], 0, &work->sDeltamu[ii][kk-1], 0);

            // update
            blasfeo_dgemv_n(nx, nx, -1.0, &work->sCholLambdaL[ii][kk-2], 0, 0,
                &work->sDeltamu[ii][kk-1], 0, 1.0, &work->sreskMod[ii][kk-2], 0,
                &work->sreskMod[ii][kk-2], 0);
        }
        blasfeo_dtrsv_lnn(nx, &work->sCholLambdaD[ii][0], 0, 0, &work->sreskMod[ii][0],
            0, &work->sDeltamu[ii][0], 0);

        // forward substitution
        for (kk = 0; kk < Nh-1; kk++) {
            // Deltamu[k+1] = inv(CholLambdaD[k]') * Deltamu[k+1]
            blasfeo_dtrsv_ltn(nx, &work->sCholLambdaD[ii][kk], 0, 0,
                &work->sDeltamu[ii][kk], 0, &work->sDeltamu[ii][kk], 0);

            // update
            blasfeo_dgemv_t(nx, nx, -1.0, &work->sCholLambdaL[ii][kk], 0, 0,
                &work->sDeltamu[ii][kk], 0, 1.0, &work->sDeltamu[ii][kk+1], 0,
                &work->sDeltamu[ii][kk+1], 0);
        }
        blasfeo_dtrsv_ltn(nx, &work->sCholLambdaD[ii][Nh-1], 0, 0,
            &work->sDeltamu[ii][Nh-1], 0, &work->sDeltamu[ii][Nh-1], 0);
        #else
        // forward substitution
        for (kk = 0; kk < Nh-1; kk++) {
            // Deltamu[k+1] = inv(CholLambdaD[k]) * res[k+1]
            blasfeo_dtrsv_lnn(nx, &work->sCholLambdaD[ii][kk], 0, 0,
                &work->sreskMod[ii][kk], 0, &work->sDeltamu[ii][kk], 0);

            // update
            blasfeo_dgemv_n(nx, nx, -1.0, &work->sCholLambdaL[ii][kk], 0, 0,
                &work->sDeltamu[ii][kk], 0, 1.0, &work->sreskMod[ii][kk+1], 0,
                &work->sreskMod[ii][kk+1], 0);
        }
        blasfeo_dtrsv_lnn(nx, &work->sCholLambdaD[ii][Nh-1], 0, 0,
            &work->sreskMod[ii][Nh-1], 0, &work->sDeltamu[ii][Nh-1], 0);

        // backward substitution
        for (kk = Nh; kk > 1; kk--) {
            // Deltamu[k] = inv(CholLambdaD[k-1]') * Deltamu[k]
            blasfeo_dtrsv_ltn(nx, &work->sCholLambdaD[ii][kk-1], 0, 0,
                &work->sDeltamu[ii][kk-1], 0, &work->sDeltamu[ii][kk-1], 0);

            // Deltamu[k-1] = Deltamu[k-1] - CholLambdaL[k-1] * Deltamu[k]
            blasfeo_dgemv_t(nx, nx, -1.0, &work->sCholLambdaL[ii][kk-2], 0, 0,
                &work->sDeltamu[ii][kk-1], 0, 1.0, &work->sDeltamu[ii][kk-2], 0,
                &work->sDeltamu[ii][kk-2], 0);
        }
        blasfeo_dtrsv_ltn(nx, &work->sCholLambdaD[ii][0], 0, 0,
            &work->sDeltamu[ii][0], 0, &work->sDeltamu[ii][0], 0);
        #endif  /* REV_CHOL */

        // printf("SCENARIO %d, MULTIPLIERS:\n", ii+1);
        // for (kk = 0; kk < Nh; kk++) {
        //     blasfeo_print_dvec(nx, &work->sDeltamu[ii][kk], 0);
        // }
    }
    #ifdef SAVE_DATA
    for (ii = 0; ii < Ns; ii++) {
        for (kk = 0; kk < Nh; kk++) {
            blasfeo_unpack_dvec(nx, &work->sDeltamu[ii][kk], 0, &Deltamu[indMu]);
            indMu += nx;
        }
    }
    write_double_vector_to_txt(rhsDynamics, Ns*Nh*nx, "examples/spring_mass_utils/rhsDynamics.txt");
    write_double_vector_to_txt(Deltamu, Ns*Nh*nx, "examples/spring_mass_utils/Deltamu.txt");
    #endif
}


double gradient_trans_times_direction(int Ns, int Nh, treeqp_sdunes_workspace *work) {
    int ii, kk;
    int nx = work->sx[0][0].m;
    double ans = 0;

    struct blasfeo_dvec *sResNonAnticip = work->sResNonAnticip;
    struct blasfeo_dvec *sDeltalambda = work->sDeltalambda;

    for (ii = 0; ii < Ns-1; ii++) {
        ans += blasfeo_ddot(sResNonAnticip[ii].m, &sResNonAnticip[ii], 0, &sDeltalambda[ii], 0);
    }

    for (ii = 0; ii < Ns; ii++) {
        for (kk = 0; kk < Nh; kk++) {
            ans += blasfeo_ddot(nx, &work->sresk[ii][kk], 0, &work->sDeltamu[ii][kk], 0);
        }
    }
    return ans;
}


double evaluate_dual_function(int Ns, int Nh, tree_ocp_qp_in *qp_in, treeqp_sdunes_workspace *work) {
    int *commonNodes = work->commonNodes;
    double *fvals = work->fvals;

    struct blasfeo_dvec *sTmpVecs = work->sTmpVecs;
    struct blasfeo_dvec *slambda = work->slambda;
    struct blasfeo_dvec *sResNonAnticip = work->sResNonAnticip;
    struct blasfeo_dmat *sA = (struct blasfeo_dmat *) qp_in->A;
    struct blasfeo_dmat *sB = (struct blasfeo_dmat *) qp_in->B;
    struct blasfeo_dvec *sb = (struct blasfeo_dvec *) qp_in->b;
    struct blasfeo_dvec *sQ = work->sQ;
    struct blasfeo_dvec *sR = work->sR;
    struct blasfeo_dvec *sq = work->sq;
    struct blasfeo_dvec *sr = work->sr;
    struct blasfeo_dvec *sQinv = work->sQinv;
    struct blasfeo_dvec *sRinv = work->sRinv;
    struct blasfeo_dvec *sxmin = (struct blasfeo_dvec *) qp_in->xmin;
    struct blasfeo_dvec *sxmax = (struct blasfeo_dvec *) qp_in->xmax;
    struct blasfeo_dvec *sumin = (struct blasfeo_dvec *) qp_in->umin;
    struct blasfeo_dvec *sumax = (struct blasfeo_dvec *) qp_in->umax;

    double fval = 0;
    int ii, kk, idx, idxp1, idxm1;
    int nu = work->su[0][0].m;
    int nx = work->sx[0][0].m;

    // TODO(dimitris): remove sTmpVecs from line search after allowing only NEW_FVAL option

    #ifdef PARALLEL
    #ifdef SPLIT_NODES
    #pragma omp parallel for
    #else
    #pragma omp parallel for private(kk, idx, idxp1, idxm1)
    #endif
    #endif
    for (ii = 0; ii < Ns; ii++) {
        fvals[ii] = 0;
        // NOTE(dimitris): unconstrained solution is used to evalaute dual function efficiently
        #ifdef PARALLEL
        #ifdef SPLIT_NODES
        #pragma omp parallel for private(idx, idxp1, idxm1)
        #endif
        #endif
        for (kk = 0; kk < Nh; kk++) {
            idxm1 = work->nodeIdx[ii][kk];
            idx = work->nodeIdx[ii][kk+1];
            // --- calculate x_opt
            if (kk < Nh-1) {
                idxp1 = work->nodeIdx[ii][kk+2];
                blasfeo_dgemv_t(nx, nx, -1.0, &sA[idxp1-1], 0, 0, &work->smu[ii][kk+1],
                    0, 1.0, &work->smu[ii][kk], 0, &work->sxUnc[ii][kk], 0);
            } else {
                blasfeo_dveccp(nx, &work->smu[ii][kk], 0, &work->sxUnc[ii][kk], 0);
            }
            blasfeo_daxpy(nx, -1.0, &sq[idx], 0, &work->sxUnc[ii][kk], 0, &work->sxUnc[ii][kk], 0);

            blasfeo_dvecmuldot(nx, &sQinv[idx], 0, &work->sxUnc[ii][kk], 0, &work->sxUnc[ii][kk], 0);

            if (work->boundsRemoved[ii][kk+1] != 1) {
                blasfeo_dveccl(nx, &sxmin[idx], 0, &work->sxUnc[ii][kk], 0,
                    &sxmax[idx], 0, &work->sx[ii][kk], 0);
            } else {
                blasfeo_dveccp(nx, &work->sxUnc[ii][kk], 0, &work->sx[ii][kk], 0);
            }

            // --- calculate u_opt
            blasfeo_dgemv_t(nx, nu, -1.0, &sB[idx-1], 0, 0, &work->smu[ii][kk], 0, -1.0,
                &sr[idxm1], 0, &work->suUnc[ii][kk], 0);

            if ((ii < Ns-1) && (kk < commonNodes[ii])) {
                blasfeo_daxpy(nu, -1.0, &slambda[ii], kk*nu, &work->suUnc[ii][kk], 0,
                    &work->suUnc[ii][kk], 0);
            }
            if ((ii > 0) && (kk < commonNodes[ii-1])) {
                blasfeo_daxpy(nu, 1.0, &slambda[ii-1], kk*nu, &work->suUnc[ii][kk], 0,
                    &work->suUnc[ii][kk], 0);
            }
            blasfeo_dvecmuldot(nu, &sRinv[idxm1], 0, &work->suUnc[ii][kk], 0,
                &work->suUnc[ii][kk], 0);

            if (work->boundsRemoved[ii][kk] != 1) {
                blasfeo_dveccl(nu, &sumin[idxm1], 0, &work->suUnc[ii][kk], 0,
                    &sumax[idxm1], 0, &work->su[ii][kk], 0);
            } else {
                blasfeo_dveccp(nu, &work->suUnc[ii][kk], 0, &work->su[ii][kk], 0);
            }

            #ifndef NEW_FVAL
            // --- recalculate residual
            if (kk == 0) {
                // res[1] = -b[0] - B[0] * u[0]
                blasfeo_dgemv_n(nx, nu, -1.0, &sB[idx-1], 0, 0, &work->su[ii][0], 0, -1.0,
                    &sb[idx-1], 0, &work->sresk[ii][0], 0);
                // res[1] = res[1] + x[1]
                blasfeo_daxpy(nx, 1.0, &work->sx[ii][0], 0, &work->sresk[ii][0], 0,
                    &work->sresk[ii][0], 0);
            } else {
                // res[k+1] = x[k+1] - A[k] * x[k]
                blasfeo_dgemv_n(nx, nx, -1.0, &sA[idx-1], 0, 0, &work->sx[ii][kk-1], 0, 1.0,
                    &work->sx[ii][kk], 0, &work->sresk[ii][kk], 0);
                // res[k+1] = res[k+1] - B[k] * u[k]
                blasfeo_dgemv_n(nx, nu, -1.0, &sB[idx-1], 0, 0, &work->su[ii][kk], 0, 1.0,
                    &work->sresk[ii][kk], 0, &work->sresk[ii][kk], 0);
                // res[k+1] = res[k+1] - b[k]
                blasfeo_daxpy(nx, -1.0, &sb[idx-1], 0,
                    &work->sresk[ii][kk], 0, &work->sresk[ii][kk], 0);
            }
            #endif

            #ifdef NEW_FVAL

            // fval[i] -= (1/2)x[k+1]' * Q[k+1] * (x[k+1] - 2*xUnc[k+1])
            blasfeo_daxpy(nx, -2.0, &work->sxUnc[ii][kk], 0, &work->sx[ii][kk], 0,
                &work->sxUnc[ii][kk], 0);
            blasfeo_dvecmuldot(nx, &sQ[idx], 0, &work->sxUnc[ii][kk], 0, &work->sxUnc[ii][kk], 0);

            fvals[ii] -= 0.5*blasfeo_ddot(nx, &work->sx[ii][kk], 0, &work->sxUnc[ii][kk], 0);

            // fval[i] -= (1/2)u[k] * R[k] * (u[k] - 2*uUnc[k])
            blasfeo_daxpy(nu, -2.0, &work->suUnc[ii][kk], 0, &work->su[ii][kk], 0,
                &work->suUnc[ii][kk], 0);
            blasfeo_dvecmuldot(nu, &sR[idxm1], 0, &work->suUnc[ii][kk], 0, &work->suUnc[ii][kk], 0);

            fvals[ii] -= 0.5*blasfeo_ddot(nu, &work->su[ii][kk], 0, &work->suUnc[ii][kk], 0);

            // fval[i] -= b[k]' *  mu[k+1]
            fvals[ii] -= blasfeo_ddot(nx, &sb[idx-1], 0, &work->smu[ii][kk], 0);

            #else
            // fval = - (1/2)x[k+1]' * Q[k+1] * x[k+1] - x[k+1]' * q[k+1]
            blasfeo_dvecmuldot(nx, &sQ[idx], 0, &work->sx[ii][kk], 0,
                &sTmpVecs[ii], 0);
            fvals[ii] -= 0.5*blasfeo_ddot(nx, &sTmpVecs[ii], 0, &work->sx[ii][kk], 0);
            fvals[ii] -= blasfeo_ddot(nx, &sq[idx], 0, &work->sx[ii][kk], 0);
            // fval -= (1/2)u[k]' * R[k] * u[k] + u[k]' * r[k]
            blasfeo_dvecmuldot(nu, &sR[idxm1], 0, &work->su[ii][kk], 0, &sTmpVecs[ii], 0);
            fvals[ii] -= 0.5*blasfeo_ddot(nu, &sTmpVecs[ii], 0, &work->su[ii][kk], 0);
            fvals[ii] -= blasfeo_ddot(nu, &sr[idxm1], 0, &work->su[ii][kk], 0);
            // fval += mu[k]' * res[k] => fval -= mu[k]' * (-x[k+1] + A[k]*x[k] + B[k]*u[k] + b[k])
            fvals[ii] += blasfeo_ddot(nx, &work->smu[ii][kk], 0, &work->sresk[ii][kk], 0);
            #endif
        }
    }

    #ifndef NEW_FVAL
    calculate_last_residual(Ns, Nh, work);
    for (ii = 0; ii < Ns - 1; ii++) {
        fvals[ii] += blasfeo_ddot(slambda[ii].m, &slambda[ii], 0, &sResNonAnticip[ii], 0);
    }
    #endif

    for (ii = 0; ii < Ns; ii++) fval += fvals[ii];
    return fval;
}


int line_search(int Ns, int Nh, tree_ocp_qp_in *qp_in, treeqp_sdunes_options_t *opts,
    treeqp_sdunes_workspace *work) {

    int ii, jj, kk;
    double dotProduct, fval, fval0;
    double tau = 1;
    double tauPrev = 0;
    int nx = work->sx[0][0].m;

    struct blasfeo_dvec *sDeltalambda = work->sDeltalambda;
    struct blasfeo_dvec *slambda = work->slambda;

    dotProduct = gradient_trans_times_direction(Ns, Nh, work);
    fval0 = evaluate_dual_function(Ns, Nh, qp_in, work);

    for (jj = 1; jj <= opts->lineSearchMaxIter; jj++) {
        // printf("LS iteration #%d\n", jj+1);
        // update multipliers

        #ifdef PARALLEL
        #ifdef SPLIT_NODES
        #pragma omp parallel for
        #else
        #pragma omp parallel for private(kk)
        #endif
        #endif
        for (ii = 0; ii < Ns; ii++) {
            // printf("scenario %d\n",ii);
            #ifdef PARALLEL
            #ifdef SPLIT_NODES
            #pragma omp parallel for
            #endif
            #endif
            for (kk = 0; kk < Nh; kk++) {
                // mu[k] = mu[k] + (tau-tauPrev)*Deltamu[k]
                blasfeo_daxpy(nx, tau-tauPrev, &work->sDeltamu[ii][kk], 0,
                    &work->smu[ii][kk], 0, &work->smu[ii][kk], 0);
                // blasfeo_print_dvec(nx, &work->smu[ii][kk],0);
            }
            if (ii < Ns-1) {
                blasfeo_daxpy(sDeltalambda[ii].m, tau-tauPrev, &sDeltalambda[ii], 0,
                    &slambda[ii], 0, &slambda[ii], 0);
                // blasfeo_print_dvec(slambda[ii].m, &slambda[ii],0);
            }
        }
        // evaluate dual function
        fval = evaluate_dual_function(Ns, Nh, qp_in, work);

        // check condition
        if (fval <= fval0 + opts->lineSearchGamma*tau*dotProduct) {
            // printf("Condition satisfied\n");
            break;
        } else {
            tauPrev = tau;
            tau = opts->lineSearchBeta*tauPrev;
        }
    }
    #ifdef SAVE_DATA
    write_double_vector_to_txt(&dotProduct, 1, "examples/spring_mass_utils/dotProduct.txt");
    write_double_vector_to_txt(&fval0, 1, "examples/spring_mass_utils/fval0.txt");
    #endif

    return jj;
}


// TODO(dimitris): time and see if it's worth to parallelize
double calculate_error_in_residuals(int Ns, int Nh, termination_t condition,
    treeqp_sdunes_workspace *work) {

    int ii, jj, kk;
    double error = 0;
    int nx = work->sx[0][0].m;

    struct blasfeo_dvec *sResNonAnticip = work->sResNonAnticip;

    if ((condition == TREEQP_SUMSQUAREDERRORS) || (condition == TREEQP_TWONORM)) {
        for (ii = 0; ii < Ns; ii++) {
            for (kk = 0; kk < Nh; kk++) {
                error += blasfeo_ddot(nx, &work->sresk[ii][kk], 0,
                    &work->sresk[ii][kk], 0);
            }
            if (ii < Ns-1) {
                error += blasfeo_ddot(sResNonAnticip[ii].m, &sResNonAnticip[ii], 0,
                    &sResNonAnticip[ii], 0);
            }
        }
        if (condition == TREEQP_TWONORM) error = sqrt(error);
    } else if (condition == TREEQP_INFNORM) {
        for (ii = 0; ii < Ns; ii++) {
            for (kk = 0; kk < Nh; kk++) {
                for (jj = 0; jj < work->sresk[ii][kk].m; jj++) {
                    error = MAX(error, ABS(BLASFEO_DVECEL(&work->sresk[ii][kk], jj)));
                }
            }
            if (ii < Ns-1) {
                for (jj = 0; jj < sResNonAnticip[ii].m; jj++) {
                    error = MAX(error, ABS(BLASFEO_DVECEL(&sResNonAnticip[ii], jj)));
                }
            }
        }
    } else {
        printf("[TREEQP] Unknown termination condition!\n");
        exit(1);
    }
    return error;
}


int treeqp_dune_scenarios_calculate_size(tree_ocp_qp_in *qp_in, treeqp_sdunes_options_t *opts)
{
    struct node *tree = (struct node *) qp_in->tree;
    int nx = qp_in->nx[1];
    int nu = qp_in->nu[0];
    int Nn = qp_in->N;
    int Nh = tree[Nn-1].stage;
    int Np = get_number_of_parent_nodes(Nn, tree);
    int Ns = Nn - Np;
    int Nr = get_robust_horizon(Nn, tree);

    int commonNodes, commonNodesNxt, maxTmpDim;
    int commonNodesMax = 0;
    int bytes = 0;

    // TODO(dimitris): run consistency checks on tree to see if compatible with algorithm

    bytes += 2*Ns*sizeof(int*);  // **nodeIdx, **boundsRemoved
    bytes += 2*Ns*(Nh+1)*sizeof(int);
    if (opts->checkLastActiveSet)
    {
        bytes += 2*Ns*sizeof(int*);  // **xasChanged, **uasChanged
        bytes += 2*Ns*(Nh+1)*sizeof(int);
    }

    bytes += (Ns-1)*sizeof(int);  // *commonNodes
    bytes += Ns*sizeof(double);  // *fvals

    // double struct pointers
    bytes += 6*Ns*sizeof(struct blasfeo_dvec*);  // x, xUnc, xas, u, uUnc, uas
    bytes += 6*Ns*Nh*sizeof(struct blasfeo_dvec);
    bytes += 2*Ns*sizeof(struct blasfeo_dvec*);  // QinvCal, RinvCal
    bytes += 2*Ns*Nh*sizeof(struct blasfeo_dvec);
    bytes += 4*Ns*sizeof(struct blasfeo_dvec*);  // res, resMod, mu, Deltamu
    bytes += 4*Ns*Nh*sizeof(struct blasfeo_dvec);
    bytes += Ns*sizeof(struct blasfeo_dmat*);  // Zbar
    bytes += Ns*Nh*sizeof(struct blasfeo_dmat);
    bytes += 2*Ns*sizeof(struct blasfeo_dmat*);  // LambdaD, CholLambdaD
    bytes += 2*Ns*Nh*sizeof(struct blasfeo_dmat);
    bytes += 2*Ns*sizeof(struct blasfeo_dmat*);  // LambdaL, CholLambdaL
    bytes += 2*Ns*(Nh-1)*sizeof(struct blasfeo_dmat);

    if (opts->checkLastActiveSet)
    {
        bytes += Ns*sizeof(struct blasfeo_dmat*);  // TmpLambdaD
        bytes += Ns*Nh*sizeof(struct blasfeo_dmat);
        bytes += 2*Ns*sizeof(struct blasfeo_dvec*);  // xasPrev, uasPrev
        bytes += 2*Ns*Nh*sizeof(struct blasfeo_dvec);
    }

    bytes += 3*Ns*Nh*blasfeo_memsize_dvec(nx);  // x, xUnc, xas
    bytes += 3*Ns*Nh*blasfeo_memsize_dvec(nu);  // u, uUnc, uas
    bytes += Ns*Nh*blasfeo_memsize_dvec(nx);  // QinvCal
    bytes += Ns*Nh*blasfeo_memsize_dvec(nu);  // RinvCal
    bytes += 4*Ns*Nh*blasfeo_memsize_dvec(nx);  // res, resMod, mu, Deltamu
    bytes += Ns*Nh*blasfeo_memsize_dmat(nx, nu);  // Zbar
    bytes += 2*Ns*Nh*blasfeo_memsize_dmat(nx, nx);  // LambdaD, CholLambdaD
    bytes += 2*Ns*(Nh-1)*blasfeo_memsize_dmat(nx, nx);  // LambdaL, CholLambdaL
    if (opts->checkLastActiveSet)
    {
        bytes += Ns*Nh*blasfeo_memsize_dmat(nx, nx);  // TmpLambdaD
        bytes += Ns*Nh*blasfeo_memsize_dvec(nx);  // xasPrev
        bytes += Ns*Nh*blasfeo_memsize_dvec(nu);  // uasPrev
    }

    // struct pointers
    bytes += 6*Nn*sizeof(struct blasfeo_dvec);  // Q, R, q, r, Qinv, Rinv
    bytes += 2*(Ns-1)*sizeof(struct blasfeo_dmat);  // JayD, CholJayD
    bytes += 2*(Ns-2)*sizeof(struct blasfeo_dmat);  // JayL, CholJayL
    bytes += 2*Ns*sizeof(struct blasfeo_dmat);  // Ut, K
    bytes += 2*(Ns-1)*sizeof(struct blasfeo_dvec);  // resNonAnticip, rhsNonAnticip
    bytes += 2*(Ns-1)*sizeof(struct blasfeo_dvec);  // lambda, Deltalambda
    bytes += 1*sizeof(struct blasfeo_dvec);  // regMat
    bytes += Ns*sizeof(struct blasfeo_dvec);  // tmpVecs
    bytes += Ns*sizeof(struct blasfeo_dmat);  // tmpMats

    for (int jj = 0; jj < Nn; jj++) {
        bytes += 3*blasfeo_memsize_dvec(qp_in->nx[jj]);  // Q, q, Qinv
        bytes += 3*blasfeo_memsize_dvec(qp_in->nu[jj]);  // R, r, Rinv
    }

    for (int ii = 0; ii < Ns-1; ii++) {
        commonNodes = get_number_of_common_nodes(Nn, Ns, Nh, ii, ii+1, tree);
        commonNodesMax = MAX(commonNodesMax, commonNodes);
        bytes += 2*blasfeo_memsize_dmat(nu*commonNodes, nu*commonNodes);  // JayD, CholJayD
        bytes += 2*blasfeo_memsize_dvec(nu*commonNodes);  // resNonAnticip, rhsNonAnticip
        bytes += 2*blasfeo_memsize_dvec(nu*commonNodes);  // lambda, Deltalambda
        if (ii < Ns-2) {
            commonNodesNxt = get_number_of_common_nodes(Nn, Ns, Nh, ii+1, ii+2, tree);
            bytes += 2*blasfeo_memsize_dmat(nu*commonNodesNxt, nu*commonNodes);  // JayL, CholJayL
        }
    }

    // maximum dimension of tmp vector to store intermediate results
    maxTmpDim = MAX(nx, nu*commonNodesMax);

    bytes += blasfeo_memsize_dvec(maxTmpDim);  // RegMat
    bytes += Ns*blasfeo_memsize_dvec(maxTmpDim);  // tmpVecs
    bytes += Ns*blasfeo_memsize_dmat(nu, nx);  // tmpMats

    bytes += Ns*blasfeo_memsize_dmat(nu*Nr, Nh*nx);  // Ut
    bytes += Ns*blasfeo_memsize_dmat(nu*Nr, nu*Nr);  // K

    make_int_multiple_of(64, &bytes);
    bytes += 1*64;

    return bytes;
}


void create_treeqp_dune_scenarios(tree_ocp_qp_in *qp_in, treeqp_sdunes_options_t *opts,
    treeqp_sdunes_workspace *work, void *ptr) {

    struct node *tree = (struct node *) qp_in->tree;
    int nx = qp_in->nx[1];
    int nu = qp_in->nu[0];
    int Nn = qp_in->N;
    int Nh = tree[Nn-1].stage;
    int Np = get_number_of_parent_nodes(Nn, tree);
    int Ns = Nn - Np;
    int Nr = get_robust_horizon(Nn, tree);

    int maxTmpDim, node, ans;
    int *commonNodes;

    // TODO(dimitris): move to workspace
    int *processedNodes = malloc(Nn*sizeof(int));
    int idx = 0;

    // store some dimensions in workspace
    work->Nr = Nr;
    work->Ns = Ns;
    work->Nh = Nh;
    work->md = tree[0].nkids;

    // char pointer
    char *c_ptr = (char *) ptr;

    // calculate number of common nodes between neighboring scenarios
    work->commonNodes = (int*) c_ptr;
    c_ptr += (Ns-1)*sizeof(int);

    commonNodes = work->commonNodes;
    build_vector_of_common_nodes(Nn, Ns, Nh, tree, commonNodes);

    // calculate maximum dimension of temp vectors
    maxTmpDim = get_maximum_vector_dimension(Ns, nx, nu, commonNodes);

    // allocate memory for the dual function evaluation of each scenario
    work->fvals = (double*) c_ptr;
    c_ptr += Ns*sizeof(double);

    // generate indexing of scenario nodes from tree and flags for removed bounds
    create_double_ptr_int(&work->nodeIdx, Ns, Nh+1, &c_ptr);
    create_double_ptr_int(&work->boundsRemoved, Ns, Nh+1, &c_ptr);

    for (int ii = 0; ii < Ns; ii++) {
        node = tree[Nn-Ns+ii].idx;
        work->nodeIdx[ii][Nh] = node;
        work->boundsRemoved[ii][Nh] = 0;
        for (int kk = Nh; kk > 0; kk--) {
            work->nodeIdx[ii][kk-1] = tree[node].dad;
            node = tree[node].dad;
            ans = node_processed(node, processedNodes, idx);
            if (ans == YES) {
                work->boundsRemoved[ii][kk-1] = 1;
            } else {
                work->boundsRemoved[ii][kk-1] = 0;
                processedNodes[idx++] = node;
            }
        }
    }

    // QP weights
    work->sQ = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    work->sR = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    work->sq = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    work->sr = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    work->sQinv = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    work->sRinv = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    // diagonal matrix (stored in vector) with regularization value
    work->regMat = (struct blasfeo_dvec *) c_ptr;
    c_ptr += 1*sizeof(struct blasfeo_dvec);
    // diagonal blocks of J (each symmetric of dimension nu*nc[k])
    work->sJayD = (struct blasfeo_dmat *) c_ptr;
    c_ptr += (Ns-1)*sizeof(struct blasfeo_dmat);
    // Cholesky factors of diagonal blocks
    work->sCholJayD = (struct blasfeo_dmat *) c_ptr;
    c_ptr += (Ns-1)*sizeof(struct blasfeo_dmat);
    // off-diagonal blocks of J (each of dim. nu*nc[k+1] x nu*nc[k])
    work->sJayL = (struct blasfeo_dmat *) c_ptr;
    c_ptr += (Ns-2)*sizeof(struct blasfeo_dmat);
    // Cholesky factors of off-diagonal blocks
    work->sCholJayL = (struct blasfeo_dmat *) c_ptr;
    c_ptr += (Ns-2)*sizeof(struct blasfeo_dmat);
    // Ut matrices to build K
    work->sUt = (struct blasfeo_dmat *) c_ptr;
    c_ptr += Ns*sizeof(struct blasfeo_dmat);
    // K matrices to build Jay
    work->sK = (struct blasfeo_dmat *) c_ptr;
    c_ptr += Ns*sizeof(struct blasfeo_dmat);
    //
    work->sResNonAnticip = (struct blasfeo_dvec *) c_ptr;
    c_ptr += (Ns-1)*sizeof(struct blasfeo_dvec);
    //
    work->sRhsNonAnticip = (struct blasfeo_dvec *) c_ptr;
    c_ptr += (Ns-1)*sizeof(struct blasfeo_dvec);
    // multipliers of non-anticipativity constraints
    work->slambda = (struct blasfeo_dvec *) c_ptr;
    c_ptr += (Ns-1)*sizeof(struct blasfeo_dvec);
    // step in multipliers of non-anticipativity constraints
    work->sDeltalambda = (struct blasfeo_dvec *) c_ptr;
    c_ptr += (Ns-1)*sizeof(struct blasfeo_dvec);

    // strmats/strvecs for intermediate results
    work->sTmpMats = (struct blasfeo_dmat *) c_ptr;
    c_ptr += Ns*sizeof(struct blasfeo_dmat);
    work->sTmpVecs = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Ns*sizeof(struct blasfeo_dvec);


    if (opts->checkLastActiveSet)
    {
        create_double_ptr_int(&work->xasChanged, Ns, Nh+1, &c_ptr);
        create_double_ptr_int(&work->uasChanged, Ns, Nh+1, &c_ptr);
    }

    create_double_ptr_strvec(&work->sx, Ns, Nh, &c_ptr);
    create_double_ptr_strvec(&work->su, Ns, Nh, &c_ptr);
    create_double_ptr_strvec(&work->sxas, Ns, Nh, &c_ptr);
    create_double_ptr_strvec(&work->suas, Ns, Nh, &c_ptr);
    create_double_ptr_strvec(&work->sxUnc, Ns, Nh, &c_ptr);
    create_double_ptr_strvec(&work->suUnc, Ns, Nh, &c_ptr);
    create_double_ptr_strvec(&work->sQinvCal, Ns, Nh, &c_ptr);
    create_double_ptr_strvec(&work->sRinvCal, Ns, Nh, &c_ptr);
    create_double_ptr_strvec(&work->sresk, Ns, Nh, &c_ptr);
    create_double_ptr_strvec(&work->sreskMod, Ns, Nh, &c_ptr);
    create_double_ptr_strvec(&work->smu, Ns, Nh, &c_ptr);
    create_double_ptr_strvec(&work->sDeltamu, Ns, Nh, &c_ptr);
    create_double_ptr_strmat(&work->sZbar, Ns, Nh, &c_ptr);
    create_double_ptr_strmat(&work->sLambdaD, Ns, Nh, &c_ptr);
    create_double_ptr_strmat(&work->sCholLambdaD, Ns, Nh, &c_ptr);
    create_double_ptr_strmat(&work->sLambdaL, Ns, Nh-1, &c_ptr);
    create_double_ptr_strmat(&work->sCholLambdaL, Ns, Nh-1, &c_ptr);
    if (opts->checkLastActiveSet)
    {
        create_double_ptr_strmat(&work->sTmpLambdaD, Ns, Nh, &c_ptr);
        create_double_ptr_strvec(&work->sxasPrev, Ns, Nh, &c_ptr);
        create_double_ptr_strvec(&work->suasPrev, Ns, Nh, &c_ptr);
    }

    // move pointer for proper alignment of blasfeo matrices and vectors
    align_char_to(64, &c_ptr);

    // strmats
    for (int ii = 0; ii < Ns; ii++)
    {
        init_strmat(nu*Nr, Nh*nx, &work->sUt[ii], &c_ptr);
        init_strmat(nu*Nr, nu*Nr, &work->sK[ii], &c_ptr);
        init_strmat(nu, nx, &work->sTmpMats[ii], &c_ptr);
        if (ii < Ns-1)
        {
            init_strmat(nu*commonNodes[ii], nu*commonNodes[ii], &work->sJayD[ii], &c_ptr);
            init_strmat(nu*commonNodes[ii], nu*commonNodes[ii], &work->sCholJayD[ii], &c_ptr);
        }
        if (ii < Ns-2)
        {
            init_strmat(nu*commonNodes[ii+1], nu*commonNodes[ii], &work->sJayL[ii], &c_ptr);
            init_strmat(nu*commonNodes[ii+1], nu*commonNodes[ii], &work->sCholJayL[ii], &c_ptr);
        }
    }

    for (int ii = 0; ii < Ns; ii++)
    {
        for (int kk = 0; kk < Nh; kk++)
        {
            init_strmat(nx, nu, &work->sZbar[ii][kk], &c_ptr);
            init_strmat(nx, nx, &work->sLambdaD[ii][kk], &c_ptr);
            init_strmat(nx, nx, &work->sCholLambdaD[ii][kk], &c_ptr);
            if (kk < Nh-1) init_strmat(nx, nx, &work->sLambdaL[ii][kk], &c_ptr);
            if (kk < Nh-1) init_strmat(nx, nx, &work->sCholLambdaL[ii][kk], &c_ptr);
            if (opts->checkLastActiveSet)
            {
                init_strmat(nx, nx, &work->sTmpLambdaD[ii][kk], &c_ptr);
            }
        }
    }

    // strvecs
    for (int ii = 0; ii < Ns; ii++)
    {
        init_strvec(maxTmpDim, &work->sTmpVecs[ii], &c_ptr);
        if (ii < Ns-1)
        {
            init_strvec(nu*commonNodes[ii], &work->sResNonAnticip[ii], &c_ptr);
            init_strvec(nu*commonNodes[ii], &work->sRhsNonAnticip[ii], &c_ptr);

            init_strvec(nu*commonNodes[ii], &work->slambda[ii], &c_ptr);
            init_strvec(nu*commonNodes[ii], &work->sDeltalambda[ii], &c_ptr);
        }
    }

    for (int ii = 0; ii < Ns; ii++)
    {
        for (int kk = 0; kk < Nh; kk++)
        {
            // NOTE(dimitris): all states are shifted by one after eliminating x0
            init_strvec(nx, &work->sx[ii][kk], &c_ptr);
            init_strvec(nu, &work->su[ii][kk], &c_ptr);
            init_strvec(nx, &work->sxUnc[ii][kk], &c_ptr);
            init_strvec(nu, &work->suUnc[ii][kk], &c_ptr);
            init_strvec(nx, &work->sxas[ii][kk], &c_ptr);
            init_strvec(nu, &work->suas[ii][kk], &c_ptr);
            init_strvec(nx, &work->sQinvCal[ii][kk], &c_ptr);
            init_strvec(nu, &work->sRinvCal[ii][kk], &c_ptr);
            init_strvec(nx, &work->smu[ii][kk], &c_ptr);
            init_strvec(nx, &work->sDeltamu[ii][kk], &c_ptr);
            init_strvec(nx, &work->sresk[ii][kk], &c_ptr);
            init_strvec(nx, &work->sreskMod[ii][kk], &c_ptr);
            if (opts->checkLastActiveSet)
            {
                init_strvec(nx, &work->sxasPrev[ii][kk], &c_ptr);
                init_strvec(nu, &work->suasPrev[ii][kk], &c_ptr);
            }
        }
    }

    init_strvec(maxTmpDim, work->regMat, &c_ptr);
    blasfeo_dvecse(maxTmpDim, opts->regValue, work->regMat, 0);

    for (int jj = 0; jj < Nn; jj++) {
        init_strvec(qp_in->nx[jj], &work->sQ[jj], &c_ptr);
        init_strvec(qp_in->nu[jj], &work->sR[jj], &c_ptr);
        init_strvec(qp_in->nx[jj], &work->sq[jj], &c_ptr);
        init_strvec(qp_in->nu[jj], &work->sr[jj], &c_ptr);
        init_strvec(qp_in->nx[jj], &work->sQinv[jj], &c_ptr);
        init_strvec(qp_in->nu[jj], &work->sRinv[jj], &c_ptr);
    }

    free(processedNodes);

    assert((char *)ptr + treeqp_dune_scenarios_calculate_size(qp_in, opts) >= c_ptr);
    // printf("memory starts at\t%p\nmemory ends at  \t%p\ndistance from the end\t%lu bytes\n",
    //     ptr, c_ptr, (char *)ptr + treeqp_dune_scenarios_calculate_size(qp_in, opts) - c_ptr);
    // exit(1);

    #ifndef REV_CHOL
    if (opts->checkLastActiveSet == 1)
    {
        printf("Error! check_last_active_set option only supported in combination with reverse Cholesky!\n");
    }
    #endif
}


int treeqp_dune_scenarios_solve(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out,
    treeqp_sdunes_options_t *opts, treeqp_sdunes_workspace *work) {

    int NewtonIter, lsIter;
    double error;
    int Nn = qp_in->N;
    int nu = qp_in->nu[0];
    int nx = qp_in->nx[1];
    return_t status = TREEQP_ERR_UNKNOWN_ERROR;

    int Nh = work->Nh;
    int Ns = work->Ns;
    int Nr = work->Nr;
    int md = work->md;

    struct blasfeo_dmat *sJayD = work->sJayD;
    struct blasfeo_dmat *sJayL = work->sJayL;

    struct blasfeo_dmat *sQnonScaled = (struct blasfeo_dmat*)qp_in->Q;
    struct blasfeo_dmat *sRnonScaled = (struct blasfeo_dmat*)qp_in->R;
    struct blasfeo_dvec *sqnonScaled = (struct blasfeo_dvec*)qp_in->q;
    struct blasfeo_dvec *srnonScaled = (struct blasfeo_dvec*)qp_in->r;

    // ------ initialization
    treeqp_timer timer;
    treeqp_tic(&timer);
    double scalingFactor;
    for (int jj = 0; jj < Nn; jj++) {
        // NOTE(dimitris): inverse of scaling factor in tree_ocp_qp_in_fill_lti_data
        scalingFactor = (double)ipow(md, MIN(qp_in->tree[jj].stage, Nr))/ipow(md, Nr);

        blasfeo_ddiaex(qp_in->nx[jj], scalingFactor, &sQnonScaled[jj], 0, 0, &work->sQ[jj], 0);
        blasfeo_ddiaex(qp_in->nu[jj], scalingFactor, &sRnonScaled[jj], 0, 0, &work->sR[jj], 0);

        blasfeo_dveccpsc(qp_in->nx[jj], scalingFactor, &sqnonScaled[jj], 0, &work->sq[jj], 0);
        blasfeo_dveccpsc(qp_in->nu[jj], scalingFactor, &srnonScaled[jj], 0, &work->sr[jj], 0);
        for (int nn = 0; nn < qp_in->nx[jj]; nn++)
            BLASFEO_DVECEL(&work->sQinv[jj], nn) = 1.0/BLASFEO_DVECEL(&work->sQ[jj], nn);
        for (int nn = 0; nn < qp_in->nu[jj]; nn++)
            BLASFEO_DVECEL(&work->sRinv[jj], nn) = 1.0/BLASFEO_DVECEL(&work->sR[jj], nn);
    }

    int idx, idxm1, idxp1;
    for (int ii = 0; ii < Ns; ii++) {
        for (int kk = 0; kk < Nh; kk++) {
            idx = work->nodeIdx[ii][kk];
            idxp1 = work->nodeIdx[ii][kk+1];

            // NOTE(dimitris): QinvCal and RinvCal of nodes with removed bounds never change
            blasfeo_dveccp(nx, &work->sQinv[idxp1], 0, &work->sQinvCal[ii][kk], 0);
            blasfeo_dveccp(nu, &work->sRinv[idx], 0, &work->sRinvCal[ii][kk], 0);

            if (opts->checkLastActiveSet)
            {
                // NOTE(dimitris): setting value outside {-1,0,1} to force full factorization at 1st it.
                blasfeo_dvecse(nx, 0.0/0.0, &work->sxasPrev[ii][kk], 0);
                blasfeo_dvecse(nu, 0.0/0.0, &work->suasPrev[ii][kk], 0);
            }
        }
    }
    double init_time = treeqp_toc(&timer);
    // printf("init. time = %f ms\n", 1e3*init_time);

    // ------ dual Newton iterations
    // NOTE(dimitris): at first iteration some matrices are initialized for opts->checkLastActiveSet
    for (NewtonIter = 0; NewtonIter < opts->maxIter; NewtonIter++) {
        #if PROFILE > 1
        treeqp_tic(&iter_tmr);
        #endif

        // --- solve stage QPs
        // - calculate unconstrained solution of stage QPs
        // - clip solution
        // - calculate Zbar
        // - calculate LambdaD and LambdaL
        #if PROFILE > 2
        treeqp_tic(&tmr);
        #endif
        solve_stage_problems(Ns, Nh, NewtonIter, qp_in, work, opts);
        #if PROFILE > 2
        stage_qps_times[NewtonIter] = treeqp_toc(&tmr);
        #endif

        // --- calculate dual gradient
        #if PROFILE > 2
        treeqp_tic(&tmr);
        #endif
        // TODO(dimitris): benchmark in linux, see if I can avoid some ifs
        calculate_residuals(Ns, Nh, qp_in, work);
        calculate_last_residual(Ns, Nh, work);
        // NOTE(dimitris): cannot parallelize last residual so better to keep the two loops separate

        // TODO(dimitris): move call inside function
        #ifdef SAVE_DATA
        save_stage_problems(Ns, Nh, Nr, md, work);
        #endif

        error = calculate_error_in_residuals(Ns, Nh, opts->termCondition, work);

        // printf("\n-------- qpdunes iteration %d, error %f \n\n", NewtonIter+1, error);
        if (error < opts->stationarityTolerance) {
            // printf("optimal solution found (error = %5.2e)\n", error);
            status = TREEQP_SUCC_OPTIMAL_SOLUTION_FOUND;
            break;
        }

        // NOTE(dimitris): inaccurate since part of dual Hessian is calculated while solving
        // the stage QPs
        #if PROFILE > 2
        build_dual_times[NewtonIter] = treeqp_toc(&tmr);
        #endif

        #if PROFILE > 2
        treeqp_tic(&tmr);
        #endif

        // --- factorize Newton system
        factorize_Lambda(Ns, Nh, opts, work);
        form_K(Ns, Nh, Nr, work);
        form_and_factorize_Jay(Ns, nu, opts, work);

        // --- calculate multipliers of non-anticipativity constraints
        form_RHS_non_anticipaticity(Ns, Nh, Nr, md, work);
        calculate_delta_lambda(Ns, Nr, md, work);

        // --- calculate multipliers of dynamics
        calculate_delta_mu(Ns, Nh, Nr, work);

        #if PROFILE > 2
        newton_direction_times[NewtonIter] = treeqp_toc(&tmr);
        #endif

        // --- line search
        #if PROFILE > 2
        treeqp_tic(&tmr);
        #endif

        lsIter = line_search(Ns, Nh, qp_in, opts, work);

        #if PROFILE > 2
        line_search_times[NewtonIter] = treeqp_toc(&tmr);
        #endif

        // --- reset data for next iteration
        // TODO(dimitris): check if it's worth parallelizing!
        for (int ii = 0; ii < Ns-1; ii++) {
            blasfeo_dgese(sJayD[ii].m, sJayD[ii].n, 0.0, &sJayD[ii], 0, 0);
        }
        for (int ii = 0; ii < Ns-2; ii++) {
            blasfeo_dgese(sJayL[ii].m, sJayL[ii].n, 0.0, &sJayL[ii], 0, 0);
        }
        #if PRINT_LEVEL > 1
        printf("iteration #%d: %d ls iterations \t\t(error %5.2e)\n", NewtonIter, lsIter, error);
        #endif
        #if PROFILE > 1
        iter_times[NewtonIter] = treeqp_toc(&iter_tmr);
        ls_iters[NewtonIter] = lsIter;
        #endif
    }

    // ------ copy solution to qp_out

    for (int ii = 0; ii < qp_in->N; ii++) {
        blasfeo_dvecse(nx, 0.0, &qp_out->lam[ii], 0);
    }

    for (int ii = 0; ii < Ns; ii++) {
        for (int kk = 0; kk < Nh; kk++) {
            idx = work->nodeIdx[ii][kk+1];
            idxm1 = work->nodeIdx[ii][kk];
            idxp1 = work->nodeIdx[ii][kk+2];
            blasfeo_daxpy(nx, 1.0, &work->smu[ii][kk], 0, &qp_out->lam[idx], 0, &qp_out->lam[idx], 0);
            if (work->boundsRemoved[ii][kk+1] == 0) {
                // printf("saving node (%d, %d) to node %d\n", ii, kk+1, work->nodeIdx[ii][kk+1]);
                blasfeo_dveccp(nx, &work->sx[ii][kk], 0, &qp_out->x[idx], 0);
                #ifdef NEW_FVAL
                // TODO(dimitris): this is copy-paste from evaluate dual function..
                if (kk < Nh-1) {
                    blasfeo_dgemv_t(nx, nx, -1.0, &qp_in->A[idxp1-1], 0, 0, &work->smu[ii][kk+1],
                    0, 1.0, &work->smu[ii][kk], 0, &work->sxUnc[ii][kk], 0);
                } else {
                    blasfeo_dveccp(nx, &work->smu[ii][kk], 0, &work->sxUnc[ii][kk], 0);
                }
                blasfeo_daxpy(nx, -1.0, &work->sq[idx], 0, &work->sxUnc[ii][kk], 0, &work->sxUnc[ii][kk], 0);
                blasfeo_dvecmuldot(nx, &work->sQinv[idx], 0, &work->sxUnc[ii][kk], 0, &work->sxUnc[ii][kk], 0);
                #endif
                blasfeo_daxpy(nx, -1., &qp_out->x[idx], 0, &work->sxUnc[ii][kk], 0, &qp_out->mu_x[idx], 0);
                blasfeo_dvecmuldot(nx, &work->sQ[idx], 0, &qp_out->mu_x[idx], 0, &qp_out->mu_x[idx], 0);
            }
            if (work->boundsRemoved[ii][kk] == 0) {
                blasfeo_dveccp(nu, &work->su[ii][kk], 0, &qp_out->u[idxm1], 0);
                #ifdef NEW_FVAL
                blasfeo_dgemv_t(nx, nu, -1.0, &qp_in->B[idx-1], 0, 0, &work->smu[ii][kk], 0, -1.0,
                    &work->sr[idxm1], 0, &work->suUnc[ii][kk], 0);
                if ((ii < Ns-1) && (kk < work->commonNodes[ii])) {
                    blasfeo_daxpy(nu, -1.0, &work->slambda[ii], kk*nu, &work->suUnc[ii][kk], 0,
                        &work->suUnc[ii][kk], 0);
                }
                if ((ii > 0) && (kk < work->commonNodes[ii-1])) {
                    blasfeo_daxpy(nu, 1.0, &work->slambda[ii-1], kk*nu, &work->suUnc[ii][kk], 0,
                        &work->suUnc[ii][kk], 0);
                }
                blasfeo_dvecmuldot(nu, &work->sRinv[idxm1], 0, &work->suUnc[ii][kk], 0,
                    &work->suUnc[ii][kk], 0);
                #endif
                blasfeo_daxpy(nu, -1., &qp_out->u[idxm1], 0, &work->suUnc[ii][kk], 0, &qp_out->mu_u[idxm1], 0);
                blasfeo_dvecmuldot(nu, &work->sR[idxm1], 0, &qp_out->mu_u[idxm1], 0, &qp_out->mu_u[idxm1], 0);
            }
        }
    }
    qp_out->info.iter = NewtonIter;

    if (qp_out->info.iter == opts->maxIter)
        status = TREEQP_ERR_MAXIMUM_ITERATIONS_REACHED;

    return status;
}


void treeqp_sdunes_set_dual_initialization(double *lam, double *mu, treeqp_sdunes_workspace *work) {
    int Ns = work->Ns;
    int Nh = work->Nh;
    int nu = work->su[0][0].m;
    int nx = work->sx[0][1].m;
    int indx;

    indx = 0;
    for (int ii = 0; ii < Ns-1; ii++)
    {
        blasfeo_pack_dvec(work->slambda[ii].m, &lam[indx], &work->slambda[ii], 0);
        indx += work->slambda[ii].m;
    }

    indx = 0;
    for (int ii = 0; ii < Ns; ii++)
    {
        for (int kk = 0; kk < Nh; kk++)
        {
            blasfeo_pack_dvec(nx, &mu[indx], &work->smu[ii][kk], 0);
            indx += nx;
        }
    }
}