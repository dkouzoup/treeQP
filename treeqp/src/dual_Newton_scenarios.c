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

// TODO(dimitris): Check if merging all loops wrt scenarios improves openmp siginficantly

#ifdef PARALLEL
#include <omp.h>
#endif

#include "treeqp/src/dual_Newton_scenarios.h"
#include "treeqp/src/tree_ocp_qp_common.h"

#include "treeqp/flags.h"

#include "treeqp/utils/types.h"
#include "treeqp/utils/blasfeo_utils.h"
#include "treeqp/utils/profiling_utils.h"
#include "treeqp/utils/tree_utils.h"
#include "treeqp/utils/utils.h"
#include "treeqp/utils/timing.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_v_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blas.h"

#ifdef _CHECK_LAST_ACTIVE_SET_
#define REV_CHOL
#endif

#define NEW_FVAL
#define SPLIT_NODES

void check_compiler_flags() {
    #ifdef PARALLEL
    #if DEBUG == 1
    // TODO(dimitris): is this for sure not possible?
    printf("\nError! Can't do detailed debugging in parallel mode\n");
    exit(66);
    #endif
    #if PROFILE > 3
    printf("\nError! Can't do detailed profiling in parallel mode\n");
    exit(66);
    #endif
    #endif
    #if PRINT_LEVEL == 2 && PROFILE > 0
    printf("\nWarning! Printing hinders timings\n");
    #endif
}


int_t calculate_dimension_of_lambda(int_t Nr, int_t md, int_t nu) {
    int_t Ns = ipow(md, Nr);

    if (Ns == 1) {
        return -1;
    } else {
        return (Nr*Ns - (Ns-1)/(md-1))*nu;
    }
}


int_t get_maximum_vector_dimension(int_t Ns, int_t nx, int_t nu, int_t *commonNodes) {
    int_t ii;
    int_t maxDimension = nx;

    for (ii = 0; ii < Ns-1; ii++) {
        if (maxDimension < nu*commonNodes[ii]) maxDimension = nu*commonNodes[ii];
    }
    return maxDimension;
}


#if DEBUG == 1

int_t get_size_of_JayD(int_t Ns, int_t nu, int_t *commonNodes) {
    int_t ii;
    int_t size = 0;

    for (ii = 0; ii < Ns-1; ii++) {
        size += ipow(nu*commonNodes[ii], 2);
    }
    return size;
}


int_t get_size_of_JayL(int_t Ns, int_t nu, int_t *commonNodes) {
    int_t ii;
    int_t size = 0;

    for (ii = 0; ii < Ns-2; ii++) {
        size += nu*commonNodes[ii+1]*nu*commonNodes[ii];
    }
    return size;
}


void save_stage_problems(int_t Ns, int_t Nh, int_t Nr, int_t md,
    treeqp_dune_scenarios_workspace *work) {

    int_t ii, kk;
    int_t nu = work->su[0][0].m;
    int_t nx = work->sx[0][0].m;
    int_t nl = calculate_dimension_of_lambda(Nr, md, nu);
    real_t residuals_k[Ns*Nh*nx], residual[nl], xit[Ns*Nh*nx], uit[Ns*Nh*nu];
    real_t QinvCal_k[Ns*Nh*nx], RinvCal_k[Ns*Nh*nu];
    real_t LambdaD[Ns*Nh*nx*nx], LambdaL[Ns*(Nh-1)*nx*nx];
    int_t indRes = 0;
    int_t indResNonAnt = 0;
    int_t indX = 0;
    int_t indU = 0;
    int_t indLambdaD = 0;
    int_t indLambdaL = 0;
    int_t indZnk = 0;
    int_t *commonNodes = work->commonNodes;

    struct d_strvec *sResNonAnticip = work->sResNonAnticip;

    for (ii = 0; ii < Ns; ii++) {
        for (kk = 0; kk < Nh; kk++) {
            d_cvt_strvec2vec(work->sresk[ii][kk].m, &work->sresk[ii][kk], 0,
                &residuals_k[indRes]);
            d_cvt_strvec2vec(nx, &work->sx[ii][kk], 0, &xit[indX]);
            d_cvt_strvec2vec(nu, &work->su[ii][kk], 0, &uit[indU]);
            d_cvt_strvec2vec(nx, &work->sQinvCal[ii][kk], 0, &QinvCal_k[indX]);
            d_cvt_strvec2vec(nu, &work->sRinvCal[ii][kk], 0, &RinvCal_k[indU]);
            d_cvt_strmat2mat(nx, nx, &work->sLambdaD[ii][kk], 0, 0, &LambdaD[indLambdaD], nx);
            if (kk < Nh-1) {
                d_cvt_strmat2mat(nx, nx, &work->sLambdaL[ii][kk], 0, 0,
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
            d_cvt_strvec2vec(sResNonAnticip[ii].m, &sResNonAnticip[ii], 0, &residual[indResNonAnt]);
            indResNonAnt += nu*commonNodes[ii];
        }
    }

    write_double_vector_to_txt(residuals_k, Ns*Nh*nx, "examples/data_spring_mass/resk.txt");
    write_double_vector_to_txt(residual, nl, "examples/data_spring_mass/res.txt");
    write_double_vector_to_txt(xit, Ns*Nh*nx, "examples/data_spring_mass/xit.txt");
    write_double_vector_to_txt(uit, Ns*Nh*nu, "examples/data_spring_mass/uit.txt");
    write_double_vector_to_txt(QinvCal_k, Ns*Nh*nx, "examples/data_spring_mass/Qit.txt");
    write_double_vector_to_txt(RinvCal_k, Ns*Nh*nu, "examples/data_spring_mass/Rit.txt");
    write_double_vector_to_txt(LambdaD, Ns*Nh*nx*nx, "examples/data_spring_mass/LambdaD.txt");
    write_double_vector_to_txt(LambdaL, Ns*(Nh-1)*nx*nx, "examples/data_spring_mass/LambdaL.txt");
}

#endif


void write_solution_to_txt(int_t Ns, int_t Nh, int_t Nr, int_t md, int_t nx, int_t nu,
    int_t NewtonIter, treeqp_dune_scenarios_workspace *work) {

    int ii, kk;

    struct d_strvec *slambda = work->slambda;

    int_t indMu = 0;
    int_t indx = 0;
    int_t indu = 0;
    int_t indLambda = 0;
    int_t nl = calculate_dimension_of_lambda(Nr, md, nu);
    real_t *muIter = malloc(Ns*Nh*nx*sizeof(real_t));
    real_t *xIter = malloc(Ns*Nh*nx*sizeof(real_t));
    real_t *uIter = malloc(Ns*Nh*nu*sizeof(real_t));
    real_t *lambdaIter = malloc(nl*sizeof(real_t));

    for (ii = 0; ii < Ns; ii++) {
        for (kk = 0; kk < Nh; kk++) {
            d_cvt_strvec2vec(nx, &work->smu[ii][kk], 0, &muIter[indMu]);
            indMu += nx;
            d_cvt_strvec2vec(nx, &work->sx[ii][kk], 0, &xIter[indx]);
            indx += nx;
            d_cvt_strvec2vec(nu, &work->su[ii][kk], 0, &uIter[indu]);
            indu += nu;
        }
        if (ii < Ns-1) {
            d_cvt_strvec2vec(slambda[ii].m, &slambda[ii], 0, &lambdaIter[indLambda]);
            indLambda += slambda[ii].m;
        }
    }
    write_double_vector_to_txt(lambdaIter, nl, "examples/data_spring_mass/lambdaIter.txt");
    write_double_vector_to_txt(muIter, Ns*Nh*nx, "examples/data_spring_mass/muIter.txt");
    write_double_vector_to_txt(xIter, Ns*Nh*nx, "examples/data_spring_mass/xIter.txt");
    write_double_vector_to_txt(uIter, Ns*Nh*nu, "examples/data_spring_mass/uIter.txt");
    write_int_vector_to_txt(&NewtonIter, 1, "examples/data_spring_mass/iter.txt");

    #if PROFILE > 0
    write_timers_to_txt();
    #endif

    free(muIter);
    free(xIter);
    free(uIter);
    free(lambdaIter);
}


answer_t node_processed(int_t node, int_t *processedNodes, int_t indx) {
    int_t ii;

    for (ii = 0; ii < indx; ii++) {
        if (node == processedNodes[ii]) return YES;
    }
    return NO;
}


int_t get_number_of_common_nodes(int_t Nn, int_t Ns, int_t Nh, int_t idx1, int_t idx2,
    struct node *tree) {

    int_t kk;
    int_t commonNodes = -1;
    int_t node1 = tree[Nn-Ns+idx1].idx;
    int_t node2 = tree[Nn-Ns+idx2].idx;

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


void build_vector_of_common_nodes(int_t Nn, int_t Ns, int_t Nh, struct node *tree,
    int_t *commonNodes) {
    int ii;

    for (ii = 0; ii < Ns-1; ii++) {
         commonNodes[ii] = get_number_of_common_nodes(Nn, Ns, Nh, ii, ii+1, tree);
    }
}

#ifdef _CHECK_LAST_ACTIVE_SET_

int_t compare_with_previous_active_set(int_t n, struct d_strvec *asNow, struct d_strvec *asBefore) {
    int_t ii;
    int_t changed = 0;

    for (ii = 0; ii < n; ii++) {
        if (DVECEL_LIBSTR(asNow, ii) != DVECEL_LIBSTR(asBefore, ii)) {
            changed = 1;
            break;
        }
    }
    dveccp_libstr(n, asNow, 0, asBefore, 0);
    return changed;
}

#endif


// TODO(dimitris): avoid some ifs in the loop maybe?
static void solve_stage_problems(int_t Ns, int_t Nh, int_t NewtonIter, tree_ocp_qp_in *qp_in,
    treeqp_dune_scenarios_workspace *work) {

    int_t ii, kk, idx, idxp1, idxm1;
    int_t nu = work->su[0][0].m;
    int_t nx = work->sx[0][0].m;
    int_t *commonNodes = work->commonNodes;
    struct d_strmat *sA = (struct d_strmat *) qp_in->A;
    struct d_strmat *sB = (struct d_strmat *) qp_in->B;
    struct d_strvec *sq = (struct d_strvec *) qp_in->q;
    struct d_strvec *sr = (struct d_strvec *) qp_in->r;
    struct d_strvec *sQinv = (struct d_strvec *) qp_in->Qinv;
    struct d_strvec *sRinv = (struct d_strvec *) qp_in->Rinv;
    struct d_strvec *sxmin = (struct d_strvec *) qp_in->xmin;
    struct d_strvec *sxmax = (struct d_strvec *) qp_in->xmax;
    struct d_strvec *sumin = (struct d_strvec *) qp_in->umin;
    struct d_strvec *sumax = (struct d_strvec *) qp_in->umax;

    struct d_strvec *slambda = work->slambda;
    int_t acc = 0;

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
            #if PROFILE > 3
            treeqp_tic(&sub_tmr);
            #endif
            idxm1 = work->nodeIdx[ii][kk];
            idx = work->nodeIdx[ii][kk+1];
            if (kk < Nh-1) {
                // x[k+1] = mu[k+1] - A[k+1]' * mu[k+2]
                idxp1 = work->nodeIdx[ii][kk+2];
                dgemv_t_libstr(nx, nx, -1.0, &sA[idxp1-1], 0, 0, &work->smu[ii][kk+1],
                    0, 1.0, &work->smu[ii][kk], 0, &work->sx[ii][kk], 0);
            } else {
                // x[Nh] = mu[Nh]
                dveccp_libstr(nx, &work->smu[ii][kk], 0, &work->sx[ii][kk], 0);
            }
            // x[k+1] = x[k+1] - q[k+1]
            daxpy_libstr(nx, -1.0, &sq[idx], 0, &work->sx[ii][kk], 0, &work->sx[ii][kk], 0);

            // x[k+1] = Q[k+1]^-1 .* x[k+1]
            dvecmuldot_libstr(nx, &sQinv[idx], 0, &work->sx[ii][kk], 0, &work->sx[ii][kk], 0);

            if (work->boundsRemoved[ii][kk+1] != 1) {
                dveccl_mask_libstr(nx, &sxmin[idx], 0, &work->sx[ii][kk], 0,
                    &sxmax[idx], 0, &work->sx[ii][kk], 0, &work->sxas[ii][kk], 0);

                #ifdef _CHECK_LAST_ACTIVE_SET_
                // NOTE(dimitris): compares with previous AS and updates previous with current
                work->xasChanged[ii][kk+1] = compare_with_previous_active_set(nx,
                    &work->sxas[ii][kk], &work->sxasPrev[ii][kk]);
                if (work->xasChanged[ii][kk+1]) {
                #endif
                // QinvCal[kk+1] = Qinv[kk+1] .* (1 - abs(xas[kk+1]))
                dvecze_libstr(nx, &work->sxas[ii][kk], 0, &sQinv[idx], 0,
                    &work->sQinvCal[ii][kk], 0);
                #ifdef _CHECK_LAST_ACTIVE_SET_
                }
                #endif
            }
            #if PROFILE > 3
            xopt_times[NewtonIter] += treeqp_toc(&sub_tmr);
            #endif

            // --- calculate u_opt
            #if PROFILE > 3
            treeqp_tic(&sub_tmr);
            #endif

            // u[k] = -B[k]' * mu[k] - r[k]
            dgemv_t_libstr(nx, nu, -1.0, &sB[idx-1], 0, 0, &work->smu[ii][kk], 0, -1.0,
                &sr[idxm1], 0, &work->su[ii][kk], 0);

            // u[k] = u[k] - C[k]' * lambda
            if ((ii < Ns-1) && (kk < commonNodes[ii])) {
                // shared multiplier with next scenario
                daxpy_libstr(nu, -1.0, &slambda[ii], kk*nu, &work->su[ii][kk], 0,
                    &work->su[ii][kk], 0);
            }
            if ((ii > 0) && (kk < commonNodes[ii-1])) {
                // shared multiplier with previous scenario
                daxpy_libstr(nu, 1.0, &slambda[ii-1], kk*nu, &work->su[ii][kk], 0,
                    &work->su[ii][kk], 0);
            }
            // u[k] = R[k]^-1 .* u[k]
            dvecmuldot_libstr(nu, &sRinv[idxm1], 0, &work->su[ii][kk], 0, &work->su[ii][kk], 0);

            if (work->boundsRemoved[ii][kk] != 1) {
                dveccl_mask_libstr(nu, &sumin[idxm1], 0, &work->su[ii][kk], 0,
                    &sumax[idxm1], 0, &work->su[ii][kk], 0, &work->suas[ii][kk], 0);
                #ifdef _CHECK_LAST_ACTIVE_SET_
                work->uasChanged[ii][kk] = compare_with_previous_active_set(nu,
                    &work->suas[ii][kk], &work->suasPrev[ii][kk]);
                if (work->uasChanged[ii][kk]) {
                #endif
                dvecze_libstr(nu, &work->suas[ii][kk], 0, &sRinv[idxm1], 0,
                    &work->sRinvCal[ii][kk], 0);
                #ifdef _CHECK_LAST_ACTIVE_SET_
                }
                #endif
            }
            #if PROFILE > 3
            uopt_times[NewtonIter] += treeqp_toc(&sub_tmr);
            #endif

            // --- calculate Zbar
            #if PROFILE > 3
            treeqp_tic(&sub_tmr);
            #endif

            #ifdef _CHECK_LAST_ACTIVE_SET_
            if (work->uasChanged[ii][kk] || NewtonIter == 0) {
            #endif
            // Zbar[k] = B[k] * RinvCal[k]
            dgemm_r_diag_libstr(nx, nu, 1.0, &sB[idx-1], 0, 0, &work->sRinvCal[ii][kk],
                0, 0.0, &work->sZbar[ii][kk], 0, 0, &work->sZbar[ii][kk], 0, 0);
            #ifdef _CHECK_LAST_ACTIVE_SET_
            }
            #endif

            #if PROFILE > 3
            Zbar_times[NewtonIter] += treeqp_toc(&sub_tmr);
            #endif

            // --- calculate Lambda blocks
            #if PROFILE > 3
            treeqp_tic(&sub_tmr);
            #endif

            #ifdef _CHECK_LAST_ACTIVE_SET_
            if ((kk == 0 && (work->uasChanged[ii][kk] || work->xasChanged[ii][kk+1])) ||
                (kk > 0 && (work->uasChanged[ii][kk] || work->xasChanged[ii][kk+1] ||
                work->xasChanged[ii][kk])) ||
                (NewtonIter == 0)) {
            #endif

            // LambdaD[k] = Zbar[k] * B[k]'
            dgemm_nt_libstr(nx, nx, nu, 1.0, &work->sZbar[ii][kk], 0, 0, &sB[idx-1],
                0, 0, 0.0, &work->sLambdaD[ii][kk], 0, 0, &work->sLambdaD[ii][kk], 0, 0);

            // LambdaD[k] = LambdaD[k] + QinvCal[k+1]
            ddiaad_libstr(nx, 1.0, &work->sQinvCal[ii][kk], 0, &work->sLambdaD[ii][kk], 0, 0);

            if (kk > 0) {
                #ifdef REV_CHOL
                // NOTE(dimitris): calculate LambdaL[k]' instead (aka upper triangular block)

                // LambdaL[k]' = A[k]'
                dgetr_libstr(nx, nx, &sA[idx-1], 0, 0, &work->sLambdaL[ii][kk-1], 0, 0);

                // LambdaL[k]' = -QinvCal[k]*A[k]'
                dgemm_l_diag_libstr(nx, nx, -1.0, &work->sQinvCal[ii][kk-1], 0,
                    &work->sLambdaL[ii][kk-1], 0, 0, 0.0, &work->sLambdaL[ii][kk-1], 0, 0,
                    &work->sLambdaL[ii][kk-1], 0, 0);

                // LambdaD[k] = LambdaD[k] - A[k]*LambdaL[k]' = LambdaD[k] + A[k]*QinvCal[k]*A[k]'
                dgemm_nn_libstr(nx, nx, nx, -1.0, &sA[idx-1], 0, 0,
                    &work->sLambdaL[ii][kk-1], 0, 0, 1.0, &work->sLambdaD[ii][kk], 0, 0,
                    &work->sLambdaD[ii][kk], 0, 0);

                #else
                // LambdaL[k] = -A[k] * QinvCal[k]
                dgemm_r_diag_libstr(nx, nx, -1.0, &sA[idx-1], 0, 0,
                    &work->sQinvCal[ii][kk-1], 0, 0.0, &work->sLambdaL[ii][kk-1], 0, 0,
                    &work->sLambdaL[ii][kk-1], 0, 0);

                // LambdaD[k] = LambdaD[k] - LambdaL[k] * A[k]'
                dgemm_nt_libstr(nx, nx, nx, -1.0, &work->sLambdaL[ii][kk-1], 0, 0,
                    &sA[idx-1], 0, 0, 1.0, &work->sLambdaD[ii][kk], 0, 0,
                    &work->sLambdaD[ii][kk], 0, 0);
                #endif
            }

            #ifdef _CHECK_LAST_ACTIVE_SET_
            // save diagonal block that will be overwritten in factorization
            dgecp_libstr(nx, nx, &work->sLambdaD[ii][kk], 0, 0, &work->sTmpLambdaD[ii][kk], 0, 0);
            } else {
                dgecp_libstr(nx, nx, &work->sTmpLambdaD[ii][kk], 0, 0,
                    &work->sLambdaD[ii][kk], 0, 0);
            }
            #endif

            #if PROFILE > 3
            Lambda_blocks_times[NewtonIter] += treeqp_toc(&sub_tmr);
            #endif
        }
        acc += commonNodes[ii];
    }
    // exit(1);
}


static void calculate_residuals(int_t Ns, int_t Nh, tree_ocp_qp_in *qp_in,
    treeqp_dune_scenarios_workspace *work) {

    int_t ii, kk, idx0, idxm1;
    int_t nu = work->su[0][0].m;
    int_t nx = work->sx[0][0].m;
    struct d_strmat *sA = (struct d_strmat *) qp_in->A;
    struct d_strmat *sB = (struct d_strmat *) qp_in->B;
    struct d_strvec *sb = (struct d_strvec *) qp_in->b;

    #ifdef PARALLEL
    #pragma omp parallel for private(kk, idx0, idxm1)
    #endif
    for (ii = 0; ii < Ns; ii++) {
        // res[1] = -b[0] - B[0] * u[0]
        // NOTE: different sign convention for b than in paper
        idx0 = work->nodeIdx[ii][1];
        dgemv_n_libstr(nx, nu, -1.0, &sB[idx0-1], 0, 0, &work->su[ii][0], 0, -1.0, &sb[idx0-1], 0,
            &work->sresk[ii][0], 0);
        // res[1] = res[1] + x[1]
        daxpy_libstr(nx, 1.0, &work->sx[ii][0], 0, &work->sresk[ii][0], 0, &work->sresk[ii][0], 0);

        for (kk = 2; kk < Nh+1; kk++) {
            idxm1 = work->nodeIdx[ii][kk];
            // printf("----> calculating residual of stage %d\n", kk);
            // res[k] = x[k] - A[k-1] * x[k-1]
            dgemv_n_libstr(nx, nx, -1.0, &sA[idxm1-1], 0, 0, &work->sx[ii][kk-2], 0, 1.0,
                &work->sx[ii][kk-1], 0, &work->sresk[ii][kk-1], 0);
            // res[k] = res[k] - B[k-1] * u[k-1]
            dgemv_n_libstr(nx, nu, -1.0, &sB[idxm1-1], 0, 0, &work->su[ii][kk-1], 0, 1.0,
                &work->sresk[ii][kk-1], 0, &work->sresk[ii][kk-1], 0);
            // res[k] = res[k] - b[k-1]
            daxpy_libstr(nx, -1.0, &sb[idxm1-1], 0, &work->sresk[ii][kk-1], 0,
                &work->sresk[ii][kk-1], 0);
        }
    }
}


static void calculate_last_residual(int_t Ns, int_t Nh, treeqp_dune_scenarios_workspace *work) {
    int_t ii, kk;
    int_t acc = 0;
    int_t nu = work->su[0][0].m;
    int_t *commonNodes = work->commonNodes;

    struct d_strvec *sResNonAnticip = work->sResNonAnticip;

    // initialize at zero
    for (ii = 0; ii < Ns-1; ii++) dvecse_libstr(sResNonAnticip[ii].m, 0.0, &sResNonAnticip[ii], 0);

    // first scenario
    for (kk = 0; kk < commonNodes[0]; kk++) {
        daxpy_libstr(nu, -1.0, &work->su[0][kk], 0, &sResNonAnticip[0], kk*nu,
            &sResNonAnticip[0], kk*nu);
    }
    acc += commonNodes[0];
    for (ii = 1; ii < Ns-1; ii++) {
        // previous scenario
        for (kk = 0; kk < commonNodes[ii-1]; kk++) {
            daxpy_libstr(nu, 1.0, &work->su[ii][kk], 0, &sResNonAnticip[ii-1], kk*nu,
                &sResNonAnticip[ii-1], kk*nu);
        }
        // next scenario
        for (kk = 0; kk < commonNodes[ii]; kk++) {
            daxpy_libstr(nu, -1.0, &work->su[ii][kk], 0, &sResNonAnticip[ii], kk*nu,
                &sResNonAnticip[ii], kk*nu);
        }
        acc += commonNodes[ii];
    }
    // last scenario
    for (kk = 0; kk < commonNodes[Ns-2]; kk++) {
        daxpy_libstr(nu, 1.0, &work->su[Ns-1][kk], 0, &sResNonAnticip[Ns-2], kk*nu,
            &sResNonAnticip[Ns-2], kk*nu);
    }
}


static void factorize_with_reg_opts(struct d_strmat *M, struct d_strmat *CholM,
    struct d_strvec *regMat, treeqp_dune_options_t *opts) {

    int_t jj;

    if (opts->regType == TREEQP_NO_REGULARIZATION) {
        // perform Cholesky  factorization
        dpotrf_l_libstr(M->m, M, 0, 0, CholM, 0, 0);
    } else if (opts->regType == TREEQP_ALWAYS_LEVENBERG_MARQUARDT) {
        // add regularization
        ddiaad_libstr(M->m, 1.0, regMat, 0, M, 0, 0);
        // perform Cholesky  factorization
        dpotrf_l_libstr(M->m, M, 0, 0, CholM, 0, 0);
    } else if (opts->regType == TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT) {
        // try Cholesky  factorization
        dpotrf_l_libstr(M->m, M, 0, 0, CholM, 0, 0);
        // check diagonal elements
        for (jj = 0; jj < M->m; jj++) {
            if (DMATEL_LIBSTR(CholM, jj, jj) <= opts->regTol) {
                // if too small, regularize
                ddiaad_libstr(M->m, 1.0, regMat, 0, M, 0, 0);
                // re-factorize
                dpotrf_l_libstr(M->m, M, 0, 0, CholM, 0, 0);
                // printf("regularized Lambda[%d][%d]\n", ii, kk);
                // exit(1);
                break;
            }
        }
    }
}

#ifdef _CHECK_LAST_ACTIVE_SET_

static void find_starting_point_of_factorization(int_t Ns, int_t Nh, int_t *idxStart,
    treeqp_dune_scenarios_workspace *work) {

    int_t ii, kk;

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

#endif

static void factorize_Lambda(int_t Ns, int_t Nh, treeqp_dune_options_t *opts, treeqp_dune_scenarios_workspace *work) {
    int_t ii, kk;
    int_t nx = work->sx[0][0].m;

    struct d_strvec *regMat = work->regMat;

    #if DEBUG == 1
    int_t indD, indL;
    real_t CholLambdaD[Ns*Nh*nx*nx], CholLambdaL[Ns*(Nh-1)*nx*nx];
    indD = 0; indL = 0;
    #endif

    #ifdef _CHECK_LAST_ACTIVE_SET_
    int_t idxStart[Ns];
    find_starting_point_of_factorization(Ns, Nh, idxStart, work);
    // for (ii = 0; ii < Ns; ii++)
    //     printf("restarting factorization of scenario %d at block %d\n", ii, idxStart[ii]);
    #endif

    // Banded Cholesky factorizations to calculate CholLambdaD[i], CholLambdaL[i]
    #ifdef PARALLEL
    #pragma omp parallel for private(kk)
    #endif
    for (ii = 0; ii < Ns; ii++) {
        #ifdef REV_CHOL
        for (kk = Nh-1; kk > 0; kk--) {
            #ifdef _CHECK_LAST_ACTIVE_SET_
            if (kk <= idxStart[ii]) {
            #endif

            // Cholesky factorization (possibly regularized)
            factorize_with_reg_opts(&work->sLambdaD[ii][kk], &work->sCholLambdaD[ii][kk],
                regMat, opts);

            // Substitution
            // NOTE(dimitris): LambdaL is already transposed (aka upper part of Lambda)
            dtrsm_rltn_libstr(nx, nx, 1.0, &work->sCholLambdaD[ii][kk], 0, 0,
                &work->sLambdaL[ii][kk-1], 0, 0, &work->sCholLambdaL[ii][kk-1], 0, 0);

            #ifdef _CHECK_LAST_ACTIVE_SET_
            }
            #endif

            #if DEBUG == 1
            d_cvt_strmat2mat(nx, nx, &work->sCholLambdaD[ii][kk], 0, 0, &CholLambdaD[indD], nx);
            // TODO(dimitris): fix debugging in matlab for reverse Cholesky
            d_cvt_strmat2mat(nx, nx, &work->sCholLambdaL[ii][kk-1], 0, 0, &CholLambdaL[indL], nx);
            indD += nx*nx; indL += nx*nx;
            #endif

            #ifdef _CHECK_LAST_ACTIVE_SET_
            if (kk <= idxStart[ii]+1) {
            #endif
            // Update (LambdaD[i][k+-1] -= CholLambdaL[i][k] * CholLambdaL[i][k]')
            dsyrk_ln_libstr(nx, nx, -1.0, &work->sCholLambdaL[ii][kk-1], 0, 0,
                &work->sCholLambdaL[ii][kk-1], 0, 0, 1.0, &work->sLambdaD[ii][kk-1], 0, 0,
                &work->sLambdaD[ii][kk-1], 0, 0);
            #ifdef _CHECK_LAST_ACTIVE_SET_
            }
            #endif
        }
        #ifdef REV_CHOL
        if (0 <= idxStart[ii]) {
        #endif
        factorize_with_reg_opts(&work->sLambdaD[ii][0], &work->sCholLambdaD[ii][0],
            regMat, opts);
        #ifdef REV_CHOL
        }
        #endif

        #if DEBUG == 1
        d_cvt_strmat2mat(nx, nx, &work->sCholLambdaD[ii][0], 0, 0, &CholLambdaD[indD], nx);
        indD += nx*nx;
        #endif

        #else  /* REV_CHOL */
        for (kk = 0; kk < Nh-1 ; kk++) {
            // Cholesky factorization (possibly regularized)
            factorize_with_reg_opts(&work->sLambdaD[ii][kk], &work->sCholLambdaD[ii][kk],
                regMat, opts);

            // Substitution
            dtrsm_rltn_libstr(nx, nx, 1.0, &work->sCholLambdaD[ii][kk], 0, 0,
                &work->sLambdaL[ii][kk], 0, 0, &work->sCholLambdaL[ii][kk], 0, 0);

            #if DEBUG == 1
            d_cvt_strmat2mat(nx, nx, &work->sCholLambdaD[ii][kk], 0, 0, &CholLambdaD[indD], nx);
            d_cvt_strmat2mat(nx, nx, &work->sCholLambdaL[ii][kk], 0, 0, &CholLambdaL[indL], nx);
            indD += nx*nx; indL += nx*nx;
            #endif

            // Update (LambdaD[i][k-1] -= CholLambdaL[i][k] * CholLambdaL[i][k]')
            dsyrk_ln_libstr(nx, nx, -1.0, &work->sCholLambdaL[ii][kk], 0, 0,
                &work->sCholLambdaL[ii][kk], 0, 0, 1.0, &work->sLambdaD[ii][kk+1], 0, 0,
                &work->sLambdaD[ii][kk+1], 0, 0);
        }
        factorize_with_reg_opts(&work->sLambdaD[ii][Nh-1], &work->sCholLambdaD[ii][Nh-1],
            regMat, opts);
        #if DEBUG == 1
        d_cvt_strmat2mat(nx, nx, &work->sCholLambdaD[ii][Nh-1], 0, 0, &CholLambdaD[indD], nx);
        indD += nx*nx;
        #endif

        #endif  /* REV_CHOL */
    }
    #if DEBUG == 1
    write_double_vector_to_txt(CholLambdaD, Ns*Nh*nx*nx, "examples/data_spring_mass/CholLambdaD.txt");
    write_double_vector_to_txt(CholLambdaL, Ns*(Nh-1)*nx*nx, "examples/data_spring_mass/CholLambdaL.txt");
    #endif
}


void form_K(int_t Ns, int_t Nh, int_t Nr, treeqp_dune_scenarios_workspace *work) {
    int_t ii, jj, kk;
    int_t indZ, indRinvCal;
    int_t nx = work->sx[0][0].m;
    int_t nu = work->su[0][0].m;

    struct d_strmat *sUt = work->sUt;
    struct d_strmat *sK = work->sK;
    struct d_strmat *sTmpMats = work->sTmpMats;

    #if DEBUG == 1
    int_t indK = 0;
    real_t K[Ns*Nr*nu*Nr*nu];
    #endif

    #ifdef PARALLEL
    #pragma omp parallel for private(jj, kk)
    #endif
    for (ii = 0; ii < Ns; ii++) {
        // ----- form U[i]'
        for (jj = 0; jj < Nr; jj++) {
            // transpose Zbar[k]
            dgetr_libstr(nx, nu, &work->sZbar[ii][jj], 0, 0, &sTmpMats[ii], 0, 0);
            #ifdef REV_CHOL
            for (kk = jj; kk >= 0; kk--) {
            #else
            for (kk = jj; kk < Nh; kk++) {
            #endif
                // matrix substitution
                // D <= B * A^{-T} , with A lower triangular employing explicit inverse of diagonal
                #ifdef LA_HIGH_PERFORMANCE
                // NOTE(dimitris): writing directly on sub-block NIY for BLASFEO_HP
                dtrsm_rltn_libstr(nu, nx, 1.0, &work->sCholLambdaD[ii][kk], 0, 0,
                    &sTmpMats[ii], 0, 0, &sTmpMats[ii], 0, 0);
                dgecp_libstr(nu, nx, &sTmpMats[ii], 0, 0, &sUt[ii], jj*nu, kk*nx);
                #else
                dtrsm_rltn_libstr(nu, nx, 1.0, &work->sCholLambdaD[ii][kk], 0, 0,
                    &sTmpMats[ii], 0, 0, &sUt[ii], jj*nu, kk*nx);
                #endif

                // update
                #ifdef REV_CHOL
                if (kk > 0) {
                    dgemm_nt_libstr(nu, nx, nx, -1.0, &sUt[ii], jj*nu, kk*nx,
                        &work->sCholLambdaL[ii][kk-1], 0, 0, 0.0, &sTmpMats[ii], 0, 0,
                        &sTmpMats[ii], 0, 0);
                }
                #else
                if (kk < Nh-1) {
                    dgemm_nt_libstr(nu, nx, nx, -1.0, &sUt[ii], jj*nu, kk*nx,
                        &work->sCholLambdaL[ii][kk], 0, 0, 0.0, &sTmpMats[ii], 0, 0,
                        &sTmpMats[ii], 0, 0);
                }
                #endif
            }
        }

        // ----- form upper right part of K[i]

        // symmetric matrix multiplication
        // TODO(dimitris): probably doing this with structure exploitation is cheaper if REV_CHOL=1
        dsyrk_ln_libstr(sUt[ii].m, sUt[ii].n, -1.0, &sUt[ii], 0, 0, &sUt[ii], 0, 0, 0.0,
            &sK[ii], 0, 0, &sK[ii], 0, 0);

        // mirror result to upper diagonal part (needed to form J properly)
        dtrtr_l_libstr(sK[ii].m, &sK[ii], 0, 0, &sK[ii], 0, 0);

        for (kk = 0; kk < Nr; kk++) {
            ddiaad_libstr(nu, 1.0, &work->sRinvCal[ii][kk], 0, &sK[ii], kk*nu, kk*nu);
        }

        #if DEBUG == 1
        d_cvt_strmat2mat(Nr*nu, Nr*nu, &sK[ii], 0, 0, &K[indK], Nr*nu);
        indK += Nr*nu*Nr*nu;
        #endif
    }

    #if DEBUG == 1
    write_double_vector_to_txt(K, Ns*Nr*nu*Nr*nu, "examples/data_spring_mass/K.txt");
    #endif
}


void form_and_factorize_Jay(int_t Ns, int_t nu, treeqp_dune_options_t *opts, treeqp_dune_scenarios_workspace *work) {
    int_t ii, dim, dimNxt;
    int_t *commonNodes = work->commonNodes;

    struct d_strvec *regMat = work->regMat;
    struct d_strmat *sK = work->sK;
    struct d_strmat *sJayD = work->sJayD;
    struct d_strmat *sJayL = work->sJayL;
    struct d_strmat *sCholJayD = work->sCholJayD;
    struct d_strmat *sCholJayL = work->sCholJayL;



    #if DEBUG == 1
    int_t indJayD, indJayL;
    int_t nJayD = get_size_of_JayD(Ns, nu, commonNodes);
    int_t nJayL = get_size_of_JayL(Ns, nu, commonNodes);
    real_t JayD[nJayD], JayL[nJayL], CholJayD[nJayD], CholJayL[nJayL];
    indJayD = 0; indJayL = 0;
    #endif

    // Banded Cholesky factorizations to calculate factor of Jay
    // NOTE: Cannot be parallelized
    for (ii = 0; ii < Ns-1; ii++) {
        dim = nu*commonNodes[ii];
        // Form JayD[i] using blocks K[i] and K[i+1]
        dgead_libstr(dim, dim, 1.0, &sK[ii], 0, 0, &sJayD[ii], 0, 0);
        dgead_libstr(dim, dim, 1.0, &sK[ii+1], 0, 0, &sJayD[ii], 0, 0);

        // Cholesky factorization (possibly regularized)
        // TODO(dimitris): remove regMat and add opts->regValue to diagonal
        factorize_with_reg_opts(&sJayD[ii], &sCholJayD[ii], regMat, opts);


        #if DEBUG == 1
        if (ii > 0) {  // undo update
            dsyrk_ln_libstr(dim, sCholJayL[ii-1].n, 1.0, &sCholJayL[ii-1], 0, 0,
                &sCholJayL[ii-1], 0, 0, 1.0, &sJayD[ii], 0, 0, &sJayD[ii], 0, 0);
        }
        d_cvt_strmat2mat(dim, dim, &sJayD[ii], 0, 0, &JayD[indJayD], dim);
        if (ii > 0) {  // redo update
            dsyrk_ln_libstr(dim, sCholJayL[ii-1].n, -1.0, &sCholJayL[ii-1], 0, 0,
                &sCholJayL[ii-1], 0, 0, 1.0, &sJayD[ii], 0, 0, &sJayD[ii], 0, 0);
        }
        d_cvt_strmat2mat(dim, dim, &sCholJayD[ii], 0, 0, &CholJayD[indJayD], dim);
        indJayD += ipow(dim, 2);
        #endif

        if (ii < Ns-2) {
            dimNxt = nu*commonNodes[ii+1];
            // Form JayL[i] using block K[i+1]
            dgead_libstr(dimNxt, dim, -1.0, &sK[ii+1], 0, 0, &sJayL[ii], 0, 0);

            // Substitution to form CholJayL[i]
            dtrsm_rltn_libstr(dimNxt, dim, 1.0, &sCholJayD[ii], 0, 0, &sJayL[ii], 0, 0,
                &sCholJayL[ii], 0, 0);

            #if DEBUG == 1
            d_cvt_strmat2mat(dimNxt, dim, &sJayL[ii], 0, 0, &JayL[indJayL], dimNxt);
            d_cvt_strmat2mat(dimNxt, dim, &sCholJayL[ii], 0, 0, &CholJayL[indJayL], dimNxt);
            indJayL += dimNxt*dim;
            #endif

            // Update for next block (NOTE: the update is added here before forming the block)
            dsyrk_ln_libstr(dimNxt, dim, -1.0, &sCholJayL[ii], 0, 0, &sCholJayL[ii], 0, 0,
                1.0, &sJayD[ii+1], 0, 0, &sJayD[ii+1], 0, 0);
        }
    }

    #if DEBUG == 1
    write_double_vector_to_txt(JayD, nJayD, "examples/data_spring_mass/JayD.txt");
    write_double_vector_to_txt(JayL, nJayL, "examples/data_spring_mass/JayL.txt");
    write_double_vector_to_txt(CholJayD, nJayD, "examples/data_spring_mass/CholJayD.txt");
    write_double_vector_to_txt(CholJayL, nJayL, "examples/data_spring_mass/CholJayL.txt");
    #endif
}


void form_RHS_non_anticipaticity(int_t Ns, int_t Nh, int_t Nr, int_t md, treeqp_dune_scenarios_workspace *work) {
    // NOTE: RHS has the opposite sign (corected when solving for Deltalambda later)

    int_t ii, jj, kk;
    int_t nx = work->sx[0][0].m;
    int_t nu = work->su[0][0].m;
    int_t *commonNodes = work->commonNodes;

    struct d_strvec *sTmpVecs = work->sTmpVecs;
    struct d_strvec *sResNonAnticip = work->sResNonAnticip;
    struct d_strvec *sRhsNonAnticip = work->sRhsNonAnticip;

    #if DEBUG == 1
    int_t ind = 0;
    int_t nl = calculate_dimension_of_lambda(Nr, md, nu);
    real_t *rhsNonAnticip = malloc(nl*sizeof(real_t));
    #endif

    #ifdef PARALLEL
    #pragma omp parallel for private(kk)
    #endif
    for (ii = 0; ii < Ns; ii++) {
        #ifdef REV_CHOL
        // tmp = res[Nh]
        dveccp_libstr(nx, &work->sresk[ii][Nh-1] , 0, &sTmpVecs[ii], 0);

        // backward substitution
        for (kk = Nh; kk > 1; kk--) {
            // resMod[k] = inv(CholLambdaD[k-1]) * tmp
            dtrsv_lnn_libstr(nx, &work->sCholLambdaD[ii][kk-1], 0, 0,
                &sTmpVecs[ii], 0, &work->sreskMod[ii][kk-1], 0);

            // update
            // tmp = res[k-1] - CholLambdaL[k-1] * resMod[k]
            dgemv_n_libstr(nx, nx, -1.0, &work->sCholLambdaL[ii][kk-2], 0, 0,
                &work->sreskMod[ii][kk-1], 0, 1.0, &work->sresk[ii][kk-2], 0, &sTmpVecs[ii], 0);
        }
        dtrsv_lnn_libstr(nx, &work->sCholLambdaD[ii][0], 0, 0, &sTmpVecs[ii],
            0, &work->sreskMod[ii][0], 0);

        // forward substitution
        for (kk = 0; kk < Nh-1; kk++) {
            // resMod[k+1] = inv(CholLambdaD[k]') * resMod[k+1]
            dtrsv_ltn_libstr(nx, &work->sCholLambdaD[ii][kk], 0, 0,
                &work->sreskMod[ii][kk], 0, &work->sreskMod[ii][kk], 0);

            // resMod[k+2] = resMod[k+2] - CholLambdaL[k+1]' * resMod[k+1]
            dgemv_t_libstr(nx, nx, -1.0, &work->sCholLambdaL[ii][kk], 0, 0,
                &work->sreskMod[ii][kk], 0, 1.0, &work->sreskMod[ii][kk+1], 0,
                &work->sreskMod[ii][kk+1], 0);
        }
        dtrsv_ltn_libstr(nx, &work->sCholLambdaD[ii][Nh-1], 0, 0,
            &work->sreskMod[ii][Nh-1], 0, &work->sreskMod[ii][Nh-1], 0);
        #else
        // tmp = res[1]
        dveccp_libstr(nx, &work->sresk[ii][0] , 0, &sTmpVecs[ii], 0);

        // forward substitution
        for (kk = 0; kk < Nh-1; kk++) {
            // resMod[k+1] = inv(CholLambdaD[k]) * tmp
            dtrsv_lnn_libstr(nx, &work->sCholLambdaD[ii][kk], 0, 0, &sTmpVecs[ii],
                0, &work->sreskMod[ii][kk], 0);

            // update
            // tmp = res[k+2] - CholLambdaL[k+1] * resMod[k+1]
            dgemv_n_libstr(nx, nx, -1.0, &work->sCholLambdaL[ii][kk], 0, 0,
                &work->sreskMod[ii][kk], 0, 1.0, &work->sresk[ii][kk+1], 0, &sTmpVecs[ii], 0);
        }
        dtrsv_lnn_libstr(nx, &work->sCholLambdaD[ii][Nh-1], 0, 0, &sTmpVecs[ii],
            0, &work->sreskMod[ii][Nh-1], 0);

        // backward substitution
        for (kk = Nh; kk > 1; kk--) {
            // resMod[k] = inv(CholLambdaD[k-1]') * resMod[k]
            dtrsv_ltn_libstr(nx, &work->sCholLambdaD[ii][kk-1], 0, 0,
                &work->sreskMod[ii][kk-1], 0, &work->sreskMod[ii][kk-1], 0);

            // resMod[k-1] = resMod[k-1] - CholLambdaL[k-1]' * resMod[k]
            dgemv_t_libstr(nx, nx, -1.0, &work->sCholLambdaL[ii][kk-2], 0, 0,
                &work->sreskMod[ii][kk-1], 0, 1.0, &work->sreskMod[ii][kk-2], 0,
                &work->sreskMod[ii][kk-2], 0);
        }
        dtrsv_ltn_libstr(nx, &work->sCholLambdaD[ii][0], 0, 0,
            &work->sreskMod[ii][0], 0, &work->sreskMod[ii][0], 0);
        #endif
    }

    // for ii == 0
    dveccp_libstr(sResNonAnticip[0].m, &sResNonAnticip[0], 0, &sRhsNonAnticip[0], 0);
    for (kk = 0; kk < commonNodes[0]; kk++) {
        dgemv_t_libstr(nx, nu, -1.0, &work->sZbar[0][kk], 0, 0,
            &work->sreskMod[0][kk], 0, 1.0, &sRhsNonAnticip[0], kk*nu,
            &sRhsNonAnticip[0], kk*nu);
    }

    for (ii = 1; ii < Ns-1; ii++) {
        dveccp_libstr(sResNonAnticip[ii].m, &sResNonAnticip[ii], 0, &sRhsNonAnticip[ii], 0);
        for (kk = 0; kk < commonNodes[ii-1]; kk++) {
            dgemv_t_libstr(nx, nu, 1.0, &work->sZbar[ii][kk], 0, 0,
                &work->sreskMod[ii][kk], 0, 1.0, &sRhsNonAnticip[ii-1], kk*nu,
                &sRhsNonAnticip[ii-1], kk*nu);
        }
        for (kk = 0; kk < commonNodes[ii]; kk++) {
            dgemv_t_libstr(nx, nu, -1.0, &work->sZbar[ii][kk], 0, 0,
                &work->sreskMod[ii][kk], 0, 1.0, &sRhsNonAnticip[ii], kk*nu,
                &sRhsNonAnticip[ii], kk*nu);
        }
    }
    // for ii == Ns-1
    for (kk = 0; kk < commonNodes[ii-1]; kk++) {
    dgemv_t_libstr(nx, nu, 1.0, &work->sZbar[Ns-1][kk], 0, 0,
        &work->sreskMod[Ns-1][kk], 0, 1.0, &sRhsNonAnticip[Ns-2], kk*nu,
        &sRhsNonAnticip[Ns-2], kk*nu);
    }

    #if DEBUG == 1
    for (ii = 0; ii < Ns-1; ii++) {
        d_cvt_strvec2vec(sRhsNonAnticip[ii].m, &sRhsNonAnticip[ii], 0, &rhsNonAnticip[ind]);
        ind += nu*commonNodes[ii];
    }
    write_double_vector_to_txt(rhsNonAnticip, nl, "examples/data_spring_mass/rhsNonAnticip.txt");
    free(rhsNonAnticip);
    #endif
    // printf("RHS:\n");
    // for (ii = 0; ii < Ns-1; ii++)
    //     d_print_strvec(commonNodes[ii]*nu, &sRhsNonAnticip[ii], 0);
}



void calculate_delta_lambda(int_t Ns, int_t Nr, int_t md, treeqp_dune_scenarios_workspace *work) {
    int_t ii, dim, dimNxt;
    int_t *commonNodes = work->commonNodes;
    int_t nu = work->su[0][0].m;

    struct d_strmat *sCholJayD = work->sCholJayD;
    struct d_strmat *sCholJayL = work->sCholJayL;
    struct d_strvec *sRhsNonAnticip = work->sRhsNonAnticip;
    struct d_strvec *sDeltalambda = work->sDeltalambda;

    #if DEBUG == 1
    int_t ind = 0;
    int_t nl = calculate_dimension_of_lambda(Nr, md, nu);
    real_t Deltalambda[nl];
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
        dveccp_libstr(dim, &sRhsNonAnticip[ii], 0, &sDeltalambda[ii], 0);
        dvecsc_libstr(dim, -1.0, &sDeltalambda[ii], 0);

        // substitution
        dtrsv_lnn_libstr(dim, &sCholJayD[ii], 0, 0, &sDeltalambda[ii], 0,
            &sRhsNonAnticip[ii], 0);

        // update
        dgemv_n_libstr(dimNxt, dim, 1.0, &sCholJayL[ii], 0, 0, &sRhsNonAnticip[ii], 0, 1.0,
            &sRhsNonAnticip[ii+1], 0, &sRhsNonAnticip[ii+1], 0);
    }
    // ii = Ns-2 (last part of the loop without the update)
    dveccp_libstr(dimNxt, &sRhsNonAnticip[ii], 0, &sDeltalambda[ii], 0);
    dvecsc_libstr(dimNxt, -1.0, &sDeltalambda[ii], 0);

    dtrsv_lnn_libstr(dimNxt, &sCholJayD[ii], 0, 0, &sDeltalambda[ii], 0,
        &sRhsNonAnticip[ii], 0);

    // ------ backward substitution
    for (ii = Ns-1; ii > 1; ii--) {
        dim = nu*commonNodes[ii-1];
        dimNxt = nu*commonNodes[ii-2];

        // substitution
        dtrsv_ltn_libstr(dim, &sCholJayD[ii-1], 0, 0, &sRhsNonAnticip[ii-1], 0,
            &sDeltalambda[ii-1], 0);

        // update
        dgemv_t_libstr(dim, dimNxt, -1.0, &sCholJayL[ii-2], 0, 0, &sDeltalambda[ii-1], 0, 1.0,
            &sRhsNonAnticip[ii-2], 0, &sRhsNonAnticip[ii-2], 0);
    }
    // ii = 1 (last part of the loop without the update)
    dtrsv_ltn_libstr(dimNxt, &sCholJayD[ii-1], 0, 0, &sRhsNonAnticip[ii-1], 0,
        &sDeltalambda[ii-1], 0);

    // printf("Delta lambdas:\n");
    // for (ii = 0; ii < Ns-1;ii++)
    //     d_print_strvec(sDeltalambda[ii].m, &sDeltalambda[ii], 0);
    #if DEBUG == 1
    for (ii = 0; ii < Ns-1; ii++) {
        d_cvt_strvec2vec(sDeltalambda[ii].m, &sDeltalambda[ii], 0, &Deltalambda[ind]);
        ind += nu*commonNodes[ii];
    }
    write_double_vector_to_txt(Deltalambda, nl, "examples/data_spring_mass/Deltalambda.txt");
    #endif
}


void calculate_delta_mu(int_t Ns, int_t Nh, int_t Nr, treeqp_dune_scenarios_workspace *work) {
    int_t ii, kk;
    int_t nx = work->sx[0][0].m;
    int_t nu = work->su[0][0].m;
    int_t *commonNodes = work->commonNodes;

    struct d_strvec *sDeltalambda = work->sDeltalambda;
    struct d_strvec *sTmpVecs = work->sTmpVecs;

    #if DEBUG == 1
    int_t indRes, indMu;
    real_t Deltamu[Ns*Nh*nx], rhsDynamics[Ns*Nh*nx];
    indRes = 0; indMu = 0;
    #endif

    #ifdef PARALLEL
    #pragma omp parallel for private(kk)
    #endif
    for (ii = 0; ii < Ns; ii++) {
        // resMod[i] = -C'[i] * Deltalambda[i] - res[i]
        for (kk = 0; kk < Nh; kk++) {
            dvecse_libstr(work->sreskMod[ii][kk].m, 0.0, &work->sreskMod[ii][kk], 0);
            if ((ii > 0) && (kk < commonNodes[ii-1])) {
                // shared multiplier with previous scenario
                daxpy_libstr(nu, 1.0, &sDeltalambda[ii-1], kk*nu, &work->sreskMod[ii][kk], 0,
                    &work->sreskMod[ii][kk], 0);
            }
            if ((ii < Ns-1) && (kk < commonNodes[ii])) {
                // shared multiplier with next scenario
                daxpy_libstr(nu, -1.0, &sDeltalambda[ii], kk*nu, &work->sreskMod[ii][kk], 0,
                    &work->sreskMod[ii][kk], 0);
            }
            if (kk < Nr) {
                // NOTE: cannot have resMod also as output!
                dgemv_n_libstr(nx, nu, 1.0, &work->sZbar[ii][kk], 0, 0,
                &work->sreskMod[ii][kk], 0, -1.0, &work->sresk[ii][kk], 0,
                    &sTmpVecs[ii], 0);
                dveccp_libstr(nx, &sTmpVecs[ii], 0, &work->sreskMod[ii][kk], 0);
            } else {
                daxpy_libstr(nx, -1.0, &work->sresk[ii][kk], 0, &work->sreskMod[ii][kk], 0,
                    &work->sreskMod[ii][kk], 0);
            }
        }
        #if DEBUG == 1
        for (kk = 0; kk < Nh; kk++) {
            d_cvt_strvec2vec(nx, &work->sreskMod[ii][kk], 0, &rhsDynamics[indRes]);
            indRes += nx;
        }
        #endif

        // ------ forward-backward substitution to calculate mu

        #ifdef REV_CHOL
        // backward substitution
        for (kk = Nh; kk > 1; kk--) {
            // Deltamu[k] = inv(CholLambdaD[k-1]) * res[k]
            dtrsv_lnn_libstr(nx, &work->sCholLambdaD[ii][kk-1], 0, 0,
                &work->sreskMod[ii][kk-1], 0, &work->sDeltamu[ii][kk-1], 0);

            // update
            dgemv_n_libstr(nx, nx, -1.0, &work->sCholLambdaL[ii][kk-2], 0, 0,
                &work->sDeltamu[ii][kk-1], 0, 1.0, &work->sreskMod[ii][kk-2], 0,
                &work->sreskMod[ii][kk-2], 0);
        }
        dtrsv_lnn_libstr(nx, &work->sCholLambdaD[ii][0], 0, 0, &work->sreskMod[ii][0],
            0, &work->sDeltamu[ii][0], 0);

        // forward substitution
        for (kk = 0; kk < Nh-1; kk++) {
            // Deltamu[k+1] = inv(CholLambdaD[k]') * Deltamu[k+1]
            dtrsv_ltn_libstr(nx, &work->sCholLambdaD[ii][kk], 0, 0,
                &work->sDeltamu[ii][kk], 0, &work->sDeltamu[ii][kk], 0);

            // update
            dgemv_t_libstr(nx, nx, -1.0, &work->sCholLambdaL[ii][kk], 0, 0,
                &work->sDeltamu[ii][kk], 0, 1.0, &work->sDeltamu[ii][kk+1], 0,
                &work->sDeltamu[ii][kk+1], 0);
        }
        dtrsv_ltn_libstr(nx, &work->sCholLambdaD[ii][Nh-1], 0, 0,
            &work->sDeltamu[ii][Nh-1], 0, &work->sDeltamu[ii][Nh-1], 0);
        #else
        // forward substitution
        for (kk = 0; kk < Nh-1; kk++) {
            // Deltamu[k+1] = inv(CholLambdaD[k]) * res[k+1]
            dtrsv_lnn_libstr(nx, &work->sCholLambdaD[ii][kk], 0, 0,
                &work->sreskMod[ii][kk], 0, &work->sDeltamu[ii][kk], 0);

            // update
            dgemv_n_libstr(nx, nx, -1.0, &work->sCholLambdaL[ii][kk], 0, 0,
                &work->sDeltamu[ii][kk], 0, 1.0, &work->sreskMod[ii][kk+1], 0,
                &work->sreskMod[ii][kk+1], 0);
        }
        dtrsv_lnn_libstr(nx, &work->sCholLambdaD[ii][Nh-1], 0, 0,
            &work->sreskMod[ii][Nh-1], 0, &work->sDeltamu[ii][Nh-1], 0);

        // backward substitution
        for (kk = Nh; kk > 1; kk--) {
            // Deltamu[k] = inv(CholLambdaD[k-1]') * Deltamu[k]
            dtrsv_ltn_libstr(nx, &work->sCholLambdaD[ii][kk-1], 0, 0,
                &work->sDeltamu[ii][kk-1], 0, &work->sDeltamu[ii][kk-1], 0);

            // Deltamu[k-1] = Deltamu[k-1] - CholLambdaL[k-1] * Deltamu[k]
            dgemv_t_libstr(nx, nx, -1.0, &work->sCholLambdaL[ii][kk-2], 0, 0,
                &work->sDeltamu[ii][kk-1], 0, 1.0, &work->sDeltamu[ii][kk-2], 0,
                &work->sDeltamu[ii][kk-2], 0);
        }
        dtrsv_ltn_libstr(nx, &work->sCholLambdaD[ii][0], 0, 0,
            &work->sDeltamu[ii][0], 0, &work->sDeltamu[ii][0], 0);
        #endif  /* REV_CHOL */

        // printf("SCENARIO %d, MULTIPLIERS:\n", ii+1);
        // for (kk = 0; kk < Nh; kk++) {
        //     d_print_strvec(nx, &work->sDeltamu[ii][kk], 0);
        // }
    }
    #if DEBUG == 1
    for (ii = 0; ii < Ns; ii++) {
        for (kk = 0; kk < Nh; kk++) {
            d_cvt_strvec2vec(nx, &work->sDeltamu[ii][kk], 0, &Deltamu[indMu]);
            indMu += nx;
        }
    }
    write_double_vector_to_txt(rhsDynamics, Ns*Nh*nx, "examples/data_spring_mass/rhsDynamics.txt");
    write_double_vector_to_txt(Deltamu, Ns*Nh*nx, "examples/data_spring_mass/Deltamu.txt");
    #endif
}


real_t gradient_trans_times_direction(int_t Ns, int_t Nh, treeqp_dune_scenarios_workspace *work) {
    int_t ii, kk;
    int_t nx = work->sx[0][0].m;
    real_t ans = 0;

    struct d_strvec *sResNonAnticip = work->sResNonAnticip;
    struct d_strvec *sDeltalambda = work->sDeltalambda;

    for (ii = 0; ii < Ns-1; ii++) {
        ans += ddot_libstr(sResNonAnticip[ii].m, &sResNonAnticip[ii], 0, &sDeltalambda[ii], 0);
    }

    for (ii = 0; ii < Ns; ii++) {
        for (kk = 0; kk < Nh; kk++) {
            ans += ddot_libstr(nx, &work->sresk[ii][kk], 0, &work->sDeltamu[ii][kk], 0);
        }
    }
    return ans;
}


real_t evaluate_dual_function(int_t Ns, int_t Nh, tree_ocp_qp_in *qp_in, treeqp_dune_scenarios_workspace *work) {
    int_t *commonNodes = work->commonNodes;
    real_t *fvals = work->fvals;

    struct d_strvec *sTmpVecs = work->sTmpVecs;
    struct d_strvec *slambda = work->slambda;
    struct d_strvec *sResNonAnticip = work->sResNonAnticip;
    struct d_strmat *sA = (struct d_strmat *) qp_in->A;
    struct d_strmat *sB = (struct d_strmat *) qp_in->B;
    struct d_strvec *sb = (struct d_strvec *) qp_in->b;
    struct d_strvec *sQ = (struct d_strvec *) qp_in->Q;
    struct d_strvec *sR = (struct d_strvec *) qp_in->R;
    struct d_strvec *sq = (struct d_strvec *) qp_in->q;
    struct d_strvec *sr = (struct d_strvec *) qp_in->r;
    struct d_strvec *sQinv = (struct d_strvec *) qp_in->Qinv;
    struct d_strvec *sRinv = (struct d_strvec *) qp_in->Rinv;
    struct d_strvec *sxmin = (struct d_strvec *) qp_in->xmin;
    struct d_strvec *sxmax = (struct d_strvec *) qp_in->xmax;
    struct d_strvec *sumin = (struct d_strvec *) qp_in->umin;
    struct d_strvec *sumax = (struct d_strvec *) qp_in->umax;

    real_t fval = 0;
    int_t ii, kk, idx, idxp1, idxm1;
    int_t nu = work->su[0][0].m;
    int_t nx = work->sx[0][0].m;

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
                dgemv_t_libstr(nx, nx, -1.0, &sA[idxp1-1], 0, 0, &work->smu[ii][kk+1],
                    0, 1.0, &work->smu[ii][kk], 0, &work->sxUnc[ii][kk], 0);
            } else {
                dveccp_libstr(nx, &work->smu[ii][kk], 0, &work->sxUnc[ii][kk], 0);
            }
            daxpy_libstr(nx, -1.0, &sq[idx], 0, &work->sxUnc[ii][kk], 0, &work->sxUnc[ii][kk], 0);

            dvecmuldot_libstr(nx, &sQinv[idx], 0, &work->sxUnc[ii][kk], 0, &work->sxUnc[ii][kk], 0);

            if (work->boundsRemoved[ii][kk+1] != 1) {
                dveccl_libstr(nx, &sxmin[idx], 0, &work->sxUnc[ii][kk], 0,
                    &sxmax[idx], 0, &work->sx[ii][kk], 0);
            } else {
                dveccp_libstr(nx, &work->sxUnc[ii][kk], 0, &work->sx[ii][kk], 0);
            }

            // --- calculate u_opt
            dgemv_t_libstr(nx, nu, -1.0, &sB[idx-1], 0, 0, &work->smu[ii][kk], 0, -1.0,
                &sr[idxm1], 0, &work->suUnc[ii][kk], 0);

            if ((ii < Ns-1) && (kk < commonNodes[ii])) {
                daxpy_libstr(nu, -1.0, &slambda[ii], kk*nu, &work->suUnc[ii][kk], 0,
                    &work->suUnc[ii][kk], 0);
            }
            if ((ii > 0) && (kk < commonNodes[ii-1])) {
                daxpy_libstr(nu, 1.0, &slambda[ii-1], kk*nu, &work->suUnc[ii][kk], 0,
                    &work->suUnc[ii][kk], 0);
            }
            dvecmuldot_libstr(nu, &sRinv[idxm1], 0, &work->suUnc[ii][kk], 0,
                &work->suUnc[ii][kk], 0);

            if (work->boundsRemoved[ii][kk] != 1) {
                dveccl_libstr(nu, &sumin[idxm1], 0, &work->suUnc[ii][kk], 0,
                    &sumax[idxm1], 0, &work->su[ii][kk], 0);
            } else {
                dveccp_libstr(nu, &work->suUnc[ii][kk], 0, &work->su[ii][kk], 0);
            }

            #ifndef NEW_FVAL
            // --- recalculate residual
            if (kk == 0) {
                // res[1] = -b[0] - B[0] * u[0]
                dgemv_n_libstr(nx, nu, -1.0, &sB[idx-1], 0, 0, &work->su[ii][0], 0, -1.0,
                    &sb[idx-1], 0, &work->sresk[ii][0], 0);
                // res[1] = res[1] + x[1]
                daxpy_libstr(nx, 1.0, &work->sx[ii][0], 0, &work->sresk[ii][0], 0,
                    &work->sresk[ii][0], 0);
            } else {
                // res[k+1] = x[k+1] - A[k] * x[k]
                dgemv_n_libstr(nx, nx, -1.0, &sA[idx-1], 0, 0, &work->sx[ii][kk-1], 0, 1.0,
                    &work->sx[ii][kk], 0, &work->sresk[ii][kk], 0);
                // res[k+1] = res[k+1] - B[k] * u[k]
                dgemv_n_libstr(nx, nu, -1.0, &sB[idx-1], 0, 0, &work->su[ii][kk], 0, 1.0,
                    &work->sresk[ii][kk], 0, &work->sresk[ii][kk], 0);
                // res[k+1] = res[k+1] - b[k]
                daxpy_libstr(nx, -1.0, &sb[idx-1], 0,
                    &work->sresk[ii][kk], 0, &work->sresk[ii][kk], 0);
            }
            #endif

            #ifdef NEW_FVAL

            // fval[i] -= (1/2)x[k+1]' * Q[k+1] * (x[k+1] - 2*xUnc[k+1])
            daxpy_libstr(nx, -2.0, &work->sxUnc[ii][kk], 0, &work->sx[ii][kk], 0,
                &work->sxUnc[ii][kk], 0);
            dvecmuldot_libstr(nx, &sQ[idx], 0, &work->sxUnc[ii][kk], 0, &work->sxUnc[ii][kk], 0);

            fvals[ii] -= 0.5*ddot_libstr(nx, &work->sx[ii][kk], 0, &work->sxUnc[ii][kk], 0);

            // fval[i] -= (1/2)u[k] * R[k] * (u[k] - 2*uUnc[k])
            daxpy_libstr(nu, -2.0, &work->suUnc[ii][kk], 0, &work->su[ii][kk], 0,
                &work->suUnc[ii][kk], 0);
            dvecmuldot_libstr(nu, &sR[idxm1], 0, &work->suUnc[ii][kk], 0, &work->suUnc[ii][kk], 0);

            fvals[ii] -= 0.5*ddot_libstr(nu, &work->su[ii][kk], 0, &work->suUnc[ii][kk], 0);

            // fval[i] -= b[k]' *  mu[k+1]
            fvals[ii] -= ddot_libstr(nx, &sb[idx-1], 0, &work->smu[ii][kk], 0);

            #else
            // fval = - (1/2)x[k+1]' * Q[k+1] * x[k+1] - x[k+1]' * q[k+1]
            dvecmuldot_libstr(nx, &sQ[idx], 0, &work->sx[ii][kk], 0,
                &sTmpVecs[ii], 0);
            fvals[ii] -= 0.5*ddot_libstr(nx, &sTmpVecs[ii], 0, &work->sx[ii][kk], 0);
            fvals[ii] -= ddot_libstr(nx, &sq[idx], 0, &work->sx[ii][kk], 0);
            // fval -= (1/2)u[k]' * R[k] * u[k] + u[k]' * r[k]
            dvecmuldot_libstr(nu, &sR[idxm1], 0, &work->su[ii][kk], 0, &sTmpVecs[ii], 0);
            fvals[ii] -= 0.5*ddot_libstr(nu, &sTmpVecs[ii], 0, &work->su[ii][kk], 0);
            fvals[ii] -= ddot_libstr(nu, &sr[idxm1], 0, &work->su[ii][kk], 0);
            // fval += mu[k]' * res[k] => fval -= mu[k]' * (-x[k+1] + A[k]*x[k] + B[k]*u[k] + b[k])
            fvals[ii] += ddot_libstr(nx, &work->smu[ii][kk], 0, &work->sresk[ii][kk], 0);
            #endif
        }
    }

    #ifndef NEW_FVAL
    calculate_last_residual(Ns, Nh, work);
    for (ii = 0; ii < Ns - 1; ii++) {
        fvals[ii] += ddot_libstr(slambda[ii].m, &slambda[ii], 0, &sResNonAnticip[ii], 0);
    }
    #endif

    for (ii = 0; ii < Ns; ii++) fval += fvals[ii];
    return fval;
}


int_t line_search(int_t Ns, int_t Nh, tree_ocp_qp_in *qp_in, treeqp_dune_options_t *opts,
    treeqp_dune_scenarios_workspace *work) {

    int_t ii, jj, kk;
    real_t dotProduct, fval, fval0;
    real_t tau = 1;
    real_t tauPrev = 0;
    int_t nx = work->sx[0][0].m;

    struct d_strvec *sDeltalambda = work->sDeltalambda;
    struct d_strvec *slambda = work->slambda;

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
                daxpy_libstr(nx, tau-tauPrev, &work->sDeltamu[ii][kk], 0,
                    &work->smu[ii][kk], 0, &work->smu[ii][kk], 0);
                // d_print_strvec(nx, &work->smu[ii][kk],0);
            }
            if (ii < Ns-1) {
                daxpy_libstr(sDeltalambda[ii].m, tau-tauPrev, &sDeltalambda[ii], 0,
                    &slambda[ii], 0, &slambda[ii], 0);
                // d_print_strvec(slambda[ii].m, &slambda[ii],0);
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
    #if DEBUG == 1
    write_double_vector_to_txt(&dotProduct, 1, "examples/data_spring_mass/dotProduct.txt");
    write_double_vector_to_txt(&fval0, 1, "examples/data_spring_mass/fval0.txt");
    #endif

    return jj;
}


// TODO(dimitris): time and see if it's worth to parallelize
real_t calculate_error_in_residuals(int_t Ns, int_t Nh, termination_t condition,
    treeqp_dune_scenarios_workspace *work) {

    int_t ii, jj, kk;
    real_t error = 0;
    int_t nx = work->sx[0][0].m;

    struct d_strvec *sResNonAnticip = work->sResNonAnticip;

    if ((condition == TREEQP_SUMSQUAREDERRORS) || (condition == TREEQP_TWONORM)) {
        for (ii = 0; ii < Ns; ii++) {
            for (kk = 0; kk < Nh; kk++) {
                error += ddot_libstr(nx, &work->sresk[ii][kk], 0,
                    &work->sresk[ii][kk], 0);
            }
            if (ii < Ns-1) {
                error += ddot_libstr(sResNonAnticip[ii].m, &sResNonAnticip[ii], 0,
                    &sResNonAnticip[ii], 0);
            }
        }
        if (condition == TREEQP_TWONORM) error = sqrt(error);
    } else if (condition == TREEQP_INFNORM) {
        for (ii = 0; ii < Ns; ii++) {
            for (kk = 0; kk < Nh; kk++) {
                for (jj = 0; jj < work->sresk[ii][kk].m; jj++) {
                    error = MAX(error, ABS(DVECEL_LIBSTR(&work->sresk[ii][kk], jj)));
                }
            }
            if (ii < Ns-1) {
                for (jj = 0; jj < sResNonAnticip[ii].m; jj++) {
                    error = MAX(error, ABS(DVECEL_LIBSTR(&sResNonAnticip[ii], jj)));
                }
            }
        }
    } else {
        printf("[TREEQP] Unknown termination condition!\n");
        exit(1);
    }
    return error;
}


int_t treeqp_dune_scenarios_calculate_size(tree_ocp_qp_in *qp_in) {
    struct node *tree = (struct node *) qp_in->tree;
    int_t nx = qp_in->nx[1];
    int_t nu = qp_in->nu[0];
    int_t Nn = qp_in->N;
    int_t Nh = tree[Nn-1].stage;
    int_t Np = get_number_of_parent_nodes(Nn, tree);
    int_t Ns = Nn - Np;
    int_t Nr = get_robust_horizon(Nn, tree);

    int_t ii, commonNodes, commonNodesNxt, maxTmpDim;
    int_t commonNodesMax = 0;
    int_t bytes = 0;

    // TODO(dimitris): run consistency checks on tree to see if compatible with algorithm

    bytes += 2*Ns*sizeof(int_t*);  // **nodeIdx, **boundsRemoved
    bytes += 2*Ns*(Nh+1)*sizeof(int_t);
    #ifdef _CHECK_LAST_ACTIVE_SET_
    bytes += 2*Ns*sizeof(int_t*);  // **xasChanged, **uasChanged
    bytes += 2*Ns*(Nh+1)*sizeof(int_t);
    #endif

    bytes += (Ns-1)*sizeof(int_t);  // *commonNodes
    bytes += Ns*sizeof(real_t);  // *fvals

    // double struct pointers
    bytes += 6*Ns*sizeof(struct d_strvec*);  // x, xUnc, xas, u, uUnc, uas
    bytes += 6*Ns*Nh*sizeof(struct d_strvec);
    bytes += 2*Ns*sizeof(struct d_strvec*);  // QinvCal, RinvCal
    bytes += 2*Ns*Nh*sizeof(struct d_strvec);
    bytes += 4*Ns*sizeof(struct d_strvec*);  // res, resMod, mu, Deltamu
    bytes += 4*Ns*Nh*sizeof(struct d_strvec);
    bytes += Ns*sizeof(struct d_strmat*);  // Zbar
    bytes += Ns*Nh*sizeof(struct d_strmat);
    bytes += 2*Ns*sizeof(struct d_strmat*);  // LambdaD, CholLambdaD
    bytes += 2*Ns*Nh*sizeof(struct d_strmat);
    bytes += 2*Ns*sizeof(struct d_strmat*);  // LambdaL, CholLambdaL
    bytes += 2*Ns*(Nh-1)*sizeof(struct d_strmat);
    #ifdef _CHECK_LAST_ACTIVE_SET_
    bytes += Ns*sizeof(struct d_strmat*);  // TmpLambdaD
    bytes += Ns*Nh*sizeof(struct d_strmat);
    bytes += 2*Ns*sizeof(struct d_strvec*);  // xasPrev, uasPrev
    bytes += 2*Ns*Nh*sizeof(struct d_strvec);
    #endif

    bytes += 3*Ns*Nh*d_size_strvec(nx);  // x, xUnc, xas
    bytes += 3*Ns*Nh*d_size_strvec(nu);  // u, uUnc, uas
    bytes += Ns*Nh*d_size_strvec(nx);  // QinvCal
    bytes += Ns*Nh*d_size_strvec(nu);  // RinvCal
    bytes += 4*Ns*Nh*d_size_strvec(nx);  // res, resMod, mu, Deltamu
    bytes += Ns*Nh*d_size_strmat(nx, nu);  // Zbar
    bytes += 2*Ns*Nh*d_size_strmat(nx, nx);  // LambdaD, CholLambdaD
    bytes += 2*Ns*(Nh-1)*d_size_strmat(nx, nx);  // LambdaL, CholLambdaL
    #ifdef _CHECK_LAST_ACTIVE_SET_
    bytes += Ns*Nh*d_size_strmat(nx, nx);  // TmpLambdaD
    bytes += Ns*Nh*d_size_strvec(nx);  // xasPrev
    bytes += Ns*Nh*d_size_strvec(nu);  // uasPrev
    #endif

    // struct pointers
    bytes += 2*(Ns-1)*sizeof(struct d_strmat);  // JayD, CholJayD
    bytes += 2*(Ns-2)*sizeof(struct d_strmat);  // JayL, CholJayL
    bytes += 2*Ns*sizeof(struct d_strmat);  // Ut, K
    bytes += 2*(Ns-1)*sizeof(struct d_strvec);  // resNonAnticip, rhsNonAnticip
    bytes += 2*(Ns-1)*sizeof(struct d_strvec);  // lambda, Deltalambda
    bytes += 1*sizeof(struct d_strvec);  // regMat
    bytes += Ns*sizeof(struct d_strvec);  // tmpVecs
    bytes += Ns*sizeof(struct d_strmat);  // tmpMats

    for (ii = 0; ii < Ns-1; ii++) {
        commonNodes = get_number_of_common_nodes(Nn, Ns, Nh, ii, ii+1, tree);
        commonNodesMax = MAX(commonNodesMax, commonNodes);
        bytes += 2*d_size_strmat(nu*commonNodes, nu*commonNodes);  // JayD, CholJayD
        bytes += 2*d_size_strvec(nu*commonNodes);  // resNonAnticip, rhsNonAnticip
        bytes += 2*d_size_strvec(nu*commonNodes);  // lambda, Deltalambda
        if (ii < Ns-2) {
            commonNodesNxt = get_number_of_common_nodes(Nn, Ns, Nh, ii+1, ii+2, tree);
            bytes += 2*d_size_strmat(nu*commonNodesNxt, nu*commonNodes);  // JayL, CholJayL
        }
    }

    // maximum dimension of tmp vector to store intermediate results
    maxTmpDim = MAX(nx, nu*commonNodesMax);

    bytes += d_size_strvec(maxTmpDim);  // RegMat
    bytes += Ns*d_size_strvec(maxTmpDim);  // tmpVecs
    bytes += Ns*d_size_strmat(nu, nx);  // tmpMats

    bytes += Ns*d_size_strmat(nu*Nr, Nh*nx);  // Ut
    bytes += Ns*d_size_strmat(nu*Nr, nu*Nr);  // K

    bytes += (bytes + 63)/64*64;  // make multiple of typical cache line size
    bytes += 64;  // align to typical cache line size

    return bytes;
}


void create_treeqp_dune_scenarios(tree_ocp_qp_in *qp_in, treeqp_dune_options_t *opts,
    treeqp_dune_scenarios_workspace *work, void *ptr_allocated_memory) {

    struct node *tree = (struct node *) qp_in->tree;
    int_t nx = qp_in->nx[1];
    int_t nu = qp_in->nu[0];
    int_t Nn = qp_in->N;
    int_t Nh = tree[Nn-1].stage;
    int_t Np = get_number_of_parent_nodes(Nn, tree);
    int_t Ns = Nn - Np;
    int_t Nr = get_robust_horizon(Nn, tree);

    int_t ii, kk, maxTmpDim, node, ans;
    int_t *commonNodes;

    // TODO(dimitris): move to workspace
    int_t *processedNodes = malloc(Nn*sizeof(int_t));
    int_t idx = 0;

    // store some dimensions in workspace
    work->Nr = Nr;
    work->Ns = Ns;
    work->Nh = Nh;
    work->md = tree[0].nkids;

    // char pointer
    char *c_ptr = (char *) ptr_allocated_memory;

    // calculate number of common nodes between neighboring scenarios
    work->commonNodes = (int_t*) c_ptr;
    c_ptr += (Ns-1)*sizeof(int_t);

    commonNodes = work->commonNodes;
    build_vector_of_common_nodes(Nn, Ns, Nh, tree, commonNodes);

    // calculate maximum dimension of temp vectors
    maxTmpDim = get_maximum_vector_dimension(Ns, nx, nu, commonNodes);

    // allocate memory for the dual function evaluation of each scenario
    work->fvals = (real_t*) c_ptr;
    c_ptr += Ns*sizeof(real_t);

    // generate indexing of scenario nodes from tree and flags for removed bounds
    create_double_ptr_int(&work->nodeIdx, Ns, Nh+1, &c_ptr);
    create_double_ptr_int(&work->boundsRemoved, Ns, Nh+1, &c_ptr);

    for (ii = 0; ii < Ns; ii++) {
        node = tree[Nn-Ns+ii].idx;
        work->nodeIdx[ii][Nh] = node;
        work->boundsRemoved[ii][Nh] = 0;
        for (kk = Nh; kk > 0; kk--) {
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

    // diagonal matrix (stored in vector) with regularization value
    work->regMat = (struct d_strvec *) c_ptr;
    c_ptr += 1*sizeof(struct d_strvec);
    // diagonal blocks of J (each symmetric of dimension nu*nc[k])
    work->sJayD = (struct d_strmat *) c_ptr;
    c_ptr += (Ns-1)*sizeof(struct d_strmat);
    // Cholesky factors of diagonal blocks
    work->sCholJayD = (struct d_strmat *) c_ptr;
    c_ptr += (Ns-1)*sizeof(struct d_strmat);
    // off-diagonal blocks of J (each of dim. nu*nc[k+1] x nu*nc[k])
    work->sJayL = (struct d_strmat *) c_ptr;
    c_ptr += (Ns-2)*sizeof(struct d_strmat);
    // Cholesky factors of off-diagonal blocks
    work->sCholJayL = (struct d_strmat *) c_ptr;
    c_ptr += (Ns-2)*sizeof(struct d_strmat);
    // Ut matrices to build K
    work->sUt = (struct d_strmat *) c_ptr;
    c_ptr += Ns*sizeof(struct d_strmat);
    // K matrices to build Jay
    work->sK = (struct d_strmat *) c_ptr;
    c_ptr += Ns*sizeof(struct d_strmat);
    //
    work->sResNonAnticip = (struct d_strvec *) c_ptr;
    c_ptr += (Ns-1)*sizeof(struct d_strvec);
    //
    work->sRhsNonAnticip = (struct d_strvec *) c_ptr;
    c_ptr += (Ns-1)*sizeof(struct d_strvec);
    // multipliers of non-anticipativity constraints
    work->slambda = (struct d_strvec *) c_ptr;
    c_ptr += (Ns-1)*sizeof(struct d_strvec);
    // step in multipliers of non-anticipativity constraints
    work->sDeltalambda = (struct d_strvec *) c_ptr;
    c_ptr += (Ns-1)*sizeof(struct d_strvec);

    // strmats/strvecs for intermediate results
    work->sTmpMats = (struct d_strmat *) c_ptr;
    c_ptr += Ns*sizeof(struct d_strmat);
    work->sTmpVecs = (struct d_strvec *) c_ptr;
    c_ptr += Ns*sizeof(struct d_strvec);


    #ifdef _CHECK_LAST_ACTIVE_SET_
    create_double_ptr_int(&work->xasChanged, Ns, Nh+1, &c_ptr);
    create_double_ptr_int(&work->uasChanged, Ns, Nh+1, &c_ptr);
    #endif

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
    #ifdef _CHECK_LAST_ACTIVE_SET_
    create_double_ptr_strmat(&work->sTmpLambdaD, Ns, Nh, &c_ptr);
    create_double_ptr_strvec(&work->sxasPrev, Ns, Nh, &c_ptr);
    create_double_ptr_strvec(&work->suasPrev, Ns, Nh, &c_ptr);
    #endif

    // move pointer for proper alignment of blasfeo matrices and vectors
    long long l_ptr = (long long) c_ptr;
	l_ptr = (l_ptr+63)/64*64;
	c_ptr = (char *) l_ptr;

    init_strvec(maxTmpDim, work->regMat, &c_ptr);
    dvecse_libstr(maxTmpDim, opts->regValue, work->regMat, 0);

    for (ii = 0; ii < Ns; ii++) {
        init_strmat(nu*Nr, Nh*nx, &work->sUt[ii], &c_ptr);
        init_strmat(nu*Nr, nu*Nr, &work->sK[ii], &c_ptr);
        init_strvec(maxTmpDim, &work->sTmpVecs[ii], &c_ptr);
        init_strmat(nu, nx, &work->sTmpMats[ii], &c_ptr);
        if (ii < Ns-1) {
            init_strmat(nu*commonNodes[ii], nu*commonNodes[ii], &work->sJayD[ii], &c_ptr);
            init_strmat(nu*commonNodes[ii], nu*commonNodes[ii], &work->sCholJayD[ii], &c_ptr);
            init_strvec(nu*commonNodes[ii], &work->sResNonAnticip[ii], &c_ptr);
            init_strvec(nu*commonNodes[ii], &work->sRhsNonAnticip[ii], &c_ptr);

            init_strvec(nu*commonNodes[ii], &work->slambda[ii], &c_ptr);
            init_strvec(nu*commonNodes[ii], &work->sDeltalambda[ii], &c_ptr);
        }
        if (ii < Ns-2) {
            init_strmat(nu*commonNodes[ii+1], nu*commonNodes[ii], &work->sJayL[ii], &c_ptr);
            init_strmat(nu*commonNodes[ii+1], nu*commonNodes[ii], &work->sCholJayL[ii], &c_ptr);
        }
    }

    for (ii = 0; ii < Ns; ii++) {
        for (kk = 0; kk < Nh; kk++) {
            init_strvec(nx, &work->sx[ii][kk], &c_ptr);
            init_strvec(nu, &work->su[ii][kk], &c_ptr);
            init_strvec(nx, &work->sxUnc[ii][kk], &c_ptr);
            init_strvec(nu, &work->suUnc[ii][kk], &c_ptr);
            init_strvec(nx, &work->sxas[ii][kk], &c_ptr);
            init_strvec(nu, &work->suas[ii][kk], &c_ptr);

            init_strvec(nx, &work->sQinvCal[ii][kk], &c_ptr);
            init_strvec(nu, &work->sRinvCal[ii][kk], &c_ptr);
            // NOTE(dimitris): all states are shifted by one after eliminating x0
            dveccp_libstr(nx, (struct d_strvec*)&qp_in->Qinv[work->nodeIdx[ii][kk+1]], 0,
                &work->sQinvCal[ii][kk], 0);
            dveccp_libstr(nu, (struct d_strvec*)&qp_in->Rinv[work->nodeIdx[ii][kk]], 0,
                &work->sRinvCal[ii][kk], 0);

            init_strvec(nx, &work->smu[ii][kk], &c_ptr);
            init_strvec(nx, &work->sDeltamu[ii][kk], &c_ptr);
            init_strvec(nx, &work->sresk[ii][kk], &c_ptr);
            init_strvec(nx, &work->sreskMod[ii][kk], &c_ptr);
            init_strmat(nx, nu, &work->sZbar[ii][kk], &c_ptr);
            init_strmat(nx, nx, &work->sLambdaD[ii][kk], &c_ptr);
            init_strmat(nx, nx, &work->sCholLambdaD[ii][kk], &c_ptr);
            if (kk < Nh-1) init_strmat(nx, nx, &work->sLambdaL[ii][kk], &c_ptr);
            if (kk < Nh-1) init_strmat(nx, nx, &work->sCholLambdaL[ii][kk], &c_ptr);
            #ifdef _CHECK_LAST_ACTIVE_SET_
            init_strmat(nx, nx, &work->sTmpLambdaD[ii][kk], &c_ptr);
            init_strvec(nx, &work->sxasPrev[ii][kk], &c_ptr);
            init_strvec(nu, &work->suasPrev[ii][kk], &c_ptr);
            // NOTE(dimitris): setting value outside {-1,0,1} to force full factorization at 1st it.
            dvecse_libstr(nx, 0.0/0.0, &work->sxasPrev[ii][kk], 0);
            dvecse_libstr(nu, 0.0/0.0, &work->suasPrev[ii][kk], 0);
            #endif
        }
    }
    free(processedNodes);
}


int_t treeqp_dune_scenarios_solve(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out,
    treeqp_dune_options_t *opts, treeqp_dune_scenarios_workspace *work) {

    int_t NewtonIter, lsIter;
    real_t error;
    int_t nu = qp_in->nu[0];
    int_t nx = qp_in->nx[1];
    return_t status = TREEQP_ERR_UNKNOWN_ERROR;

    int_t Nh = work->Nh;
    int_t Ns = work->Ns;
    int_t Nr = work->Nr;
    int_t md = work->md;

    struct d_strmat *sJayD = work->sJayD;
    struct d_strmat *sJayL = work->sJayL;

    // ------ dual Newton iterations
    // NOTE(dimitris): at first iteration some matrices are initialized for _CHECK_LAST_ACTIVE_SET_
    for (NewtonIter = 0; NewtonIter < opts->maxIter; NewtonIter++) {
        #if PROFILE > 1
        treeqp_tic(&iter_tmr);
        #endif
        #if PROFILE > 3
        reset_accumulative_timers(NewtonIter);
        #endif

        // ------ solve stage QPs
        // - calculate unconstrained solution of stage QPs
        // - clip solution
        // - calculate Zbar
        // - calculate LambdaD and LambdaL
        #if PROFILE > 2
        treeqp_tic(&tmr);
        #endif
        solve_stage_problems(Ns, Nh, NewtonIter, qp_in, work);
        #if PROFILE > 2
        stage_qps_times[NewtonIter] = treeqp_toc(&tmr);
        #endif

        // ------ calculate dual gradient
        #if PROFILE > 2
        treeqp_tic(&tmr);
        #endif
        // TODO(dimitris): benchmark in linux, see if I can avoid some ifs
        calculate_residuals(Ns, Nh, qp_in, work);
        calculate_last_residual(Ns, Nh, work);
        // NOTE(dimitris): cannot parallelize last residual so better to keep the two loops separate

        // TODO(dimitris): move call inside function
        #if DEBUG == 1
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

        // ------ factorize Newton system
        factorize_Lambda(Ns, Nh, opts, work);
        form_K(Ns, Nh, Nr, work);
        form_and_factorize_Jay(Ns, nu, opts, work);

        // ------ calculate multipliers of non-anticipativity constraints
        form_RHS_non_anticipaticity(Ns, Nh, Nr, md, work);
        calculate_delta_lambda(Ns, Nr, md, work);

        // ------ calculate multipliers of dynamics
        calculate_delta_mu(Ns, Nh, Nr, work);

        #if PROFILE > 2
        newton_direction_times[NewtonIter] = treeqp_toc(&tmr);
        #endif

        // ------ line search
        #if PROFILE > 2
        treeqp_tic(&tmr);
        #endif

        lsIter = line_search(Ns, Nh, qp_in, opts, work);

        #if PROFILE > 2
        line_search_times[NewtonIter] = treeqp_toc(&tmr);
        #endif

        // ------ reset data for next iteration
        // TODO(dimitris): check if it's worth parallelizing!
        for (int_t ii = 0; ii < Ns-1; ii++) {
            dgese_libstr(sJayD[ii].m, sJayD[ii].n, 0.0, &sJayD[ii], 0, 0);
        }
        for (int_t ii = 0; ii < Ns-2; ii++) {
            dgese_libstr(sJayL[ii].m, sJayL[ii].n, 0.0, &sJayL[ii], 0, 0);
        }
        #if PRINT_LEVEL > 1
        printf("iteration #%d: %d ls iterations \t\t(error %5.2e)\n", NewtonIter, lsIter, error);
        #endif
        #if PROFILE > 1
        iter_times[NewtonIter] = treeqp_toc(&iter_tmr);
        ls_iters[NewtonIter] = lsIter;
        #endif
    }

    for (int_t ii = 0; ii < Ns; ii++) {
        for (int_t kk = 0; kk < Nh; kk++) {
            if (work->boundsRemoved[ii][kk+1] == 0) {
                // printf("saving node (%d, %d) to node %d\n", ii, kk+1, work->nodeIdx[ii][kk+1]);
                dveccp_libstr(nx, &work->sx[ii][kk], 0, &qp_out->x[work->nodeIdx[ii][kk+1]], 0);
            }
            if (work->boundsRemoved[ii][kk] == 0) {
                dveccp_libstr(nu, &work->su[ii][kk], 0, &qp_out->u[work->nodeIdx[ii][kk]], 0);
            }
        }
    }

    qp_out->info.iter = NewtonIter;

    if (qp_out->info.iter == opts->maxIter)
        status = TREEQP_ERR_MAXIMUM_ITERATIONS_REACHED;

    return status;
}
