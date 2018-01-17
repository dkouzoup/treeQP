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

// TODO(dimitris): VALGRIND CODE
// TODO(dimitris): Try open MPI interface (message passing)
// TODO(dimitris): FIX BUG WITH LA=HP AND !MERGE_SUBS (HAPPENS ONLY IF Nr, md > 1)
// TODO(dimitris): find out why algorithm is not scale-invariant
// TODO(dimitris): on-the-fly regularization
// TODO(dimitris): different types of line-search
// TODO(dimitris): why saving so many Chol factorizations does not imporove cpu time?
// TODO(dimitris): ask Gianluca if I can overwrite Chol. and check if it makes sense

#include "treeqp/src/dual_Newton_tree.h"
#include "treeqp/src/tree_ocp_qp_common.h"

#include "treeqp/flags.h"

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
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_v_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blas.h"

#include <qpOASES_e.h>

#define _MERGE_FACTORIZATION_WITH_SUBSTITUTION_

treeqp_tdunes_options_t treeqp_tdunes_default_options(int Nn)
{
    treeqp_tdunes_options_t opts;
    termination_t cond = TREEQP_INFNORM;

    opts.maxIter = 100;
    opts.termCondition = cond;
    opts.stationarityTolerance = 1.0e-12;

    // TODO(dimitris): replace with calculate_size/create for args
    opts.qp_solver = malloc(Nn*sizeof(stage_qp_t));
    for (int ii = 0; ii < Nn; ii++) opts.qp_solver[ii] = TREEQP_CLIPPING_SOLVER;

    opts.lineSearchMaxIter = 50;
    opts.lineSearchGamma = 0.1;
    opts.lineSearchBeta = 0.6;

    // TODO(dimitris): implement on the fly regularization
    opts.regType  = TREEQP_ALWAYS_LEVENBERG_MARQUARDT;
    // opts.regTol   = 1.0e-12;
    opts.regValue = 1.0e-8;

    return opts;
}


static void setup_npar(int Nh, int Nn, struct node *tree, int *npar) {
    // initialize vector to zero
    for (int kk = 0; kk < Nh; kk++) {
        npar[kk] = 0;
    }
    // enumerate nodes per stage
    for (int kk = 0; kk < Nn; kk++) {
        npar[tree[kk].stage]++;
    }
}


static void setup_idxpos(tree_ocp_qp_in *qp_in, int *idxpos) {
    int Nn = qp_in->N;
    int idxdad;

    struct node *tree = (struct node *)qp_in->tree;

    for (int kk = 0; kk < Nn; kk++) {
        idxdad = tree[kk].dad;
        idxpos[kk] = 0;
        for (int ii = 0; ii < tree[kk].idxkid; ii++) {
            idxpos[kk] += qp_in->nx[tree[idxdad].kids[ii]];
        }
    }
    // for (int kk = 0; kk < Nn; kk++) {
    //     printf("kk = %d, idxpos = %d\n", kk, idxpos[kk]);
    // }
}


static answer_t is_clipping_solver_applicable(tree_ocp_qp_in *qp_in, int node_index)
{
    answer_t ans = YES;

    if (is_strmat_diagonal(&qp_in->Q[node_index]) == NO)
    {
        ans = NO;
    }

    return ans;
}



static int maximum_hessian_block_dimension(tree_ocp_qp_in *qp_in) {
    int maxDim = 0;
    int currDim, idxkid;

    for (int ii = 0; ii < qp_in->N; ii++) {
        currDim = 0;
        for (int jj = 0; jj < qp_in->tree[ii].nkids; jj++) {
            idxkid = qp_in->tree[ii].kids[jj];
            currDim += qp_in->nx[idxkid];
        }
        maxDim = MAX(maxDim, currDim);
    }
    return maxDim;
}


static void solve_stage_problems(tree_ocp_qp_in *qp_in, treeqp_tdunes_workspace *work) {
    int idxkid, idxdad, idxpos;
    int Nn = qp_in->N;
    int *nx = (int *)qp_in->nx;
    int *nu = (int *)qp_in->nu;

    struct node *tree = (struct node *)qp_in->tree;

    struct blasfeo_dvec *slambda = work->slambda;
    struct blasfeo_dvec *sx = (struct blasfeo_dvec *) work->sx;
    struct blasfeo_dvec *su = (struct blasfeo_dvec *) work->su;
    struct blasfeo_dvec *sxUnc = (struct blasfeo_dvec *) work->sxUnc;
    struct blasfeo_dvec *suUnc = (struct blasfeo_dvec *) work->suUnc;
    struct blasfeo_dvec *sxas = (struct blasfeo_dvec *) work->sxas;
    struct blasfeo_dvec *suas = (struct blasfeo_dvec *) work->suas;

    struct blasfeo_dmat *sA = (struct blasfeo_dmat *) qp_in->A;
    struct blasfeo_dmat *sB = (struct blasfeo_dmat *) qp_in->B;

    struct blasfeo_dvec *sq = (struct blasfeo_dvec *) qp_in->q;
    struct blasfeo_dvec *sr = (struct blasfeo_dvec *) qp_in->r;
    struct blasfeo_dvec *sQinv = work->sQinv;
    struct blasfeo_dvec *sRinv = work->sRinv;
    struct blasfeo_dvec *sQinvCal = work->sQinvCal;
    struct blasfeo_dvec *sRinvCal = work->sRinvCal;
    struct blasfeo_dvec *sqmod = work->sqmod;
    struct blasfeo_dvec *srmod = work->srmod;

    struct blasfeo_dvec *sxmin = (struct blasfeo_dvec *) qp_in->xmin;
    struct blasfeo_dvec *sxmax = (struct blasfeo_dvec *) qp_in->xmax;
    struct blasfeo_dvec *sumin = (struct blasfeo_dvec *) qp_in->umin;
    struct blasfeo_dvec *sumax = (struct blasfeo_dvec *) qp_in->umax;

    #if DEBUG == 1
    int indh = 0;
    int indx = 0;
    int indu = 0;
    int dimh = number_of_primal_variables(qp_in);
    int dimx = number_of_states(qp_in);
    int dimu = number_of_controls(qp_in);
    double *hmod = malloc(dimh*sizeof(double));
    double *xit = malloc(dimx*sizeof(double));
    double *uit = malloc(dimu*sizeof(double));
    double *QinvCal = malloc(dimx*sizeof(double));
    double *RinvCal = malloc(dimu*sizeof(double));
    #endif

    #ifdef PARALLEL
    #pragma omp parallel for private(idxkid, idxdad, idxpos)
    #endif
    for (int kk = 0; kk < Nn; kk++) {
        idxdad = tree[kk].dad;
        idxpos = work->idxpos[kk];

        // --- update QP gradient

        // qmod[k] = - q[k] + lambda[k]
        if (kk == 0) {
            // lambda[0] = 0
            for (int jj = 0; jj < nx[kk]; jj++) DVECEL_LIBSTR(&sqmod[kk], jj) = 0.0;
            blasfeo_daxpy(nx[kk], -1.0, &sq[kk], 0, &sqmod[kk], 0, &sqmod[kk], 0);
        } else {
            blasfeo_daxpy(nx[kk], -1.0, &sq[kk], 0, &slambda[idxdad], idxpos, &sqmod[kk], 0);
        }

        // rmod[k] = - r[k]
        blasfeo_dveccp(nu[kk], &sr[kk], 0, &srmod[kk], 0);
        blasfeo_dvecsc(nu[kk], -1.0, &srmod[kk], 0);

        for (int ii = 0; ii < tree[kk].nkids; ii++) {
            idxkid = tree[kk].kids[ii];
            idxdad = tree[idxkid].dad;
            idxpos = work->idxpos[idxkid];

            // qmod[k] -= A[jj]' * lambda[jj]
            blasfeo_dgemv_t(nx[idxkid], nx[idxdad], -1.0, &sA[idxkid-1], 0, 0,
                &slambda[idxdad], idxpos, 1.0, &sqmod[kk], 0, &sqmod[kk], 0);
            // rmod[k] -= B[jj]' * lambda[jj]
            blasfeo_dgemv_t(nx[idxkid], nu[idxdad], -1.0, &sB[idxkid-1], 0, 0,
                &slambda[idxdad], idxpos, 1.0, &srmod[kk], 0, &srmod[kk], 0);
        }

        // --- solve QP
        // x[k] = Q[k]^-1 .* qmod[k] (NOTE: minus sign already in mod. gradient)
        blasfeo_dvecmuldot(nx[kk], &sQinv[kk], 0, &sqmod[kk], 0, &sxUnc[kk], 0);

        // x[k] = median(xmin, x[k], xmax), xas[k] = active set
        blasfeo_dveccl_mask(nx[kk], &sxmin[kk], 0, &sxUnc[kk], 0, &sxmax[kk], 0,
            &sx[kk], 0, &sxas[kk], 0);

        // QinvCal[kk] = Qinv[kk] .* (1 - abs(xas[kk])), aka elimination matrix
        blasfeo_dvecze(nx[kk], &sxas[kk], 0, &sQinv[kk], 0, &sQinvCal[kk], 0);

        // u[k] = R[k]^-1 .* rmod[k]
        blasfeo_dvecmuldot(nu[kk], &sRinv[kk], 0, &srmod[kk], 0, &suUnc[kk], 0);

        // u[k] = median(umin, u[k], umax), uas[k] = active set
        blasfeo_dveccl_mask(nu[kk], &sumin[kk], 0, &suUnc[kk], 0, &sumax[kk], 0, &su[kk], 0,
            &suas[kk], 0);

        // RinvCal[kk] = Rinv[kk] .* (1 - abs(uas[kk]))
        blasfeo_dvecze(nu[kk], &suas[kk], 0, &sRinv[kk], 0, &sRinvCal[kk], 0);
    }

    #if DEBUG == 1
    for (int kk = 0; kk < Nn; kk++) {
        blasfeo_unpack_dvec(sqmod[kk].m, &sqmod[kk], 0, &hmod[indh]);
        blasfeo_unpack_dvec(sx[kk].m, &sx[kk], 0, &xit[indx]);
        blasfeo_unpack_dvec(sQinvCal[kk].m, &sQinvCal[kk], 0, &QinvCal[indx]);
        indh += sqmod[kk].m;
        indx += sx[kk].m;
        blasfeo_unpack_dvec(srmod[kk].m, &srmod[kk], 0, &hmod[indh]);
        blasfeo_unpack_dvec(su[kk].m, &su[kk], 0, &uit[indu]);
        blasfeo_unpack_dvec(sRinvCal[kk].m, &sRinvCal[kk], 0, &RinvCal[indu]);
        indh += srmod[kk].m;
        indu += su[kk].m;
    }
    // printf("dimh = %d, indh = %d\n", dimh, indh);
    write_double_vector_to_txt(hmod, dimh, "examples/data_spring_mass/hmod.txt");
    write_double_vector_to_txt(xit, dimx, "examples/data_spring_mass/xit.txt");
    write_double_vector_to_txt(uit, dimu, "examples/data_spring_mass/uit.txt");
    write_double_vector_to_txt(QinvCal, dimx, "examples/data_spring_mass/Qinvcal.txt");
    write_double_vector_to_txt(RinvCal, dimu, "examples/data_spring_mass/Rinvcal.txt");
    free(hmod);
    free(xit);
    free(uit);
    free(QinvCal);
    free(RinvCal);
    #endif
}


#ifdef _CHECK_LAST_ACTIVE_SET_
static void compare_with_previous_active_set(int isLeaf, int indx, treeqp_tdunes_workspace *work) {

    int *xasChanged = work->xasChanged;
    int *uasChanged = work->uasChanged;

    struct blasfeo_dvec *sxas = &work->sxas[indx];
    struct blasfeo_dvec *suas = &work->suas[indx];
    struct blasfeo_dvec *sxasPrev = &work->sxasPrev[indx];
    struct blasfeo_dvec *suasPrev = &work->suasPrev[indx];

    xasChanged[indx] = 0;
    for (int ii = 0; ii < sxas->m; ii++) {
        if (DVECEL_LIBSTR(sxas, ii) != DVECEL_LIBSTR(sxasPrev, ii)) {
            xasChanged[indx] = 1;
            break;
        }
    }
    blasfeo_dveccp(sxas->m, sxas, 0, sxasPrev, 0);

    if (!isLeaf) {
        uasChanged[indx] = 0;
        for (int ii = 0; ii < suas->m; ii++) {
            if (DVECEL_LIBSTR(suas, ii) != DVECEL_LIBSTR(suasPrev, ii)) {
                uasChanged[indx] = 1;
                break;
            }
        }
        blasfeo_dveccp(suas->m, suas, 0, suasPrev, 0);
    }
}


static int find_starting_point_of_factorization(struct node *tree, treeqp_tdunes_workspace *work) {
    int idxdad, asDadChanged;
    int Np = work->Np;
    int idxFactorStart = Np;
    int *xasChanged = work->xasChanged;
    int *uasChanged = work->uasChanged;
    int *blockChanged = work->blockChanged;

    for (int kk = 0; kk < Np; kk++) {
        blockChanged[kk] = 0;
    }

    // TODO(dimitris):check if it's worth parallelizing
    // --> CAREFULLY THOUGH since multiple threads write on same memory
    for (int kk = work->Nn-1; kk > 0; kk--) {
        idxdad = tree[kk].dad;
        asDadChanged = xasChanged[idxdad] | uasChanged[idxdad];

        if (asDadChanged || xasChanged[kk]) blockChanged[idxdad] = 1;
    }
    for (int kk = Np-1; kk >= 0; kk--) {
        if (!blockChanged[kk]) {
            idxFactorStart--;
        } else {
            break;
        }
    }
    return idxFactorStart;
}

#endif  /* _CHECK_LAST_ACTIVE_SET_ */

// TODO(dimitris): one, two, inf norms efficiently in blasfeo?
// TODO(dimitris): benchmark different stopping criteria
// TODO(dimitris): check if it is slower or faster when parallelized
static double calculate_error_in_residuals(termination_t condition, treeqp_tdunes_workspace *work) {
    double error = 0;
    int Np = work->Np;
    struct blasfeo_dvec *sres = work->sres;

    if ((condition == TREEQP_SUMSQUAREDERRORS) || (condition == TREEQP_TWONORM)) {
        for (int kk = 0; kk < Np; kk++) {
            error += blasfeo_ddot(sres[kk].m, &sres[kk], 0, &sres[kk], 0);
        }
        if (condition == TREEQP_TWONORM) error = sqrt(error);
    } else if (condition == TREEQP_INFNORM) {
        for (int kk = 0; kk < Np; kk++) {
            for (int ii = 0; ii < sres[kk].m; ii++) {
                error = MAX(error, ABS(DVECEL_LIBSTR(&sres[kk], ii)));
            }
        }
    } else {
        printf("[TREEQP] Unknown termination condition!\n");
        exit(1);
    }
    // printf("error=%2.3e\n",error);
    return error;
}


static return_t build_dual_problem(tree_ocp_qp_in *qp_in, int *idxFactorStart,
    treeqp_tdunes_options_t *opts, treeqp_tdunes_workspace *work) {

    int idxdad, idxpos, idxsib, idxii, ns, isLeaf, asDadChanged;
    double error;

    int *nx = (int *)qp_in->nx;
    int *nu = (int *)qp_in->nu;

    #ifdef _CHECK_LAST_ACTIVE_SET_
    int *xasChanged = work->xasChanged;
    int *uasChanged = work->uasChanged;
    struct blasfeo_dmat *sWdiag = work->sWdiag;
    #endif

    int Nn = work->Nn;
    int Np = work->Np;

    struct blasfeo_dmat *sA = (struct blasfeo_dmat *) qp_in->A;
    struct blasfeo_dmat *sB = (struct blasfeo_dmat *) qp_in->B;
    struct blasfeo_dvec *sb = (struct blasfeo_dvec *) qp_in->b;
    struct node *tree = (struct node *)qp_in->tree;

    struct blasfeo_dvec *sQinvCal = work->sQinvCal;
    struct blasfeo_dvec *sRinvCal = work->sRinvCal;

    struct blasfeo_dvec *sx = work->sx;
    struct blasfeo_dvec *su = work->su;

    struct blasfeo_dmat *sM = work->sM;
    struct blasfeo_dmat *sW = work->sW;
    struct blasfeo_dmat *sUt = work->sUt;
    struct blasfeo_dvec *sres = work->sres;
    struct blasfeo_dvec *sresMod = work->sresMod;
    struct blasfeo_dvec *regMat = work->regMat;

    *idxFactorStart = -1;

    #if DEBUG == 1
    int indres = 0;
    int dimres = number_of_states(qp_in) - qp_in->nx[0];
    double res[dimres];
    int dimW = 0;
    int dimUt = 0;
    for (int kk = 0; kk < Np; kk++) {
        dimW += sW[kk].n*sW[kk].n;  // NOTE(dimitris): not m, as it may be equal to n+1
        if (kk > 0) dimUt += sUt[kk-1].m*sUt[kk-1].n;
    }
    double W[dimW], Ut[dimUt];
    int indW = 0;
    int indUt = 0;
    #endif

    #ifdef _CHECK_LAST_ACTIVE_SET_
    // TODO(dimitris): check if it's worth to parallelize
    for (int kk = Nn-1; kk >= 0; kk--) {
        isLeaf = (tree[kk].nkids > 0 ? 0:1);
        // NOTE(dimitris): updates both xasChanged/uasChanged and xasPrev/uasPrev
        compare_with_previous_active_set(isLeaf, kk, work);
    }
    // TODO(dimitris): double check that this indx is correct (not higher s.t. we loose efficiency)
    *idxFactorStart = find_starting_point_of_factorization(tree, work);
    #endif

    #ifdef PARALLEL
    #pragma omp parallel for private(idxdad, idxpos)
    #endif
    // Calculate dual gradient
    // TODO(dimitris): can we merge with solution of stage QPs without problems in parallelizing?
    for (int kk = Nn-1; kk > 0; kk--) {
        idxdad = tree[kk].dad;
        idxpos = work->idxpos[kk];

        // TODO(dimitris): decide on convention for comments (+offset or not)

        // res[k] = b[k] - x[k]
        blasfeo_daxpy(nx[kk], -1.0, &sx[kk], 0, &sb[kk-1], 0, &sres[idxdad], idxpos);

        // res[k] += A[k]*x[idxdad]
        blasfeo_dgemv_n(nx[kk], nx[idxdad], 1.0, &sA[kk-1], 0, 0, &sx[idxdad], 0, 1.0, &sres[idxdad],
            idxpos, &sres[idxdad], idxpos);

        // res[k] += B[k]*u[idxdad]
        blasfeo_dgemv_n(nx[kk], nu[idxdad], 1.0, &sB[kk-1], 0, 0, &su[idxdad], 0, 1.0, &sres[idxdad],
            idxpos, &sres[idxdad], idxpos);

        // resMod[k] = res[k]
        blasfeo_dveccp(nx[kk], &sres[idxdad], idxpos, &sresMod[idxdad], idxpos);
    }

    // Check termination condition
    error = calculate_error_in_residuals(opts->termCondition, work);
    if (error < opts->stationarityTolerance) {
        return TREEQP_SUCC_OPTIMAL_SOLUTION_FOUND;
    }
    #ifdef PARALLEL
    #pragma omp parallel for private(idxdad, idxpos, idxsib, idxii, ns, asDadChanged)
    #endif
    // Calculate dual Hessian
    for (int kk = Nn-1; kk > 0; kk--) {
        idxdad = tree[kk].dad;
        idxpos = work->idxpos[kk];

        #ifdef _CHECK_LAST_ACTIVE_SET_
        asDadChanged = xasChanged[idxdad] | uasChanged[idxdad];
        #endif

        // Filling W[idxdad] and Ut[idxdad-1]

        #ifdef _CHECK_LAST_ACTIVE_SET_
        // TODO(dimitris): if only xasChanged, remove QinvCalPrev and add new
        if (asDadChanged || xasChanged[kk]) {
        #endif

        // --- intermediate result (used both for Ut and W)

        // M = A[k] * Qinvcal[idxdad]
                blasfeo_dgemm_nd(nx[kk], nx[idxdad], 1.0,  &sA[kk-1], 0, 0, &sQinvCal[idxdad], 0, 0.0,
            &sM[kk], 0, 0, &sM[kk], 0, 0);

            // --- hessian contribution of parent (Ut)

        #ifdef _CHECK_LAST_ACTIVE_SET_
        if (asDadChanged && tree[idxdad].dad >= 0) {
        #else
        if (tree[idxdad].dad >= 0) {
        #endif
            // Ut[idxdad]+offset = M' = - A[k] *  Qinvcal[idxdad]
            blasfeo_dgetr(nx[kk], nx[idxdad], &sM[kk], 0, 0, &sUt[idxdad-1], 0, idxpos);
            blasfeo_dgesc(nx[idxdad], nx[kk], -1.0, &sUt[idxdad-1], 0, idxpos);
        }

        // --- hessian contribution of node (diagonal block of W)

        // W[idxdad]+offset = A[k]*M^T = A[k]*Qinvcal[idxdad]*A[k]'
        blasfeo_dsyrk_ln(nx[kk], nx[idxdad], 1.0, &sA[kk-1], 0, 0, &sM[kk], 0, 0, 0.0, &sW[idxdad],
            idxpos, idxpos, &sW[idxdad], idxpos, idxpos);

        // M = B[k]*Rinvcal[idxdad]
        blasfeo_dgemm_nd(nx[kk], nu[idxdad], 1.0,  &sB[kk-1], 0, 0, &sRinvCal[idxdad], 0, 0.0,
            &sM[kk], 0, 0, &sM[kk], 0, 0);

        // W[idxdad]+offset += B[k]*M^T = B[k]*Rinvcal[idxdad]*B[k]'
        blasfeo_dsyrk_ln(nx[kk], nu[idxdad], 1.0, &sB[kk-1], 0, 0, &sM[kk], 0, 0, 1.0, &sW[idxdad],
            idxpos, idxpos, &sW[idxdad], idxpos, idxpos);

        // W[idxdad]+offset += Qinvcal[k]
        blasfeo_ddiaad(nx[kk], 1.0, &sQinvCal[kk], 0, &sW[idxdad], idxpos, idxpos);

        // W[idxdad]+offset += regMat (regularization)
        blasfeo_ddiaad(nx[kk], 1.0, regMat, 0, &sW[idxdad], idxpos, idxpos);

        #ifdef _CHECK_LAST_ACTIVE_SET_
        // save diagonal block that will be overwritten in factorization
        blasfeo_dgecp(nx[kk], nx[kk], &sW[idxdad], idxpos, idxpos, &sWdiag[kk], 0, 0);
        #endif

        // --- hessian contribution of preceding siblings (off-diagonal blocks of W)

        #ifdef _CHECK_LAST_ACTIVE_SET_
        if (asDadChanged) {
        #endif
        ns = tree[idxdad].nkids - 1;  // number of siblings
        idxii = 0;
        for (int ii = 0; ii < ns; ii++) {
            idxsib = tree[idxdad].kids[ii];
            if (idxsib == kk) break;  // completed all preceding siblings

            // M = A[idxsib] * Qinvcal[idxdad]
            blasfeo_dgemm_nd(nx[idxsib], nx[idxdad], 1.0,  &sA[idxsib-1], 0, 0,
                &sQinvCal[idxdad], 0, 0.0, &sM[kk], 0, 0, &sM[kk], 0, 0);

            // W[idxdad]+offset = A[k]*M^T = A[k]*Qinvcal[idxdad]*A[idxsib]'
            blasfeo_dgemm_nt(nx[kk], nx[idxsib], nx[idxdad], 1.0, &sA[kk-1], 0, 0,
                &sM[kk], 0, 0, 0.0, &sW[idxdad], idxpos, idxii, &sW[idxdad], idxpos, idxii);

            // M = B[idxsib]*Rinvcal[idxdad]
            blasfeo_dgemm_nd(nx[idxsib], nu[idxdad], 1.0, &sB[idxsib-1], 0, 0,
                &sRinvCal[idxdad], 0, 0.0, &sM[kk], 0, 0, &sM[kk], 0, 0);

            // W[idxdad]+offset += B[k]*M^T = B[k]*Rinvcal[idxdad]*B[idxsib]'
            blasfeo_dgemm_nt(nx[kk], nx[idxsib], nu[idxdad], 1.0, &sB[kk-1], 0, 0,
                &sM[kk], 0, 0, 1.0, &sW[idxdad], idxpos, idxii, &sW[idxdad], idxpos, idxii);

            // idxiiOLD = ii*qp_in->nx[1];
            idxii += nx[idxsib];
        }
        #ifdef _CHECK_LAST_ACTIVE_SET_
        }
        #endif

        #ifdef _CHECK_LAST_ACTIVE_SET_
        } else {
            blasfeo_dgecp(nx[kk], nx[kk], &sWdiag[kk], 0, 0, &sW[idxdad], idxpos, idxpos);
        }
        #endif
    }

    #if DEBUG == 1
    for (int kk = 0; kk < Np; kk++) {
        blasfeo_unpack_dvec(sres[kk].m, &sres[kk], 0, &res[indres]);
        indres += sres[kk].m;
        blasfeo_unpack_dmat(sW[kk].n, sW[kk].n, &sW[kk], 0, 0, &W[indW], sW[kk].n);
        indW += sW[kk].n*sW[kk].n;
        if (kk > 0) {
            blasfeo_unpack_dmat( sUt[kk-1].m, sUt[kk-1].n, &sUt[kk-1], 0, 0,
                &Ut[indUt], sUt[kk-1].m);
            indUt += sUt[kk-1].m*sUt[kk-1].n;
        }
    }
    write_double_vector_to_txt(res, dimres, "examples/data_spring_mass/res.txt");
    write_double_vector_to_txt(W, dimW, "examples/data_spring_mass/W.txt");
    write_double_vector_to_txt(Ut, dimUt, "examples/data_spring_mass/Ut.txt");
    #endif

    return TREEQP_OK;
}


static void calculate_delta_lambda(tree_ocp_qp_in *qp_in, int idxFactorStart,
    treeqp_tdunes_workspace *work) {

    struct node *tree = (struct node *)qp_in->tree;
    int idxdad, idxpos;
    int Nn = qp_in->N;
    int Nh = tree[Nn-1].stage;
    int Np = work->Np;
    int icur = Np-1;
    int *npar = work->npar;
    int *nx = (int *)qp_in->nx;

    struct blasfeo_dmat *sW = work->sW;
    struct blasfeo_dmat *sUt = work->sUt;
    struct blasfeo_dmat *sCholW = work->sCholW;
    struct blasfeo_dmat *sCholUt = work->sCholUt;
    struct blasfeo_dvec *sresMod = work->sresMod;
    struct blasfeo_dvec *sDeltalambda = work->sDeltalambda;

    #if DEBUG == 1
    int dimlam = number_of_states(qp_in) - qp_in->nx[0];
    double deltalambda[dimlam];
    int indlam = 0;
    #endif

    // --- Cholesky factorization merged with backward substitution

    for (int kk = Nh-1; kk > 0; kk--) {
        #if PRINT_LEVEL > 2
        printf("\n--------- New (parallel) factorization branch  ---------\n");
        #endif
        #ifdef PARALLEL
        #pragma omp parallel for private(idxdad, idxpos)
        #endif
        for (int ii = icur; ii > icur-npar[kk]; ii--) {

            // NOTE(dimitris): result of backward substitution saved in deltalambda
            // NOTE(dimitris): substitution for free if dual[ii].W not multiple of 4 (in LA=HP)
            #ifdef _MERGE_FACTORIZATION_WITH_SUBSTITUTION_

            #ifdef _CHECK_LAST_ACTIVE_SET_
            if (ii < idxFactorStart) {
            #endif

            // add resMod in last row of matrix W
            blasfeo_drowin(sresMod[ii].m, 1.0, &sresMod[ii], 0, &sW[ii], sW[ii].m-1, 0);
            // perform Cholesky factorization and backward substitution together
            blasfeo_dpotrf_l_mn(sW[ii].m, sW[ii].n, &sW[ii], 0, 0, &sCholW[ii], 0, 0);
            // extract result of substitution
            blasfeo_drowex(sDeltalambda[ii].m, 1.0, &sCholW[ii], sCholW[ii].m-1, 0,
                &sDeltalambda[ii], 0);

            #ifdef _CHECK_LAST_ACTIVE_SET_
            } else {
            // perform only vector substitution
            blasfeo_dtrsv_lnn(sresMod[ii].m, &sCholW[ii], 0, 0, &sresMod[ii], 0,
                &sDeltalambda[ii], 0);
            }
            #endif

            #else  /* _MERGE_FACTORIZATION_WITH_SUBSTITUTION_ */

            #ifdef _CHECK_LAST_ACTIVE_SET_
            if (ii < idxFactorStart) {
            #endif
            // Cholesky factorization to calculate factor of current diagonal block
            blasfeo_dpotrf_l(sW[ii].n, &sW[ii], 0, 0, &sCholW[ii], 0, 0);
            #ifdef _CHECK_LAST_ACTIVE_SET_
            }  // TODO(dimitris): we can probably skip more calculations (see scenarios)
            #endif

            // vector substitution
            blasfeo_dtrsv_lnn(sresMod[ii].m, &sCholW[ii], 0, 0, &sresMod[ii], 0,
                &sDeltalambda[ii], 0);

            #endif  /* _MERGE_FACTORIZATION_WITH_SUBSTITUTION_ */

            // Matrix substitution to calculate transposed factor of parent block
            blasfeo_dtrsm_rltn( sUt[ii-1].m , sUt[ii-1].n, 1.0, &sCholW[ii], 0, 0,
                &sUt[ii-1], 0, 0, &sCholUt[ii-1], 0, 0);

            // Symmetric matrix multiplication to update diagonal block of parent
            // NOTE(dimitris): use blasfeo_dgemm_nt if dsyrk not implemented yet
            idxdad = tree[ii].dad;
            idxpos = work->idxpos[ii];

            blasfeo_dsyrk_ln(sCholUt[ii-1].m, sCholUt[ii-1].n, -1.0,
                &sCholUt[ii-1], 0, 0, &sCholUt[ii-1], 0, 0, 1.0,
                &sW[idxdad], idxpos, idxpos, &sW[idxdad], idxpos, idxpos);

            // Matrix vector multiplication to update vector of parent
            blasfeo_dgemv_n(sCholUt[ii-1].m, sCholUt[ii-1].n, -1.0, &sCholUt[ii-1], 0, 0,
                &sDeltalambda[ii], 0, 1.0, &sresMod[idxdad], idxpos, &sresMod[idxdad], idxpos);
        }
        icur -= npar[kk];
    }
    #ifdef _MERGE_FACTORIZATION_WITH_SUBSTITUTION_
    // add resMod in last row of matrix W
    blasfeo_drowin(sresMod[0].m, 1.0, &sresMod[0], 0, &sW[0], sW[0].m-1, 0);
    // perform Cholesky factorization and backward substitution together
    blasfeo_dpotrf_l_mn(sW[0].m, sW[0].n, &sW[0], 0, 0, &sCholW[0], 0, 0);
    // extract result of substitution
    blasfeo_drowex(sDeltalambda[0].m, 1.0, &sCholW[0], sCholW[0].m-1, 0, &sDeltalambda[0], 0);
    #else
    // calculate Cholesky factor of root block
    blasfeo_dpotrf_l(sW[0].m, &sW[0], 0, 0, &sCholW[0], 0, 0);

    // calculate last elements of backward substitution
    blasfeo_dtrsv_lnn(sresMod[0].m, &sCholW[0], 0, 0, &sresMod[0], 0, &sDeltalambda[0], 0);
    #endif

    // --- Forward substitution

    icur = 1;

    blasfeo_dtrsv_ltn(sDeltalambda[0].m, &sCholW[0], 0, 0, &sDeltalambda[0], 0, &sDeltalambda[0], 0);

    for (int kk = 1; kk < Nh; kk++) {
        #ifdef PARALLEL
        #pragma omp parallel for private(idxdad, idxpos)
        #endif
        for (int ii = icur; ii < icur+npar[kk]; ii++) {
            idxdad = tree[ii].dad;
            idxpos = work->idxpos[ii];

            blasfeo_dgemv_t(sCholUt[ii-1].m, sCholUt[ii-1].n, -1.0, &sCholUt[ii-1], 0, 0,
                &sDeltalambda[idxdad], idxpos, 1.0, &sDeltalambda[ii], 0, &sDeltalambda[ii], 0);

            blasfeo_dtrsv_ltn(sDeltalambda[ii].m, &sCholW[ii], 0, 0, &sDeltalambda[ii], 0,
                &sDeltalambda[ii], 0);
        }
        icur += npar[kk];
    }

    #if PRINT_LEVEL > 2
    for (int ii = 0; ii < Np; ii++) {
        printf("\nCholesky factor of diagonal block #%d as strmat: \n\n", ii+1);
        blasfeo_print_dmat( sCholW[ii].m, sCholW[ii].n, &sCholW[ii], 0, 0);
    }
    for (int ii = 1; ii < Np; ii++) {
        printf("\nTransposed Cholesky factor of parent block #%d as strmat: \n\n", ii+1);
        blasfeo_print_dmat(sCholUt[ii-1].m, sCholUt[ii-1].n, &sCholUt[ii-1], 0, 0);
    }

    printf("\nResult of backward substitution:\n\n");
    for (int ii = 0; ii < Np; ii++) {
        blasfeo_print_dvec(sDeltalambda[0].m, &sDeltalambda[0], 0);
    }

    printf("\nResult of forward substitution (aka final result):\n\n");
    for (int ii = 0; ii < Np; ii++) {
        blasfeo_print_dvec(sDeltalambda[ii].m, &sDeltalambda[ii], 0);
    }
    #endif

    #if DEBUG == 1
    for (int kk = 0; kk < Np; kk++) {
        blasfeo_unpack_dvec(sDeltalambda[kk].m, &sDeltalambda[kk], 0, &deltalambda[indlam]);
        indlam += sDeltalambda[kk].m;
    }
    write_double_vector_to_txt(deltalambda, dimlam, "examples/data_spring_mass/deltalambda.txt");
    #endif
}


static double gradient_trans_times_direction(treeqp_tdunes_workspace *work) {
    double ans = 0;
    struct blasfeo_dvec *sres = work->sres;
    struct blasfeo_dvec *sDeltalambda = work->sDeltalambda;

    for (int kk = 0; kk < work->Np; kk++) {
        ans += blasfeo_ddot(sres[kk].m, &sres[kk], 0, &sDeltalambda[kk], 0);
    }
    // NOTE(dimitris): res has was -gradient above
    return -ans;
}


static double evaluate_dual_function(tree_ocp_qp_in *qp_in, treeqp_tdunes_workspace *work) {
    int ii, jj, kk, idxkid, idxpos, idxdad;
    double fval = 0;

    int Nn = work->Nn;
    int Np = work->Np;

    int *nx = (int *)qp_in->nx;
    int *nu = (int *)qp_in->nu;

    double *fvals = work->fval;
    double *cmod = work->cmod;

    struct blasfeo_dvec *sx = work->sx;
    struct blasfeo_dvec *su = work->su;
    struct blasfeo_dvec *sxas = work->sxas;
    struct blasfeo_dvec *suas = work->suas;

    struct blasfeo_dmat *sA = (struct blasfeo_dmat *) qp_in->A;
    struct blasfeo_dmat *sB = (struct blasfeo_dmat *) qp_in->B;
    struct blasfeo_dvec *sb = (struct blasfeo_dvec *) qp_in->b;

    struct blasfeo_dvec *sQ = (struct blasfeo_dvec *) work->sQ;
    struct blasfeo_dvec *sR = (struct blasfeo_dvec *) work->sR;
    struct blasfeo_dvec *sq = (struct blasfeo_dvec *) qp_in->q;
    struct blasfeo_dvec *sr = (struct blasfeo_dvec *) qp_in->r;
    struct blasfeo_dvec *sQinv = work->sQinv;
    struct blasfeo_dvec *sRinv = work->sRinv;
    struct blasfeo_dvec *sqmod = work->sqmod;
    struct blasfeo_dvec *srmod = work->srmod;

    struct blasfeo_dvec *sxmin = (struct blasfeo_dvec *) qp_in->xmin;
    struct blasfeo_dvec *sxmax = (struct blasfeo_dvec *) qp_in->xmax;
    struct blasfeo_dvec *sumin = (struct blasfeo_dvec *) qp_in->umin;
    struct blasfeo_dvec *sumax = (struct blasfeo_dvec *) qp_in->umax;

    struct node *tree = (struct node *)qp_in->tree;

    struct blasfeo_dvec *slambda = work->slambda;
    #ifdef PARALLEL
    #pragma omp parallel for private(ii, jj, idxkid, idxpos, idxdad)
    #endif
    // NOTE: same code as in solve_stage_problems but:
    // - without calculating as
    // - without calculating elimination matrix
    // - with calculating modified constant term
    for (kk = 0; kk < Nn; kk++) {
        idxdad = tree[kk].dad;
        idxpos = work->idxpos[kk];

        // --- update QP gradient

        // qmod[k] = - q[k] + lambda[k]
        if (kk == 0) {
            // lambda[0] = 0
            for (jj = 0; jj < nx[kk]; jj++) DVECEL_LIBSTR(&sqmod[kk], jj) = 0.0;
            blasfeo_daxpy(nx[kk], -1.0, &sq[kk], 0, &sqmod[kk], 0, &sqmod[kk], 0);
        } else {
            blasfeo_daxpy(nx[kk], -1.0, &sq[kk], 0, &slambda[idxdad], idxpos, &sqmod[kk], 0);
        }

        // rmod[k] = - r[k]
        if (kk < Np) {
            blasfeo_dveccp(nu[kk], &sr[kk], 0, &srmod[kk], 0);
            blasfeo_dvecsc(nu[kk], -1.0, &srmod[kk], 0);
        }

        // cmod[k] = 0
        cmod[kk] = 0.;

        for (ii = 0; ii < tree[kk].nkids; ii++) {
            idxkid = tree[kk].kids[ii];
            idxdad = tree[idxkid].dad;
            idxpos = work->idxpos[idxkid];

            // cmod[k] += b[jj]' * lambda[jj]
            cmod[kk] += blasfeo_ddot(nx[kk], &sb[idxkid-1], 0, &slambda[idxdad], idxpos);

            // return x^T * y

            // qmod[k] -= A[jj]' * lambda[jj]
            blasfeo_dgemv_t(nx[idxkid], nx[idxdad], -1.0, &sA[idxkid-1], 0, 0,
                &slambda[idxdad], idxpos, 1.0, &sqmod[kk], 0, &sqmod[kk], 0);
            if (kk < Np) {
                // rmod[k] -= B[jj]' * lambda[jj]
                blasfeo_dgemv_t(nx[idxkid], nu[idxdad], -1.0, &sB[idxkid-1], 0, 0,
                    &slambda[idxdad], idxpos, 1.0, &srmod[kk], 0, &srmod[kk], 0);
            }
        }

        // --- solve QP
        // x[k] = Q[k]^-1 .* qmod[k] (NOTE: minus sign already in mod. gradient)
        blasfeo_dvecmuldot(nx[kk], &sQinv[kk], 0, &sqmod[kk], 0, &sx[kk], 0);

        // x[k] = median(xmin, x[k], xmax)
        blasfeo_dveccl(nx[kk], &sxmin[kk], 0, &sx[kk], 0, &sxmax[kk], 0, &sx[kk], 0);

        if (kk < Np) {
            // u[k] = R[k]^-1 .* rmod[k]
            blasfeo_dvecmuldot(nu[kk], &sRinv[kk], 0, &srmod[kk], 0, &su[kk], 0);
            // u[k] = median(umin, u[k], umax)
            blasfeo_dveccl(nu[kk], &sumin[kk], 0, &su[kk], 0, &sumax[kk], 0, &su[kk], 0);
        }

        // --- calculate dual function term

        // feval = - (1/2)x[k]' * Q[k] * x[k] + x[k]' * qmod[k] - cmod[k]
        // NOTE: qmod[k] has already a minus sign
        // NOTE: xas used as workspace
        blasfeo_dvecmuldot(nx[kk], &sQ[kk], 0, &sx[kk], 0, &sxas[kk], 0);
        fvals[kk] = -0.5*blasfeo_ddot(nx[kk], &sxas[kk], 0, &sx[kk], 0) - cmod[kk];
        fvals[kk] += blasfeo_ddot(nx[kk], &sqmod[kk], 0, &sx[kk], 0);

        if (kk < Np) {
            // feval -= (1/2)u[k]' * R[k] * u[k] - u[k]' * rmod[k]
            blasfeo_dvecmuldot(nu[kk], &sR[kk], 0, &su[kk], 0, &suas[kk], 0);
            fvals[kk] -= 0.5*blasfeo_ddot(nu[kk], &suas[kk], 0, &su[kk], 0);
            fvals[kk] += blasfeo_ddot(nu[kk], &srmod[kk], 0, &su[kk], 0);
        }
    }

    for (kk = 0; kk < Nn; kk++) fval += fvals[kk];

    return fval;
}


static int line_search(tree_ocp_qp_in *qp_in, treeqp_tdunes_options_t *opts,
    treeqp_tdunes_workspace *work) {

    int Nn = qp_in->N;
    int Np = work->Np;

    struct node *tree = (struct node *)qp_in->tree;

    #if DEBUG == 1
    int dimlam = number_of_states(qp_in) - qp_in->nx[0];
    double *lambda = malloc(dimlam*sizeof(double));
    int indlam = 0;
    #endif

    struct blasfeo_dvec *slambda = work->slambda;
    struct blasfeo_dvec *sDeltalambda = work->sDeltalambda;

    double dotProduct, fval, fval0;
    double tau = 1;
    double tauPrev = 0;

    dotProduct = gradient_trans_times_direction(work);
    fval0 = evaluate_dual_function(qp_in, work);
    // printf(" dot_product = %f\n", dotProduct);
    // printf(" dual_function = %f\n", fval0);

    int lsIter;

    for (lsIter = 1; lsIter <= opts->lineSearchMaxIter; lsIter++) {
        // update multipliers
        #ifdef PARALLEL
        #pragma omp parallel for
        #endif
        for (int kk = 0; kk < Np; kk++) {
            blasfeo_daxpy( sDeltalambda[kk].m, tau-tauPrev, &sDeltalambda[kk], 0, &slambda[kk], 0,
                &slambda[kk], 0);
        }

        // evaluate dual function
        fval = evaluate_dual_function(qp_in, work);
        // printf("LS iteration #%d (fval = %f <? %f )\n", lsIter, fval, fval0 + opts->lineSearchGamma*tau*dotProduct);

        // check condition
        if (fval < fval0 + opts->lineSearchGamma*tau*dotProduct) {
            // printf("Condition satisfied at iteration %d\n", lsIter);
            break;
        } else {
            tauPrev = tau;
            tau = opts->lineSearchBeta*tauPrev;
        }
    }
    #if DEBUG == 1
    for (int kk = 0; kk < Np; kk++) {
        blasfeo_unpack_dvec( slambda[kk].m, &slambda[kk], 0, &lambda[indlam]);
        indlam += slambda[kk].m;
    }
    write_double_vector_to_txt(lambda, dimlam, "examples/data_spring_mass/lambda_opt.txt");
    write_double_vector_to_txt(&dotProduct, 1, "examples/data_spring_mass/dotProduct.txt");
    write_double_vector_to_txt(&fval0, 1, "examples/data_spring_mass/fval0.txt");
    write_int_vector_to_txt(&lsIter, 1, "examples/data_spring_mass/lsiter.txt");
    free(lambda);
    #endif

    return lsIter;
}


void write_solution_to_txt(tree_ocp_qp_in *qp_in, int Np, int iter, struct node *tree,
    treeqp_tdunes_workspace *work) {

    int kk, indx, indu, ind;

    int Nn = qp_in->N;
    int dimx = number_of_states(qp_in);
    int dimu = number_of_controls(qp_in);
    int dimlam = dimx - qp_in->nx[0];

    struct blasfeo_dvec *sx = work->sx;
    struct blasfeo_dvec *su = work->su;

    struct blasfeo_dvec *slambda = work->slambda;
    struct blasfeo_dvec *sDeltalambda = work->sDeltalambda;

    double *x = malloc(dimx*sizeof(double));
    double *u = malloc(dimu*sizeof(double));
    double *deltalambda = malloc(dimlam*sizeof(double));
    double *lambda = malloc(dimlam*sizeof(double));

    indx = 0; indu = 0;
    for (kk = 0; kk < Nn; kk++) {
        blasfeo_unpack_dvec(sx[kk].m, &sx[kk], 0, &x[indx]);
        indx += sx[kk].m;
        if (kk < Np) {
            blasfeo_unpack_dvec(su[kk].m, &su[kk], 0, &u[indu]);
            indu += su[kk].m;
        }
    }

    ind = 0;
    for (kk = 0; kk < Np; kk++) {
        blasfeo_unpack_dvec(sDeltalambda[kk].m, &sDeltalambda[kk], 0, &deltalambda[ind]);
        blasfeo_unpack_dvec(slambda[kk].m, &slambda[kk], 0, &lambda[ind]);
        ind += slambda[kk].m;
    }

    write_double_vector_to_txt(x, dimx, "examples/data_spring_mass/x_opt.txt");
    write_double_vector_to_txt(u, dimu, "examples/data_spring_mass/u_opt.txt");
    write_double_vector_to_txt(lambda, dimlam, "examples/data_spring_mass/deltalambda_opt.txt");
    write_double_vector_to_txt(lambda, dimlam, "examples/data_spring_mass/lambda_opt.txt");
    write_int_vector_to_txt(&iter, 1, "examples/data_spring_mass/iter.txt");

    #if PROFILE > 0
    write_timers_to_txt();
    #endif

    free(x);
    free(u);
    free(deltalambda);
    free(lambda);
}

int treeqp_tdunes_solve(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out,
    treeqp_tdunes_options_t *opts, treeqp_tdunes_workspace *work) {

    int status;
    int idxFactorStart;  // TODO(dimitris): move to workspace
    int lsIter;

    treeqp_timer solver_tmr, interface_tmr;

    int *nx = (int *)qp_in->nx;
    int *nu = (int *)qp_in->nu;

    int NewtonIter;

    struct node *tree = (struct node *)qp_in->tree;

    int Nn = work->Nn;
    int Nh = qp_in->tree[Nn-1].stage;
    int Np = work->Np;
    int *npar = work->npar;
    struct blasfeo_dvec *regMat = work->regMat;

    // ------ initialization
    treeqp_tic(&interface_tmr);

    treeqp_tdunes_clipping_solver_data *clipping_solver_data;
    treeqp_tdunes_qpoases_solver_data *qpoases_solver_data;

    for (int kk = 0; kk < Nn; kk++)
    {
        if (opts->qp_solver[kk] == TREEQP_CLIPPING_SOLVER)
        {
            // TODO(dimitris): TO BE REMOVED!
            blasfeo_ddiaex(nx[kk], 1.0, (struct blasfeo_dmat *)&qp_in->Q[kk], 0, 0, &work->sQ[kk], 0);
            blasfeo_ddiaex(nu[kk], 1.0, (struct blasfeo_dmat *)&qp_in->R[kk], 0, 0, &work->sR[kk], 0);

            for (int nn = 0; nn < qp_in->nx[kk]; nn++)
                DVECEL_LIBSTR(&work->sQinv[kk], nn) = 1.0/DVECEL_LIBSTR(&work->sQ[kk], nn);
            for (int nn = 0; nn < qp_in->nu[kk]; nn++)
                DVECEL_LIBSTR(&work->sRinv[kk], nn) = 1.0/DVECEL_LIBSTR(&work->sR[kk], nn);

            clipping_solver_data = (treeqp_tdunes_clipping_solver_data *)work->stage_qp_data[kk];
            blasfeo_ddiaex(nx[kk], 1.0, (struct blasfeo_dmat *)&qp_in->Q[kk], 0, 0, clipping_solver_data->sQ, 0);
            blasfeo_ddiaex(nu[kk], 1.0, (struct blasfeo_dmat *)&qp_in->R[kk], 0, 0, clipping_solver_data->sR, 0);

            for (int nn = 0; nn < qp_in->nx[kk]; nn++)
                DVECEL_LIBSTR(clipping_solver_data->sQinv, nn) = 1.0/DVECEL_LIBSTR(clipping_solver_data->sQ, nn);
            for (int nn = 0; nn < qp_in->nu[kk]; nn++)
                DVECEL_LIBSTR(clipping_solver_data->sRinv, nn) = 1.0/DVECEL_LIBSTR(clipping_solver_data->sR, nn);
        }

        #ifdef _CHECK_LAST_ACTIVE_SET_
        blasfeo_dvecse(work->sxasPrev[kk].m, 0.0/0.0, &work->sxasPrev[kk], 0);
        if (kk < Np)
            blasfeo_dvecse(work->suasPrev[kk].m, 0.0/0.0, &work->suasPrev[kk], 0);
        #endif
    }

    qp_out->info.interface_time = treeqp_toc(&interface_tmr);
    treeqp_tic(&solver_tmr);

    // ------ dual Newton iterations
    for (NewtonIter = 0; NewtonIter < opts->maxIter; NewtonIter++) {
        #if PROFILE > 1
        treeqp_tic(&iter_tmr);
        #endif

        // solve stage QPs, update active sets, calculate elimination matrices
        #if PROFILE > 2
        treeqp_tic(&tmr);
        #endif
        solve_stage_problems(qp_in, work);
        #if PROFILE > 2
        stage_qps_times[NewtonIter] = treeqp_toc(&tmr);
        #endif

        // calculate gradient and Hessian of the dual problem
        #if PROFILE > 2
        treeqp_tic(&tmr);
        #endif
        status = build_dual_problem(qp_in, &idxFactorStart, opts, work);
        #if PROFILE > 2
        build_dual_times[NewtonIter] = treeqp_toc(&tmr);
        #endif
        if (status == TREEQP_SUCC_OPTIMAL_SOLUTION_FOUND) {
            // printf("optimal solution found\n", 1);
            break;
        }

        // factorize Newton matrix and calculate step direction
        #if PROFILE > 2
        treeqp_tic(&tmr);
        #endif
        calculate_delta_lambda(qp_in, idxFactorStart, work);
        #if PROFILE > 2
        newton_direction_times[NewtonIter] = treeqp_toc(&tmr);
        #endif

        // line-search
        // NOTE: line-search overwrites xas, uas (used as workspace)
        #if PROFILE > 2
        treeqp_tic(&tmr);
        #endif
        lsIter = line_search(qp_in, opts, work);
        #if PROFILE > 2
        line_search_times[NewtonIter] = treeqp_toc(&tmr);
        #endif

        #if PRINT_LEVEL > 1
        printf("iteration #%d: %d ls iterations\n", NewtonIter, lsIter);
        #endif
        #if PROFILE > 1
        iter_times[NewtonIter] = treeqp_toc(&iter_tmr);
        ls_iters[NewtonIter] = lsIter;
        #endif
    }

    qp_out->info.solver_time = treeqp_toc(&solver_tmr);
    treeqp_tic(&interface_tmr);

    // ------ copy solution to qp_out

    for (int kk = 0; kk < Nn; kk++) {
        blasfeo_dveccp(nx[kk], &work->sx[kk], 0, &qp_out->x[kk], 0);
        blasfeo_dveccp(nu[kk], &work->su[kk], 0, &qp_out->u[kk], 0);

        if (kk > 0) {
            blasfeo_dveccp(nx[kk], &work->slambda[tree[kk].dad], work->idxpos[kk],
                &qp_out->lam[kk], 0);
        }
        if (opts->qp_solver[kk] == TREEQP_CLIPPING_SOLVER) {
            // mu_x[kk] = (xUnc[k] - x[k])*Q[k] = -(Q[k]*x[k]+q[k])*abs(xas[k])
            blasfeo_daxpy(nx[kk], -1., &qp_out->x[kk], 0, &work->sxUnc[kk], 0, &qp_out->mu_x[kk], 0);
            blasfeo_daxpy(nu[kk], -1., &qp_out->u[kk], 0, &work->suUnc[kk], 0, &qp_out->mu_u[kk], 0);
            blasfeo_dvecmuldot(nx[kk], &work->sQ[kk], 0, &qp_out->mu_x[kk], 0, &qp_out->mu_x[kk], 0);
            blasfeo_dvecmuldot(nu[kk], &work->sR[kk], 0, &qp_out->mu_u[kk], 0, &qp_out->mu_u[kk], 0);
        }
    }
    qp_out->info.iter = NewtonIter;

    qp_out->info.interface_time += treeqp_toc(&interface_tmr);

    if (qp_out->info.iter == opts->maxIter)
        status = TREEQP_ERR_MAXIMUM_ITERATIONS_REACHED;

    return status;  // TODO(dimitris): return correct status
}


static void update_M_dimensions(int idx, tree_ocp_qp_in *qp_in, int *rowsM, int *colsM){

    int idxdad = qp_in->tree[idx].dad;
    int idxsib;

    if (idx == 0) {
        *rowsM = 0;
        *colsM = 0;
    } else {
        *colsM = MAX(qp_in->nx[idxdad], qp_in->nu[idxdad]);
        *rowsM = 0;

        for (int jj = 0; jj < qp_in->tree[idxdad].nkids; jj++) {
            idxsib = qp_in->tree[idxdad].kids[jj];
            *rowsM = MAX(*rowsM, MAX(qp_in->nx[idxsib], qp_in->nu[idxsib]));
        }
    }
}



static int stage_qp_calculate_size(int nx, int nu, stage_qp_t qp_solver)
{
    int bytes  = 0;

    switch (qp_solver)
    {
        case TREEQP_CLIPPING_SOLVER:
            bytes += sizeof(treeqp_tdunes_clipping_solver_data);
            bytes += 6*sizeof(struct blasfeo_dvec);  // Q, R, Qinv, Rinv, QinvCal, RinvCal
            bytes += 3*blasfeo_memsize_dvec(nx);  // Q, Qinv, QinvCal
            bytes += 3*blasfeo_memsize_dvec(nu);  // R, Rinv, RinvCal
            break;
        case TREEQP_QPOASES_SOLVER:
            // TODO(dimitris)
            break;
        default:
            printf("[TREEQP] Error! Unknown solver.\n");
            exit(1);
    }
    return bytes;
}



static void stage_qp_assign_structs(stage_qp_t qp_solver, void **stage_qp_data, char **c_double_ptr)
{
    treeqp_tdunes_clipping_solver_data *clipping_solver_data;
    treeqp_tdunes_qpoases_solver_data *qpoases_solver_data;

    switch (qp_solver)
    {
        case TREEQP_CLIPPING_SOLVER:

            clipping_solver_data = (treeqp_tdunes_clipping_solver_data *)*c_double_ptr;
            *c_double_ptr += sizeof(treeqp_tdunes_clipping_solver_data);

            clipping_solver_data->sQ = (struct blasfeo_dvec *)*c_double_ptr;
            *c_double_ptr += sizeof(struct blasfeo_dvec);

            clipping_solver_data->sR = (struct blasfeo_dvec *)*c_double_ptr;
            *c_double_ptr += sizeof(struct blasfeo_dvec);

            clipping_solver_data->sQinv = (struct blasfeo_dvec *)*c_double_ptr;
            *c_double_ptr += sizeof(struct blasfeo_dvec);

            clipping_solver_data->sRinv = (struct blasfeo_dvec *)*c_double_ptr;
            *c_double_ptr += sizeof(struct blasfeo_dvec);

            clipping_solver_data->sQinvCal = (struct blasfeo_dvec *)*c_double_ptr;
            *c_double_ptr += sizeof(struct blasfeo_dvec);

            clipping_solver_data->sRinvCal = (struct blasfeo_dvec *)*c_double_ptr;
            *c_double_ptr += sizeof(struct blasfeo_dvec);

            *stage_qp_data = (void *) clipping_solver_data;
            break;
        case TREEQP_QPOASES_SOLVER:
            // TODO(dimitris)
            break;
        default:
            printf("[TREEQP] Error! Unknown solver.\n");
            exit(1);
    }
}



// NOTE(dimitris): structs and data are assigned separately due to alignment requirements
static void stage_qp_assign_data(int nx, int nu, stage_qp_t qp_solver,
    void *stage_qp_data, char **c_double_ptr)
{
    treeqp_tdunes_clipping_solver_data *clipping_solver_data;
    treeqp_tdunes_qpoases_solver_data *qpoases_solver_data;

    switch (qp_solver)
    {
        case TREEQP_CLIPPING_SOLVER:

            clipping_solver_data = (treeqp_tdunes_clipping_solver_data *)stage_qp_data;

            init_strvec(nx, clipping_solver_data->sQ, c_double_ptr);
            init_strvec(nu, clipping_solver_data->sR, c_double_ptr);
            init_strvec(nx, clipping_solver_data->sQinv, c_double_ptr);
            init_strvec(nu, clipping_solver_data->sRinv, c_double_ptr);
            init_strvec(nx, clipping_solver_data->sQinvCal, c_double_ptr);
            init_strvec(nu, clipping_solver_data->sRinvCal, c_double_ptr);
            break;
        case TREEQP_QPOASES_SOLVER:
            // TODO(dimitris)
            break;
        default:
            printf("[TREEQP] Error! Unknown solver.\n");
            exit(1);
    }
}



int treeqp_tdunes_calculate_size(tree_ocp_qp_in *qp_in, treeqp_tdunes_options_t *opts)
{
    struct node *tree = (struct node *) qp_in->tree;
    int bytes = 0;
    int Nn = qp_in->N;
    int Nh = tree[Nn-1].stage;
    int Np = get_number_of_parent_nodes(Nn, tree);
    int regDim = maximum_hessian_block_dimension(qp_in);
    int dim, idxkid;
    int rowsM, colsM;

    // int pointers
    bytes += Nh*sizeof(int);  // npar
    bytes += Nn*sizeof(int);  // idxpos

    #ifdef _CHECK_LAST_ACTIVE_SET_
    bytes += 2*Nn*sizeof(int);  // xasChanged, uasChanged
    bytes += Np*sizeof(int);  // blockChanged
    #endif

    // double pointers
    bytes += 2*Nn*sizeof(double);  // fval, cmod

    // stage QP solvers
    bytes += Nn*sizeof(void *);  // stage_qp_data
    for (int ii = 0; ii < Nn; ii++)
    {
        bytes += stage_qp_calculate_size(qp_in->nx[ii], qp_in->nu[ii], opts->qp_solver[ii]);
    }

    // TODO(dimitris): TO BE REMOVED!
    bytes += 6*Nn*sizeof(struct blasfeo_dvec);  // Q, R, Qinv, Rinv, QinvCal, RinvCal

    // struct pointers
    bytes += 2*Nn*sizeof(struct blasfeo_dvec);  // qmod, rmod
    #ifdef _CHECK_LAST_ACTIVE_SET_
    bytes += Nn*sizeof(struct blasfeo_dmat);  // Wdiag
    #endif
    bytes += 1*sizeof(struct blasfeo_dvec);  // regMat
    bytes += Nn*sizeof(struct blasfeo_dmat);  // M
    bytes += 2*Np*sizeof(struct blasfeo_dmat);  // W, CholW
    bytes += 2*(Np-1)*sizeof(struct blasfeo_dmat);  // Ut, CholUt
    bytes += 4*Np*sizeof(struct blasfeo_dvec);  // res, resMod, lambda, Deltalambda

    bytes += 3*Nn*sizeof(struct blasfeo_dvec);  // x, xUnc, xas
    bytes += 3*Nn*sizeof(struct blasfeo_dvec);  // u, uUnc, uas

    #ifdef _CHECK_LAST_ACTIVE_SET_
    bytes += Nn*sizeof(struct blasfeo_dvec);  // xasPrev
    bytes += Nn*sizeof(struct blasfeo_dvec);  // uasPrev
    #endif

    // structs
    bytes += blasfeo_memsize_dvec(regDim);  // regMat

    for (int ii = 0; ii < Nn; ii++)
    {
        // TODO(dimitris): TO BE REMOVED!
        if (opts->qp_solver[ii] == TREEQP_CLIPPING_SOLVER)
        {
            bytes += 3*blasfeo_memsize_dvec(qp_in->nx[ii]);  // Q, Qinv, QinvCal
            bytes += 3*blasfeo_memsize_dvec(qp_in->nu[ii]);  // R, Rinv, RinvCal
        }

        bytes += blasfeo_memsize_dvec(qp_in->nx[ii]);  // qmod
        bytes += blasfeo_memsize_dvec(qp_in->nu[ii]);  // rmod

        bytes += 3*blasfeo_memsize_dvec(qp_in->nx[ii]);  // x, xUnc, xas
        #ifdef _CHECK_LAST_ACTIVE_SET_
        bytes += blasfeo_memsize_dmat(qp_in->nx[ii], qp_in->nx[ii]);  // Wdiag
        bytes += blasfeo_memsize_dvec(qp_in->nx[ii]);  // xasPrev
        #endif

        bytes += 3*blasfeo_memsize_dvec(qp_in->nu[ii]);  // u, uUnc, uas
        #ifdef _CHECK_LAST_ACTIVE_SET_
        bytes += blasfeo_memsize_dvec(qp_in->nu[ii]);  // uasPrev
        #endif

        update_M_dimensions(ii, qp_in, &rowsM, &colsM);
        bytes += blasfeo_memsize_dmat(rowsM, colsM);  // M

        if (ii < Np)
        {   // NOTE(dimitris): for constant dimensions dim = tree[ii].nkids*nx
            dim = 0;
            for (int jj = 0; jj < tree[ii].nkids; jj++)
            {
                idxkid = tree[ii].kids[jj];
                dim += qp_in->nx[idxkid];
            }

            #ifdef _MERGE_FACTORIZATION_WITH_SUBSTITUTION_
            bytes += 2*blasfeo_memsize_dmat(dim + 1, dim);  // W, CholW
            #else
            bytes += 2*blasfeo_memsize_dmat(dim, dim);  // W, CholW
            #endif
            bytes += 4*blasfeo_memsize_dvec(dim);  // res, resMod, lambda, Deltalambda
            if (ii > 0)
            {
                bytes += 2*blasfeo_memsize_dmat(qp_in->nx[ii], dim);  // Ut, CholUt
            }
        }
    }

    make_int_multiple_of(64, &bytes);
    bytes += 2*64;

    return bytes;
}


void create_treeqp_tdunes(tree_ocp_qp_in *qp_in, treeqp_tdunes_options_t *opts,
    treeqp_tdunes_workspace *work, void *ptr) {

    struct node *tree = (struct node *) qp_in->tree;
    int Nn = qp_in->N;
    int Nh = tree[Nn-1].stage;
    int Np = get_number_of_parent_nodes(Nn, tree);
    int regDim = maximum_hessian_block_dimension(qp_in);
    int dim, idxkid;
    int rowsM, colsM;

    // save some useful dimensions to workspace
    work->Nn = Nn;
    work->Np = Np;

    // char pointer
    char *c_ptr = (char *) ptr;

    // TODO(dimitris): these destroy alignment of doubles maybe..

    // pointers
    work->npar = (int *) c_ptr;
    c_ptr += Nh*sizeof(int);
    setup_npar(Nh, Nn, tree, work->npar);

    work->idxpos = (int *) c_ptr;
    c_ptr += Nn*sizeof(int);
    setup_idxpos(qp_in, work->idxpos);

    // stage QP solvers
    work->stage_qp_data = (void **) c_ptr;
    c_ptr += Nn*sizeof(void *);

    for (int ii = 0; ii < Nn; ii++)
    {
        if (opts->qp_solver[ii] == TREEQP_CLIPPING_SOLVER)
        {
            if (is_clipping_solver_applicable(qp_in, ii) == NO)
            {
                printf("[TREEQP]: Error! Specified stage QP solver (clipping) not applicable.\n");
                exit(1);
            }
        }
        stage_qp_assign_structs(opts->qp_solver[ii], &work->stage_qp_data[ii], &c_ptr);
    }

    #ifdef _CHECK_LAST_ACTIVE_SET_
    work->xasChanged = (int *) c_ptr;
    c_ptr += Nn*sizeof(int);

    work->uasChanged = (int *) c_ptr;
    c_ptr += Nn*sizeof(int);

    work->blockChanged = (int *) c_ptr;
    c_ptr += Np*sizeof(int);
    #endif

    // TODO(dimitris): TO BE REMOVED!

    work->sQ = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->sR = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->sQinv = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->sRinv = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->sQinvCal = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->sRinvCal = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->sqmod = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->srmod = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->regMat = (struct blasfeo_dvec *) c_ptr;
    c_ptr += 1*sizeof(struct blasfeo_dvec);

    work->sM = (struct blasfeo_dmat *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dmat);

    #ifdef _CHECK_LAST_ACTIVE_SET_
    work->sWdiag = (struct blasfeo_dmat *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dmat);
    #endif

    work->sW = (struct blasfeo_dmat *) c_ptr;
    c_ptr += Np*sizeof(struct blasfeo_dmat);

    work->sCholW = (struct blasfeo_dmat *) c_ptr;
    c_ptr += Np*sizeof(struct blasfeo_dmat);

    work->sUt = (struct blasfeo_dmat *) c_ptr;
    c_ptr += (Np-1)*sizeof(struct blasfeo_dmat);

    work->sCholUt = (struct blasfeo_dmat *) c_ptr;
    c_ptr += (Np-1)*sizeof(struct blasfeo_dmat);

    work->sres = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Np*sizeof(struct blasfeo_dvec);

    work->sresMod = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Np*sizeof(struct blasfeo_dvec);

    work->slambda = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Np*sizeof(struct blasfeo_dvec);

    work->sDeltalambda = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Np*sizeof(struct blasfeo_dvec);

    work->sx = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->su = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->sxUnc = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->suUnc = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->sxas = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->suas = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    #ifdef _CHECK_LAST_ACTIVE_SET_
    work->sxasPrev = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->suasPrev = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    #endif

    // move pointer for proper alignment of doubles and blasfeo matrices/vectors
    align_char_to(64, &c_ptr);

    // first assign blasfeo-based solvers, then the rest, and then align again
    for (int ii = 0; ii < Nn; ii++)
    {
        if (opts->qp_solver[ii] != TREEQP_QPOASES_SOLVER)
            stage_qp_assign_data(qp_in->nx[ii], qp_in->nu[ii], opts->qp_solver[ii], work->stage_qp_data[ii], &c_ptr);
    }
    for (int ii = 0; ii < Nn; ii++)
    {
        if (opts->qp_solver[ii] == TREEQP_QPOASES_SOLVER)
            stage_qp_assign_data(qp_in->nx[ii], qp_in->nu[ii], opts->qp_solver[ii], work->stage_qp_data[ii], &c_ptr);
    }

    align_char_to(64, &c_ptr);

    init_strvec(regDim, work->regMat, &c_ptr);
    blasfeo_dvecse(regDim, opts->regValue, work->regMat, 0);

    for (int ii = 0; ii < Nn; ii++)
    {
        // TODO(dimitris): TO BE REMOVED!
        init_strvec(qp_in->nx[ii], &work->sQ[ii], &c_ptr);
        init_strvec(qp_in->nu[ii], &work->sR[ii], &c_ptr);
        init_strvec(qp_in->nx[ii], &work->sQinv[ii], &c_ptr);
        init_strvec(qp_in->nu[ii], &work->sRinv[ii], &c_ptr);
        init_strvec(qp_in->nx[ii], &work->sQinvCal[ii], &c_ptr);
        init_strvec(qp_in->nu[ii], &work->sRinvCal[ii], &c_ptr);

        init_strvec(qp_in->nx[ii], &work->sqmod[ii], &c_ptr);
        init_strvec(qp_in->nu[ii], &work->srmod[ii], &c_ptr);

        init_strvec(qp_in->nx[ii], &work->sx[ii], &c_ptr);
        init_strvec(qp_in->nx[ii], &work->sxUnc[ii], &c_ptr);
        init_strvec(qp_in->nx[ii], &work->sxas[ii], &c_ptr);
        #ifdef _CHECK_LAST_ACTIVE_SET_
        init_strvec(qp_in->nx[ii], &work->sxasPrev[ii], &c_ptr);
        init_strmat(qp_in->nx[ii], qp_in->nx[ii], &work->sWdiag[ii], &c_ptr);
        #endif

        update_M_dimensions(ii, qp_in, &rowsM, &colsM);
        init_strmat(rowsM, colsM, &work->sM[ii], &c_ptr);

        init_strvec(qp_in->nu[ii], &work->su[ii], &c_ptr);
        init_strvec(qp_in->nu[ii], &work->suUnc[ii], &c_ptr);
        init_strvec(qp_in->nu[ii], &work->suas[ii], &c_ptr);
        #ifdef _CHECK_LAST_ACTIVE_SET_
        init_strvec(qp_in->nu[ii], &work->suasPrev[ii], &c_ptr);
        #endif

        if (ii < Np) {
            dim = 0;
            for (int jj = 0; jj < tree[ii].nkids; jj++) {
                idxkid = tree[ii].kids[jj];
                dim += qp_in->nx[idxkid];
            }

            #ifdef _MERGE_FACTORIZATION_WITH_SUBSTITUTION_
            init_strmat(dim+1, dim, &work->sW[ii], &c_ptr);
            init_strmat(dim+1, dim, &work->sCholW[ii], &c_ptr);
            #else
            init_strmat(dim, dim, &work->sW[ii], &c_ptr);
            init_strmat(dim, dim, &work->sCholW[ii], &c_ptr);
            #endif
            init_strvec(dim, &work->sres[ii], &c_ptr);
            init_strvec(dim, &work->sresMod[ii], &c_ptr);
            init_strvec(dim, &work->slambda[ii], &c_ptr);
            init_strvec(dim, &work->sDeltalambda[ii], &c_ptr);
            if (ii > 0) {
                init_strmat(qp_in->nx[ii], dim, &work->sUt[ii-1], &c_ptr);
                init_strmat(qp_in->nx[ii], dim, &work->sCholUt[ii-1], &c_ptr);
            }
        }
    }

    work->fval = (double *) c_ptr;
    c_ptr += Nn*sizeof(double);

    work->cmod = (double *) c_ptr;
    c_ptr += Nn*sizeof(double);

    assert((char *)ptr + treeqp_tdunes_calculate_size(qp_in, opts) >= c_ptr);
    // printf("memory starts at\t%p\nmemory ends at  \t%p\ndistance from the end\t%lu bytes\n",
    //     ptr, c_ptr, (char *)ptr + treeqp_tdunes_calculate_size(qp_in, opts) - c_ptr);
    // exit(1);
}



// write dual initial point to workspace ( _AFTER_ creating it )
void treeqp_tdunes_set_dual_initialization(double *lambda, treeqp_tdunes_workspace *work) {
    int indx = 0;

    for (int ii = 0; ii < work->Np; ii++) {
        blasfeo_pack_dvec(work->slambda[ii].m, &lambda[indx], &work->slambda[ii], 0);
        indx += work->slambda[ii].m;
    }
}
