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

// TODO(dimitris): test variable dimensions and pruned trees
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

#include "treeqp/utils/types.h"
#include "treeqp/utils/blasfeo_utils.h"
#include "treeqp/utils/profiling_utils.h"
#include "treeqp/utils/tree_utils.h"
#include "treeqp/utils/utils.h"
#include "treeqp/utils/timing.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_v_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blas.h"

#define _MERGE_FACTORIZATION_WITH_SUBSTITUTION_


static void setup_npar(int_t Nh, int_t Nn, struct node *tree, int_t *npar) {
    // initialize vector to zero
    for (int_t kk = 0; kk < Nh; kk++) {
        npar[kk] = 0;
    }
    // enumerate nodes per stage
    for (int_t kk = 0; kk < Nn; kk++) {
        npar[tree[kk].stage]++;
    }
}


static int_t maximum_hessian_block_dimension(tree_ocp_qp_in *qp_in) {
    int_t maxDim = 0;
    int_t currDim, idxkid;

    for (int_t ii = 0; ii < qp_in->N; ii++) {
        currDim = 0;
        for (int_t jj = 0; jj < qp_in->tree[ii].nkids; jj++) {
            idxkid = qp_in->tree[ii].kids[jj];
            currDim += qp_in->nx[idxkid];
        }
        maxDim = MAX(maxDim, currDim);
    }
    return maxDim;
}


static void solve_stage_problems(tree_ocp_qp_in *qp_in, treeqp_tdunes_workspace *work) {
    int_t idxkid, idxdad, idxpos;
    int_t Nn = qp_in->N;
    int_t *nx = (int_t *)qp_in->nx;
    int_t *nu = (int_t *)qp_in->nu;

    struct node *tree = (struct node *)qp_in->tree;

    struct d_strvec *slambda = work->slambda;
    struct d_strvec *sx = (struct d_strvec *) work->sx;
    struct d_strvec *su = (struct d_strvec *) work->su;
    struct d_strvec *sxas = (struct d_strvec *) work->sxas;
    struct d_strvec *suas = (struct d_strvec *) work->suas;

    struct d_strmat *sA = (struct d_strmat *) qp_in->A;
    struct d_strmat *sB = (struct d_strmat *) qp_in->B;

    struct d_strvec *sq = (struct d_strvec *) qp_in->q;
    struct d_strvec *sr = (struct d_strvec *) qp_in->r;
    struct d_strvec *sQinv = work->sQinv;
    struct d_strvec *sRinv = work->sRinv;
    struct d_strvec *sQinvCal = work->sQinvCal;
    struct d_strvec *sRinvCal = work->sRinvCal;
    struct d_strvec *sqmod = work->sqmod;
    struct d_strvec *srmod = work->srmod;

    struct d_strvec *sxmin = (struct d_strvec *) qp_in->xmin;
    struct d_strvec *sxmax = (struct d_strvec *) qp_in->xmax;
    struct d_strvec *sumin = (struct d_strvec *) qp_in->umin;
    struct d_strvec *sumax = (struct d_strvec *) qp_in->umax;

    #if DEBUG == 1
    int_t indh = 0;
    int_t indx = 0;
    int_t indu = 0;
    int_t dimh = number_of_primal_variables(qp_in);
    int_t dimx = number_of_states(qp_in);
    int_t dimu = number_of_controls(qp_in);
    real_t *hmod = malloc(dimh*sizeof(real_t));
    real_t *xit = malloc(dimx*sizeof(real_t));
    real_t *uit = malloc(dimu*sizeof(real_t));
    real_t *QinvCal = malloc(dimx*sizeof(real_t));
    real_t *RinvCal = malloc(dimu*sizeof(real_t));
    #endif

    #ifdef PARALLEL
    #pragma omp parallel for private(idxkid, idxdad, idxpos)
    #endif
    for (int_t kk = 0; kk < Nn; kk++) {
        idxdad = tree[kk].dad;

        // TODO(dimitris): check that idxpos here, below and in LS is correct for varying dimensions
        idxpos = 0;
        for (int_t ii = 0; ii < tree[kk].idxkid; ii++) {
            idxpos += qp_in->nx[tree[idxdad].kids[ii]];
        }
        // int_t idxposOLD = tree[kk].idxkid*qp_in->nx[1];
        // assert(idxpos == idxposOLD);

        // --- update QP gradient

        // qmod[k] = - q[k] + lambda[k]
        if (kk == 0) {
            // lambda[0] = 0
            for (int_t jj = 0; jj < nx[kk]; jj++) DVECEL_LIBSTR(&sqmod[kk], jj) = 0.0;
            daxpy_libstr(nx[kk], -1.0, &sq[kk], 0, &sqmod[kk], 0, &sqmod[kk], 0);
        } else {
            daxpy_libstr(nx[kk], -1.0, &sq[kk], 0, &slambda[idxdad], idxpos, &sqmod[kk], 0);
        }

        // rmod[k] = - r[k]
        if (tree[kk].nkids > 0) {
            dveccp_libstr(nu[kk], &sr[kk], 0, &srmod[kk], 0);
            dvecsc_libstr(nu[kk], -1.0, &srmod[kk], 0);
        }

        for (int_t ii = 0; ii < tree[kk].nkids; ii++) {
            idxkid = tree[kk].kids[ii];
            idxdad = tree[idxkid].dad;
            idxpos = 0;
            for (int_t ii = 0; ii < tree[idxkid].idxkid; ii++) {
                idxpos += qp_in->nx[tree[idxdad].kids[ii]];
            }
            // idxposOLD = tree[idxkid].idxkid*qp_in->nx[1];
            // assert(idxpos == idxposOLD);

            // qmod[k] -= A[jj]' * lambda[jj]
            dgemv_t_libstr(nx[idxkid], nx[idxdad], -1.0, &sA[idxkid-1], 0, 0,
                &slambda[idxdad], idxpos, 1.0, &sqmod[kk], 0, &sqmod[kk], 0);
            if (tree[kk].nkids > 0) {
                // rmod[k] -= B[jj]' * lambda[jj]
                dgemv_t_libstr(nx[idxkid], nu[idxdad], -1.0, &sB[idxkid-1], 0, 0,
                    &slambda[idxdad], idxpos, 1.0, &srmod[kk], 0, &srmod[kk], 0);
            }
        }

        // --- solve QP
        // x[k] = Q[k]^-1 .* qmod[k] (NOTE: minus sign already in mod. gradient)
        dvecmuldot_libstr(nx[kk], &sQinv[kk], 0, &sqmod[kk], 0, &sx[kk], 0);

        // x[k] = median(xmin, x[k], xmax), xas[k] = active set
        dveccl_mask_libstr(nx[kk], &sxmin[kk], 0, &sx[kk], 0, &sxmax[kk], 0,
            &sx[kk], 0, &sxas[kk], 0);

        // QinvCal[kk] = Qinv[kk] .* (1 - abs(xas[kk])), aka elimination matrix
        dvecze_libstr(nx[kk], &sxas[kk], 0, &sQinv[kk], 0, &sQinvCal[kk], 0);

        if (tree[kk].nkids > 0) {
            // u[k] = R[k]^-1 .* rmod[k]
            dvecmuldot_libstr(nu[kk], &sRinv[kk], 0, &srmod[kk], 0, &su[kk], 0);
            // u[k] = median(umin, u[k], umax), uas[k] = active set
            dveccl_mask_libstr(nu[kk], &sumin[kk], 0, &su[kk], 0, &sumax[kk], 0, &su[kk], 0,
                &suas[kk], 0);

            // RinvCal[kk] = Rinv[kk] .* (1 - abs(uas[kk]))
            dvecze_libstr(nu[kk], &suas[kk], 0, &sRinv[kk], 0, &sRinvCal[kk], 0);
        }
    }

    #if DEBUG == 1
    for (int_t kk = 0; kk < Nn; kk++) {
        d_cvt_strvec2vec(sqmod[kk].m, &sqmod[kk], 0, &hmod[indh]);
        d_cvt_strvec2vec(sx[kk].m, &sx[kk], 0, &xit[indx]);
        d_cvt_strvec2vec(sQinvCal[kk].m, &sQinvCal[kk], 0, &QinvCal[indx]);
        indh += sqmod[kk].m;
        indx += sx[kk].m;
        d_cvt_strvec2vec(srmod[kk].m, &srmod[kk], 0, &hmod[indh]);
        d_cvt_strvec2vec(su[kk].m, &su[kk], 0, &uit[indu]);
        d_cvt_strvec2vec(sRinvCal[kk].m, &sRinvCal[kk], 0, &RinvCal[indu]);
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
static void compare_with_previous_active_set(int_t isLeaf, int_t indx, treeqp_tdunes_workspace *work) {

    int_t *xasChanged = work->xasChanged;
    int_t *uasChanged = work->uasChanged;

    struct d_strvec *sxas = &work->sxas[indx];
    struct d_strvec *suas = &work->suas[indx];
    struct d_strvec *sxasPrev = &work->sxasPrev[indx];
    struct d_strvec *suasPrev = &work->suasPrev[indx];

    xasChanged[indx] = 0;
    for (int_t ii = 0; ii < sxas->m; ii++) {
        if (DVECEL_LIBSTR(sxas, ii) != DVECEL_LIBSTR(sxasPrev, ii)) {
            xasChanged[indx] = 1;
            break;
        }
    }
    dveccp_libstr(sxas->m, sxas, 0, sxasPrev, 0);

    if (!isLeaf) {
        uasChanged[indx] = 0;
        for (int_t ii = 0; ii < suas->m; ii++) {
            if (DVECEL_LIBSTR(suas, ii) != DVECEL_LIBSTR(suasPrev, ii)) {
                uasChanged[indx] = 1;
                break;
            }
        }
        dveccp_libstr(suas->m, suas, 0, suasPrev, 0);
    }
}


static int_t find_starting_point_of_factorization(struct node *tree, treeqp_tdunes_workspace *work) {
    int_t idxdad, asDadChanged;
    int_t Np = work->Np;
    int_t idxFactorStart = Np;
    int_t *xasChanged = work->xasChanged;
    int_t *uasChanged = work->uasChanged;
    int_t *blockChanged = work->blockChanged;

    for (int_t kk = 0; kk < Np; kk++) {
        blockChanged[kk] = 0;
    }

    // TODO(dimitris):check if it's worth parallelizing
    // --> CAREFULLY THOUGH since multiple threads write on same memory
    for (int_t kk = work->Nn-1; kk > 0; kk--) {
        idxdad = tree[kk].dad;
        asDadChanged = xasChanged[idxdad] | uasChanged[idxdad];

        if (asDadChanged || xasChanged[kk]) blockChanged[idxdad] = 1;
    }
    for (int_t kk = Np-1; kk >= 0; kk--) {
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
static real_t calculate_error_in_residuals(termination_t condition, treeqp_tdunes_workspace *work) {
    real_t error = 0;
    int_t Np = work->Np;
    struct d_strvec *sres = work->sres;

    if ((condition == TREEQP_SUMSQUAREDERRORS) || (condition == TREEQP_TWONORM)) {
        for (int_t kk = 0; kk < Np; kk++) {
            error += ddot_libstr(sres[kk].m, &sres[kk], 0, &sres[kk], 0);
        }
        if (condition == TREEQP_TWONORM) error = sqrt(error);
    } else if (condition == TREEQP_INFNORM) {
        for (int_t kk = 0; kk < Np; kk++) {
            for (int_t ii = 0; ii < sres[kk].m; ii++) {
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


static return_t build_dual_problem(tree_ocp_qp_in *qp_in, int_t *idxFactorStart,
    treeqp_tdunes_options_t *opts, treeqp_tdunes_workspace *work) {

    int_t idxdad, idxpos, idxsib, idxii, ns, isLeaf, asDadChanged;
    real_t error;

    int_t *nx = (int_t *)qp_in->nx;
    int_t *nu = (int_t *)qp_in->nu;

    #ifdef _CHECK_LAST_ACTIVE_SET_
    int_t *xasChanged = work->xasChanged;
    int_t *uasChanged = work->uasChanged;
    struct d_strmat *sWdiag = work->sWdiag;
    #endif

    int_t Nn = work->Nn;
    int_t Np = work->Np;

    struct d_strmat *sA = (struct d_strmat *) qp_in->A;
    struct d_strmat *sB = (struct d_strmat *) qp_in->B;
    struct d_strvec *sb = (struct d_strvec *) qp_in->b;
    struct node *tree = (struct node *)qp_in->tree;

    struct d_strvec *sQinvCal = work->sQinvCal;
    struct d_strvec *sRinvCal = work->sRinvCal;

    struct d_strvec *sx = work->sx;
    struct d_strvec *su = work->su;

    struct d_strmat *sM = work->sM;
    struct d_strmat *sW = work->sW;
    struct d_strmat *sUt = work->sUt;
    struct d_strvec *sres = work->sres;
    struct d_strvec *sresMod = work->sresMod;
    struct d_strvec *regMat = work->regMat;

    *idxFactorStart = -1;

    #if DEBUG == 1
    int_t indres = 0;
    int_t dimres = number_of_states(qp_in) - qp_in->nx[0];
    real_t res[dimres];
    int_t dimW = 0;
    int_t dimUt = 0;
    for (int_t kk = 0; kk < Np; kk++) {
        dimW += sW[kk].n*sW[kk].n;  // NOTE(dimitris): not m, as it may be equal to n+1
        if (kk > 0) dimUt += sUt[kk-1].m*sUt[kk-1].n;
    }
    real_t W[dimW], Ut[dimUt];
    int_t indW = 0;
    int_t indUt = 0;
    #endif

    #ifdef _CHECK_LAST_ACTIVE_SET_
    // TODO(dimitris): check if it's worth to parallelize
    for (int_t kk = Nn-1; kk >= 0; kk--) {
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
    for (int_t kk = Nn-1; kk > 0; kk--) {
        idxdad = tree[kk].dad;
        idxpos = 0;
        for (int_t ii = 0; ii < tree[kk].idxkid; ii++) {
            idxpos += qp_in->nx[tree[idxdad].kids[ii]];
        }

        // TODO(dimitris): decide on convention for comments (+offset or not)

        // res[k] = b[k] - x[k]
        daxpy_libstr(nx[kk], -1.0, &sx[kk], 0, &sb[kk-1], 0, &sres[idxdad], idxpos);

        // res[k] += A[k]*x[idxdad]
        dgemv_n_libstr(nx[kk], nx[idxdad], 1.0, &sA[kk-1], 0, 0, &sx[idxdad], 0, 1.0, &sres[idxdad],
            idxpos, &sres[idxdad], idxpos);

        // res[k] += B[k]*u[idxdad]
        dgemv_n_libstr(nx[kk], nu[idxdad], 1.0, &sB[kk-1], 0, 0, &su[idxdad], 0, 1.0, &sres[idxdad],
            idxpos, &sres[idxdad], idxpos);

        // resMod[k] = res[k]
        dveccp_libstr(nx[kk], &sres[idxdad], idxpos, &sresMod[idxdad], idxpos);
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
    for (int_t kk = Nn-1; kk > 0; kk--) {
        idxdad = tree[kk].dad;
        idxpos = 0;
        for (int_t ii = 0; ii < tree[kk].idxkid; ii++) {
            idxpos += qp_in->nx[tree[idxdad].kids[ii]];
        }

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
        dgemm_r_diag_libstr(nx[kk], nx[idxdad], 1.0,  &sA[kk-1], 0, 0, &sQinvCal[idxdad], 0, 0.0,
            &sM[kk], 0, 0, &sM[kk], 0, 0);

        // --- hessian contribution of parent (Ut)

        #ifdef _CHECK_LAST_ACTIVE_SET_
        if (asDadChanged && tree[idxdad].dad >= 0) {
        #else
        if (tree[idxdad].dad >= 0) {
        #endif
            // Ut[idxdad]+offset = M' = - A[k] *  Qinvcal[idxdad]
            // TODO(dimitris): check that this is correct for varying dimensions
            dgetr_libstr(nx[kk], nx[idxdad], &sM[kk], 0, 0, &sUt[idxdad-1], 0, idxpos);
            dgesc_libstr(nx[idxdad], nx[kk], -1.0, &sUt[idxdad-1], 0, idxpos);
        }

        // --- hessian contribution of node (diagonal block of W)

        // W[idxdad]+offset = A[k]*M^T = A[k]*Qinvcal[idxdad]*A[k]'
        dsyrk_ln_libstr(nx[kk], nx[idxdad], 1.0, &sA[kk-1], 0, 0, &sM[kk], 0, 0, 0.0, &sW[idxdad],
            idxpos, idxpos, &sW[idxdad], idxpos, idxpos);

        // M = B[k]*Rinvcal[idxdad]
        dgemm_r_diag_libstr(nx[kk], nu[idxdad], 1.0,  &sB[kk-1], 0, 0, &sRinvCal[idxdad], 0, 0.0,
            &sM[kk], 0, 0, &sM[kk], 0, 0);

        // W[idxdad]+offset += B[k]*M^T = B[k]*Rinvcal[idxdad]*B[k]'
        dsyrk_ln_libstr(nx[kk], nu[idxdad], 1.0, &sB[kk-1], 0, 0, &sM[kk], 0, 0, 1.0, &sW[idxdad],
            idxpos, idxpos, &sW[idxdad], idxpos, idxpos);

        // W[idxdad]+offset += Qinvcal[k]
        ddiaad_libstr(nx[kk], 1.0, &sQinvCal[kk], 0, &sW[idxdad], idxpos, idxpos);

        // W[idxdad]+offset += regMat (regularization)
        ddiaad_libstr(nx[kk], 1.0, regMat, 0, &sW[idxdad], idxpos, idxpos);

        #ifdef _CHECK_LAST_ACTIVE_SET_
        // save diagonal block that will be overwritten in factorization
        dgecp_libstr(nx[kk], nx[kk], &sW[idxdad], idxpos, idxpos, &sWdiag[kk], 0, 0);
        #endif

        // --- hessian contribution of preceding siblings (off-diagonal blocks of W)

        #ifdef _CHECK_LAST_ACTIVE_SET_
        if (asDadChanged) {
        #endif
        ns = tree[idxdad].nkids - 1;  // number of siblings
        idxii = 0;
        for (int_t ii = 0; ii < ns; ii++) {
            idxsib = tree[idxdad].kids[ii];
            if (idxsib == kk) break;  // completed all preceding siblings

            // M = A[idxsib] * Qinvcal[idxdad]
            dgemm_r_diag_libstr(nx[idxsib], nx[idxdad], 1.0,  &sA[idxsib-1], 0, 0,
                &sQinvCal[idxdad], 0, 0.0, &sM[kk], 0, 0, &sM[kk], 0, 0);

            // W[idxdad]+offset = A[k]*M^T = A[k]*Qinvcal[idxdad]*A[idxsib]'
            dgemm_nt_libstr(nx[kk], nx[idxsib], nx[idxdad], 1.0, &sA[kk-1], 0, 0,
                &sM[kk], 0, 0, 0.0, &sW[idxdad], idxpos, idxii, &sW[idxdad], idxpos, idxii);

            // M = B[idxsib]*Rinvcal[idxdad]
            dgemm_r_diag_libstr(nx[idxsib], nu[idxdad], 1.0, &sB[idxsib-1], 0, 0,
                &sRinvCal[idxdad], 0, 0.0, &sM[kk], 0, 0, &sM[kk], 0, 0);

            // W[idxdad]+offset += B[k]*M^T = B[k]*Rinvcal[idxdad]*B[idxsib]'
            dgemm_nt_libstr(nx[kk], nx[idxsib], nu[idxdad], 1.0, &sB[kk-1], 0, 0,
                &sM[kk], 0, 0, 1.0, &sW[idxdad], idxpos, idxii, &sW[idxdad], idxpos, idxii);

            // TODO(dimitris): check that this is correct with varying dims
            assert(idxii == ii*qp_in->nx[1]);
            idxii += nx[idxsib];
        }
        #ifdef _CHECK_LAST_ACTIVE_SET_
        }
        #endif

        #ifdef _CHECK_LAST_ACTIVE_SET_
        } else {
            dgecp_libstr(nx[kk], nx[kk], &sWdiag[kk], 0, 0, &sW[idxdad], idxpos, idxpos);
        }
        #endif
    }

    #if DEBUG == 1
    for (int_t kk = 0; kk < Np; kk++) {
        d_cvt_strvec2vec(sres[kk].m, &sres[kk], 0, &res[indres]);
        indres += sres[kk].m;
        d_cvt_strmat2mat(sW[kk].n, sW[kk].n, &sW[kk], 0, 0, &W[indW], sW[kk].n);
        indW += sW[kk].n*sW[kk].n;
        if (kk > 0) {
            d_cvt_strmat2mat( sUt[kk-1].m, sUt[kk-1].n, &sUt[kk-1], 0, 0,
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


static void calculate_delta_lambda(tree_ocp_qp_in *qp_in, int_t idxFactorStart,
    treeqp_tdunes_workspace *work) {

    struct node *tree = (struct node *)qp_in->tree;
    int_t idxdad, idxpos;
    int_t Nn = qp_in->N;
    int_t Nh = tree[Nn-1].stage;
    int_t Np = work->Np;
    int_t icur = Np-1;
    int_t *npar = work->npar;
    int_t *nx = (int_t *)qp_in->nx;

    struct d_strmat *sW = work->sW;
    struct d_strmat *sUt = work->sUt;
    struct d_strmat *sCholW = work->sCholW;
    struct d_strmat *sCholUt = work->sCholUt;
    struct d_strvec *sresMod = work->sresMod;
    struct d_strvec *sDeltalambda = work->sDeltalambda;

    #if DEBUG == 1
    int_t dimlam = number_of_states(qp_in) - qp_in->nx[0];
    real_t deltalambda[dimlam];
    int_t indlam = 0;
    #endif

    // --- Cholesky factorization merged with backward substitution

    for (int_t kk = Nh-1; kk > 0; kk--) {
        #if PRINT_LEVEL > 2
        printf("\n--------- New (parallel) factorization branch  ---------\n");
        #endif
        #ifdef PARALLEL
        #pragma omp parallel for private(idxdad, idxpos)
        #endif
        for (int_t ii = icur; ii > icur-npar[kk]; ii--) {

            // NOTE(dimitris): result of backward substitution saved in deltalambda
            // NOTE(dimitris): substitution for free if dual[ii].W not multiple of 4 (in LA=HP)
            #ifdef _MERGE_FACTORIZATION_WITH_SUBSTITUTION_

            #ifdef _CHECK_LAST_ACTIVE_SET_
            if (ii < idxFactorStart) {
            #endif

            // add resMod in last row of matrix W
            drowin_libstr(sresMod[ii].m, 1.0, &sresMod[ii], 0, &sW[ii], sW[ii].m-1, 0);
            // perform Cholesky factorization and backward substitution together
            dpotrf_l_mn_libstr(sW[ii].m, sW[ii].n, &sW[ii], 0, 0, &sCholW[ii], 0, 0);
            // extract result of substitution
            drowex_libstr(sDeltalambda[ii].m, 1.0, &sCholW[ii], sCholW[ii].m-1, 0,
                &sDeltalambda[ii], 0);

            #ifdef _CHECK_LAST_ACTIVE_SET_
            } else {
            // perform only vector substitution
            dtrsv_lnn_libstr(sresMod[ii].m, &sCholW[ii], 0, 0, &sresMod[ii], 0,
                &sDeltalambda[ii], 0);
            }
            #endif

            #else  /* _MERGE_FACTORIZATION_WITH_SUBSTITUTION_ */

            #ifdef _CHECK_LAST_ACTIVE_SET_
            if (ii < idxFactorStart) {
            #endif
            // Cholesky factorization to calculate factor of current diagonal block
            dpotrf_l_libstr(sW[ii].n, &sW[ii], 0, 0, &sCholW[ii], 0, 0);
            #ifdef _CHECK_LAST_ACTIVE_SET_
            }  // TODO(dimitris): we can probably skip more calculations (see scenarios)
            #endif

            // vector substitution
            dtrsv_lnn_libstr(sresMod[ii].m, &sCholW[ii], 0, 0, &sresMod[ii], 0,
                &sDeltalambda[ii], 0);

            #endif  /* _MERGE_FACTORIZATION_WITH_SUBSTITUTION_ */

            // Matrix substitution to calculate transposed factor of parent block
            dtrsm_rltn_libstr( sUt[ii-1].m , sUt[ii-1].n, 1.0, &sCholW[ii], 0, 0,
                &sUt[ii-1], 0, 0, &sCholUt[ii-1], 0, 0);

            // Symmetric matrix multiplication to update diagonal block of parent
            // NOTE(dimitris): use dgemm_nt_libstr if dsyrk not implemented yet
            idxdad = tree[ii].dad;
            // TODO(dimitris): either store this somewhere or put it in a function
            idxpos = 0;
            for (int_t jj = 0; jj < tree[ii].idxkid; jj++) {
                idxpos += qp_in->nx[tree[idxdad].kids[jj]];
            }

            dsyrk_ln_libstr(sCholUt[ii-1].m, sCholUt[ii-1].n, -1.0,
                &sCholUt[ii-1], 0, 0, &sCholUt[ii-1], 0, 0, 1.0,
                &sW[idxdad], idxpos, idxpos, &sW[idxdad], idxpos, idxpos);

            // Matrix vector multiplication to update vector of parent
            dgemv_n_libstr(sCholUt[ii-1].m, sCholUt[ii-1].n, -1.0, &sCholUt[ii-1], 0, 0,
                &sDeltalambda[ii], 0, 1.0, &sresMod[idxdad], idxpos, &sresMod[idxdad], idxpos);
        }
        icur -= npar[kk];
    }
    #ifdef _MERGE_FACTORIZATION_WITH_SUBSTITUTION_
    // add resMod in last row of matrix W
    drowin_libstr(sresMod[0].m, 1.0, &sresMod[0], 0, &sW[0], sW[0].m-1, 0);
    // perform Cholesky factorization and backward substitution together
    dpotrf_l_mn_libstr(sW[0].m, sW[0].n, &sW[0], 0, 0, &sCholW[0], 0, 0);
    // extract result of substitution
    drowex_libstr(sDeltalambda[0].m, 1.0, &sCholW[0], sCholW[0].m-1, 0, &sDeltalambda[0], 0);
    #else
    // calculate Cholesky factor of root block
    dpotrf_l_libstr(sW[0].m, &sW[0], 0, 0, &sCholW[0], 0, 0);

    // calculate last elements of backward substitution
    dtrsv_lnn_libstr(sresMod[0].m, &sCholW[0], 0, 0, &sresMod[0], 0, &sDeltalambda[0], 0);
    #endif

    // --- Forward substitution

    icur = 1;

    dtrsv_ltn_libstr(sDeltalambda[0].m, &sCholW[0], 0, 0, &sDeltalambda[0], 0, &sDeltalambda[0], 0);

    for (int_t kk = 1; kk < Nh; kk++) {
        #ifdef PARALLEL
        #pragma omp parallel for private(idxdad, idxpos)
        #endif
        for (int_t ii = icur; ii < icur+npar[kk]; ii++) {
            idxdad = tree[ii].dad;
            idxpos = 0;
            for (int_t jj = 0; jj < tree[ii].idxkid; jj++) {
                idxpos += qp_in->nx[tree[idxdad].kids[jj]];
            }

            dgemv_t_libstr(sCholUt[ii-1].m, sCholUt[ii-1].n, -1.0, &sCholUt[ii-1], 0, 0,
                &sDeltalambda[idxdad], idxpos, 1.0, &sDeltalambda[ii], 0, &sDeltalambda[ii], 0);

            dtrsv_ltn_libstr(sDeltalambda[ii].m, &sCholW[ii], 0, 0, &sDeltalambda[ii], 0,
                &sDeltalambda[ii], 0);
        }
        icur += npar[kk];
    }

    #if PRINT_LEVEL > 2
    for (int_t ii = 0; ii < Np; ii++) {
        printf("\nCholesky factor of diagonal block #%d as strmat: \n\n", ii+1);
        d_print_strmat( sCholW[ii].m, sCholW[ii].n, &sCholW[ii], 0, 0);
    }
    for (int_t ii = 1; ii < Np; ii++) {
        printf("\nTransposed Cholesky factor of parent block #%d as strmat: \n\n", ii+1);
        d_print_strmat(sCholUt[ii-1].m, sCholUt[ii-1].n, &sCholUt[ii-1], 0, 0);
    }

    printf("\nResult of backward substitution:\n\n");
    for (int_t ii = 0; ii < Np; ii++) {
        d_print_strvec(sDeltalambda[0].m, &sDeltalambda[0], 0);
    }

    printf("\nResult of forward substitution (aka final result):\n\n");
    for (int_t ii = 0; ii < Np; ii++) {
        d_print_strvec(sDeltalambda[ii].m, &sDeltalambda[ii], 0);
    }
    #endif

    #if DEBUG == 1
    for (int_t kk = 0; kk < Np; kk++) {
        d_cvt_strvec2vec(sDeltalambda[kk].m, &sDeltalambda[kk], 0, &deltalambda[indlam]);
        indlam += sDeltalambda[kk].m;
    }
    write_double_vector_to_txt(deltalambda, dimlam, "examples/data_spring_mass/deltalambda.txt");
    #endif
}


static real_t gradient_trans_times_direction(treeqp_tdunes_workspace *work) {
    real_t ans = 0;
    struct d_strvec *sres = work->sres;
    struct d_strvec *sDeltalambda = work->sDeltalambda;

    for (int_t kk = 0; kk < work->Np; kk++) {
        ans += ddot_libstr(sres[kk].m, &sres[kk], 0, &sDeltalambda[kk], 0);
    }
    // NOTE(dimitris): res has was -gradient above
    return -ans;
}


static real_t evaluate_dual_function(tree_ocp_qp_in *qp_in, treeqp_tdunes_workspace *work) {
    int_t ii, jj, kk, idxkid, idxpos, idxdad;
    real_t fval = 0;

    int_t Nn = work->Nn;
    int_t Np = work->Np;

    int_t *nx = (int_t *)qp_in->nx;
    int_t *nu = (int_t *)qp_in->nu;

    real_t *fvals = work->fval;
    real_t *cmod = work->cmod;

    struct d_strvec *sx = work->sx;
    struct d_strvec *su = work->su;
    struct d_strvec *sxas = work->sxas;
    struct d_strvec *suas = work->suas;

    struct d_strmat *sA = (struct d_strmat *) qp_in->A;
    struct d_strmat *sB = (struct d_strmat *) qp_in->B;
    struct d_strvec *sb = (struct d_strvec *) qp_in->b;

    struct d_strvec *sQ = (struct d_strvec *) qp_in->Q;
    struct d_strvec *sR = (struct d_strvec *) qp_in->R;
    struct d_strvec *sq = (struct d_strvec *) qp_in->q;
    struct d_strvec *sr = (struct d_strvec *) qp_in->r;
    struct d_strvec *sQinv = work->sQinv;
    struct d_strvec *sRinv = work->sRinv;
    struct d_strvec *sqmod = work->sqmod;
    struct d_strvec *srmod = work->srmod;

    struct d_strvec *sxmin = (struct d_strvec *) qp_in->xmin;
    struct d_strvec *sxmax = (struct d_strvec *) qp_in->xmax;
    struct d_strvec *sumin = (struct d_strvec *) qp_in->umin;
    struct d_strvec *sumax = (struct d_strvec *) qp_in->umax;

    struct node *tree = (struct node *)qp_in->tree;

    struct d_strvec *slambda = work->slambda;
    #ifdef PARALLEL
    #pragma omp parallel for private(ii, jj, idxkid, idxpos, idxdad)
    #endif
    // NOTE: same code as in solve_stage_problems but:
    // - without calculating as
    // - without calculating elimination matrix
    // - with calculating modified constant term
    for (kk = 0; kk < Nn; kk++) {
        idxdad = tree[kk].dad;
        idxpos = 0;
        for (int_t ii = 0; ii < tree[kk].idxkid; ii++) {
            idxpos += qp_in->nx[tree[idxdad].kids[ii]];
        }

        // --- update QP gradient

        // qmod[k] = - q[k] + lambda[k]
        if (kk == 0) {
            // lambda[0] = 0
            for (jj = 0; jj < nx[kk]; jj++) DVECEL_LIBSTR(&sqmod[kk], jj) = 0.0;
            daxpy_libstr(nx[kk], -1.0, &sq[kk], 0, &sqmod[kk], 0, &sqmod[kk], 0);
        } else {
            daxpy_libstr(nx[kk], -1.0, &sq[kk], 0, &slambda[idxdad], idxpos, &sqmod[kk], 0);
        }

        // rmod[k] = - r[k]
        if (kk < Np) {
            dveccp_libstr(nu[kk], &sr[kk], 0, &srmod[kk], 0);
            dvecsc_libstr(nu[kk], -1.0, &srmod[kk], 0);
        }

        // cmod[k] = 0
        cmod[kk] = 0.;

        for (ii = 0; ii < tree[kk].nkids; ii++) {
            idxkid = tree[kk].kids[ii];
            idxdad = tree[idxkid].dad;
            idxpos = 0;
            for (int_t ii = 0; ii < tree[idxkid].idxkid; ii++) {
                idxpos += qp_in->nx[tree[idxdad].kids[ii]];
            }

            // cmod[k] += b[jj]' * lambda[jj]
            cmod[kk] += ddot_libstr(nx[kk], &sb[idxkid-1], 0, &slambda[idxdad], idxpos);

            // return x^T * y

            // qmod[k] -= A[jj]' * lambda[jj]
            dgemv_t_libstr(nx[idxkid], nx[idxdad], -1.0, &sA[idxkid-1], 0, 0,
                &slambda[idxdad], idxpos, 1.0, &sqmod[kk], 0, &sqmod[kk], 0);
            if (kk < Np) {
                // rmod[k] -= B[jj]' * lambda[jj]
                dgemv_t_libstr(nx[idxkid], nu[idxdad], -1.0, &sB[idxkid-1], 0, 0,
                    &slambda[idxdad], idxpos, 1.0, &srmod[kk], 0, &srmod[kk], 0);
            }
        }

        // --- solve QP
        // x[k] = Q[k]^-1 .* qmod[k] (NOTE: minus sign already in mod. gradient)
        dvecmuldot_libstr(nx[kk], &sQinv[kk], 0, &sqmod[kk], 0, &sx[kk], 0);

        // x[k] = median(xmin, x[k], xmax)
        dveccl_libstr(nx[kk], &sxmin[kk], 0, &sx[kk], 0, &sxmax[kk], 0, &sx[kk], 0);

        if (kk < Np) {
            // u[k] = R[k]^-1 .* rmod[k]
            dvecmuldot_libstr(nu[kk], &sRinv[kk], 0, &srmod[kk], 0, &su[kk], 0);
            // u[k] = median(umin, u[k], umax)
            dveccl_libstr(nu[kk], &sumin[kk], 0, &su[kk], 0, &sumax[kk], 0, &su[kk], 0);
        }

        // --- calculate dual function term

        // feval = - (1/2)x[k]' * Q[k] * x[k] + x[k]' * qmod[k] - cmod[k]
        // NOTE: qmod[k] has already a minus sign
        // NOTE: xas used as workspace
        dvecmuldot_libstr(nx[kk], &sQ[kk], 0, &sx[kk], 0, &sxas[kk], 0);
        fvals[kk] = -0.5*ddot_libstr(nx[kk], &sxas[kk], 0, &sx[kk], 0) - cmod[kk];
        fvals[kk] += ddot_libstr(nx[kk], &sqmod[kk], 0, &sx[kk], 0);

        if (kk < Np) {
            // feval -= (1/2)u[k]' * R[k] * u[k] - u[k]' * rmod[k]
            dvecmuldot_libstr(nu[kk], &sR[kk], 0, &su[kk], 0, &suas[kk], 0);
            fvals[kk] -= 0.5*ddot_libstr(nu[kk], &suas[kk], 0, &su[kk], 0);
            fvals[kk] += ddot_libstr(nu[kk], &srmod[kk], 0, &su[kk], 0);
        }
    }

    for (kk = 0; kk < Nn; kk++) fval += fvals[kk];

    return fval;
}


static int_t line_search(tree_ocp_qp_in *qp_in, treeqp_tdunes_options_t *opts,
    treeqp_tdunes_workspace *work) {

    int_t Nn = qp_in->N;
    int_t Np = work->Np;

    struct node *tree = (struct node *)qp_in->tree;

    #if DEBUG == 1
    int_t dimlam = number_of_states(qp_in) - qp_in->nx[0];
    real_t *lambda = malloc(dimlam*sizeof(real_t));
    int_t indlam = 0;
    #endif

    struct d_strvec *slambda = work->slambda;
    struct d_strvec *sDeltalambda = work->sDeltalambda;

    real_t dotProduct, fval, fval0;
    real_t tau = 1;
    real_t tauPrev = 0;

    dotProduct = gradient_trans_times_direction(work);
    fval0 = evaluate_dual_function(qp_in, work);
    // printf(" dot_product = %f\n", dotProduct);
    // printf(" dual_function = %f\n", fval0);

    int_t lsIter;

    for (lsIter = 1; lsIter <= opts->lineSearchMaxIter; lsIter++) {
        // update multipliers
        #ifdef PARALLEL
        #pragma omp parallel for
        #endif
        for (int_t kk = 0; kk < Np; kk++) {
            daxpy_libstr( sDeltalambda[kk].m, tau-tauPrev, &sDeltalambda[kk], 0, &slambda[kk], 0,
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
    for (int_t kk = 0; kk < Np; kk++) {
        d_cvt_strvec2vec( slambda[kk].m, &slambda[kk], 0, &lambda[indlam]);
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


void write_solution_to_txt(tree_ocp_qp_in *qp_in, int_t Np, int_t iter, struct node *tree,
    treeqp_tdunes_workspace *work) {

    int_t kk, indx, indu, ind;

    int_t Nn = qp_in->N;
    int_t dimx = number_of_states(qp_in);
    int_t dimu = number_of_controls(qp_in);
    int_t dimlam = dimx - qp_in->nx[0];

    struct d_strvec *sx = work->sx;
    struct d_strvec *su = work->su;

    struct d_strvec *slambda = work->slambda;
    struct d_strvec *sDeltalambda = work->sDeltalambda;

    // TODO(dimitris): maybe use Np for u instead of Nn in other places too to avoid confusion
    real_t *x = malloc(dimx*sizeof(real_t));
    real_t *u = malloc(dimu*sizeof(real_t));
    real_t *deltalambda = malloc(dimlam*sizeof(real_t));
    real_t *lambda = malloc(dimlam*sizeof(real_t));

    indx = 0; indu = 0;
    for (kk = 0; kk < Nn; kk++) {
        d_cvt_strvec2vec(sx[kk].m, &sx[kk], 0, &x[indx]);
        indx += sx[kk].m;
        if (kk < Np) {
            d_cvt_strvec2vec(su[kk].m, &su[kk], 0, &u[indu]);
            indu += su[kk].m;
        }
    }

    ind = 0;
    for (kk = 0; kk < Np; kk++) {
        d_cvt_strvec2vec(sDeltalambda[kk].m, &sDeltalambda[kk], 0, &deltalambda[ind]);
        d_cvt_strvec2vec(slambda[kk].m, &slambda[kk], 0, &lambda[ind]);
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

int_t treeqp_tdunes_solve(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out,
    treeqp_tdunes_options_t *opts, treeqp_tdunes_workspace *work) {

    int status;
    int idxFactorStart;  // TODO(dimitris): move to workspace
    int lsIter;

    int_t NewtonIter;

    struct node *tree = (struct node *)qp_in->tree;

    int_t Nn = work->Nn;
    int_t Nh = qp_in->tree[Nn-1].stage;
    int_t Np = work->Np;
    int_t *npar = work->npar;
    struct d_strvec *regMat = work->regMat;

    // ------ initialization
    for (int_t ii = 0; ii < Nn; ii++) {
        for (int_t nn = 0; nn < qp_in->nx[ii]; nn++)
            DVECEL_LIBSTR(&work->sQinv[ii], nn) = 1.0/DVECEL_LIBSTR(&qp_in->Q[ii], nn);
        for (int_t nn = 0; nn < qp_in->nu[ii]; nn++)
            DVECEL_LIBSTR(&work->sRinv[ii], nn) = 1.0/DVECEL_LIBSTR(&qp_in->R[ii], nn);

        #ifdef _CHECK_LAST_ACTIVE_SET_
        dvecse_libstr(work->sxasPrev[ii].m, 0.0/0.0, &work->sxasPrev[ii], 0);
        if (ii < Np)
            dvecse_libstr(work->suasPrev[ii].m, 0.0/0.0, &work->suasPrev[ii], 0);
        #endif
    }

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

    // ------ copy solution to qp_out
    for (int_t ii = 0; ii < Nn; ii++) {
        dveccp_libstr(qp_in->nx[ii], &work->sx[ii], 0, &qp_out->x[ii], 0);
        dveccp_libstr(qp_in->nu[ii], &work->su[ii], 0, &qp_out->u[ii], 0);
    }
    qp_out->info.iter = NewtonIter;

    if (qp_out->info.iter == opts->maxIter)
        status = TREEQP_ERR_MAXIMUM_ITERATIONS_REACHED;

    return status;  // TODO(dimitris): return correct status
}


int_t treeqp_tdunes_calculate_size(tree_ocp_qp_in *qp_in) {
    struct node *tree = (struct node *) qp_in->tree;
    int_t bytes = 0;
    int_t Nn = qp_in->N;
    int_t Nh = tree[Nn-1].stage;
    int_t Np = get_number_of_parent_nodes(Nn, tree);
    int_t regDim = maximum_hessian_block_dimension(qp_in);
    int_t dim, idxkid;

    // int pointers
    bytes += Nh*sizeof(int_t);  // npar

    #ifdef _CHECK_LAST_ACTIVE_SET_
    bytes += 2*Nn*sizeof(int_t);  // xasChanged, uasChanged
    bytes += Np*sizeof(int_t);  // blockChanged
    #endif

    // real_t pointers
    bytes += 2*Nn*sizeof(real_t);  // fval, cmod

    // struct pointers
    bytes += 6*Nn*sizeof(struct d_strvec);  // Qinv, Rinv, QinvCal, RinvCal, qmod, rmod
    #ifdef _CHECK_LAST_ACTIVE_SET_
    bytes += Nn*sizeof(struct d_strmat);  // Wdiag
    #endif
    bytes += 1*sizeof(struct d_strvec);  // regMat
    bytes += Nn*sizeof(struct d_strmat);  // M
    bytes += 2*Np*sizeof(struct d_strmat);  // W, CholW
    bytes += 2*(Np-1)*sizeof(struct d_strmat);  // Ut, CholUt
    bytes += 4*Np*sizeof(struct d_strvec);  // res, resMod, lambda, Deltalambda

    // TODO(dimitris): allow nu[N] > 0?
    bytes += 2*Nn*sizeof(struct d_strvec);  // x, xas
    bytes += 2*Nn*sizeof(struct d_strvec);  // u, uas

    #ifdef _CHECK_LAST_ACTIVE_SET_
    bytes += Nn*sizeof(struct d_strvec);  // xasPrev
    bytes += Nn*sizeof(struct d_strvec);  // uasPrev
    #endif

    // structs
    bytes += d_size_strvec(regDim);  // regMat

    for (int_t ii = 0; ii < Nn; ii++) {
        bytes += 3*d_size_strvec(qp_in->nx[ii]);  // Qinv, QinvCal, qmod
        bytes += 3*d_size_strvec(qp_in->nu[ii]);  // Rinv, RinvCal, rmod

        bytes += 2*d_size_strvec(qp_in->nx[ii]);  // x, xas
        #ifdef _CHECK_LAST_ACTIVE_SET_
        bytes += d_size_strmat(qp_in->nx[ii], qp_in->nx[ii]);  // Wdiag
        bytes += d_size_strvec(qp_in->nx[ii]);  // xasPrev
        #endif

        bytes += 2*d_size_strvec(qp_in->nu[ii]);  // u, uas
        #ifdef _CHECK_LAST_ACTIVE_SET_
        bytes += d_size_strvec(qp_in->nu[ii]);  // uasPrev
        #endif

        bytes +=  // M
            d_size_strmat(MAX(qp_in->nx[ii], qp_in->nu[ii]), MAX(qp_in->nx[ii], qp_in->nu[ii]));

        if (ii < Np) {
            // NOTE(dimitris): for constant dimensions dim = tree[ii].nkids*nx
            // TODO(dimitris): check that this is correct for varying dimensions
            dim = 0;
            for (int_t jj = 0; jj < tree[ii].nkids; jj++) {
                idxkid = tree[ii].kids[jj];
                dim += qp_in->nx[idxkid];
            }

            #ifdef _MERGE_FACTORIZATION_WITH_SUBSTITUTION_
            bytes += 2*d_size_strmat(dim + 1, dim);  // W, CholW
            #else
            bytes += 2*d_size_strmat(dim, dim);  // W, CholW
            #endif
            bytes += 4*d_size_strvec(dim);  // res, resMod, lambda, Deltalambda
            if (ii > 0) {
                bytes += 2*d_size_strmat(qp_in->nx[ii], dim);  // Ut, CholUt
            }
        }
    }

    bytes = (bytes + 63)/64*64;
    bytes += 64;

    return bytes;
}


void create_treeqp_tdunes(tree_ocp_qp_in *qp_in, treeqp_tdunes_options_t *opts,
    treeqp_tdunes_workspace *work, void *ptr) {

    struct node *tree = (struct node *) qp_in->tree;
    int_t Nn = qp_in->N;
    int_t Nh = tree[Nn-1].stage;
    int_t Np = get_number_of_parent_nodes(Nn, tree);
    int_t regDim = maximum_hessian_block_dimension(qp_in);
    int_t dim, idxkid;

    // save some useful dimensions to workspace
    work->Nn = Nn;
    work->Np = Np;

    // char pointer
    char *c_ptr = (char *) ptr;

    // pointers
    work->npar = (int_t *) c_ptr;
    c_ptr += Nh*sizeof(int_t);
    setup_npar(Nh, Nn, tree, work->npar);

    #ifdef _CHECK_LAST_ACTIVE_SET_
    work->xasChanged = (int_t *) c_ptr;
    c_ptr += Nn*sizeof(int_t);

    work->uasChanged = (int_t *) c_ptr;
    c_ptr += Nn*sizeof(int_t);

    work->blockChanged = (int_t *) c_ptr;
    c_ptr += Np*sizeof(int_t);
    #endif

    work->sQinv = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    work->sRinv = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    work->sQinvCal = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    work->sRinvCal = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    work->sqmod = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    work->srmod = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    work->regMat = (struct d_strvec *) c_ptr;
    c_ptr += 1*sizeof(struct d_strvec);

    work->sM = (struct d_strmat *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strmat);

    #ifdef _CHECK_LAST_ACTIVE_SET_
    work->sWdiag = (struct d_strmat *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strmat);
    #endif

    work->sW = (struct d_strmat *) c_ptr;
    c_ptr += Np*sizeof(struct d_strmat);

    work->sCholW = (struct d_strmat *) c_ptr;
    c_ptr += Np*sizeof(struct d_strmat);

    work->sUt = (struct d_strmat *) c_ptr;
    c_ptr += (Np-1)*sizeof(struct d_strmat);

    work->sCholUt = (struct d_strmat *) c_ptr;
    c_ptr += (Np-1)*sizeof(struct d_strmat);

    work->sres = (struct d_strvec *) c_ptr;
    c_ptr += Np*sizeof(struct d_strvec);

    work->sresMod = (struct d_strvec *) c_ptr;
    c_ptr += Np*sizeof(struct d_strvec);

    work->slambda = (struct d_strvec *) c_ptr;
    c_ptr += Np*sizeof(struct d_strvec);

    work->sDeltalambda = (struct d_strvec *) c_ptr;
    c_ptr += Np*sizeof(struct d_strvec);

    work->sx = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    work->su = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    work->sxas = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    work->suas = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    #ifdef _CHECK_LAST_ACTIVE_SET_
    work->sxasPrev = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    work->suasPrev = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);
    #endif

    // move pointer for proper alignment of doubles and blasfeo matrices/vectors
    long long l_ptr = (long long) c_ptr;
    l_ptr = (l_ptr+63)/64*64;
    c_ptr = (char *) l_ptr;

    work->fval = (real_t *) c_ptr;
    c_ptr += Nn*sizeof(real_t);

    work->cmod = (real_t *) c_ptr;
    c_ptr += Nn*sizeof(real_t);

    // TODO(dimitris): asserts for mem. alignment
    init_strvec(regDim, work->regMat, &c_ptr);
    dvecse_libstr(regDim, opts->regValue, work->regMat, 0);

    for (int_t ii = 0; ii < Nn; ii++) {
        init_strvec(qp_in->nx[ii], &work->sQinv[ii], &c_ptr);
        init_strvec(qp_in->nu[ii], &work->sRinv[ii], &c_ptr);
        init_strvec(qp_in->nx[ii], &work->sQinvCal[ii], &c_ptr);
        init_strvec(qp_in->nu[ii], &work->sRinvCal[ii], &c_ptr);
        init_strvec(qp_in->nx[ii], &work->sqmod[ii], &c_ptr);
        init_strvec(qp_in->nu[ii], &work->srmod[ii], &c_ptr);

        init_strvec(qp_in->nx[ii], &work->sx[ii], &c_ptr);
        init_strvec(qp_in->nx[ii], &work->sxas[ii], &c_ptr);
        #ifdef _CHECK_LAST_ACTIVE_SET_
        init_strvec(qp_in->nx[ii], &work->sxasPrev[ii], &c_ptr);
        init_strmat(qp_in->nx[ii], qp_in->nx[ii], &work->sWdiag[ii], &c_ptr);
        #endif
        init_strmat(MAX(qp_in->nx[ii], qp_in->nu[ii]), MAX(qp_in->nx[ii], qp_in->nu[ii]),
            &work->sM[ii], &c_ptr);

        init_strvec(qp_in->nu[ii], &work->su[ii], &c_ptr);
        init_strvec(qp_in->nu[ii], &work->suas[ii], &c_ptr);
        #ifdef _CHECK_LAST_ACTIVE_SET_
        init_strvec(qp_in->nu[ii], &work->suasPrev[ii], &c_ptr);
        #endif

        if (ii < Np) {
            dim = 0;
            for (int_t jj = 0; jj < tree[ii].nkids; jj++) {
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
    #ifdef  RUNTIME_CHECKS
    char *ptrStart = (char *) ptr;
    char *ptrEnd = c_ptr;
    int_t bytes = treeqp_tdunes_calculate_size(qp_in);
    assert(ptrEnd <= ptrStart + bytes);
    // printf("memory starts at\t%p\nmemory ends at  \t%p\ndistance from the end\t%lu bytes\n",
    //     ptrStart, ptrEnd, ptrStart + bytes - ptrEnd);
    // exit(1);
    #endif
}


// write dual initial point to workspace ( _AFTER_ creating it )
void treeqp_tdunes_set_dual_initialization(real_t *lambda, treeqp_tdunes_workspace *work) {
    int_t indx = 0;

    for (int_t ii = 0; ii < work->Np; ii++) {
        d_cvt_vec2strvec(work->slambda[ii].m, &lambda[indx], &work->slambda[ii], 0);
        indx += work->slambda[ii].m;
    }
}
