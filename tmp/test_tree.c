#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // for sqrt in 2-norm
#ifdef PARALLEL
#include <omp.h>
#endif

// TODO(dimitris): Try open MPI interface (message passing)
// TODO(dimitris): FIX BUG WITH LA=HP AND !MERGE_SUBS (HAPPENS ONLY IF Nr, md > 1)

#include "treeqp/src/dual_Newton_tree.h"

#include "./test_tree.h"

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

#include "./data.c"

// TODO(dimitris): on-the-fly regularization
// TODO(dimitris): different types of line-search
// TODO(dimitris): why saving so many Chol factorizations does not imporove cpu time?
// TODO(dimitris): ask Gianluca if I can overwrite Chol. and check if it makes sense

treeqp_tdunes_options_t set_default_options(void) {
    treeqp_tdunes_options_t opts;
    termination_t cond = TREEQP_INFNORM;

    opts.maxIter = 100;
    opts.termCondition = cond;
    opts.stationarityTolerance = 1.0e-6;

    opts.lineSearchMaxIter = 50;
    opts.lineSearchGamma = 0.1;
    opts.lineSearchBeta = 0.8;

    opts.regType  = TREEQP_ALWAYS_LEVENBERG_MARQUARDT;
    // opts.regTol   = 1.0e-12;
    opts.regValue = 1.0e-6;

    return opts;
}


// static tree_options_t read_matlab_options(void) {
//     tree_options_t opts;

//     opts.maxIter = maxit;
//     opts.stationarityTolerance = termTolerance;

//     opts.lineSearchMaxIter = maxIterLS;
//     opts.lineSearchGamma = gammaLS;
//     opts.lineSearchBeta = betaLS;

//     #if REGULARIZATION == 1
//     opts.regType  = TREEQP_ALWAYS_LEVENBERG_MARQUARDT;
//     opts.regValue = valueREG;
//     #else
//     // TODO(dimitris): do not even add regularization instead of adding zero
//     opts.regType  = TREEQP_NO_REGULARIZATION;
//     opts.regValue = 0.0;
//     #endif

//     return opts;
// }


static void setup_npar(int_t Nh, int_t Nn, struct node *tree, int_t *npar) {
    int kk;

    for (kk = 0; kk < Nh; kk++) npar[kk] = 0;

    for (kk = 0; kk < Nn; kk++) {
        npar[tree[kk].stage]++;
    }

}


static int_t calculate_blasfeo_memory_size_tree(int_t Nh, int_t Nr, int_t md, int_t nx, int_t nu,
    struct node *tree) {

    int_t size = 0;
    int_t Nn = calculate_number_of_nodes(md, Nr, Nh);
    int_t Np = Nn - ipow(md, Nr);
    int_t kk;

    // objective
    size += 3*(Nh+1)*d_size_strvec(nx);  // Q, Qinv, q
    size += 3*Nh*d_size_strvec(nu);  // R, Rinv, r

    // dynamics
    size += md*d_size_strmat(nx, nx);  // A
    size += md*d_size_strmat(nx, nu);  // B
    size += md*d_size_strvec(nx);  // b

    // bounds
    size += 2*d_size_strvec(nx);  // xmin, xmax
    size += 2*d_size_strvec(nu);  // umin, umax

    // primal iterates
    size += Nn*d_size_strvec(nx);
    size += Np*d_size_strvec(nu);

    // intermediate results
    size += Nn*d_size_strvec(nx);  // xas
    size += Np*d_size_strvec(nu);  // uas
    #ifdef _CHECK_LAST_ACTIVE_SET_
    size += Nn*d_size_strvec(nx);  // xasPrev
    size += Np*d_size_strvec(nu);  // uasPrev
    size += Nn*d_size_strmat(nx, nx);  // Wdiag
    #endif
    size += Nn*d_size_strvec(nx);  // qmod
    size += Np*d_size_strvec(nu);  // rmod
    size += Nn*d_size_strvec(nx);  // QinvCal
    size += Np*d_size_strvec(nu);  // RinvCal
    size += Nn*d_size_strmat(MAX(nx, nu), MAX(nx, nu));  // M

    // Newton system
    for (kk = 0; kk < Np; kk++) {
        size += 4*d_size_strvec(tree[kk].nkids*nx);  // res, resMod, lambda, deltalambda
        #ifdef _MERGE_FACTORIZATION_WITH_SUBSTITUTION_
        size += 2*d_size_strmat(tree[kk].nkids*nx+1, tree[kk].nkids*nx);  // W, CholW
        #else
        size += 2*d_size_strmat(tree[kk].nkids*nx, tree[kk].nkids*nx);  // W, CholW
        #endif
        if (kk > 0) size += 2*d_size_strmat(nx, tree[kk].nkids*nx);  // Ut, CholUt
    }

    // misc
    size += d_size_strvec(md*nx);  // regMat
    size += d_size_strvec(nx);  // x0

    return size;
}


static void solve_stage_problems(int_t Nn, stage_QP *QP, dual_block *dual, struct node *tree) {
    int_t ii, jj, kk, idxkid, idxdad, idxpos;

    #if DEBUG == 1
    int_t indh, indx, indu;
    real_t hmod[Nn*(nx+nu)];
    real_t xit[Nn*nx];
    real_t uit[Nn*nu];
    real_t QinvCal[Nn*nx];
    real_t RinvCal[Nn*nu];
    indh = 0; indx = 0; indu = 0;
    #endif

    #ifdef PARALLEL
    #pragma omp parallel for private(ii, jj, idxkid, idxdad, idxpos)
    #endif
    for (kk = 0; kk < Nn; kk++) {
        idxdad = tree[kk].dad;
        idxpos = tree[kk].idxkid*nx;
        // --- update QP gradient

        // qmod[k] = - q[k] + lambda[k]
        if (kk == 0) {
            // lambda[0] = 0
            for (jj = 0; jj < nx; jj++) DVECEL_LIBSTR(QP[kk].qmod, jj) = 0.0;
            daxpy_libstr(nx, -1.0, QP[kk].q, 0, QP[kk].qmod, 0, QP[kk].qmod, 0);
        } else {
            daxpy_libstr(nx, -1.0, QP[kk].q, 0, dual[idxdad].lambda, idxpos, QP[kk].qmod, 0);
        }

        // rmod[k] = - r[k]
        if (tree[kk].nkids > 0) {
            dveccp_libstr(nu, QP[kk].r, 0, QP[kk].rmod, 0);
            dvecsc_libstr(nu, -1.0, QP[kk].rmod, 0);
        }

        for (ii = 0; ii < tree[kk].nkids; ii++) {
            idxkid = tree[kk].kids[ii];
            idxdad = tree[idxkid].dad;
            idxpos = tree[idxkid].idxkid*nx;

            // qmod[k] -= A[jj]' * lambda[jj]
            dgemv_t_libstr(nx, nx, -1.0, QP[idxkid].A, 0, 0, dual[idxdad].lambda, idxpos, 1.0,
                 QP[kk].qmod, 0, QP[kk].qmod, 0);
            if (tree[kk].nkids > 0) {
                // rmod[k] -= B[jj]' * lambda[jj]
                dgemv_t_libstr(nx, nu, -1.0, QP[idxkid].B, 0, 0, dual[idxdad].lambda, idxpos, 1.0,
                    QP[kk].rmod, 0, QP[kk].rmod, 0);
            }
        }

        // --- solve QP
        // x[k] = Q[k]^-1 .* qmod[k] (NOTE: minus sign already in mod. gradient)
        dvecmuldot_libstr(nx, QP[kk].Qinv, 0, QP[kk].qmod, 0, QP[kk].x, 0);

        // x[k] = median(xmin, x[k], xmax), xas[k] = active set
        dveccl_mask_libstr(nx, QP[kk].xmin, 0, QP[kk].x, 0, QP[kk].xmax, 0, QP[kk].x, 0,
            QP[kk].xas, 0);

        // QinvCal[kk] = Qinv[kk] .* (1 - abs(xas[kk])), aka elimination matrix
        dvecze_libstr(nx, QP[kk].xas, 0, QP[kk].Qinv, 0, QP[kk].QinvCal, 0);

        if (tree[kk].nkids > 0) {
            // u[k] = R[k]^-1 .* rmod[k]
            dvecmuldot_libstr(nu, QP[kk].Rinv, 0, QP[kk].rmod, 0, QP[kk].u, 0);
            // u[k] = median(umin, u[k], umax), uas[k] = active set
            dveccl_mask_libstr(nu, QP[kk].umin, 0, QP[kk].u, 0, QP[kk].umax, 0, QP[kk].u, 0,
                QP[kk].uas, 0);

            // RinvCal[kk] = Rinv[kk] .* (1 - abs(uas[kk]))
            dvecze_libstr(nu, QP[kk].uas, 0, QP[kk].Rinv, 0, QP[kk].RinvCal, 0);
        }
    }

    #if DEBUG == 1
    for (kk = 0; kk < Nn; kk++) {
        d_cvt_strvec2vec(nx, QP[kk].qmod, 0, &hmod[indh]);
        d_cvt_strvec2vec(nx, QP[kk].x, 0, &xit[indx]);
        d_cvt_strvec2vec(nx, QP[kk].QinvCal, 0, &QinvCal[indx]);
        indh += nx;
        indx += nx;
        if (tree[kk].nkids > 0) {
            d_cvt_strvec2vec(nu, QP[kk].rmod, 0, &hmod[indh]);
            d_cvt_strvec2vec(nu, QP[kk].u, 0, &uit[indu]);
            d_cvt_strvec2vec(nx, QP[kk].RinvCal, 0, &RinvCal[indu]);
            indh += nu;
            indu += nu;
        }
    }
    write_double_vector_to_txt(hmod, Nn*(nx+nu), "data_tree/hmod.txt");
    write_double_vector_to_txt(xit, Nn*nx, "data_tree/xit.txt");
    write_double_vector_to_txt(uit, Nn*nu, "data_tree/uit.txt");
    write_double_vector_to_txt(QinvCal, Nn*nx, "data_tree/Qinvcal.txt");
    write_double_vector_to_txt(RinvCal, Nn*nu, "data_tree/Rinvcal.txt");
    #endif
}


#ifdef _CHECK_LAST_ACTIVE_SET_

static void compare_with_previous_active_set(stage_QP *QP, int_t isLeaf) {
    int_t ii;

    QP->xasChanged = 0;
    for (ii = 0; ii < QP->x->m; ii++) {
        if ( DVECEL_LIBSTR(QP->xas, ii) != DVECEL_LIBSTR(QP->xasPrev, ii)) {
            QP->xasChanged = 1;
            break;
        }
    }
    dveccp_libstr(QP->x->m, QP->xas, 0, QP->xasPrev, 0);

    if (!isLeaf) {
        QP->uasChanged = 0;
        for (ii = 0; ii < QP->u->m; ii++) {
            if ( DVECEL_LIBSTR(QP->uas, ii) != DVECEL_LIBSTR(QP->uasPrev, ii)) {
                QP->uasChanged = 1;
                break;
            }
        }
        dveccp_libstr(QP->u->m, QP->uas, 0, QP->uasPrev, 0);
    }
}


static int_t find_starting_point_of_factorization(int_t Np, int_t Nn, stage_QP *QP, dual_block *dual, struct node *tree) {
    int_t kk, nx, idxdad, asDadChanged;
    int_t idxFactorStart = Np;
    for (kk = 0; kk < Np; kk++) {
        dual[kk].blockChanged = 0;
    }

    // TODO(dimitris):check if it's worth parallelizing
    // --> CAREFULLY THOUGH since multiple threads right on same memory
    for (kk = Nn-1; kk > 0; kk--) {
        idxdad = tree[kk].dad;
        asDadChanged = QP[idxdad].xasChanged | QP[idxdad].uasChanged;

        if (asDadChanged || QP[kk].xasChanged) dual[idxdad].blockChanged = 1;
    }
    for (kk = Np-1; kk >= 0; kk--) {
        if (!dual[kk].blockChanged) {
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
static real_t calculate_error_in_residuals(int_t Np, dual_block *dual, termination_t condition) {
    int_t ii, kk;
    real_t error = 0;

    if ((condition == TREEQP_SUMSQUAREDERRORS) || (condition == TREEQP_TWONORM)) {
        for (kk = 0; kk < Np; kk++) {
            error += ddot_libstr(dual[kk].res->m, dual[kk].res, 0, dual[kk].res, 0);
        }
        if (condition == TREEQP_TWONORM) error = sqrt(error);
    } else if (condition == TREEQP_INFNORM) {
        for (kk = 0; kk < Np; kk++) {
            for (ii = 0; ii < dual[kk].res->m; ii++) {
                error = MAX(error, ABS(DVECEL_LIBSTR(dual[kk].res, ii)));
            }
        }
    } else {
        printf("[TREEQP] Unknown termination condition!\n");
        exit(1);
    }
    // printf("error=%2.3e\n",error);
    return error;
}


static return_t build_dual_problem(int_t Nn, int_t Np, stage_QP *QP, dual_block *dual, struct node *tree,
    struct d_strvec *regMat, int_t *idxFactorStart, treeqp_tdunes_options_t *opts) {

    int_t ii, kk, idxdad, idxpos, idxsib, ns, isLeaf, asDadChanged;
    real_t error;

    *idxFactorStart = -1;

    #if DEBUG == 1
    int_t indres = 0;
    real_t res[(Nn-1)*nx];
    int_t dimW = 0;
    int_t dimUt = 0;
    for (kk = 0; kk < Np; kk++) {
        dimW += dual[kk].W->n*dual[kk].W->n;  // NOTE(dimitris): not m, as it may be equal to n+1
        if (kk > 0) dimUt += dual[kk].Ut->m*dual[kk].Ut->n;
    }
    real_t W[dimW], Ut[dimUt];
    int_t indW = 0;
    int_t indUt = 0;
    #endif

    #ifdef _CHECK_LAST_ACTIVE_SET_
    // TODO(dimitris): check if it's worth to parallelize
    for (kk = Nn-1; kk >= 0; kk--) {
        isLeaf = (tree[kk].nkids > 0 ? 0:1);
        // NOTE(dimitris): updates both xasChanged/uasChanged and xasPrev/uasPrev
        compare_with_previous_active_set(&QP[kk], isLeaf);
    }
    // TODO(dimitris): double check that this indx is correct (not higher s.t. we loose efficiency)
    *idxFactorStart = find_starting_point_of_factorization(Np, Nn, QP, dual, tree);
    #endif

    #ifdef PARALLEL
    #pragma omp parallel for private(idxdad, idxpos)
    #endif
    // Calculate dual gradient
    // TODO(dimitris): can we merge with solution of stage QPs without problems in parallelizing?
    for (kk = Nn-1; kk > 0; kk--) {
        idxdad = tree[kk].dad;
        idxpos = tree[kk].idxkid*nx;

        // TODO(dimitris): decide on convention for comments (+offset or not)

        // res[k] = b[k] - x[k]
        daxpy_libstr(nx, -1.0, QP[kk].x, 0, QP[kk].b, 0, dual[idxdad].res, idxpos);

        // res[k] += A[k]*x[idxdad]
        dgemv_n_libstr(nx, nx, 1.0, QP[kk].A, 0, 0, QP[idxdad].x, 0, 1.0, dual[idxdad].res,
            idxpos, dual[idxdad].res, idxpos);

        // res[k] += B[k]*u[idxdad]
        dgemv_n_libstr(nx, nu, 1.0, QP[kk].B, 0, 0, QP[idxdad].u, 0, 1.0, dual[idxdad].res,
            idxpos, dual[idxdad].res, idxpos);

        // resMod[k] = res[k]
        dveccp_libstr(nx, dual[idxdad].res, idxpos, dual[idxdad].resMod, idxpos);

    }

    // Check termination condition
    error = calculate_error_in_residuals(Np, dual, opts->termCondition);
    if (error < opts->stationarityTolerance) {
        return TREEQP_SUCC_OPTIMAL_SOLUTION_FOUND;
    }

    #ifdef PARALLEL
    #pragma omp parallel for private(ii, idxdad, idxpos, idxsib, ns, asDadChanged)
    #endif
    // Calculate dual Hessian
    for (kk = Nn-1; kk > 0; kk--) {
        idxdad = tree[kk].dad;
        idxpos = tree[kk].idxkid*nx;
        #ifdef _CHECK_LAST_ACTIVE_SET_
        asDadChanged = QP[idxdad].xasChanged | QP[idxdad].uasChanged;
        #endif

        // Filling dual[idxdad].W and dual[idxdad].Ut

        #ifdef _CHECK_LAST_ACTIVE_SET_
        // TODO(dimitris): if only xasChanged, remove QinvCalPrev and add new
        if (asDadChanged || QP[kk].xasChanged) {
        #endif

        // --- intermediate result (used both for Ut and W)

        // M = A[k] * Qinvcal[idxdad]
        dgemm_r_diag_libstr(nx, nx, 1.0,  QP[kk].A, 0, 0, QP[idxdad].QinvCal, 0, 0.0,
            QP[kk].M, 0, 0, QP[kk].M, 0, 0);

        // --- hessian contribution of parent (Ut)

        #ifdef _CHECK_LAST_ACTIVE_SET_
        if (asDadChanged && tree[idxdad].dad >= 0) {
        #else
        if (tree[idxdad].dad >= 0) {
        #endif
            // Ut[idxdad]+offset = M' = - A[k] *  Qinvcal[idxdad]
            // dgetr_libstr(nx, nx, -1.0, QP[kk].M, 0, 0, dual[idxdad].Ut, 0, idxpos);
            dgetr_libstr(nx, nx, QP[kk].M, 0, 0, dual[idxdad].Ut, 0, idxpos);
            dgesc_libstr(nx, nx, -1.0, dual[idxdad].Ut, 0, idxpos);
        }

        // --- hessian contribution of node (diagonal block of W)

        // W[idxdad]+offset = A[k]*M^T = A[k]*Qinvcal[idxdad]*A[k]'
        dsyrk_ln_libstr(nx, nx, 1.0, QP[kk].A, 0, 0, QP[kk].M, 0, 0, 0.0, dual[idxdad].W,
            idxpos, idxpos, dual[idxdad].W, idxpos, idxpos);

        // M = B[k]*Rinvcal[idxdad]
        dgemm_r_diag_libstr(nx, nu, 1.0,  QP[kk].B, 0, 0, QP[idxdad].RinvCal, 0, 0.0,
            QP[kk].M, 0, 0, QP[kk].M, 0, 0);

        // W[idxdad]+offset += B[k]*M^T = B[k]*Rinvcal[idxdad]*B[k]'
        dsyrk_ln_libstr(nx, nu, 1.0, QP[kk].B, 0, 0, QP[kk].M, 0, 0, 1.0, dual[idxdad].W,
            idxpos, idxpos, dual[idxdad].W, idxpos, idxpos);

        // W[idxdad]+offset += Qinvcal[k]
        ddiaad_libstr(nx, 1.0, QP[kk].QinvCal, 0, dual[idxdad].W, idxpos, idxpos);

        // W[idxdad]+offset += regMat (regularization)
        ddiaad_libstr(nx, 1.0, regMat, 0, dual[idxdad].W, idxpos, idxpos);

        #ifdef _CHECK_LAST_ACTIVE_SET_
        // save diagonal block that will be overwritten in factorization
        dgecp_libstr(nx, nx, dual[idxdad].W, idxpos, idxpos, QP[kk].Wdiag, 0, 0);
        #endif

        // --- hessian contribution of preceding siblings (off-diagonal blocks of W)

        #ifdef _CHECK_LAST_ACTIVE_SET_
        if (asDadChanged) {
        #endif
        ns = tree[idxdad].nkids - 1;  // number of siblings
        for (ii = 0; ii < ns; ii++) {
            idxsib = tree[idxdad].kids[ii];
            if (idxsib == kk) break;  // completed all preceding siblings

            // M = A[idxsib] * Qinvcal[idxdad]
            dgemm_r_diag_libstr(nx, nx, 1.0,  QP[idxsib].A, 0, 0, QP[idxdad].QinvCal, 0, 0.0,
                QP[kk].M, 0, 0, QP[kk].M, 0, 0);

            // W[idxdad]+offset = A[k]*M^T = A[k]*Qinvcal[idxdad]*A[idxsib]'
            dgemm_nt_libstr(nx, nx, nx, 1.0, QP[kk].A, 0, 0, QP[kk].M, 0, 0, 0.0, dual[idxdad].W,
                idxpos, ii*nx, dual[idxdad].W, idxpos, ii*nx);

            // M = B[idxsib]*Rinvcal[idxdad]
            dgemm_r_diag_libstr(nx, nu, 1.0,  QP[idxsib].B, 0, 0, QP[idxdad].RinvCal, 0, 0.0,
                QP[kk].M, 0, 0, QP[kk].M, 0, 0);

            // W[idxdad]+offset += B[k]*M^T = B[k]*Rinvcal[idxdad]*B[idxsib]'
            dgemm_nt_libstr(nx, nx, nu, 1.0, QP[kk].B, 0, 0, QP[kk].M, 0, 0, 1.0, dual[idxdad].W,
                idxpos, ii*nx, dual[idxdad].W, idxpos, ii*nx);
        }
        #ifdef _CHECK_LAST_ACTIVE_SET_
        }
        #endif

        #ifdef _CHECK_LAST_ACTIVE_SET_
        } else {
            dgecp_libstr(nx, nx, QP[kk].Wdiag, 0, 0, dual[idxdad].W, idxpos, idxpos);
        }
        #endif
    }

    #if DEBUG == 1
    for (kk = 0; kk < Np; kk++) {
        d_cvt_strvec2vec(dual[kk].res->m, dual[kk].res, 0, &res[indres]);
        indres += dual[kk].res->m;
        d_cvt_strmat2mat(dual[kk].W->n, dual[kk].W->n, dual[kk].W, 0, 0, &W[indW], dual[kk].W->n);
        indW += dual[kk].W->n*dual[kk].W->n;
        if (kk > 0) {
            d_cvt_strmat2mat(dual[kk].Ut->m, dual[kk].Ut->n, dual[kk].Ut, 0, 0,
                &Ut[indUt], dual[kk].Ut->m);
            indUt += dual[kk].Ut->m*dual[kk].Ut->n;
        }
    }
    write_double_vector_to_txt(res, (Nn-1)*nx, "data_tree/res.txt");
    write_double_vector_to_txt(W, dimW, "data_tree/W.txt");
    write_double_vector_to_txt(Ut, dimUt, "data_tree/Ut.txt");
    #endif

    return TREEQP_OK;
}


static void calculate_delta_lambda(int_t Np, int_t Nh, int_t idxFactorStart, int_t *npar,
    dual_block *dual, struct node *tree) {

    int_t ii, kk, idxdad, idxpos;
    int_t icur = Np-1;

    #if DEBUG == 1
    int_t dimlam = 0;  // aka (Nn-1)*nx
    for (kk = 0; kk < Np; kk++) {
        dimlam += dual[kk].deltalambda->m;
    }
    real_t deltalambda[dimlam];
    int_t indlam = 0;
    #endif

    // --- Cholesky factorization merged with backward substitution

    for (kk = Nh-1; kk > 0; kk--) {
        #if PRINT_LEVEL > 2
        printf("\n--------- New (parallel) factorization branch  ---------\n");
        #endif
        #ifdef PARALLEL
        #pragma omp parallel for private(idxdad, idxpos)
        #endif
        for (ii = icur; ii > icur-npar[kk]; ii--) {

            // NOTE(dimitris): result of backward substitution saved in deltalambda
            // NOTE(dimitris): substitution for free if dual[ii].W not multiple of 4 (in LA=HP)
            #ifdef _MERGE_FACTORIZATION_WITH_SUBSTITUTION_

            #ifdef _CHECK_LAST_ACTIVE_SET_
            if (ii < idxFactorStart) {
            #endif

            // add resMod in last row of matrix W
            drowin_libstr(dual[ii].resMod->m, 1.0, dual[ii].resMod, 0, dual[ii].W, dual[ii].W->m-1, 0);
            // perform Cholesky factorization and backward substitution together
            dpotrf_l_mn_libstr(dual[ii].W->m, dual[ii].W->n, dual[ii].W, 0, 0, dual[ii].CholW, 0, 0);
            // extract result of substitution
            drowex_libstr(dual[ii].deltalambda->m, 1.0, dual[ii].CholW, dual[ii].CholW->m-1, 0, dual[ii].deltalambda, 0);

            #ifdef _CHECK_LAST_ACTIVE_SET_
            } else {
            // perform only vector substitution
            dtrsv_lnn_libstr(dual[ii].resMod->m, dual[ii].CholW, 0, 0, dual[ii].resMod, 0,
                dual[ii].deltalambda, 0);
            }
            #endif

            #else  /* _MERGE_FACTORIZATION_WITH_SUBSTITUTION_ */

            #ifdef _CHECK_LAST_ACTIVE_SET_
            if (ii < idxFactorStart) {
            #endif
            // Cholesky factorization to calculate factor of current diagonal block
            dpotrf_l_libstr(dual[ii].W->n, dual[ii].W, 0, 0, dual[ii].CholW, 0, 0);
            #ifdef _CHECK_LAST_ACTIVE_SET_
            }  // TODO(dimitris): we can probably skip more calculations (see scenarios)
            #endif

            // vector substitution
            dtrsv_lnn_libstr(dual[ii].resMod->m, dual[ii].CholW, 0, 0, dual[ii].resMod, 0,
                dual[ii].deltalambda, 0);

            #endif  /* _MERGE_FACTORIZATION_WITH_SUBSTITUTION_ */

            // Matrix substitution to calculate transposed factor of parent block
            dtrsm_rltn_libstr(dual[ii].Ut->m, dual[ii].Ut->n, 1.0, dual[ii].CholW, 0, 0,
                dual[ii].Ut, 0, 0, dual[ii].CholUt, 0, 0);

            // Symmetric matrix multiplication to update diagonal block of parent
            // NOTE(dimitris): use dgemm_nt_libstr if dsyrk not implemented yet
            idxdad = tree[ii].dad;
            idxpos = tree[ii].idxkid*nx;
            dsyrk_ln_libstr(dual[ii].CholUt->m, dual[ii].CholUt->n, -1.0,
                dual[ii].CholUt, 0, 0, dual[ii].CholUt, 0, 0, 1.0, dual[idxdad].W, idxpos, idxpos,
                dual[idxdad].W, idxpos, idxpos);

            // Matrix vector multiplication to update vector of parent
            dgemv_n_libstr(dual[ii].CholUt->m, dual[ii].CholUt->n, -1.0, dual[ii].CholUt, 0, 0,
                dual[ii].deltalambda, 0, 1.0, dual[idxdad].resMod, idxpos, dual[idxdad].resMod, idxpos);
        }
        icur -= npar[kk];
    }
    #ifdef _MERGE_FACTORIZATION_WITH_SUBSTITUTION_
    // add resMod in last row of matrix W
    drowin_libstr(dual[0].resMod->m, 1.0, dual[0].resMod, 0, dual[0].W, dual[0].W->m-1, 0);
    // perform Cholesky factorization and backward substitution together
    dpotrf_l_mn_libstr(dual[0].W->m, dual[0].W->n, dual[0].W, 0, 0, dual[0].CholW, 0, 0);
    // extract result of substitution
    drowex_libstr(dual[0].deltalambda->m, 1.0, dual[0].CholW, dual[0].CholW->m-1, 0, dual[0].deltalambda, 0);
    #else
    // calculate Cholesky factor of root block
    dpotrf_l_libstr(dual[0].W->m, dual[0].W, 0, 0, dual[0].CholW, 0, 0);

    // calculate last elements of backward substitution
    dtrsv_lnn_libstr(dual[0].resMod->m, dual[0].CholW, 0, 0, dual[0].resMod, 0,
        dual[0].deltalambda, 0);
    #endif

    // --- Forward substitution

    icur = 1;

    dtrsv_ltn_libstr(dual[0].deltalambda->m, dual[0].CholW, 0, 0, dual[0].deltalambda, 0,
        dual[0].deltalambda, 0);

    for (kk = 1; kk < Nh; kk++) {
        #ifdef PARALLEL
        #pragma omp parallel for private(idxdad, idxpos)
        #endif
        for (ii = icur; ii < icur+npar[kk]; ii++) {
            idxdad = tree[ii].dad;
            idxpos = tree[ii].idxkid*nx;
            dgemv_t_libstr(dual[ii].CholUt->m, dual[ii].CholUt->n, -1.0, dual[ii].CholUt, 0, 0,
                dual[idxdad].deltalambda, idxpos, 1.0, dual[ii].deltalambda, 0, dual[ii].deltalambda, 0);

            dtrsv_ltn_libstr(dual[ii].deltalambda->m, dual[ii].CholW, 0, 0, dual[ii].deltalambda, 0,
                dual[ii].deltalambda, 0);
        }
        icur += npar[kk];
    }

    #if PRINT_LEVEL > 2
    for (ii = 0; ii < Np; ii++) {
        printf("\nCholesky factor of diagonal block #%d as strmat: \n\n", ii+1);
        d_print_strmat(dual[ii].CholW->m, dual[ii].CholW->n, dual[ii].CholW, 0, 0);
    }
    for (ii = 1; ii < Np; ii++) {
        printf("\nTransposed Cholesky factor of parent block #%d as strmat: \n\n", ii+1);
        d_print_strmat(dual[ii].CholUt->m, dual[ii].CholUt->n, dual[ii].CholUt, 0, 0);
    }

    printf("\nResult of backward substitution:\n\n");
    for (ii = 0; ii < Np; ii++) {
        d_print_strvec(dual[0].deltalambda->m, dual[0].deltalambda, 0);
    }

    printf("\nResult of forward substitution (aka final result):\n\n");
    for (ii = 0; ii < Np; ii++) {
        d_print_strvec(dual[ii].deltalambda->m, dual[ii].deltalambda, 0);
    }
    #endif

    #if DEBUG == 1
    for (kk = 0; kk < Np; kk++) {
        d_cvt_strvec2vec(dual[kk].deltalambda->m, dual[kk].deltalambda, 0, &deltalambda[indlam]);
        indlam += dual[kk].deltalambda->m;
    }
    write_double_vector_to_txt(deltalambda, dimlam, "data_tree/deltalambda.txt");
    #endif
}


static real_t gradient_trans_times_direction(int_t Np, dual_block *dual) {
    int_t kk;
    real_t ans = 0;

    for (kk = 0; kk < Np; kk++) {
        ans += ddot_libstr(dual[kk].res->m, dual[kk].res, 0, dual[kk].deltalambda, 0);
    }
    // NOTE: res has was -gradient above
    return -ans;
}


static real_t evaluate_dual_function(int_t Nn, int_t Np, stage_QP *QP, dual_block *dual, struct node *tree) {
    int_t ii, jj, kk, idxkid, idxpos, idxdad;
    real_t fval = 0;

    #ifdef PARALLEL
    #pragma omp parallel for private(ii, jj, idxkid, idxpos, idxdad)
    #endif
    // NOTE: same code as in solve_stage_problems but:
    // - without calculating as
    // - without calculating elimination matrix
    // - with calculating modified constant term
    for (kk = 0; kk < Nn; kk++) {
        idxdad = tree[kk].dad;
        idxpos = tree[kk].idxkid*nx;
        // --- update QP gradient

        // qmod[k] = - q[k] + lambda[k]
        if (kk == 0) {
            // lambda[0] = 0
            for (jj = 0; jj < nx; jj++) DVECEL_LIBSTR(QP[kk].qmod, jj) = 0.0;
            daxpy_libstr(nx, -1.0, QP[kk].q, 0, QP[kk].qmod, 0, QP[kk].qmod, 0);
        } else {
            daxpy_libstr(nx, -1.0, QP[kk].q, 0, dual[idxdad].lambda, idxpos, QP[kk].qmod, 0);
        }

        // rmod[k] = - r[k]
        if (kk < Np) {
            dveccp_libstr(nu, QP[kk].r, 0, QP[kk].rmod, 0);
            dvecsc_libstr(nu, -1.0, QP[kk].rmod, 0);
        }

        // cmod[k] = 0
        QP[kk].cmod = 0.;

        for (ii = 0; ii < tree[kk].nkids; ii++) {
            idxkid = tree[kk].kids[ii];
            idxdad = tree[idxkid].dad;
            idxpos = tree[idxkid].idxkid*nx;

            // cmod[k] += b[jj]' * lambda[jj]
            QP[kk].cmod += ddot_libstr(nx, QP[idxkid].b, 0, dual[idxdad].lambda, idxpos);

            // return x^T * y

            // qmod[k] -= A[jj]' * lambda[jj]
            dgemv_t_libstr(nx, nx, -1.0, QP[idxkid].A, 0, 0, dual[idxdad].lambda, idxpos, 1.0,
                 QP[kk].qmod, 0, QP[kk].qmod, 0);
            if (kk < Np) {
                // rmod[k] -= B[jj]' * lambda[jj]
                dgemv_t_libstr(nx, nu, -1.0, QP[idxkid].B, 0, 0, dual[idxdad].lambda, idxpos, 1.0,
                    QP[kk].rmod, 0, QP[kk].rmod, 0);
            }
        }

        // --- solve QP
        // x[k] = Q[k]^-1 .* qmod[k] (NOTE: minus sign already in mod. gradient)
        dvecmuldot_libstr(nx, QP[kk].Qinv, 0, QP[kk].qmod, 0, QP[kk].x, 0);

        // x[k] = median(xmin, x[k], xmax)
        dveccl_libstr(nx, QP[kk].xmin, 0, QP[kk].x, 0, QP[kk].xmax, 0, QP[kk].x, 0);

        if (kk < Np) {
            // u[k] = R[k]^-1 .* rmod[k]
            dvecmuldot_libstr(nu, QP[kk].Rinv, 0, QP[kk].rmod, 0, QP[kk].u, 0);
            // u[k] = median(umin, u[k], umax)
            dveccl_libstr(nu, QP[kk].umin, 0, QP[kk].u, 0, QP[kk].umax, 0, QP[kk].u, 0);
        }

        // --- calculate dual function term

        // feval = - (1/2)x[k]' * Q[k] * x[k] + x[k]' * qmod[k] - cmod[k]
        // NOTE: qmod[k] has already a minus sign
        // NOTE: xas used as workspace
        dvecmuldot_libstr(nx, QP[kk].Q, 0, QP[kk].x, 0, QP[kk].xas, 0);
        QP[kk].fval = -0.5*ddot_libstr(nx, QP[kk].xas, 0, QP[kk].x, 0) - QP[kk].cmod;
        QP[kk].fval += ddot_libstr(nx, QP[kk].qmod, 0, QP[kk].x, 0);

        if (kk < Np) {
            // feval -= (1/2)u[k]' * R[k] * u[k] - u[k]' * rmod[k]
            dvecmuldot_libstr(nu, QP[kk].R, 0, QP[kk].u, 0, QP[kk].uas, 0);
            QP[kk].fval -= 0.5*ddot_libstr(nu, QP[kk].uas, 0, QP[kk].u, 0);
            QP[kk].fval += ddot_libstr(nu, QP[kk].rmod, 0, QP[kk].u, 0);
        }
    }

    for (kk = 0; kk < Nn; kk++) fval += QP[kk].fval;

    return fval;
}


static int_t line_search(int_t Nn, int_t Np, stage_QP *QP, dual_block *dual, struct node *tree,
    treeqp_tdunes_options_t *opts) {
    int_t jj, kk;

    #if DEBUG == 1
    real_t lambda[(Nn-1)*nx];
    int_t indlam = 0;
    #endif

    real_t dotProduct, fval, fval0;
    real_t tau = 1;
    real_t tauPrev = 0;

    dotProduct = gradient_trans_times_direction(Np, dual);
    fval0 = evaluate_dual_function(Nn, Np, QP, dual, tree);
    // printf(" dot_product = %f\n", dotProduct);
    // printf(" dual_function = %f\n", fval0);

    for (jj = 1; jj <= opts->lineSearchMaxIter; jj++) {
        // update multipliers
        #ifdef PARALLEL
        #pragma omp parallel for
        #endif
        for (kk = 0; kk < Np; kk++) {
            daxpy_libstr(dual[kk].deltalambda->m, tau-tauPrev, dual[kk].deltalambda, 0,
                dual[kk].lambda, 0, dual[kk].lambda, 0);
        }

        // evaluate dual function
        fval = evaluate_dual_function(Nn, Np, QP, dual, tree);
        // printf("LS iteration #%d (fval = %f <? %f )\n", jj, fval, fval0 + opts->lineSearchGamma*tau*dotProduct);

        // check condition
        if (fval < fval0 + opts->lineSearchGamma*tau*dotProduct) {
            // printf("Condition satisfied at iteration %d\n", jj);
            break;
        } else {
            tauPrev = tau;
            tau = opts->lineSearchBeta*tauPrev;
        }
    }
    #if DEBUG == 1
    for (kk = 0; kk < Np; kk++) {
        d_cvt_strvec2vec(dual[kk].lambda->m, dual[kk].lambda, 0, &lambda[indlam]);
        indlam += dual[kk].lambda->m;
    }
    write_double_vector_to_txt(lambda, (Nn-1)*nx, "data_tree/lambda_opt.txt");
    write_double_vector_to_txt(&dotProduct, 1, "data_tree/dotProduct.txt");
    write_double_vector_to_txt(&fval0, 1, "data_tree/fval0.txt");
    write_int_vector_to_txt(&jj, 1, "data_tree/lsiter.txt");
    #endif

    return jj;
}


void write_solution_to_txt(int_t Nn, int_t Np, int_t iter, stage_QP *QP, dual_block *dual,
    struct node *tree) {

    int_t kk, indx, indu, ind;

    // TODO(dimitris): maybe use Np for u instead of Nn in other places too to avoid confusion
    real_t *x = malloc(Nn*nx*sizeof(real_t));
    real_t *u = malloc(Np*nu*sizeof(real_t));
    real_t *deltalambda = malloc((Nn-1)*nx*sizeof(real_t));
    real_t *lambda = malloc((Nn-1)*nx*sizeof(real_t));

    indx = 0; indu = 0;
    for (kk = 0; kk < Nn; kk++) {
        d_cvt_strvec2vec(nx, QP[kk].x, 0, &x[indx]);
        indx += nx;
        if (kk < Np) {
            d_cvt_strvec2vec(nu, QP[kk].u, 0, &u[indu]);
            indu += nu;
        }
    }

    ind = 0;
    for (kk = 0; kk < Np; kk++) {
        d_cvt_strvec2vec(dual[kk].deltalambda->m, dual[kk].deltalambda, 0, &deltalambda[ind]);
        d_cvt_strvec2vec(dual[kk].lambda->m, dual[kk].lambda, 0, &lambda[ind]);
        ind += dual[kk].lambda->m;
    }

    write_double_vector_to_txt(x, Nn*nx, "data_tree/x_opt.txt");
    write_double_vector_to_txt(u, Np*nu, "data_tree/u_opt.txt");
    write_double_vector_to_txt(lambda, (Nn-1)*nx, "data_tree/deltalambda_opt.txt");
    write_double_vector_to_txt(lambda, (Nn-1)*nx, "data_tree/lambda_opt.txt");
    write_int_vector_to_txt(&iter, 1, "data_tree/iter.txt");

    #if PROFILE > 0
    write_timers_to_txt( );
    #endif

    free(x);
    free(u);
    free(deltalambda);
    free(lambda);
}

int_t treeqp_tdunes_solve(struct d_strvec *regMat, stage_QP *stage_QPs, dual_block *dual,
    tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, treeqp_tdunes_options_t *opts, treeqp_tdunes_workspace *work) {

    int status;
    int idxFactorStart;
    int lsIter;

    int_t NewtonIter;

    struct node *tree = (struct node *)qp_in->tree;

    int_t Nn = work->Nn;
    int_t Np = work->Np;
    int_t *npar = work->npar;

    // TEMP!!
    for (int_t ii = 0; ii < Nn; ii++) {
        if (ii < Np) {
            dual[ii].W = &work->sW[ii];
        }
    }

    // dual Newton iterations
    for (NewtonIter = 0; NewtonIter < opts->maxIter; NewtonIter++) {
        #if PROFILE > 1
        treeqp_tic(&iter_tmr);
        #endif

        // solve stage QPs, update active sets, calculate elimination matrices
        #if PROFILE > 2
        treeqp_tic(&tmr);
        #endif
        solve_stage_problems(Nn, stage_QPs, dual, tree);
        #if PROFILE > 2
        stage_qps_times[NewtonIter] = treeqp_toc(&tmr);
        #endif

        // calculate gradient and Hessian of the dual problem
        #if PROFILE > 2
        treeqp_tic(&tmr);
        #endif
        status = build_dual_problem(Nn, Np, stage_QPs, dual, tree, regMat, &idxFactorStart, opts);
        #if PROFILE > 2
        build_dual_times[NewtonIter] = treeqp_toc(&tmr);
        #endif
        if (status == TREEQP_SUCC_OPTIMAL_SOLUTION_FOUND) {
            // if (ll == NRUNS-1) {
            //     printf("Convergence achieved in %d iterations \t(error %5.2e)\n", NewtonIter, error);
            // }
            // printf("optimal solution found\n", 1);
            break;
        }

        // factorize Newton matrix and calculate step direction
        #if PROFILE > 2
        treeqp_tic(&tmr);
        #endif
        calculate_delta_lambda(Np, Nh, idxFactorStart, npar, dual, tree);
        #if PROFILE > 2
        newton_direction_times[NewtonIter] = treeqp_toc(&tmr);
        #endif

        // line-search
        // NOTE: line-search overwrites xas, uas (used as workspace)
        #if PROFILE > 2
        treeqp_tic(&tmr);
        #endif
        lsIter = line_search(Nn, Np, stage_QPs, dual, tree, opts);
        #if PROFILE > 2
        line_search_times[NewtonIter] = treeqp_toc(&tmr);
        #endif

        #if PRINT_LEVEL > 1
        if (NewtonIter == 0) printf("\n--- RUN #%d ---\n\n", ll+1);
        printf("iteration #%d: %d ls iterations \t\t(error %5.2e)\n", NewtonIter, lsIter, error);
        #endif
        #if PROFILE > 1
        iter_times[NewtonIter] = treeqp_toc(&iter_tmr);
        ls_iters[NewtonIter] = lsIter;
        #endif
    }

    qp_out->info.iter = NewtonIter;

    if (qp_out->info.iter == opts->maxIter)
        status = TREEQP_ERR_MAXIMUM_ITERATIONS_REACHED;

    return status;  // TODO(dimitris): return correct status
}


int_t treeqp_tdunes_calculate_size(tree_ocp_qp_in *qp_in) {
    struct node *tree = (struct node *) qp_in->tree;
    int_t Nn = qp_in->N;
    int_t Nh = tree[Nn-1].stage;
    int_t Np = get_number_of_parent_nodes(Nn, tree);
    int_t bytes = 0;
    int_t dim;

    int_t nx = qp_in->nx[0];

    bytes += Nh*sizeof(int);  // npar

    // struct pointers
    bytes += Np*sizeof(struct d_strmat);

    for (int_t ii = 0; ii < Nn; ii++) {

        if (ii < Np) {
            dim = tree[ii].nkids*nx;
            #ifdef _MERGE_FACTORIZATION_WITH_SUBSTITUTION_
            bytes += d_size_strmat(dim + 1, dim);  // W
            #else
            bytes += d_size_strmat(dim, dim);  // W
            #endif
        }
    }

    bytes += (bytes + 63)/64*64;
    bytes += 64;

    return bytes;
}


void create_treeqp_tdunes(tree_ocp_qp_in *qp_in, treeqp_tdunes_options_t *opts,
    treeqp_tdunes_workspace *work, void *ptr) {

    struct node *tree = (struct node *) qp_in->tree;
    int_t Nn = qp_in->N;
    int_t Nh = tree[Nn-1].stage;
    int_t Np = get_number_of_parent_nodes(Nn, tree);
    int_t dim;

    // save some useful dimensions to workspace
    work->Nn = Nn;
    work->Np = Np;

    // char pointer
    char *c_ptr = (char *) ptr;

    // pointers
    work->npar = (int_t *) c_ptr;
    c_ptr += Nh*sizeof(int_t);
    setup_npar(Nh, Nn, tree, work->npar);

    work->sW = (struct d_strmat *) c_ptr;
    c_ptr += Np*sizeof(struct d_strmat);

    // move pointer for proper alignment of blasfeo matrices and vectors
    long long l_ptr = (long long) c_ptr;
    l_ptr = (l_ptr+63)/64*64;
    c_ptr = (char *) l_ptr;

    for (int_t ii = 0; ii < Nn; ii++) {
        if (ii < Np) {
            dim = tree[ii].nkids*nx;
            #ifdef _MERGE_FACTORIZATION_WITH_SUBSTITUTION_
            init_strmat(dim+1, dim, &work->sW[ii], &c_ptr);
            #else
            init_strmat(dim, dim, &work->sW[ii], &c_ptr);
            #endif
        }
    }
}


int main(int argc, char const *argv[]) {

    int_t Nn = calculate_number_of_nodes(md, Nr, Nh);

    treeqp_tdunes_options_t opts = set_default_options();

    // TODO(dimitris): implement those
    // read initial point from txt file
    // read constraint on x0 from txt file

    // setup scenario tree
    struct node *tree = malloc(Nn*sizeof(struct node));
    setup_tree(md, Nr, Nh, Nn, tree);

    // setup QP
    tree_ocp_qp_in qp_in;

    int_t *nxVec = malloc(Nn*sizeof(int_t));
    int_t *nuVec = malloc(Nn*sizeof(int_t));

    for (int_t ii = 0; ii < Nn; ii++) {
        // state and input dimensions on each node (only different at root/leaves)
        if (ii > 0) {
            nxVec[ii] = nx;
        } else {
            // TODO(dimitris): support both with and without eliminating x0
            nxVec[ii] = nx;
        }

        if (tree[ii].nkids > 0) {  // not a leaf
            nuVec[ii] = nu;
        } else {
            nuVec[ii] = 0;
        }
    }

    int_t qp_in_size = tree_ocp_qp_in_calculate_size(Nn, nxVec, nuVec, tree);
    void *qp_in_memory = malloc(qp_in_size);
    create_tree_ocp_qp_in(Nn, nxVec, nuVec, tree, &qp_in, qp_in_memory);
    free(nxVec); free(nuVec);

    // setup output
    tree_ocp_qp_out qp_out;

    // setup QP solver
    treeqp_tdunes_workspace work;

    int_t treeqp_size = treeqp_tdunes_calculate_size(&qp_in);
    void *qp_solver_memory = malloc(treeqp_size);
    create_treeqp_tdunes(&qp_in, &opts, &work, qp_solver_memory);

    int_t ii, jj, kk, ll, status_OLD, real, dim, indlam, lsIter, idxFactorStart;
    real_t prob;
    return_t status;

    int_t Np = get_number_of_parent_nodes(Nn, tree);  // = Nn - ipow(md, Nr), for the standard case

    stage_QP *stage_QPs = malloc(Nn*sizeof(stage_QP));

    dual_block *dual = malloc(Np*sizeof(dual_block));

    int_t npar[Nh];  // number of parallel factorizations per stage

    setup_npar(Nh, Nn, tree, npar);

    real_t dQinv[nx], dRinv[nu], dPinv[nx];

    for (ii = 0; ii < nx; ii++) dQinv[ii] = 1./dQ[ii];
    for (ii = 0; ii < nx; ii++) dPinv[ii] = 1./dP[ii];
    for (ii = 0; ii < nu; ii++) dRinv[ii] = 1./dR[ii];

    int_t nl = Nn*nx;  // number of dual variables

    real_t error;

    #ifndef _HARDCODE_INITIAL_POINT_
    real_t x0[nx], lambda[nl];  // TODO(dimitris): use malloc here too.

    status = read_double_vector_from_txt(x0, nx, "data_tree/x0.txt");
    if (status != TREEQP_OK) return status;
    status = read_double_vector_from_txt(lambda, nl, "data_tree/lambda0.txt");
    if (status != TREEQP_OK) return status;
    #endif

    struct d_strvec regMat, sx0;

    struct d_strmat sA_in[md], sB_in[md];
    struct d_strvec sb_in[md];
    struct d_strvec sQ[Nh+1], sQinv[Nh+1], sq[Nh+1], sR[Nh], sRinv[Nh], sr[Nh];
    struct d_strvec sxmin_in,  sxmax_in;
    struct d_strvec sumin_in, sumax_in;

    // TODO(dimitris): do the same for scenarios
    struct d_strmat *sW = malloc(Np*sizeof(struct d_strmat));
    struct d_strmat *sUt = malloc((Np-1)*sizeof(struct d_strmat));
    struct d_strmat *sCholW = malloc(Np*sizeof(struct d_strmat));
    struct d_strmat *sCholUt = malloc((Np-1)*sizeof(struct d_strmat));

    struct d_strvec *sres = malloc(Np*sizeof(struct d_strvec));
    struct d_strvec *sresMod = malloc(Np*sizeof(struct d_strvec));
    struct d_strvec *slambda = malloc(Np*sizeof(struct d_strvec));
    struct d_strvec *sDeltalambda = malloc(Np*sizeof(struct d_strvec));

    // q,r + contribution of current dual multipliers
    struct d_strvec *sqmod = malloc(Nn*sizeof(struct d_strvec));
    struct d_strvec *srmod = malloc(Nn*sizeof(struct d_strvec));

    struct d_strvec *sx = malloc(Nn*sizeof(struct d_strvec));
    struct d_strvec *su = malloc(Np*sizeof(struct d_strvec));
    struct d_strvec *sxas = malloc(Nn*sizeof(struct d_strvec));
    struct d_strvec *suas = malloc(Np*sizeof(struct d_strvec));

    struct d_strvec *sQinvCal = malloc(Nn*sizeof(struct d_strvec));
    struct d_strvec *sRinvCal = malloc(Np*sizeof(struct d_strvec));

    #ifdef _CHECK_LAST_ACTIVE_SET_
    struct d_strvec *sxasPrev = malloc(Nn*sizeof(struct d_strvec));
    struct d_strvec *suasPrev = malloc(Np*sizeof(struct d_strvec));
    struct d_strmat *sWdiag = malloc(Nn*sizeof(struct d_strmat));
    #endif

    struct d_strmat *sM = malloc(Nn*sizeof(struct d_strmat));

    int_t memorySize = calculate_blasfeo_memory_size_tree(Nh, Nr, md, nx, nu, tree);
    #if PRINT_LEVEL > 0
    printf("\n-------- Blasfeo requires %d bytes of memory\n\n", memorySize);
    #endif
    #if PRINT_LEVEL > 0
    printf("\n-------- treeQP workspace requires %d bytes \n", treeqp_size);
    #endif

    void *tmpBlasfeoPtr;
    v_zeros_align(&tmpBlasfeoPtr, memorySize);
    char *blasfeoPtr = (char *) tmpBlasfeoPtr;

    for (ii = 0; ii < md; ii++) {
        // NOTE: first dynamics in data file are the nominal ones (skipped here)
        wrapper_mat_to_strmat(nx, nx, &A[(ii+1)*nx*nx], &sA_in[ii], &blasfeoPtr);
        wrapper_mat_to_strmat(nx, nu, &B[(ii+1)*nx*nu], &sB_in[ii], &blasfeoPtr);
        wrapper_vec_to_strvec(nx, &b[(ii+1)*nx], &sb_in[ii], &blasfeoPtr);
    }

    for (kk = 0; kk < Nh; kk++) {
        // calculate stage-wise scaling to minimize average performance
        if (kk < Nr) {
            prob = 1.0*ipow(md, Nr - kk);
        } else {
            prob = 1.0;
        }
        wrapper_vec_to_strvec(nx, dQ, &sQ[kk], &blasfeoPtr);
        wrapper_vec_to_strvec(nx, dQinv, &sQinv[kk], &blasfeoPtr);
        wrapper_vec_to_strvec(nx, q, &sq[kk], &blasfeoPtr);
        wrapper_vec_to_strvec(nu, dR, &sR[kk], &blasfeoPtr);
        wrapper_vec_to_strvec(nu, dRinv, &sRinv[kk], &blasfeoPtr);
        wrapper_vec_to_strvec(nu, r, &sr[kk], &blasfeoPtr);

        for (jj = 0; jj < nx; jj++) DVECEL_LIBSTR(&sQ[kk], jj) *= prob;
        for (jj = 0; jj < nx; jj++) DVECEL_LIBSTR(&sQinv[kk], jj) *= 1.0/prob;
        for (jj = 0; jj < nx; jj++) DVECEL_LIBSTR(&sq[kk], jj) *= prob;
        for (jj = 0; jj < nu; jj++) DVECEL_LIBSTR(&sR[kk], jj) *= prob;
        for (jj = 0; jj < nu; jj++) DVECEL_LIBSTR(&sRinv[kk], jj) *= 1.0/prob;
        for (jj = 0; jj < nu; jj++) DVECEL_LIBSTR(&sr[kk], jj) *= prob;
    }
    // TODO(dimitris): update prob for random trees?
    prob = 1.0;
    wrapper_vec_to_strvec(nx, dP, &sQ[Nh], &blasfeoPtr);
    wrapper_vec_to_strvec(nx, dPinv, &sQinv[Nh], &blasfeoPtr);
    wrapper_vec_to_strvec(nx, p, &sq[Nh], &blasfeoPtr);
    for (jj = 0; jj < nx; jj++) DVECEL_LIBSTR(&sQ[Nh], jj) *= prob;
    for (jj = 0; jj < nx; jj++) DVECEL_LIBSTR(&sQinv[Nh], jj) *= 1.0/prob;
    for (jj = 0; jj < nx; jj++) DVECEL_LIBSTR(&sq[Nh], jj) *= prob;

    // TODO(dimitris): find out why algorithm is not scale-invariant

    wrapper_vec_to_strvec(nx, x0, &sx0, &blasfeoPtr);
    wrapper_vec_to_strvec(nx, xmin, &sxmin_in, &blasfeoPtr);
    wrapper_vec_to_strvec(nx, xmax, &sxmax_in, &blasfeoPtr);
    wrapper_vec_to_strvec(nu, umin, &sumin_in, &blasfeoPtr);
    wrapper_vec_to_strvec(nu, umax, &sumax_in, &blasfeoPtr);
    init_strvec(md*nx, &regMat, &blasfeoPtr);
    dvecse_libstr(md*nx, opts.regValue, &regMat, 0);

    for (kk = 0; kk < Nn; kk++) {
            init_strvec(nx, &sx[kk], &blasfeoPtr);
            init_strvec(nx, &sxas[kk], &blasfeoPtr);
            #ifdef _CHECK_LAST_ACTIVE_SET_
            init_strvec(nx, &sxasPrev[kk], &blasfeoPtr);
            // NOTE(dimitris): set a value outside {-1, 0, 1} to force full factorization at 1st it.
            dvecse_libstr(nx, 0.0/0.0, &sxasPrev[kk], 0);
            init_strmat(nx, nx, &sWdiag[kk], &blasfeoPtr);
            #endif
            init_strvec(nx, &sqmod[kk], &blasfeoPtr);
            init_strvec(nx, &sQinvCal[kk], &blasfeoPtr);
            init_strmat(MAX(nx, nu), MAX(nx, nu), &sM[kk], &blasfeoPtr);
        if (kk < Np) {
            init_strvec(nu, &su[kk], &blasfeoPtr);
            init_strvec(nu, &suas[kk], &blasfeoPtr);
            #ifdef _CHECK_LAST_ACTIVE_SET_
            init_strvec(nu, &suasPrev[kk], &blasfeoPtr);
            dvecse_libstr(nu, 0.0/0.0, &suasPrev[kk], 0);
            #endif
            init_strvec(nu, &srmod[kk], &blasfeoPtr);
            init_strvec(nu, &sRinvCal[kk], &blasfeoPtr);
        }
    }

    indlam = nx;  // NOTE: we skip lambda[0] which is zero by convention
    for (kk = 0; kk < Np; kk++) {
        dim = tree[kk].nkids*nx;
        init_strvec(dim, &sres[kk], &blasfeoPtr);
        dual[kk].res = &sres[kk];
        init_strvec(dim, &sresMod[kk], &blasfeoPtr);
        dual[kk].resMod = &sresMod[kk];
        wrapper_vec_to_strvec(dim, &lambda[indlam], &slambda[kk], &blasfeoPtr);
        indlam += dim;
        dual[kk].lambda = &slambda[kk];
        init_strvec(dim, &sDeltalambda[kk], &blasfeoPtr);
        dual[kk].deltalambda = &sDeltalambda[kk];
        #ifdef _MERGE_FACTORIZATION_WITH_SUBSTITUTION_
        init_strmat(dim+1, dim, &sW[kk], &blasfeoPtr);
        dual[kk].W = &sW[kk];
        init_strmat(dim+1, dim, &sCholW[kk], &blasfeoPtr);
        dual[kk].CholW = &sCholW[kk];
        #else
        init_strmat(dim, dim, &sW[kk], &blasfeoPtr);
        dual[kk].W = &sW[kk];
        init_strmat(dim, dim, &sCholW[kk], &blasfeoPtr);
        dual[kk].CholW = &sCholW[kk];
        #endif
        if (kk > 0) {
            init_strmat(nx, dim, &sUt[kk-1], &blasfeoPtr);
            dual[kk].Ut = &sUt[kk-1];
            init_strmat(nx, dim, &sCholUt[kk-1], &blasfeoPtr);
            dual[kk].CholUt = &sCholUt[kk-1];
        } else {
            dual[kk].Ut = NULL;
            dual[kk].CholUt = NULL;
        }
    }

    // build stage_QPs
    for (kk = 0; kk < Nn; kk++) {
        // iterates and intermediate results
        stage_QPs[kk].x = &sx[kk];
        stage_QPs[kk].xas = &sxas[kk];
        #ifdef _CHECK_LAST_ACTIVE_SET_
        stage_QPs[kk].xasPrev = &sxasPrev[kk];
        stage_QPs[kk].Wdiag = &sWdiag[kk];
        #endif
        stage_QPs[kk].qmod = &sqmod[kk];
        stage_QPs[kk].QinvCal = &sQinvCal[kk];
        stage_QPs[kk].M = &sM[kk];

        if (kk < Np) {
            stage_QPs[kk].u = &su[kk];
            stage_QPs[kk].uas = &suas[kk];
            #ifdef _CHECK_LAST_ACTIVE_SET_
            stage_QPs[kk].uasPrev = &suasPrev[kk];
            #endif
            stage_QPs[kk].rmod = &srmod[kk];
            stage_QPs[kk].RinvCal = &sRinvCal[kk];
        } else {
            stage_QPs[kk].u = NULL;
            stage_QPs[kk].uas = NULL;
            #ifdef _CHECK_LAST_ACTIVE_SET_
            stage_QPs[kk].uasPrev = NULL;
            #endif
            stage_QPs[kk].rmod = NULL;
            stage_QPs[kk].RinvCal = NULL;
        }
        // input data
        if (kk > 0) {
            real = tree[kk].real;
            stage_QPs[kk].A = &sA_in[real];
            stage_QPs[kk].B = &sB_in[real];
            stage_QPs[kk].b = &sb_in[real];
        } else {  // NOTE: dynamics assigned to children nodes
            stage_QPs[kk].A = NULL;
            stage_QPs[kk].B = NULL;
            stage_QPs[kk].b = NULL;
        }

        stage_QPs[kk].Q = &sQ[tree[kk].stage];
        stage_QPs[kk].Qinv = &sQinv[tree[kk].stage];
        stage_QPs[kk].q = &sq[tree[kk].stage];

        if (kk < Np) {
            stage_QPs[kk].R = &sR[tree[kk].stage];
            stage_QPs[kk].Rinv = &sRinv[tree[kk].stage];
            stage_QPs[kk].r = &sr[tree[kk].stage];

            if (kk > 0) {
                stage_QPs[kk].xmin = &sxmin_in;
                stage_QPs[kk].xmax = &sxmax_in;
            } else {
                stage_QPs[kk].xmin = &sx0;
                stage_QPs[kk].xmax = &sx0;
            }
            stage_QPs[kk].umin = &sumin_in;
            stage_QPs[kk].umax = &sumax_in;
        } else {  // leaf node
            stage_QPs[kk].R = NULL;
            stage_QPs[kk].Rinv = NULL;
            stage_QPs[kk].r = NULL;

            stage_QPs[kk].xmin = &sxmin_in;
            stage_QPs[kk].xmax = &sxmax_in;
            stage_QPs[kk].umin = NULL;
            stage_QPs[kk].umax = NULL;
        }
    }

    #if PROFILE > 0
    initialize_timers( );
    #endif

    for (ll = 0; ll < NRUNS; ll++) {

        #if PROFILE > 0
        treeqp_tic(&tot_tmr);
        #endif

        treeqp_tdunes_solve(&regMat, stage_QPs, dual, &qp_in, &qp_out, &opts, &work);

        #if PROFILE > 0
        total_time = treeqp_toc(&tot_tmr);
        update_min_timers(ll);
        #endif

        // ------ prepare data for next run
        if (ll < NRUNS-1) {
            indlam = nx;
            for (kk = 0; kk < Np; kk++) {
                d_cvt_vec2strvec(dual[kk].lambda->m, &lambda[indlam], dual[kk].lambda, 0);
                indlam += dual[kk].lambda->m;
            }
        }
    }

    write_solution_to_txt(Nn, Np, qp_out.info.iter, stage_QPs, dual, tree);

    #if PROFILE > 0 && PRINT_LEVEL > 0
    print_timers(qp_out.info.iter);
    #endif

    // Free memory
    free_tree(md, Nr, Nh, Nn, tree);
    free(tree);
    free(stage_QPs);
    free(dual);
    free(sqmod);
    free(srmod);
    free(sW);
    free(sUt);
    free(sCholW);
    free(sCholUt);
    free(sres);
    free(sresMod);
    free(slambda);
    free(sDeltalambda);
    free(sx);
    free(su);
    free(sxas);
    free(suas);
    free(sQinvCal);
    free(sRinvCal);
    #ifdef _CHECK_LAST_ACTIVE_SET_
    free(sxasPrev);
    free(suasPrev);
    free(sWdiag);
    #endif
    free(sM);

    // TODO(dimitris): why do I get an error when I use c_align/free instead?
    v_free(tmpBlasfeoPtr);

    // for (kk = 0; kk < Np; kk++) d_print_strvec(dual[kk].lambda->m, dual[kk].lambda,0);
    // printf("\n");
    // d_print_tran_strvec(dual[Np-1].lambda->m, dual[Np-1].lambda,0);
    return 0;
}
