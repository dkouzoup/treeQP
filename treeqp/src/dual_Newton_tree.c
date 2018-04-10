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

#include "treeqp/src/dual_Newton_common.h"
#include "treeqp/src/dual_Newton_tree.h"
#include "treeqp/src/dual_Newton_tree_clipping.h"
#include "treeqp/src/dual_Newton_tree_qpoases.h"
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
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_v_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blas.h"

#define _MERGE_FACTORIZATION_WITH_SUBSTITUTION_

treeqp_tdunes_options_t treeqp_tdunes_default_options(int Nn)
{
    treeqp_tdunes_options_t opts;
    termination_t cond = TREEQP_INFNORM;

    opts.maxIter = 100;
    opts.termCondition = cond;
    opts.stationarityTolerance = 1.0e-12;

    opts.checkLastActiveSet = 1;

    // TODO(dimitris): replace with calculate_size/create for args
    opts.qp_solver = malloc(Nn*sizeof(stage_qp_t));

    // for (int ii = 0; ii < Nn; ii++) opts.qp_solver[ii] = TREEQP_CLIPPING_SOLVER;
    for (int ii = 0; ii < Nn; ii++)
    {
        if (0) // (ii % 2 == 0)
            opts.qp_solver[ii] = TREEQP_QPOASES_SOLVER;
        else
            opts.qp_solver[ii] = TREEQP_CLIPPING_SOLVER;
    }

    opts.lineSearchMaxIter = 50;
    opts.lineSearchGamma = 0.1;
    opts.lineSearchBeta = 0.6;

    opts.regType  = TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT;
    opts.regTol   = 1.0e-12;
    opts.regValue = 1.0e-8;

    return opts;
}



void stage_qp_set_fcn_ptrs(stage_qp_fcn_ptrs *ptrs, stage_qp_t qp_solver)
{
    switch (qp_solver)
    {
        case TREEQP_CLIPPING_SOLVER:
            ptrs->is_applicable = stage_qp_clipping_is_applicable;
            ptrs->calculate_size = stage_qp_clipping_calculate_size;
            ptrs->assign_structs = stage_qp_clipping_assign_structs;
            ptrs->assign_blasfeo_data = stage_qp_clipping_assign_blasfeo_data;
            ptrs->assign_data = stage_qp_clipping_assign_data;
            ptrs->init = stage_qp_clipping_init;
            ptrs->solve_extended = stage_qp_clipping_solve_extended;
            ptrs->solve = stage_qp_clipping_solve;
            ptrs->set_CmPnCmT = stage_qp_clipping_set_CmPnCmT;
            ptrs->add_EPmE = stage_qp_clipping_add_EPmE;
            ptrs->add_CmPnCkT = stage_qp_clipping_add_CmPnCkT;
            ptrs->eval_dual_term = stage_qp_clipping_eval_dual_term;
            ptrs->export_mu = stage_qp_clipping_export_mu;
            break;
        case TREEQP_QPOASES_SOLVER:
            ptrs->is_applicable = stage_qp_qpoases_is_applicable;
            ptrs->calculate_size = stage_qp_qpoases_calculate_size;
            ptrs->assign_structs = stage_qp_qpoases_assign_structs;
            ptrs->assign_blasfeo_data = stage_qp_qpoases_assign_blasfeo_data;
            ptrs->assign_data = stage_qp_qpoases_assign_data;
            ptrs->init = stage_qp_qpoases_init;
            ptrs->solve_extended = stage_qp_qpoases_solve_extended;
            ptrs->solve = stage_qp_qpoases_solve;
            ptrs->set_CmPnCmT = stage_qp_qpoases_set_CmPnCmT;
            ptrs->add_EPmE = stage_qp_qpoases_add_EPmE;
            ptrs->add_CmPnCkT = stage_qp_qpoases_add_CmPnCkT;
            ptrs->eval_dual_term = stage_qp_qpoases_eval_dual_term;
            ptrs->export_mu = stage_qp_qpoases_export_mu;
            break;
        default:
            printf("[TREEQP] Error! Unknown stage QP solver specified.\n");
            exit(1);
    }
}



static void setup_npar(int Nh, int Nn, struct node *tree, int *npar)
{
    // initialize vector to zero
    for (int kk = 0; kk < Nh; kk++) npar[kk] = 0;

    // enumerate nodes per stage
    for (int kk = 0; kk < Nn; kk++) npar[tree[kk].stage]++;
}



static void setup_idxpos(tree_ocp_qp_in *qp_in, int *idxpos)
{
    int Nn = qp_in->N;
    int idxdad;

    struct node *tree = qp_in->tree;

    for (int kk = 0; kk < Nn; kk++)
    {
        idxdad = tree[kk].dad;
        idxpos[kk] = 0;
        for (int ii = 0; ii < tree[kk].idxkid; ii++)
        {
            idxpos[kk] += qp_in->nx[tree[idxdad].kids[ii]];
        }
    }
    // for (int kk = 0; kk < Nn; kk++) printf("kk = %d, idxpos = %d\n", kk, idxpos[kk]);
}



static int maximum_hessian_block_dimension(tree_ocp_qp_in *qp_in)
{
    int maxDim = 0;
    int currDim, idxkid;

    for (int ii = 0; ii < qp_in->N; ii++)
    {
        currDim = 0;
        for (int jj = 0; jj < qp_in->tree[ii].nkids; jj++)
        {
            idxkid = qp_in->tree[ii].kids[jj];
            currDim += qp_in->nx[idxkid];
        }
        maxDim = MAX(maxDim, currDim);
    }
    return maxDim;
}



static void solve_stage_problems(tree_ocp_qp_in *qp_in, treeqp_tdunes_workspace *work)
{
    int idxkid, idxdad, idxpos;
    int Nn = qp_in->N;
    int *nx = (int *)qp_in->nx;
    int *nu = (int *)qp_in->nu;

    struct node *tree = (struct node *)qp_in->tree;

    struct blasfeo_dvec *slambda = work->slambda;

    struct blasfeo_dmat *sA = (struct blasfeo_dmat *) qp_in->A;
    struct blasfeo_dmat *sB = (struct blasfeo_dmat *) qp_in->B;

    struct blasfeo_dvec *sq = (struct blasfeo_dvec *) qp_in->q;
    struct blasfeo_dvec *sr = (struct blasfeo_dvec *) qp_in->r;

    struct blasfeo_dvec *sqmod = work->sqmod;
    struct blasfeo_dvec *srmod = work->srmod;

    #ifdef SAVE_DATA
    int indh = 0;
    int indx = 0;
    int indu = 0;
    int dimh = number_of_primal_variables(qp_in);
    int dimx = number_of_states(qp_in);
    int dimu = number_of_controls(qp_in);
    double *hmod = malloc(dimh*sizeof(double));
    double *xit = malloc(dimx*sizeof(double));
    double *uit = malloc(dimu*sizeof(double));
    struct blasfeo_dvec *sx = (struct blasfeo_dvec *) work->sx;
    struct blasfeo_dvec *su = (struct blasfeo_dvec *) work->su;
    #endif

    #ifdef PARALLEL
    #pragma omp parallel for private(idxkid, idxdad, idxpos)
    #endif
    for (int kk = 0; kk < Nn; kk++)
    {
        idxdad = tree[kk].dad;
        idxpos = work->idxpos[kk];

        // --- update QP gradient

        // qmod[k] = - q[k] + lambda[k]
        if (kk == 0)
        {   // lambda[0] = 0
            for (int jj = 0; jj < nx[kk]; jj++) BLASFEO_DVECEL(&sqmod[kk], jj) = 0.0;
            blasfeo_daxpy(nx[kk], -1.0, &sq[kk], 0, &sqmod[kk], 0, &sqmod[kk], 0);
        }
        else
        {
            blasfeo_daxpy(nx[kk], -1.0, &sq[kk], 0, &slambda[idxdad], idxpos, &sqmod[kk], 0);
        }

        // rmod[k] = - r[k]
        blasfeo_dveccpsc(nu[kk], -1.0, &sr[kk], 0, &srmod[kk], 0);

        for (int ii = 0; ii < tree[kk].nkids; ii++)
        {
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
        //
        // (a) clipping:    - solve stage QP
        //                  - store unconstrained solution (to calculate multipliers later)
        //                  - calculate QinvCal, RinvCal vectors (to build dual blocks later)
        // (b) qpoases:     - solve stage QP
        //                  - TODO(dimitris): what else?

        work->stage_qp_ptrs[kk].solve_extended(qp_in, kk, work);
    }

    #ifdef SAVE_DATA
    for (int kk = 0; kk < Nn; kk++)
    {
        blasfeo_unpack_dvec(sqmod[kk].m, &sqmod[kk], 0, &hmod[indh]);
        blasfeo_unpack_dvec(sx[kk].m, &sx[kk], 0, &xit[indx]);
        indh += sqmod[kk].m;
        indx += sx[kk].m;
        blasfeo_unpack_dvec(srmod[kk].m, &srmod[kk], 0, &hmod[indh]);
        blasfeo_unpack_dvec(su[kk].m, &su[kk], 0, &uit[indu]);
        indh += srmod[kk].m;
        indu += su[kk].m;
    }
    // printf("dimh = %d, indh = %d\n", dimh, indh);
    write_double_vector_to_txt(hmod, dimh, "examples/spring_mass_utils/hmod.txt");
    write_double_vector_to_txt(xit, dimx, "examples/spring_mass_utils/xit.txt");
    write_double_vector_to_txt(uit, dimu, "examples/spring_mass_utils/uit.txt");
    free(hmod);
    free(xit);
    free(uit);
    #endif
}



static void compare_with_previous_active_set(int isLeaf, int indx, treeqp_tdunes_workspace *work) {

    int *xasChanged = work->xasChanged;
    int *uasChanged = work->uasChanged;

    struct blasfeo_dvec *sxas = &work->sxas[indx];
    struct blasfeo_dvec *suas = &work->suas[indx];
    struct blasfeo_dvec *sxasPrev = &work->sxasPrev[indx];
    struct blasfeo_dvec *suasPrev = &work->suasPrev[indx];

    xasChanged[indx] = 0;
    for (int ii = 0; ii < sxas->m; ii++) {
        if (BLASFEO_DVECEL(sxas, ii) != BLASFEO_DVECEL(sxasPrev, ii)) {
            xasChanged[indx] = 1;
            break;
        }
    }
    blasfeo_dveccp(sxas->m, sxas, 0, sxasPrev, 0);

    if (!isLeaf) {
        uasChanged[indx] = 0;
        for (int ii = 0; ii < suas->m; ii++) {
            if (BLASFEO_DVECEL(suas, ii) != BLASFEO_DVECEL(suasPrev, ii)) {
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
                error = MAX(error, ABS(BLASFEO_DVECEL(&sres[kk], ii)));
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
    treeqp_tdunes_options_t *opts, treeqp_tdunes_workspace *work)
{

    int idxdad, idxpos, idxsib, idxii, ns, isLeaf, asDadChanged;
    double error;

    int *nx = (int *)qp_in->nx;
    int *nu = (int *)qp_in->nu;

    int *xasChanged, *uasChanged;
    struct blasfeo_dmat *sWdiag;

    if (opts->checkLastActiveSet)
    {
        xasChanged = work->xasChanged;
        uasChanged = work->uasChanged;
        sWdiag = work->sWdiag;
    }

    int Nn = work->Nn;
    int Np = work->Np;

    struct blasfeo_dmat *sA = (struct blasfeo_dmat *) qp_in->A;
    struct blasfeo_dmat *sB = (struct blasfeo_dmat *) qp_in->B;
    struct blasfeo_dvec *sb = (struct blasfeo_dvec *) qp_in->b;
    struct node *tree = (struct node *)qp_in->tree;

    struct blasfeo_dvec *sx = work->sx;
    struct blasfeo_dvec *su = work->su;

    struct blasfeo_dmat *sM = work->sM;
    struct blasfeo_dmat *sW = work->sW;
    struct blasfeo_dmat *sUt = work->sUt;
    struct blasfeo_dvec *sres = work->sres;
    struct blasfeo_dvec *sresMod = work->sresMod;

    *idxFactorStart = -1;

    #ifdef SAVE_DATA
    int indres = 0;
    int dimres = number_of_states(qp_in) - qp_in->nx[0];
    double res[dimres];
    int dimW = 0;
    int dimUt = 0;
    for (int kk = 0; kk < Np; kk++)
    {
        dimW += sW[kk].n*sW[kk].n;  // NOTE(dimitris): not m, as it may be equal to n+1
        if (kk > 0) dimUt += sUt[kk-1].m*sUt[kk-1].n;
    }
    double W[dimW], Ut[dimUt];
    int indW = 0;
    int indUt = 0;
    #endif

    if (opts->checkLastActiveSet)
    {
        // TODO(dimitris): check if it's worth to parallelize
        for (int kk = Nn-1; kk >= 0; kk--)
        {
            isLeaf = (tree[kk].nkids > 0 ? 0:1);
            // NOTE(dimitris): updates both xasChanged/uasChanged and xasPrev/uasPrev
            compare_with_previous_active_set(isLeaf, kk, work);
        }
        // TODO(dimitris): double check that this indx is correct (not higher s.t. we loose efficiency)
        *idxFactorStart = find_starting_point_of_factorization(tree, work);
    }

    #ifdef PARALLEL
    #pragma omp parallel for private(idxdad, idxpos)
    #endif
    // Calculate dual gradient
    // TODO(dimitris): can we merge with solution of stage QPs without problems in parallelizing?
    for (int kk = Nn-1; kk > 0; kk--)
    {
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
    if (error < opts->stationarityTolerance)
    {
        return TREEQP_SUCC_OPTIMAL_SOLUTION_FOUND;
    }
    #ifdef PARALLEL
    #pragma omp parallel for private(idxdad, idxpos, idxsib, idxii, ns, asDadChanged)
    #endif
    // Calculate dual Hessian
    for (int kk = Nn-1; kk > 0; kk--)
    {
        idxdad = tree[kk].dad;
        idxpos = work->idxpos[kk];

        if (opts->checkLastActiveSet)
        {
            asDadChanged = xasChanged[idxdad] | uasChanged[idxdad];
        }
        // Filling W[idxdad] and Ut[idxdad-1]

        // TODO(dimitris): if only xasChanged, remove QinvCalPrev and add new
        if ((opts->checkLastActiveSet == 0) || (asDadChanged || xasChanged[kk]))
        {
            // --- hessian contribution of node (diagonal block of W)

            // W[idxdad] + offset = C[k] * P[idxdad] * C[k]', with C[k] = [A[k] B[k]]
            work->stage_qp_ptrs[idxdad].set_CmPnCmT(qp_in, kk, idxdad, idxpos, work);

            // W[idxdad] + offset += E * P[k] * E', with E = [I 0]
            work->stage_qp_ptrs[kk].add_EPmE(qp_in, kk, idxdad, idxpos, work);

            if (opts->checkLastActiveSet)
            {
                // save diagonal block that will be overwritten in factorization
                blasfeo_dgecp(nx[kk], nx[kk], &sW[idxdad], idxpos, idxpos, &sWdiag[kk], 0, 0);
            }

            // --- hessian contribution of parent (Ut)

            if (tree[idxdad].dad >= 0)
            {
                if ((opts->checkLastActiveSet == 0) || (asDadChanged && opts->checkLastActiveSet))
                {
                    // Ut[idxdad] + offset = M' = - Qinvcal[idxdad] * A[k]'
                    blasfeo_dgetr(nx[kk], nx[idxdad], &sM[kk], 0, 0, &sUt[idxdad-1], 0, idxpos);
                    blasfeo_dgesc(nx[idxdad], nx[kk], -1.0, &sUt[idxdad-1], 0, idxpos);
                }
            }

            // --- hessian contribution of preceding siblings (off-diagonal blocks of W)

            if ((opts->checkLastActiveSet == 0) || (asDadChanged))
            {
                ns = tree[idxdad].nkids - 1;  // number of siblings
                idxii = 0;
                for (int ii = 0; ii < ns; ii++)
                {
                    idxsib = tree[idxdad].kids[ii];
                    if (idxsib == kk) break;  // completed all preceding siblings

                    // W[idxdad] + offset += C[k] * P[idxdad] * C[idxsib]'
                    work->stage_qp_ptrs[idxdad].add_CmPnCkT(qp_in, kk, idxsib, idxdad, idxpos, idxii, work);

                    // idxiiOLD = ii*qp_in->nx[1];
                    idxii += nx[idxsib];
                }
            }

        }
        else
        {
            blasfeo_dgecp(nx[kk], nx[kk], &sWdiag[kk], 0, 0, &sW[idxdad], idxpos, idxpos);
        }
    }

    #ifdef SAVE_DATA
    for (int kk = 0; kk < Np; kk++)
    {
        blasfeo_unpack_dvec(sres[kk].m, &sres[kk], 0, &res[indres]);
        indres += sres[kk].m;
        blasfeo_unpack_dmat(sW[kk].n, sW[kk].n, &sW[kk], 0, 0, &W[indW], sW[kk].n);
        indW += sW[kk].n*sW[kk].n;
        if (kk > 0)
        {
            blasfeo_unpack_dmat( sUt[kk-1].m, sUt[kk-1].n, &sUt[kk-1], 0, 0,
                &Ut[indUt], sUt[kk-1].m);
            indUt += sUt[kk-1].m*sUt[kk-1].n;
        }
    }
    write_double_vector_to_txt(res, dimres, "examples/spring_mass_utils/res.txt");
    write_double_vector_to_txt(W, dimW, "examples/spring_mass_utils/W.txt");
    write_double_vector_to_txt(Ut, dimUt, "examples/spring_mass_utils/Ut.txt");
    #endif

    return TREEQP_OK;
}



static void calculate_delta_lambda(tree_ocp_qp_in *qp_in, int idxFactorStart,
    treeqp_tdunes_workspace *work, treeqp_tdunes_options_t *opts)
{
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

    #ifdef SAVE_DATA
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

            if ((opts->checkLastActiveSet == 0) || (ii < idxFactorStart))
            {
                // add resMod in last row of matrix W
                blasfeo_drowin(sresMod[ii].m, 1.0, &sresMod[ii], 0, &sW[ii], sW[ii].m-1, 0);
                // perform Cholesky factorization and backward substitution together
                // blasfeo_dpotrf_l_mn(sW[ii].m, sW[ii].n, &sW[ii], 0, 0, &sCholW[ii], 0, 0);
                treeqp_dpotrf_l_mn_with_reg_opts(&sW[ii], &sCholW[ii], opts->regType, opts->regTol, opts->regValue);

                // extract result of substitution
                blasfeo_drowex(sDeltalambda[ii].m, 1.0, &sCholW[ii], sCholW[ii].m-1, 0,
                    &sDeltalambda[ii], 0);

            }
            else if (opts->checkLastActiveSet)
            {
            // perform only vector substitution
            blasfeo_dtrsv_lnn(sresMod[ii].m, &sCholW[ii], 0, 0, &sresMod[ii], 0,
                &sDeltalambda[ii], 0);
            }

            #else  /* _MERGE_FACTORIZATION_WITH_SUBSTITUTION_ */

            if ((opts->checkLastActiveSet == 0) || (ii < idxFactorStart))
            {
                // Cholesky factorization to calculate factor of current diagonal block
                // blasfeo_dpotrf_l(sW[ii].n, &sW[ii], 0, 0, &sCholW[ii], 0, 0);
                treeqp_dpotrf_l_with_reg_opts(&sW[ii], &sCholW[ii], opts->regType, opts->regTol, opts->regValue);

            }  // TODO(dimitris): we can probably skip more calculations (see scenarios)

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
    // blasfeo_dpotrf_l_mn(sW[0].m, sW[0].n, &sW[0], 0, 0, &sCholW[0], 0, 0);
    treeqp_dpotrf_l_mn_with_reg_opts(&sW[0], &sCholW[0], opts->regType, opts->regTol, opts->regValue);

    // extract result of substitution
    blasfeo_drowex(sDeltalambda[0].m, 1.0, &sCholW[0], sCholW[0].m-1, 0, &sDeltalambda[0], 0);
    #else
    // calculate Cholesky factor of root block
    // blasfeo_dpotrf_l(sW[0].m, &sW[0], 0, 0, &sCholW[0], 0, 0);
    treeqp_dpotrf_l_with_reg_opts(&sW[0], &sCholW[0], opts->regType, opts->regTol, opts->regValue);

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

    #ifdef SAVE_DATA
    for (int kk = 0; kk < Np; kk++) {
        blasfeo_unpack_dvec(sDeltalambda[kk].m, &sDeltalambda[kk], 0, &deltalambda[indlam]);
        indlam += sDeltalambda[kk].m;
    }
    write_double_vector_to_txt(deltalambda, dimlam, "examples/spring_mass_utils/deltalambda.txt");
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


static double evaluate_dual_function(tree_ocp_qp_in *qp_in, treeqp_tdunes_workspace *work)
{
    int idxkid, idxpos, idxdad;
    double fval = 0;

    int Nn = work->Nn;
    int Np = work->Np;

    int *nx = (int *)qp_in->nx;
    int *nu = (int *)qp_in->nu;

    double *fvals = work->fval;
    double *cmod = work->cmod;

    struct blasfeo_dmat *sA = (struct blasfeo_dmat *) qp_in->A;
    struct blasfeo_dmat *sB = (struct blasfeo_dmat *) qp_in->B;
    struct blasfeo_dvec *sb = (struct blasfeo_dvec *) qp_in->b;

    struct blasfeo_dvec *sq = (struct blasfeo_dvec *) qp_in->q;
    struct blasfeo_dvec *sr = (struct blasfeo_dvec *) qp_in->r;

    struct blasfeo_dvec *sqmod = work->sqmod;
    struct blasfeo_dvec *srmod = work->srmod;
    struct blasfeo_dvec *slambda = work->slambda;

    struct node *tree = (struct node *)qp_in->tree;

    #ifdef PARALLEL
    #pragma omp parallel for private(idxkid, idxpos, idxdad)
    #endif
    // NOTE: same code as in solve_stage_problems but:
    // - without calculating as
    // - without calculating elimination matrix
    // - with calculating modified constant term
    for (int kk = 0; kk < Nn; kk++)
    {
        idxdad = tree[kk].dad;
        idxpos = work->idxpos[kk];

        // --- update QP gradient

        // qmod[k] = - q[k] + lambda[k]
        if (kk == 0) {
            // lambda[0] = 0
            for (int jj = 0; jj < nx[kk]; jj++) BLASFEO_DVECEL(&sqmod[kk], jj) = 0.0;
            blasfeo_daxpy(nx[kk], -1.0, &sq[kk], 0, &sqmod[kk], 0, &sqmod[kk], 0);
        } else {
            blasfeo_daxpy(nx[kk], -1.0, &sq[kk], 0, &slambda[idxdad], idxpos, &sqmod[kk], 0);
        }

        // rmod[k] = - r[k]
        blasfeo_dveccpsc(nu[kk], -1.0, &sr[kk], 0, &srmod[kk], 0);

        // cmod[k] = 0
        cmod[kk] = 0.;

        for (int ii = 0; ii < tree[kk].nkids; ii++)
        {
            idxkid = tree[kk].kids[ii];
            idxdad = tree[idxkid].dad;
            idxpos = work->idxpos[idxkid];

            // cmod[k] += b[jj]' * lambda[jj]
            cmod[kk] += blasfeo_ddot(nx[kk], &sb[idxkid-1], 0, &slambda[idxdad], idxpos);

            // return x^T * y

            // qmod[k] -= A[jj]' * lambda[jj]
            blasfeo_dgemv_t(nx[idxkid], nx[idxdad], -1.0, &sA[idxkid-1], 0, 0,
                &slambda[idxdad], idxpos, 1.0, &sqmod[kk], 0, &sqmod[kk], 0);

            // rmod[k] -= B[jj]' * lambda[jj]
            blasfeo_dgemv_t(nx[idxkid], nu[idxdad], -1.0, &sB[idxkid-1], 0, 0,
                &slambda[idxdad], idxpos, 1.0, &srmod[kk], 0, &srmod[kk], 0);
        }

        // --- solve QP
        work->stage_qp_ptrs[kk].solve(qp_in, kk, work);

        // --- calculate dual function term
        work->stage_qp_ptrs[kk].eval_dual_term(qp_in, kk, work);
    }

    for (int kk = 0; kk < Nn; kk++) fval += fvals[kk];

    return fval;
}



static int line_search(tree_ocp_qp_in *qp_in, treeqp_tdunes_options_t *opts,
    treeqp_tdunes_workspace *work) {

    int Nn = qp_in->N;
    int Np = work->Np;

    struct node *tree = (struct node *)qp_in->tree;

    #ifdef SAVE_DATA
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
    #ifdef SAVE_DATA
    for (int kk = 0; kk < Np; kk++) {
        blasfeo_unpack_dvec( slambda[kk].m, &slambda[kk], 0, &lambda[indlam]);
        indlam += slambda[kk].m;
    }
    write_double_vector_to_txt(lambda, dimlam, "examples/spring_mass_utils/lambda_opt.txt");
    write_double_vector_to_txt(&dotProduct, 1, "examples/spring_mass_utils/dotProduct.txt");
    write_double_vector_to_txt(&fval0, 1, "examples/spring_mass_utils/fval0.txt");
    write_int_vector_to_txt(&lsIter, 1, "examples/spring_mass_utils/lsiter.txt");
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
        blasfeo_unpack_dvec(su[kk].m, &su[kk], 0, &u[indu]);
        indu += su[kk].m;
    }

    ind = 0;
    for (kk = 0; kk < Np; kk++) {
        blasfeo_unpack_dvec(sDeltalambda[kk].m, &sDeltalambda[kk], 0, &deltalambda[ind]);
        blasfeo_unpack_dvec(slambda[kk].m, &slambda[kk], 0, &lambda[ind]);
        ind += slambda[kk].m;
    }

    write_double_vector_to_txt(x, dimx, "examples/spring_mass_utils/x_opt.txt");
    write_double_vector_to_txt(u, dimu, "examples/spring_mass_utils/u_opt.txt");
    write_double_vector_to_txt(lambda, dimlam, "examples/spring_mass_utils/deltalambda_opt.txt");
    write_double_vector_to_txt(lambda, dimlam, "examples/spring_mass_utils/lambda_opt.txt");
    write_int_vector_to_txt(&iter, 1, "examples/spring_mass_utils/iter.txt");

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

    // ------ initialization
    treeqp_tic(&interface_tmr);

    for (int kk = 0; kk < Nn; kk++)
    {
        // TODO(dimitris): Clean this up! At root, opts->qp_solver[-1] gives segfault
        if (kk > 0)
        {
            work->stage_qp_ptrs[kk].init(qp_in, kk, opts->qp_solver[tree[kk].dad], work);
        }
        else
        {
            work->stage_qp_ptrs[kk].init(qp_in, kk, TREEQP_CLIPPING_SOLVER, work);
        }

        if (opts->checkLastActiveSet)
        {
            blasfeo_dvecse(work->sxasPrev[kk].m, 0.0/0.0, &work->sxasPrev[kk], 0);
            blasfeo_dvecse(work->suasPrev[kk].m, 0.0/0.0, &work->suasPrev[kk], 0);
        }
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
        calculate_delta_lambda(qp_in, idxFactorStart, work, opts);
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

    for (int kk = 0; kk < Nn; kk++)
    {
        blasfeo_dveccp(nx[kk], &work->sx[kk], 0, &qp_out->x[kk], 0);
        blasfeo_dveccp(nu[kk], &work->su[kk], 0, &qp_out->u[kk], 0);

        if (kk > 0)
        {
            blasfeo_dveccp(nx[kk], &work->slambda[tree[kk].dad], work->idxpos[kk],
                &qp_out->lam[kk], 0);
        }

        work->stage_qp_ptrs[kk].export_mu(qp_out, kk, work);
    }
    qp_out->info.iter = NewtonIter;

    qp_out->info.interface_time += treeqp_toc(&interface_tmr);

    if (qp_out->info.iter == opts->maxIter)
        status = TREEQP_ERR_MAXIMUM_ITERATIONS_REACHED;

    return status;  // TODO(dimitris): return correct status
}



static void update_M_dimensions(int idx, tree_ocp_qp_in *qp_in, int *rowsM, int *colsM)
{
    int idxdad = qp_in->tree[idx].dad;
    int idxsib;

    if (idx == 0)
    {
        *rowsM = 0;
        *colsM = 0;
    } else
    {
        *colsM = qp_in->nx[idxdad] + qp_in->nu[idxdad];
        *rowsM = 0;

        for (int jj = 0; jj < qp_in->tree[idxdad].nkids; jj++)
        {
            idxsib = qp_in->tree[idxdad].kids[jj];
            *rowsM = MAX(*rowsM, qp_in->nx[idxsib]);
            // TODO(dimitris): test that old code below was indeed wrong (currently nx > nu always)
            // *rowsM = MAX(*rowsM, MAX(qp_in->nx[idxsib], qp_in->nu[idxsib]));
            if (qp_in->nx[idxsib] < qp_in->nu[idxsib])
            {
                assert(1 == 0 && "Case not tested yet! Comment out and check if code seg. faults.");
            }
        }
    }
}



int treeqp_tdunes_calculate_size(tree_ocp_qp_in *qp_in, treeqp_tdunes_options_t *opts)
{
    struct node *tree = (struct node *) qp_in->tree;
    int bytes = 0;
    int Nn = qp_in->N;
    int Nh = tree[Nn-1].stage;
    int Np = get_number_of_parent_nodes(Nn, tree);
    // int regDim = maximum_hessian_block_dimension(qp_in);
    int dim, idxkid, ncolAB;
    int rowsM, colsM;

    // int pointers
    bytes += Nh*sizeof(int);  // npar
    bytes += Nn*sizeof(int);  // idxpos

    if (opts->checkLastActiveSet)
    {
        bytes += 2*Nn*sizeof(int);  // xasChanged, uasChanged
        bytes += Np*sizeof(int);  // blockChanged
    }

    // double pointers
    bytes += 2*Nn*sizeof(double);  // fval, cmod

    // stage QP solvers
    bytes += Nn*sizeof(void *);  // stage_qp_data
    bytes += Nn*sizeof(stage_qp_fcn_ptrs);  // stage_qp_ptrs

    stage_qp_fcn_ptrs stage_qp_ptrs;
    for (int ii = 0; ii < Nn; ii++)
    {
        stage_qp_set_fcn_ptrs(&stage_qp_ptrs, opts->qp_solver[ii]);
        bytes += stage_qp_ptrs.calculate_size(qp_in->nx[ii], qp_in->nu[ii]);
    }

    // struct pointers
    bytes += 2*Nn*sizeof(struct blasfeo_dvec);  // qmod, rmod
    if (opts->checkLastActiveSet)
    {
        bytes += Nn*sizeof(struct blasfeo_dmat);  // Wdiag
    }
    bytes += (Nn-1)*sizeof(struct blasfeo_dmat);  // AB
    bytes += Nn*sizeof(struct blasfeo_dmat);  // M
    bytes += 2*Np*sizeof(struct blasfeo_dmat);  // W, CholW
    bytes += 2*(Np-1)*sizeof(struct blasfeo_dmat);  // Ut, CholUt
    bytes += 4*Np*sizeof(struct blasfeo_dvec);  // res, resMod, lambda, Deltalambda

    bytes += 3*Nn*sizeof(struct blasfeo_dvec);  // x, xUnc, xas
    bytes += 3*Nn*sizeof(struct blasfeo_dvec);  // u, uUnc, uas

    if (opts->checkLastActiveSet)
    {
        bytes += Nn*sizeof(struct blasfeo_dvec);  // xasPrev
        bytes += Nn*sizeof(struct blasfeo_dvec);  // uasPrev
    }

    // structs

    for (int ii = 0; ii < Nn; ii++)
    {
        bytes += blasfeo_memsize_dvec(qp_in->nx[ii]);  // qmod
        bytes += blasfeo_memsize_dvec(qp_in->nu[ii]);  // rmod

        bytes += 3*blasfeo_memsize_dvec(qp_in->nx[ii]);  // x, xUnc, xas
        if (opts->checkLastActiveSet)
        {
            bytes += blasfeo_memsize_dmat(qp_in->nx[ii], qp_in->nx[ii]);  // Wdiag
            bytes += blasfeo_memsize_dvec(qp_in->nx[ii]);  // xasPrev
        }

        bytes += 3*blasfeo_memsize_dvec(qp_in->nu[ii]);  // u, uUnc, uas

        if (opts->checkLastActiveSet)
        {
            bytes += blasfeo_memsize_dvec(qp_in->nu[ii]);  // uasPrev
        }

        if (ii > 0)
        {
            ncolAB = qp_in->nx[tree[ii].dad] + qp_in->nu[tree[ii].dad];
            bytes += blasfeo_memsize_dmat(qp_in->nx[ii], ncolAB);  // AB
        }

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
    treeqp_tdunes_workspace *work, void *ptr)
{
    struct node *tree = (struct node *) qp_in->tree;
    int Nn = qp_in->N;
    int Nh = tree[Nn-1].stage;
    int Np = get_number_of_parent_nodes(Nn, tree);
    int regDim = maximum_hessian_block_dimension(qp_in);
    int dim, idxkid, ncolAB;
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

    work->stage_qp_ptrs = (stage_qp_fcn_ptrs *) c_ptr;
    c_ptr += Nn*sizeof(stage_qp_fcn_ptrs);

    for (int ii = 0; ii < Nn; ii++)
    {
        stage_qp_set_fcn_ptrs(&work->stage_qp_ptrs[ii], opts->qp_solver[ii]);
        work->stage_qp_ptrs[ii].is_applicable(qp_in, ii);
        work->stage_qp_ptrs[ii].assign_structs(&work->stage_qp_data[ii], &c_ptr);
    }

    if (opts->checkLastActiveSet)
    {
        work->xasChanged = (int *) c_ptr;
        c_ptr += Nn*sizeof(int);

        work->uasChanged = (int *) c_ptr;
        c_ptr += Nn*sizeof(int);

        work->blockChanged = (int *) c_ptr;
        c_ptr += Np*sizeof(int);
    }

    work->sqmod = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->srmod = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->sAB = (struct blasfeo_dmat *) c_ptr;
    c_ptr += (Nn-1)*sizeof(struct blasfeo_dmat);

    work->sM = (struct blasfeo_dmat *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dmat);

    if (opts->checkLastActiveSet)
    {
        work->sWdiag = (struct blasfeo_dmat *) c_ptr;
        c_ptr += Nn*sizeof(struct blasfeo_dmat);
    }

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

    if (opts->checkLastActiveSet)
    {
        work->sxasPrev = (struct blasfeo_dvec *) c_ptr;
        c_ptr += Nn*sizeof(struct blasfeo_dvec);

        work->suasPrev = (struct blasfeo_dvec *) c_ptr;
        c_ptr += Nn*sizeof(struct blasfeo_dvec);
    }

    // move pointer for proper alignment of doubles and blasfeo matrices/vectors
    align_char_to(64, &c_ptr);

    // first assign blasfeo-based solvers, then the rest, and then align again
    // TODO(dimitris): the distinction should actually be blasfeo_dmats and the rest (currently works because there are either only strvecs or only strmats in modules)
    for (int ii = 0; ii < Nn; ii++)
    {
        work->stage_qp_ptrs[ii].assign_blasfeo_data(qp_in->nx[ii], qp_in->nu[ii],
            work->stage_qp_data[ii], &c_ptr);
    }
    for (int ii = 0; ii < Nn; ii++)
    {
        work->stage_qp_ptrs[ii].assign_data(qp_in->nx[ii], qp_in->nu[ii],
            work->stage_qp_data[ii], &c_ptr);
    }

    align_char_to(64, &c_ptr);

    // strmats
    for (int ii = 0; ii < Nn; ii++)
    {
        if (opts->checkLastActiveSet)
        {
            init_strmat(qp_in->nx[ii], qp_in->nx[ii], &work->sWdiag[ii], &c_ptr);
        }

        if (ii > 0)
        {
            ncolAB = qp_in->nx[tree[ii].dad] + qp_in->nu[tree[ii].dad];
            init_strmat(qp_in->nx[ii], ncolAB, &work->sAB[ii-1], &c_ptr);
        }

        update_M_dimensions(ii, qp_in, &rowsM, &colsM);
        init_strmat(rowsM, colsM, &work->sM[ii], &c_ptr);

        if (ii < Np)
        {
            dim = 0;
            for (int jj = 0; jj < tree[ii].nkids; jj++)
            {
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
            if (ii > 0)
            {
                init_strmat(qp_in->nx[ii], dim, &work->sUt[ii-1], &c_ptr);
                init_strmat(qp_in->nx[ii], dim, &work->sCholUt[ii-1], &c_ptr);
            }
        }
    }

    // strvecs

    for (int ii = 0; ii < Nn; ii++)
    {
        init_strvec(qp_in->nx[ii], &work->sqmod[ii], &c_ptr);
        init_strvec(qp_in->nu[ii], &work->srmod[ii], &c_ptr);

        init_strvec(qp_in->nx[ii], &work->sx[ii], &c_ptr);
        init_strvec(qp_in->nx[ii], &work->sxUnc[ii], &c_ptr);
        init_strvec(qp_in->nx[ii], &work->sxas[ii], &c_ptr);
        if (opts->checkLastActiveSet)
        {
            init_strvec(qp_in->nx[ii], &work->sxasPrev[ii], &c_ptr);
        }

        init_strvec(qp_in->nu[ii], &work->su[ii], &c_ptr);
        init_strvec(qp_in->nu[ii], &work->suUnc[ii], &c_ptr);
        init_strvec(qp_in->nu[ii], &work->suas[ii], &c_ptr);
        if (opts->checkLastActiveSet)
        {
            init_strvec(qp_in->nu[ii], &work->suasPrev[ii], &c_ptr);
        }

        if (ii < Np)
        {
            dim = 0;
            for (int jj = 0; jj < tree[ii].nkids; jj++)
            {
                idxkid = tree[ii].kids[jj];
                dim += qp_in->nx[idxkid];
            }
            init_strvec(dim, &work->sres[ii], &c_ptr);
            init_strvec(dim, &work->sresMod[ii], &c_ptr);
            init_strvec(dim, &work->slambda[ii], &c_ptr);
            init_strvec(dim, &work->sDeltalambda[ii], &c_ptr);
        }
    }

    // doubles
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
