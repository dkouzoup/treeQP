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

// NOTE(dimitris): Current limitations
// - simple bounds, diagonal weights
// - x0 eliminated (no MHE)
// - not varying nx, nu
// - no arbitrary trees

// TODO(dimitris): valgrind code

#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/src/dual_Newton_scenarios.h"
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

#include "examples/data_spring_mass/data.c"

scen_options_t set_default_options(void) {
    scen_options_t opts;
    termination_t cond = TREEQP_INFNORM;

    opts.maxIter = 100;
    opts.termCondition = cond;
    opts.stationarityTolerance = 1.0e-12;

    opts.lineSearchMaxIter = 50;
    opts.lineSearchGamma = 0.1;
    opts.lineSearchBeta = 0.6;

    opts.regType  = TREEQP_ALWAYS_LEVENBERG_MARQUARDT;
    opts.regTol   = 1.0e-12;
    opts.regValue = 1.0e-8;

    return opts;
}


void write_dual_initial_point_to_workspace(int_t Ns, int_t Nh, real_t *lambda, real_t *mu,
    treeqp_workspace *work) {

    int_t ii, kk, indx;
    int_t nu = work->su[0][0].m;
    int_t nx = work->sx[0][1].m;

    indx = 0;
    for (ii = 0; ii < Ns-1; ii++) {
        d_cvt_vec2strvec(nu*work->commonNodes[ii], &lambda[indx], &work->slambda[ii], 0);
        indx += nu*work->commonNodes[ii];
    }

    indx = 0;
    for (ii = 0; ii < Ns; ii++) {
        for (kk = 0; kk < Nh; kk++) {
            d_cvt_vec2strvec(nx, &mu[indx], &work->smu[ii][kk], 0);
            indx += nx;
        }
    }
}


int main() {
    int_t ii, jj, kk;
    int_t real;
    return_t status;

    int_t nl = get_dimension_of_lambda(Nr, md, NU);
    int_t Nn = get_number_of_nodes(md, Nr, Nh);
    int_t Ns = ipow(md, Nr);

    scen_options_t opts = set_default_options();
    treeqp_info_t info;

    check_compiler_flags();

    // read initial point from txt file
    real_t *mu = malloc(Ns*Nh*NX*sizeof(real_t));
    real_t *lambda = malloc(nl*sizeof(real_t));
    status = read_double_vector_from_txt(mu, Ns*Nh*NX, "examples/data_spring_mass/mu0.txt");
    if (status != 0) return -1;
    status = read_double_vector_from_txt(lambda, nl, "examples/data_spring_mass/lambda0.txt");
    if (status != 0) return -1;

    // read constraint on x0 from txt file
    real_t x0[NX];
    status = read_double_vector_from_txt(x0, NX, "examples/data_spring_mass/x0.txt");
    if (status != 0) return -1;

    // calculate inverse weights
    // TODO(dimitris): hide this inside workspace
    real_t dQinv[NX], dRinv[NU], dPinv[NX];

    for (ii = 0; ii < NX; ii++) dQinv[ii] = 1./dQ[ii];
    for (ii = 0; ii < NX; ii++) dPinv[ii] = 1./dP[ii];
    for (ii = 0; ii < NU; ii++) dRinv[ii] = 1./dR[ii];

    struct node *tree = malloc(Nn*sizeof(struct node));

    struct d_strvec sQ, sP, sR, sq, sp, sr;  // NOTE(dimitris): time invariant for this example
    struct d_strvec sQinv, sPinv, sRinv;

    struct d_strmat sA[md], sB[md];
    struct d_strvec sb[md];

    struct d_strvec sx0, sxmin, sxmax, sumin, sumax;
    struct d_strvec sb0[md];

    setup_tree(md, Nr, Nh, Nn, tree);

    tree_ocp_qp_in qp_in;

    d_allocate_strvec(NX, &sQ);
    d_cvt_vec2strvec(NX, dQ, &sQ, 0);
    d_allocate_strvec(NX, &sP);
    d_cvt_vec2strvec(NX, dP, &sP, 0);
    d_allocate_strvec(NU, &sR);
    d_cvt_vec2strvec(NU, dR, &sR, 0);
    d_allocate_strvec(NX, &sq);
    d_cvt_vec2strvec(NX, q, &sq, 0);
    d_allocate_strvec(NX, &sp);
    d_cvt_vec2strvec(NX, p, &sp, 0);
    d_allocate_strvec(NU, &sr);
    d_cvt_vec2strvec(NU, r, &sr, 0);
    d_allocate_strvec(NX, &sQinv);
    d_cvt_vec2strvec(NX, dQinv, &sQinv, 0);
    d_allocate_strvec(NX, &sPinv);
    d_cvt_vec2strvec(NX, dPinv, &sPinv, 0);
    d_allocate_strvec(NU, &sRinv);
    d_cvt_vec2strvec(NU, dRinv, &sRinv, 0);
    d_allocate_strvec(NX, &sx0);
    d_cvt_vec2strvec(NX, x0, &sx0, 0);

    for (ii = 0; ii < md; ii++) {
        // NOTE(dimitris): first dynamics in data file are the nominal ones, skipped here
        d_allocate_strmat(NX, NX, &sA[ii]);
        d_cvt_mat2strmat(NX, NX, &A[(ii+1)*NX*NX], NX, &sA[ii], 0, 0);
        d_allocate_strmat(NX, NU, &sB[ii]);
        d_cvt_mat2strmat(NX, NU, &B[(ii+1)*NX*NU], NX, &sB[ii], 0, 0);
        d_allocate_strvec(NX, &sb[ii]);
        d_cvt_vec2strvec(NX, &b[(ii+1)*NX], &sb[ii], 0);

        // set up constraint on x0
        d_allocate_strvec(NX, &sb0[ii]);
        // b0[0] = b[0] + A[0]*x[0]
        dgemv_n_libstr(NX, NX, 1.0, &sA[ii], 0, 0, &sx0, 0, 1.0, &sb[ii], 0, &sb0[ii], 0);
    }
    d_allocate_strvec(NX, &sxmin);
    d_cvt_vec2strvec(NX, xmin, &sxmin, 0);
    d_allocate_strvec(NX, &sxmax);
    d_cvt_vec2strvec(NX, xmax, &sxmax, 0);
    d_allocate_strvec(NU, &sumin);
    d_cvt_vec2strvec(NU, umin, &sumin, 0);
    d_allocate_strvec(NU, &sumax);
    d_cvt_vec2strvec(NU, umax, &sumax, 0);

    int_t *nx = malloc(Nn*sizeof(int_t));
    int_t *nu = malloc(Nn*sizeof(int_t));

    struct d_strmat *hA = malloc((Nn-1)*sizeof(struct d_strmat));
    struct d_strmat *hB = malloc((Nn-1)*sizeof(struct d_strmat));
    struct d_strvec *hb = malloc((Nn-1)*sizeof(struct d_strvec));

    struct d_strvec *hQ = malloc(Nn*sizeof(struct d_strvec));
    struct d_strvec *hR = malloc(Nn*sizeof(struct d_strvec));
    struct d_strvec *hq = malloc(Nn*sizeof(struct d_strvec));
    struct d_strvec *hr = malloc(Nn*sizeof(struct d_strvec));
    struct d_strvec *hQinv = malloc(Nn*sizeof(struct d_strvec));
    struct d_strvec *hRinv = malloc(Nn*sizeof(struct d_strvec));

    struct d_strvec *hxmin = malloc(Nn*sizeof(struct d_strvec));
    struct d_strvec *hxmax = malloc(Nn*sizeof(struct d_strvec));
    struct d_strvec *humin = malloc(Nn*sizeof(struct d_strvec));
    struct d_strvec *humax = malloc(Nn*sizeof(struct d_strvec));

    // set up vectors of QP dimensions
    for (ii = 0; ii < Nn; ii++) {
        // dimensions
        if (ii > 0) {
            nx[ii] = NX;
        } else {
            nx[ii] = 0;  // NOTE(dimitris): x0 variable is eliminated
        }

        if (tree[ii].nkids > 0) {  // not a leaf
            nu[ii] = NU;
        } else {
            nu[ii] = 0;
        }

        // objective
        if (tree[ii].nkids > 0) {
            hQ[ii] = sQ;
            hR[ii] = sR;
            hq[ii] = sq;
            hr[ii] = sr;
            hQinv[ii] = sQinv;
            hRinv[ii] = sRinv;
        } else {
            hQ[ii] = sP;
            hq[ii] = sp;
            hQinv[ii] = sPinv;
        }

        // dynamics
        real = tree[ii].real;
        if (ii > 0) {
            hA[ii-1] = sA[real];
            hB[ii-1] = sB[real];
            if (ii <= md) {
                hb[ii-1] = sb0[real];
            } else {
                hb[ii-1] = sb[real];
            }
        }

        // bounds
        if (ii > 0) {
            hxmin[ii] = sxmin;
            hxmax[ii] = sxmax;
        }
        if (tree[ii].nkids > 0) {
            humin[ii] = sumin;
            humax[ii] = sumax;
        }
    }

    qp_in.N = Nn;

    qp_in.nx = (const int *) nx;
    qp_in.nu = (const int *) nu;

    // TODO(dimitris): scaling factor missing
    qp_in.Q = (const struct d_strvec *) hQ;
    qp_in.R = (const struct d_strvec *) hR;
    qp_in.q = (const struct d_strvec *) hq;
    qp_in.r = (const struct d_strvec *) hr;
    qp_in.Qinv = (const struct d_strvec *) hQinv;
    qp_in.Rinv = (const struct d_strvec *) hRinv;

    qp_in.A = (const struct d_strmat *) hA;
    qp_in.B = (const struct d_strmat *) hB;
    qp_in.b = (const struct d_strvec *) hb;

    qp_in.xmin = (const struct d_strvec *) hxmin;
    qp_in.xmax = (const struct d_strvec *) hxmax;
    qp_in.umin = (const struct d_strvec *) humin;
    qp_in.umax = (const struct d_strvec *) humax;

    qp_in.tree = tree;

    // print_tree_ocp_qp_in(&qp_in);
    // exit(1);

    // create workspace of QP solver

    treeqp_workspace work;

    int_t treeqp_work_size = treeqp_calculate_workspace_size(Nn, Ns, Nh, Nr, NX, NU, tree);
    void *allocated_memory = malloc(treeqp_work_size);

    treeqp_create_workspace(Nn, Ns, Nr, &qp_in, &opts, &work, allocated_memory);

    #if PRINT_LEVEL > 0
    printf("\n-------- treeQP workspace size: %d bytes \n", treeqp_work_size);
    #endif

    #if PROFILE > 0
    initialize_timers();
    #endif

    for (jj = 0; jj < NRUNS; jj++) {
        write_dual_initial_point_to_workspace(Ns, Nh, lambda, mu, &work);

        #if PROFILE > 0
        treeqp_tic(&tot_tmr);
        #endif

        status = treeqp_dual_newton_scenarios(Ns, Nh, Nr, md, &qp_in, &opts, &info, &work);

        // printf("QP solver status at run %d: %d\n", jj, status);

        #if PROFILE > 0
        total_time = treeqp_toc(&tot_tmr);
        update_min_timers(jj);
        #endif
    }  // end NRUNS

    // d_print_strvec(NU, &work.su[0][3], 0);

    write_solution_to_txt(Ns, Nh, Nr, md, NX, NU, info.NewtonIter, &work);

    #if PROFILE > 0 && PRINT_LEVEL > 0
    print_timers(info.NewtonIter);
    #endif

    // Free allocated memory

    d_free_strvec(&sQ);
    d_free_strvec(&sP);
    d_free_strvec(&sR);
    d_free_strvec(&sq);
    d_free_strvec(&sp);
    d_free_strvec(&sr);
    d_free_strvec(&sQinv);
    d_free_strvec(&sPinv);
    d_free_strvec(&sRinv);
    d_free_strvec(&sx0);

    for (ii = 0; ii < md; ii++) {
        d_free_strmat(&sA[ii]);
        d_free_strmat(&sB[ii]);
        d_free_strvec(&sb[ii]);
        d_free_strvec(&sb0[ii]);
    }

    d_free_strvec(&sxmin);
    d_free_strvec(&sxmax);
    d_free_strvec(&sumin);
    d_free_strvec(&sumax);

    free(nx);
    free(nu);
    free(hQ);
    free(hR);
    free(hq);
    free(hr);
    free(hQinv);
    free(hRinv);
    free(hA);
    free(hB);
    free(hb);
    free(hxmin);
    free(hxmax);
    free(humin);
    free(humax);

    free(allocated_memory);

    free_tree(md, Nr, Nh, Nn, tree);
    free(tree);


    free(mu);
    free(lambda);

    return 0;
}
