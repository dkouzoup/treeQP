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
#include <assert.h>

#include "treeqp/src/hpmpc_tree.h"
#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/utils/blasfeo_utils.h"
#include "treeqp/utils/timing.h"
#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"

#include "hpmpc/include/target.h"
#include "hpmpc/include/tree.h"
#include "hpmpc/include/mpc_solvers.h"

#define INF 1e10 // TODO(dimitris): option instead of hardcoded here?


treeqp_hpmpc_options_t treeqp_hpmpc_default_options() {
    treeqp_hpmpc_options_t opts;

    opts.maxIter = 20;
	opts.mu0 = 2.0;
	opts.mu_tol = 1e-12;
	opts.alpha_min = 1e-8;
	opts.warm_start = 0;
    opts.compute_mult = 1;

    return opts;
}


int_t number_of_bounds(const struct blasfeo_dvec *vmin, const struct blasfeo_dvec *vmax) {
    int_t nb = 0;
    int_t n = vmin->m;
    assert(vmin->m == vmax->m);

    for (int_t ii = 0; ii < n; ii++) {
        if (DVECEL_LIBSTR(vmin, ii) > -INF ||
            DVECEL_LIBSTR(vmax, ii) < INF) {
            nb += 1;
        }
    }
    return nb;
}


int_t get_size_idxb(tree_ocp_qp_in *qp_in) {
    int_t size = 0;
    int_t Nn = qp_in->N;

    for (int_t ii = 0; ii < Nn; ii++) {
        size += number_of_bounds(&qp_in->umin[ii], &qp_in->umax[ii]);
        size += number_of_bounds(&qp_in->xmin[ii], &qp_in->xmax[ii]);
    }

    return size;
}


void setup_nb(tree_ocp_qp_in *qp_in, int *nb) {
    int_t Nn = qp_in->N;

    for (int_t ii = 0; ii < Nn; ii++) {
        nb[ii] = 0;
        nb[ii] += number_of_bounds(&qp_in->umin[ii], &qp_in->umax[ii]);
        nb[ii] += number_of_bounds(&qp_in->xmin[ii], &qp_in->xmax[ii]);
    }
}


void setup_nb_idxb(tree_ocp_qp_in *qp_in, int *nb, int **idxb) {
    int_t Nn = qp_in->N;
    int_t kk;

    for (int_t ii = 0; ii < Nn; ii++) {
        nb[ii] = 0;
        kk = 0;
        for (int_t jj = 0; jj < qp_in->nu[ii]; jj++) {
            if (DVECEL_LIBSTR(&qp_in->umin[ii], jj) > -INF ||
                DVECEL_LIBSTR(&qp_in->umax[ii], jj) < INF) {
                nb[ii] += 1;
                idxb[ii][kk++] = jj;
            }
        }
        for (int_t jj = 0; jj < qp_in->nx[ii]; jj++) {
            if (DVECEL_LIBSTR(&qp_in->xmin[ii], jj) > -INF ||
                DVECEL_LIBSTR(&qp_in->xmax[ii], jj) < INF) {
                nb[ii] += 1;
                idxb[ii][kk++] = jj + qp_in->nu[ii];
            }

        }
    }
}


void setup_ng(tree_ocp_qp_in *qp_in, int *ng) {
    int_t Nn = qp_in->N;

    for (int_t ii = 0; ii < Nn; ii++) {
        ng[ii] = 0;  // TODO(dimitris): update once polyhedral constraints are supported
    }
}


int_t treeqp_hpmpc_calculate_size(tree_ocp_qp_in *qp_in, treeqp_hpmpc_options_t *opts) {
    int_t bytes = 0;
    int_t Nn = qp_in->N;
    int_t idxp;

    // TODO(dimitris): can we avoid memory allocation in here?
    int_t *nb = (int_t *)malloc(Nn*sizeof(int_t));
    int_t *ng = (int_t *)malloc(Nn*sizeof(int_t));
    setup_nb(qp_in, nb);
    setup_ng(qp_in, ng);

    bytes += 2*Nn*sizeof(int);  // nb, ng

    bytes += Nn*sizeof(int_t*);  // idxb
    bytes += get_size_idxb(qp_in)*sizeof(int_t);

    bytes += 3*Nn*sizeof(struct blasfeo_dvec);  // sux, slam, sst

    bytes += Nn*sizeof(struct blasfeo_dmat);  // sRSQrq
    bytes += (Nn-1)*sizeof(struct blasfeo_dmat);  // sBAbt

    bytes += Nn*sizeof(struct blasfeo_dmat);  // sDCt
    bytes += Nn*sizeof(struct blasfeo_dvec);  // sd

    for (int_t ii = 0; ii < Nn; ii++) {
        bytes += blasfeo_memsize_dvec(qp_in->nx[ii] + qp_in->nu[ii]);  // sux
        bytes += 2*blasfeo_memsize_dvec(2*nb[ii] + 2*ng[ii]);  // slam, sst

        bytes += blasfeo_memsize_dmat(qp_in->nx[ii] + qp_in->nu[ii] + 1, qp_in->nx[ii] + qp_in->nu[ii]);  // sRSQrq

        if (ii > 0) {
            idxp = qp_in->tree[ii].dad;
            bytes += blasfeo_memsize_dmat(qp_in->nx[idxp] + qp_in->nu[idxp] + 1, qp_in->nx[ii]);  // sABbt
        }

        // TODO(dimitris): this has not been tested
        bytes += blasfeo_memsize_dmat(qp_in->nx[ii] + qp_in->nu[ii], ng[ii]);  // sDCt
        bytes += blasfeo_memsize_dvec(2*nb[ii] + 2*ng[ii]);  // sd
    }

    bytes += 5*opts->maxIter*sizeof(double);  // status

    bytes += d_tree_ip2_res_mpc_hard_work_space_size_bytes_libstr(Nn,
        (struct node *) qp_in->tree, (int *) qp_in->nx, (int *) qp_in->nu, nb, ng);

    make_int_multiple_of(64, &bytes);
    bytes += 2*64;

    free(nb);
    free(ng);

    return bytes;
}


void create_treeqp_hpmpc(tree_ocp_qp_in *qp_in, treeqp_hpmpc_options_t *opts,
    treeqp_hpmpc_workspace *work, void *ptr) {

    struct node *tree = (struct node *) qp_in->tree;
    int_t Nn = qp_in->N;
    int_t idxp;

    // char pointer
    char *c_ptr = (char *) ptr;

    // double pointers
    work->idxb = (int **) c_ptr;
    c_ptr += Nn*sizeof(int *);
    for (int_t ii = 0; ii < Nn; ii++) {
        work->idxb[ii] = (int *) c_ptr;
        c_ptr += number_of_bounds(&qp_in->umin[ii], &qp_in->umax[ii])*sizeof(int);
        c_ptr += number_of_bounds(&qp_in->xmin[ii], &qp_in->xmax[ii])*sizeof(int);
    }

    // pointers
    work->nb = (int *) c_ptr;
    c_ptr += Nn*sizeof(int);

    setup_nb_idxb(qp_in, work->nb, work->idxb);

    work->ng = (int *) c_ptr;
    c_ptr += Nn*sizeof(int);
    setup_ng(qp_in, work->ng);

    work->sux = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->slam = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->sst = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    work->sRSQrq = (struct blasfeo_dmat *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dmat);

    work->sBAbt = (struct blasfeo_dmat *) c_ptr;
    c_ptr += (Nn-1)*sizeof(struct blasfeo_dmat);

    work->sDCt = (struct blasfeo_dmat *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dmat);

    work->sd = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    // move pointer for proper alignment of doubles and blasfeo matrices/vectors
    align_char_to(64, &c_ptr);

    for (int_t ii = 0; ii < Nn; ii++) {
        init_strvec(qp_in->nx[ii] + qp_in->nu[ii], &work->sux[ii], &c_ptr);
        init_strvec(2*work->nb[ii] + 2*work->ng[ii], &work->slam[ii], &c_ptr);
        init_strvec(2*work->nb[ii] + 2*work->ng[ii], &work->sst[ii], &c_ptr);

        init_strmat(qp_in->nx[ii]+qp_in->nu[ii]+1, qp_in->nx[ii]+qp_in->nu[ii], &work->sRSQrq[ii], &c_ptr);

        if (ii > 0) {
            idxp = tree[ii].dad;
            init_strmat(qp_in->nx[idxp]+qp_in->nu[idxp]+1, qp_in->nx[ii], &work->sBAbt[ii-1], &c_ptr);
        }
        init_strmat(qp_in->nx[ii]+qp_in->nu[ii], work->ng[ii], &work->sDCt[ii], &c_ptr);
        init_strvec(2*work->nb[ii] + 2*work->ng[ii], &work->sd[ii], &c_ptr);
    }

    work->status = (double *) c_ptr;
    c_ptr += 5*opts->maxIter*sizeof(double);

    // TODO(dimitris): Maybe realign not needed
    align_char_to(64, &c_ptr);

    work->internal = (void *) c_ptr;
    c_ptr += d_tree_ip2_res_mpc_hard_work_space_size_bytes_libstr(Nn, tree,
        (int *) qp_in->nx, (int *) qp_in->nu, work->nb, work->ng);


    assert((char *)ptr + treeqp_hpmpc_calculate_size(qp_in, opts) >= c_ptr);
    // printf("memory starts at\t%p\nmemory ends at  \t%p\ndistance from the end\t%lu bytes\n",
    //     ptr, c_ptr, (char *)ptr + treeqp_hpmpc_calculate_size(qp_in, opts) - c_ptr);
    // exit(1);
}

int_t treeqp_hpmpc_solve(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out,
    treeqp_hpmpc_options_t *opts, treeqp_hpmpc_workspace *work) {

    int_t Nn = qp_in->N;
    int_t *nx = (int_t *)qp_in->nx;
    int_t *nu = (int_t *)qp_in->nu;

    treeqp_timer solver_tmr, interface_tmr;

    struct node *tree = (struct node *)qp_in->tree;

    struct blasfeo_dmat *sA = (struct blasfeo_dmat *) qp_in->A;
    struct blasfeo_dmat *sB = (struct blasfeo_dmat *) qp_in->B;
    struct blasfeo_dvec *sb = (struct blasfeo_dvec *) qp_in->b;

    struct blasfeo_dmat *sQ = (struct blasfeo_dmat *) qp_in->Q;
    struct blasfeo_dmat *sR = (struct blasfeo_dmat *) qp_in->R;
    struct blasfeo_dvec *sq = (struct blasfeo_dvec *) qp_in->q;
    struct blasfeo_dvec *sr = (struct blasfeo_dvec *) qp_in->r;

    // convert input to HPMPC format
    int_t idxp, idxb;

    treeqp_tic(&interface_tmr);

    for (int_t ii = 0; ii < Nn; ii++) {

        // TODO(dimitris): Add S' (nx x nu) term to lower diagonal part
        blasfeo_dgecp(nu[ii], nu[ii], &sR[ii], 0, 0, &work->sRSQrq[ii], 0, 0);
        blasfeo_dgecp(nx[ii], nx[ii], &sQ[ii], 0, 0, &work->sRSQrq[ii], nu[ii], nu[ii]);

        blasfeo_drowin(nu[ii], 1.0, &sr[ii], 0, &work->sRSQrq[ii], nu[ii] + nx[ii], 0);
        blasfeo_drowin(nx[ii], 1.0, &sq[ii], 0, &work->sRSQrq[ii], nu[ii] + nx[ii], nu[ii]);

        if (ii > 0) {
            idxp = tree[ii].dad;
            blasfeo_dgetr(nx[ii], nu[idxp], &sB[ii-1], 0, 0, &work->sBAbt[ii-1], 0, 0);
            blasfeo_dgetr(nx[ii], nx[idxp], &sA[ii-1], 0, 0, &work->sBAbt[ii-1], nu[idxp], 0);
            blasfeo_drowin(nx[ii], 1.0, &sb[ii-1], 0, &work->sBAbt[ii-1], nx[idxp] + nu[idxp], 0);
        }

        for (int_t jj = 0; jj < work->nb[ii]; jj++) {
            idxb = work->idxb[ii][jj];
            if (idxb < nu[ii]) {
                DVECEL_LIBSTR(&work->sd[ii], jj) = DVECEL_LIBSTR(&qp_in->umin[ii], idxb);
                DVECEL_LIBSTR(&work->sd[ii], jj + work->nb[ii]) = DVECEL_LIBSTR(&qp_in->umax[ii], idxb);
            } else {
                DVECEL_LIBSTR(&work->sd[ii], jj) = DVECEL_LIBSTR(&qp_in->xmin[ii], idxb - nu[ii]);
                DVECEL_LIBSTR(&work->sd[ii], jj + work->nb[ii]) = DVECEL_LIBSTR(&qp_in->xmax[ii], idxb - nu[ii]);
            }
        }
    }

    qp_out->info.interface_time = treeqp_toc(&interface_tmr);
    treeqp_tic(&solver_tmr);

    // solve QP
    int_t status = d_tree_ip2_res_mpc_hard_libstr(&qp_out->info.iter, opts->maxIter, opts->mu0,
            opts->mu_tol, opts->alpha_min, opts->warm_start, work->status, qp_in->N, tree,
            nx, nu, work->nb, work->idxb, work->ng, work->sBAbt, work->sRSQrq, work->sDCt, work->sd,
            work->sux, opts->compute_mult, qp_out->lam, work->slam, work->sst, work->internal);

    qp_out->info.solver_time = treeqp_toc(&solver_tmr);

    // copy results to qp_out struct
    treeqp_tic(&interface_tmr);

    // TODO(dimitris): COPY ALSO MULTIPLIERS!
    for (int_t ii = 0; ii < Nn; ii++) {
        blasfeo_dveccp(nu[ii], &work->sux[ii], 0, &qp_out->u[ii], 0);
        blasfeo_dveccp(nx[ii], &work->sux[ii], nu[ii], &qp_out->x[ii], 0);
    }

    qp_out->info.interface_time += treeqp_toc(&interface_tmr);

    return status;
}