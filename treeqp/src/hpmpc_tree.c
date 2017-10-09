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

#include <stdlib.h>
#include <assert.h>

#include "treeqp/src/hpmpc_tree.h"
#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/utils/blasfeo_utils.h"
#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"

#include "hpmpc/include/target.h"
#include "hpmpc/include/tree.h"
#include "hpmpc/include/mpc_solvers.h"

#define INF 1e10  // TODO(dimitris): option instead of hardcoded here?


int_t number_of_bounds(const struct d_strvec *vmin, const struct d_strvec *vmax) {
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


void treeqp_hpmpc_set_default_options(treeqp_hpmpc_options_t *opts) {
    opts->maxIter = 20;
	opts->mu0 = 2.0;
	opts->mu_tol = 1e-12;
	opts->alpha_min = 1e-8;
	opts->warm_start = 0;
	opts->compute_mult = 1;
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

    bytes += 3*Nn*sizeof(struct d_strvec);  // sux, slam, sst

    bytes += Nn*sizeof(struct d_strmat);  // sRSQrq
    bytes += (Nn-1)*sizeof(struct d_strmat);  // sBAbt

    bytes += Nn*sizeof(struct d_strmat);  // sDCt
    bytes += Nn*sizeof(struct d_strvec);  // sd

    for (int_t ii = 0; ii < Nn; ii++) {
        bytes += d_size_strvec(qp_in->nx[ii] + qp_in->nu[ii]);  // sux
        bytes += 2*d_size_strvec(2*nb[ii] + 2*ng[ii]);  // slam, sst

        bytes += d_size_strmat(qp_in->nx[ii] + qp_in->nu[ii] + 1, qp_in->nx[ii] + qp_in->nu[ii]);  // sRSQrq

        if (ii > 0) {
            idxp = qp_in->tree[ii].dad;
            bytes += d_size_strmat(qp_in->nx[idxp] + qp_in->nu[idxp] + 1, qp_in->nx[ii]);  // sABbt
        }

        // TODO(dimitris): this has not been tested
        bytes += d_size_strmat(qp_in->nx[ii] + qp_in->nu[ii], ng[ii]);  // sDCt
        bytes += d_size_strvec(2*qp_in->nx[ii] + 2*qp_in->nu[ii] + 2*ng[ii]);  // sd
    }

    bytes += 5*opts->maxIter*sizeof(double);  // status

    bytes += d_tree_ip2_res_mpc_hard_work_space_size_bytes_libstr(Nn,
        (struct node *) qp_in->tree, (int *) qp_in->nx, (int *) qp_in->nu, nb, ng);

    bytes = (bytes + 63)/64*64;
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

    work->sux = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    work->slam = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    work->sst = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    work->sRSQrq = (struct d_strmat *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strmat);

    work->sBAbt = (struct d_strmat *) c_ptr;
    c_ptr += (Nn-1)*sizeof(struct d_strmat);

    work->sDCt = (struct d_strmat *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strmat);

    work->sd = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    // move pointer for proper alignment of doubles and blasfeo matrices/vectors
    // TODO(dimitris): put in a function and use size_t
    long long l_ptr = (long long) c_ptr;
    l_ptr = (l_ptr+63)/64*64;
    c_ptr = (char *) l_ptr;

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
        init_strvec(2*qp_in->nx[ii] + 2*qp_in->nu[ii] + 2*work->ng[ii], &work->sd[ii], &c_ptr);
    }

    work->status = (double *) c_ptr;
    c_ptr += 5*opts->maxIter*sizeof(double);

    // TODO(dimitris): Maybe realign not needed
    l_ptr = (long long) c_ptr;
    l_ptr = (l_ptr+63)/64*64;
    c_ptr = (char *) l_ptr;

    int_t memsize = d_tree_ip2_res_mpc_hard_work_space_size_bytes_libstr(Nn, tree,
        (int *) qp_in->nx, (int *) qp_in->nu, work->nb, work->ng);

    work->internal = (void *) c_ptr;
    c_ptr += memsize;

    // // TODO(dimitris): probably zeroing memory not needed
    // char *tmp = (char *) work->internal;
    // for (int_t ii = 0; ii < memsize; ii++) {
    //     tmp[ii] = 0;
    // }

    #ifdef  RUNTIME_CHECKS
    char *ptrStart = (char *) ptr;
    char *ptrEnd = c_ptr;
    int_t bytes = treeqp_hpmpc_calculate_size(qp_in, opts);
    assert(ptrEnd <= ptrStart + bytes);
    // printf("memory starts at\t%p\nmemory ends at  \t%p\ndistance from the end\t%lu bytes\n",
    //     ptrStart, ptrEnd, ptrStart + bytes - ptrEnd);
    // exit(1);
    #endif
}