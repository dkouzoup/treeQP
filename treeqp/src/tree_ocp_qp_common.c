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

#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/utils/blasfeo_utils.h"
#include "treeqp/utils/tree_utils.h"
#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blas.h"


int_t tree_ocp_qp_in_workspace_size(int_t Nn, int_t *nx, int_t *nu, struct node *tree) {
    int_t bytes = 0;

    bytes += 2*Nn*sizeof(int_t);  // nx, nu
    bytes += 2*(Nn-1)*sizeof(struct d_strmat);  // A, B
    bytes += (Nn-1)*sizeof(struct d_strvec);  // b

    bytes += 6*Nn*sizeof(struct d_strvec);  // q, r, xmin, xmax, umin, umax
    bytes += 4*Nn*sizeof(struct d_strvec);  // Q, R, Qinv, Rinv

    int_t idx, idxp;
    for (int_t ii = 0; ii < Nn; ii++) {
        idx = ii;
        idxp = tree[idx].dad;

        if (ii > 0) {
            bytes += d_size_strmat(nx[idx], nx[idxp]);  // A
            bytes += d_size_strmat(nx[idx], nu[idxp]);  // B
            bytes += d_size_strvec(nx[idx]);  // b
        }

        bytes += 3*d_size_strvec(nx[idx]);  // Q, q, Qinv
        bytes += 3*d_size_strvec(nu[idx]);  // R, r, Rinv

        bytes += 2*d_size_strvec(nx[idx]);  // xmin, xmax
        bytes += 2*d_size_strvec(nu[idx]);  // umin, umax
    }

    bytes += (bytes + 63)/64*64;
    bytes += 64;

    return bytes;
}


void tree_ocp_qp_in_create_workspace(int_t Nn, int_t *nx, int_t *nu, tree_ocp_qp_in *qp_in,
    struct node *tree, void *ptr) {

    qp_in->N = Nn;
    qp_in->tree = tree;

    // char pointer
    char *c_ptr = (char *) ptr;

    // copy dimensions to workspace
    for (int_t ii = 0; ii < Nn; ii++)
        c_ptr[ii*sizeof(int_t)] = nx[ii];
    for (int_t ii = 0; ii < Nn; ii++)
        c_ptr[(ii + Nn)*sizeof(int_t)] = nu[ii];

    qp_in->nx = (int_t *) c_ptr;
    c_ptr += Nn*sizeof(int_t);
    qp_in->nu = (int_t *) c_ptr;
    c_ptr += Nn*sizeof(int_t);

    // for (int_t ii = 0; ii < Nn; ii++)
    //     printf("NODE %d: NX = %d NU = %d\n", ii, qp_in->nx[ii], qp_in->nu[ii]);

    qp_in->A = (struct d_strmat *) c_ptr;
    c_ptr += (Nn-1)*sizeof(struct d_strmat);
    qp_in->B = (struct d_strmat *) c_ptr;
    c_ptr += (Nn-1)*sizeof(struct d_strmat);
    qp_in->b = (struct d_strvec *) c_ptr;
    c_ptr += (Nn-1)*sizeof(struct d_strvec);

    // TODO(dimitris): scaling factor missing
    qp_in->Q = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);
    qp_in->q = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);
    qp_in->R = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);
    qp_in->r = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);
    qp_in->Qinv = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);
    qp_in->Rinv = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    qp_in->xmin = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);
    qp_in->xmax = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);
    qp_in->umin = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);
    qp_in->umax = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    // align pointer
    long long l_ptr = (long long) c_ptr;
	l_ptr = (l_ptr+63)/64*64;
	c_ptr = (char *) l_ptr;

    int_t idx, idxp;
    for (int_t ii = 0; ii < Nn; ii++) {
        idx = ii;
        idxp = tree[idx].dad;
        if (ii > 0) {
            init_strmat(nx[idx], nx[idxp], (struct d_strmat *) &qp_in->A[idx-1], &c_ptr);
            init_strmat(nx[idx], nu[idxp], (struct d_strmat *) &qp_in->B[idx-1], &c_ptr);
            init_strvec(nx[idx], (struct d_strvec *) &qp_in->b[idx-1], &c_ptr);
        }
        init_strvec(nx[idx], (struct d_strvec *) &qp_in->Q[idx], &c_ptr);
        init_strvec(nx[idx], (struct d_strvec *) &qp_in->q[idx], &c_ptr);
        init_strvec(nu[idx], (struct d_strvec *) &qp_in->R[idx], &c_ptr);
        init_strvec(nu[idx], (struct d_strvec *) &qp_in->r[idx], &c_ptr);
        init_strvec(nx[idx], (struct d_strvec *) &qp_in->Qinv[idx], &c_ptr);
        init_strvec(nu[idx], (struct d_strvec *) &qp_in->Rinv[idx], &c_ptr);

        init_strvec(nx[idx], (struct d_strvec *) &qp_in->xmin[idx], &c_ptr);
        init_strvec(nx[idx], (struct d_strvec *) &qp_in->xmax[idx], &c_ptr);
        init_strvec(nu[idx], (struct d_strvec *) &qp_in->umin[idx], &c_ptr);
        init_strvec(nu[idx], (struct d_strvec *) &qp_in->umax[idx], &c_ptr);
    }
}


void tree_ocp_qp_in_fill_lti_data(double *A, double *B, double *b, double *Q, double *q, double *P,
    double *p, double *R, double *r, double *xmin, double *xmax, double *umin, double *umax,
    double *x0, tree_ocp_qp_in *qp_in) {

    struct node *tree = (struct node *) qp_in->tree;
    int_t Nn = qp_in->N;
    int_t re, nx, nu, nxp, nup;

    // check that x0 is eliminated
    if (qp_in->nx[0] != 0) {
        printf("[TREEQP]: Error! tree_ocp_qp_in_fill_lti_data function assumes x0 is eliminated\n");
        exit(1);
    }

    // TODO(dimitris): check that nx and nu are constant for the lti case (except root and leaves)

    // TODO(dimitris): avoid allocating memory here
    struct d_strmat sA;
    struct d_strvec sx0;
    d_allocate_strmat(qp_in->nx[1], qp_in->nx[1], &sA);
    d_allocate_strvec(qp_in->nx[1], &sx0);
    d_cvt_vec2strvec(qp_in->nx[1], x0, &sx0, 0);

    for (int_t ii = 0; ii < Nn; ii++) {
        nx = qp_in->nx[ii];
        nu = qp_in->nu[ii];
        if (ii > 0) {
            nxp = qp_in->nx[tree[ii].dad];
            nup = qp_in->nu[tree[ii].dad];
            re = tree[ii].real;
            d_cvt_mat2strmat(nx, nxp, &A[re*nx*nxp], nx, (struct d_strmat *) &qp_in->A[ii-1], 0, 0);
            d_cvt_mat2strmat(nx, nup, &B[re*nx*nup], nx, (struct d_strmat *) &qp_in->B[ii-1], 0, 0);
            if (tree[ii].dad == 0) {
                d_cvt_vec2strvec(nx, &b[re*nx], (struct d_strvec *) &qp_in->b[ii-1], 0);
                d_cvt_mat2strmat(nx, nx, &A[re*nx*nx], nx, &sA, 0, 0);
                dgemv_n_libstr(sA.m, sA.n, 1.0, &sA, 0, 0, &sx0, 0, 1.0,
                    (struct d_strvec *) &qp_in->b[ii-1], 0, (struct d_strvec *) &qp_in->b[ii-1], 0);
            } else {
                d_cvt_vec2strvec(nx, &b[re*nx], (struct d_strvec *) &qp_in->b[ii-1], 0);
            }
        }
        if (tree[ii].nkids > 0) {
            d_cvt_vec2strvec(nx, Q, (struct d_strvec *) &qp_in->Q[ii], 0);
            d_cvt_vec2strvec(nx, q, (struct d_strvec *) &qp_in->q[ii], 0);
        } else {
            d_cvt_vec2strvec(nx, P, (struct d_strvec *) &qp_in->Q[ii], 0);
            d_cvt_vec2strvec(nx, p, (struct d_strvec *) &qp_in->q[ii], 0);
        }
        d_cvt_vec2strvec(nu, R, (struct d_strvec *) &qp_in->R[ii], 0);
        d_cvt_vec2strvec(nu, r, (struct d_strvec *) &qp_in->r[ii], 0);

        // TODO(dimitris): move from qp_in to workspace
        for (int_t jj = 0; jj < nx; jj++)
            DVECEL_LIBSTR(&qp_in->Qinv[ii], jj) = 1.0/DVECEL_LIBSTR(&qp_in->Q[ii], jj);
        for (int_t jj = 0; jj < nu; jj++)
            DVECEL_LIBSTR(&qp_in->Rinv[ii], jj) = 1.0/R[jj];

        d_cvt_vec2strvec(nx, xmin, (struct d_strvec *) &qp_in->xmin[ii], 0);
        d_cvt_vec2strvec(nx, xmax, (struct d_strvec *) &qp_in->xmax[ii], 0);
        d_cvt_vec2strvec(nu, umin, (struct d_strvec *) &qp_in->umin[ii], 0);
        d_cvt_vec2strvec(nu, umax, (struct d_strvec *) &qp_in->umax[ii], 0);
    }
    d_free_strmat(&sA);
    d_free_strvec(&sx0);
}


int_t tree_ocp_qp_out_workspace_size(tree_ocp_qp_in *qp_in) {
    int_t Nn = qp_in->N;
    int_t bytes = 2*Nn*sizeof(struct d_strvec);  // x, u

    // TODO(dimitris): think again about convention of N and N+1
    for (int_t kk = 0; kk < Nn; kk++) {
        bytes += d_size_strvec(qp_in->nx[kk]);
        bytes += d_size_strvec(qp_in->nu[kk]);
    }

    bytes += (bytes + 63)/64*64;
    bytes += 64;

    return bytes;
}


void tree_ocp_qp_out_create_workspace(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, void *ptr) {
    int_t Nn = qp_in->N;
    // char pointer
    char *c_ptr = (char *) ptr;

    qp_out->x = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);
    qp_out->u = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    long long l_ptr = (long long) c_ptr;
	l_ptr = (l_ptr+63)/64*64;
	c_ptr = (char *) l_ptr;

    for (int_t kk = 0; kk < Nn; kk++) {
        init_strvec(qp_in->nx[kk], &qp_out->x[kk], &c_ptr);
        init_strvec(qp_in->nu[kk], &qp_out->u[kk], &c_ptr);
    }
}


real_t maximum_error_in_dynamic_constraints(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out) {
    // calculate maximum state dimension
    int_t nxMax = 0;
    for (int ii = 1; ii < qp_in->N; ii++)
        nxMax = MAX(nxMax, qp_in->nx[ii]);

    // allocate vector of size nxMax for intermediate result
    struct d_strvec tmp;
    d_allocate_strvec(nxMax, &tmp);

    // calculate maximum error
    int_t idx, idxp;
    real_t error = -1.0;

    for (int ii = 1; ii < qp_in->N; ii++) {
        idx = ii;
        idxp = qp_in->tree[idx].dad;
        // tmp = A[idx-1]*x[p(idx)] + b[idx-1]
        dgemv_n_libstr(qp_in->nx[idx], qp_in->nx[idxp], 1.0, (struct d_strmat*) &qp_in->A[idx-1],
            0, 0, &qp_out->x[idxp], 0, 1.0, (struct d_strvec*) &qp_in->b[idx-1], 0, &tmp, 0);
        // tmp = tmp + B[idx-1]*u[p(idx)]
        dgemv_n_libstr(qp_in->nx[idx], qp_in->nu[idxp], 1.0, (struct d_strmat*)&qp_in->B[idx-1],
            0, 0, &qp_out->u[idxp], 0, 1.0, &tmp, 0, &tmp, 0);

        // d_print_tran_strvec(qp_in->nx[idx], &tmp, 0);
        // d_print_tran_strvec(qp_in->nx[idx], &qp_out->x[idx], 0);

        // tmp = tmp - x[idx], aka error
        daxpy_libstr(qp_in->nx[idx], -1.0, &qp_out->x[idx], 0, &tmp, 0, &tmp, 0);

        // printf("error at node %d:\n", ii);
        // d_print_e_tran_strvec(qp_in->nx[idx], &tmp, 0);

        for (int_t jj = 0; jj < qp_in->nx[idx]; jj++)
            error = MAX(error, ABS(DVECEL_LIBSTR(&tmp, jj)));
    }

    d_free_strvec(&tmp);
    return error;
}


void print_tree_ocp_qp_in(tree_ocp_qp_in *qp_in) {
    int_t ii, jj;
    int_t Nn = qp_in->N;
    real_t min, max;

    for (ii = 0; ii < Nn; ii++) {
        printf("* Node %d/%d (nx = %d, nu = %d) ---------------------------------\n\n",
            ii, Nn-1, qp_in->nx[ii],  qp_in->nu[ii]);

        // print bounds on x
        for (jj = 0; jj < qp_in->nx[ii]; jj++) {
            min = DVECEL_LIBSTR(&qp_in->xmin[ii], jj);
            if (min > -1e10) {  // TODO(dimitris): check opts->inf instead
                printf("%5.2f  ", min);
            } else {
                printf("-INF   ");
            }
            printf("<=  x_%d  <=  ", jj);
            max = DVECEL_LIBSTR(&qp_in->xmax[ii], jj);
            if (max < 1e10) {
                printf("%5.2f\n", max);
            } else {
                printf("  INF\n");
            }
        }
        printf("\n");

        // print bounds on u
        for (jj = 0; jj < qp_in->nu[ii]; jj++) {
            min = DVECEL_LIBSTR(&qp_in->umin[ii], jj);
            if (min > -1e10) {
                printf("%5.2f  ", min);
            } else {
                printf("-INF   ");
            }
            printf("<=  u_%d  <=  ", jj);
            max = DVECEL_LIBSTR(&qp_in->umax[ii], jj);
            if (max < 1e10) {
                printf("%5.2f\n", max);
            } else {
                printf("  INF\n");
            }
        }
        printf("\n\n");

        printf("Q[%d] = \n", ii);
        d_print_tran_strvec(qp_in->nx[ii], (struct d_strvec *) &qp_in->Q[ii], 0);
        printf("q[%d] = \n", ii);
        d_print_tran_strvec(qp_in->nx[ii], (struct d_strvec *) &qp_in->q[ii], 0);

        printf("R[%d] = \n", ii);
        d_print_tran_strvec(qp_in->nu[ii], (struct d_strvec *) &qp_in->R[ii], 0);
        printf("r[%d] = \n", ii);
        d_print_tran_strvec(qp_in->nu[ii], (struct d_strvec *) &qp_in->r[ii], 0);

        // TODO(dimitris): add scaling factor for weights
        // printf("real = %d\n\n", qp_in->tree[ii].real);
        if (ii > 0) {
            // TODO(dimitris): check that .m/.n of structs coincide with nx/nu
            jj = qp_in->tree[ii].dad;
            printf("A[%d] = \n", ii-1);
            d_print_strmat(qp_in->nx[ii], qp_in->nx[jj], (struct d_strmat*) &qp_in->A[ii-1], 0, 0);
            printf("B[%d] = \n", ii-1);
            d_print_strmat(qp_in->nx[ii], qp_in->nu[jj], (struct d_strmat *) &qp_in->B[ii-1], 0, 0);
            printf("b[%d] = \n", ii-1);
            d_print_tran_strvec(qp_in->nx[ii], (struct d_strvec *) &qp_in->b[ii-1], 0);
        }
    }
}
