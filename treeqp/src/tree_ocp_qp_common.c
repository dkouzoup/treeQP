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

#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/utils/blasfeo_utils.h"
#include "treeqp/utils/tree_utils.h"
#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blas.h"


int_t tree_ocp_qp_in_calculate_size(int_t Nn, int_t *nx, int_t *nu, struct node *tree) {
    int_t bytes = 0;

    bytes += 2*Nn*sizeof(int_t);  // nx, nu
    bytes += 2*(Nn-1)*sizeof(struct d_strmat);  // A, B
    bytes += (Nn-1)*sizeof(struct d_strvec);  // b

    bytes += 3*Nn*sizeof(struct d_strmat);  // Q, R, S
    bytes += 6*Nn*sizeof(struct d_strvec);  // q, r, xmin, xmax, umin, umax

    int_t idx, idxp;
    for (int_t ii = 0; ii < Nn; ii++) {
        idx = ii;
        idxp = tree[idx].dad;

        if (ii > 0) {
            bytes += d_size_strmat(nx[idx], nx[idxp]);  // A
            bytes += d_size_strmat(nx[idx], nu[idxp]);  // B
            bytes += d_size_strvec(nx[idx]);  // b
        }

        bytes += d_size_strmat(nx[idx], nx[idx]);  // Q
        bytes += d_size_strmat(nu[idx], nu[idx]);  // R
        bytes += d_size_strmat(nu[idx], nx[idx]);  // S

        bytes += d_size_strvec(nx[idx]);  // q
        bytes += d_size_strvec(nu[idx]);  // r

        bytes += 2*d_size_strvec(nx[idx]);  // xmin, xmax
        bytes += 2*d_size_strvec(nu[idx]);  // umin, umax
    }

    bytes = (bytes + 63)/64*64;
    bytes += 64;

    return bytes;
}


void create_tree_ocp_qp_in(int_t Nn, int_t *nx, int_t *nu, struct node *tree, tree_ocp_qp_in *qp_in,
    void *ptr) {

    qp_in->N = Nn;
    qp_in->tree = tree;

    // char pointer
    char *c_ptr = (char *) ptr;

    // copy dimensions to allocated memory
    qp_in->nx = (int_t *) c_ptr;
    c_ptr += Nn*sizeof(int_t);
    qp_in->nu = (int_t *) c_ptr;
    c_ptr += Nn*sizeof(int_t);

    int_t *hnx = (int_t *)qp_in->nx;
    int_t *hnu = (int_t *)qp_in->nu;

    for (int_t ii = 0; ii < Nn; ii++)
        hnx[ii] = nx[ii];
    for (int_t ii = 0; ii < Nn; ii++)
        hnu[ii] = nu[ii];

    // for (int_t ii = 0; ii < Nn; ii++)
    //     printf("NODE %d: NX = %d NU = %d\n", ii, qp_in->nx[ii], qp_in->nu[ii]);

    qp_in->A = (struct d_strmat *) c_ptr;
    c_ptr += (Nn-1)*sizeof(struct d_strmat);
    qp_in->B = (struct d_strmat *) c_ptr;
    c_ptr += (Nn-1)*sizeof(struct d_strmat);
    qp_in->b = (struct d_strvec *) c_ptr;
    c_ptr += (Nn-1)*sizeof(struct d_strvec);

    qp_in->Q = (struct d_strmat *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strmat);
    qp_in->R = (struct d_strmat *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strmat);
    qp_in->S = (struct d_strmat *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strmat);

    qp_in->q = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);
    qp_in->r = (struct d_strvec *) c_ptr;
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

        init_strmat(nx[idx], nx[idx], (struct d_strmat *) &qp_in->Q[idx], &c_ptr);
        init_strmat(nu[idx], nu[idx], (struct d_strmat *) &qp_in->R[idx], &c_ptr);
        init_strmat(nu[idx], nx[idx], (struct d_strmat *) &qp_in->S[idx], &c_ptr);

        init_strvec(nx[idx], (struct d_strvec *) &qp_in->q[idx], &c_ptr);
        init_strvec(nu[idx], (struct d_strvec *) &qp_in->r[idx], &c_ptr);

        init_strvec(nx[idx], (struct d_strvec *) &qp_in->xmin[idx], &c_ptr);
        init_strvec(nx[idx], (struct d_strvec *) &qp_in->xmax[idx], &c_ptr);
        init_strvec(nu[idx], (struct d_strvec *) &qp_in->umin[idx], &c_ptr);
        init_strvec(nu[idx], (struct d_strvec *) &qp_in->umax[idx], &c_ptr);
    }
#ifdef  RUNTIME_CHECKS
    char *ptrStart = (char *) ptr;
    char *ptrEnd = c_ptr;
    int_t bytes = tree_ocp_qp_in_calculate_size(Nn, nx, nu, tree);
    assert(ptrEnd <= ptrStart + bytes);
    // printf("memory starts at\t%p\nmemory ends at  \t%p\ndistance from the end\t%lu bytes\n",
    //     ptrStart, ptrEnd, ptrStart + bytes - ptrEnd);
    // exit(1);
#endif
}


int_t tree_ocp_qp_out_calculate_size(int_t Nn, int_t *nx, int_t *nu) {

    int_t bytes = 5*Nn*sizeof(struct d_strvec);  // x, u, lam, mu_x, mu_u

    for (int_t kk = 0; kk < Nn; kk++) {
        bytes += 3*d_size_strvec(nx[kk]);  // x, lam, mu_x
        bytes += 2*d_size_strvec(nu[kk]);  // u, mu_u
    }

    bytes = (bytes + 63)/64*64;
    bytes += 64;

    return bytes;
}


void create_tree_ocp_qp_out(int_t Nn, int_t *nx, int_t *nu, tree_ocp_qp_out *qp_out, void *ptr) {

    // char pointer
    char *c_ptr = (char *) ptr;

    qp_out->x = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);
    qp_out->u = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);
    qp_out->lam = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);
    qp_out->mu_x = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);
    qp_out->mu_u = (struct d_strvec *) c_ptr;
    c_ptr += Nn*sizeof(struct d_strvec);

    long long l_ptr = (long long) c_ptr;
	l_ptr = (l_ptr+63)/64*64;
	c_ptr = (char *) l_ptr;

    for (int_t kk = 0; kk < Nn; kk++) {
        init_strvec(nx[kk], &qp_out->x[kk], &c_ptr);
        init_strvec(nu[kk], &qp_out->u[kk], &c_ptr);
        init_strvec(nx[kk], &qp_out->lam[kk], &c_ptr);
        init_strvec(nx[kk], &qp_out->mu_x[kk], &c_ptr);
        init_strvec(nu[kk], &qp_out->mu_u[kk], &c_ptr);
    }
#ifdef  RUNTIME_CHECKS
    char *ptrStart = (char *) ptr;
    char *ptrEnd = c_ptr;
    int_t bytes = tree_ocp_qp_out_calculate_size(Nn, nx, nu);
    assert(ptrEnd <= ptrStart + bytes);
    // printf("memory starts at\t%p\nmemory ends at  \t%p\ndistance from the end\t%lu bytes\n",
    //     ptrStart, ptrEnd, ptrStart + bytes - ptrEnd);
    // exit(1);
#endif
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


// TODO(dimitris): add complementarity
void calculate_KKT_residuals(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, real_t *res) {

    int_t Nn = qp_in->N;
    int_t nz = number_of_primal_variables(qp_in);

    // initialize to NaN
    for (int_t ii = 0; ii < nz; ii++) {
        res[ii] = 0.0/0.0;
    }

    int_t *nx = (int_t *)qp_in->nx;
    int_t *nu = (int_t *)qp_in->nu;

    struct d_strvec *sx = (struct d_strvec *)qp_out->x;
    struct d_strvec *su = (struct d_strvec *)qp_out->u;
    struct d_strmat *sQ = (struct d_strmat *)qp_in->Q;
    struct d_strmat *sR = (struct d_strmat *)qp_in->R;
    struct d_strmat *sS = (struct d_strmat *)qp_in->S;
    struct d_strvec *sq = (struct d_strvec *)qp_in->q;
    struct d_strvec *sr = (struct d_strvec *)qp_in->r;
    struct d_strmat *sA = (struct d_strmat *)qp_in->A;
    struct d_strmat *sB = (struct d_strmat *)qp_in->B;
    struct node *tree = (struct node *) qp_in->tree;

    struct d_strvec tmp_x, tmp_u;

    int_t idxkid;
    int_t idx = 0;

    for (int_t ii = 0; ii < Nn; ii++) {

        // TODO(dimitris): allocate max dim outside
        d_allocate_strvec(nx[ii], &tmp_x);
        d_allocate_strvec(nu[ii], &tmp_u);

        // TODO(dimitris): NOT tested for non-zero S terms

        // tmp_x = Q[ii]*x[ii] + q[ii]
        dgemv_n_libstr(nx[ii], nx[ii], 1.0, &sQ[ii], 0, 0, &sx[ii], 0, 1.0, &sq[ii], 0, &tmp_x, 0);
        // tmp_x += S[ii]*u[ii]
        dgemv_n_libstr(nx[ii], nu[ii], 1.0, &sS[ii], 0, 0, &su[ii], 0, 1.0, &tmp_x, 0, &tmp_x, 0);
        // tmp_x += mu_x[ii]
        daxpy_libstr(nx[ii], 1.0, &qp_out->mu_x[ii], 0, &tmp_x, 0, &tmp_x, 0);
        // tmp_x += lam[ii]
        daxpy_libstr(nx[ii], -1.0, &qp_out->lam[ii], 0, &tmp_x, 0, &tmp_x, 0);
        // tmp_u = R[ii]*u[ii] + r[ii]
        dgemv_n_libstr(nu[ii], nu[ii], 1.0, &sR[ii], 0, 0, &su[ii], 0, 1.0, &sr[ii], 0, &tmp_u, 0);
        // tmp_u += S[ii]'*x[ii]
        dgemv_t_libstr(nu[ii], nx[ii], 1.0, &sS[ii], 0, 0, &sx[ii], 0, 1.0, &tmp_u, 0, &tmp_u, 0);
        // tmp_u += mu_u[ii]
        daxpy_libstr(nu[ii], 1.0, &qp_out->mu_u[ii], 0, &tmp_u, 0, &tmp_u, 0);

        for (int_t jj = 0; jj < tree[ii].nkids; jj++) {
            idxkid = tree[ii].kids[jj];
            // tmp_x -= A[s(ii)]' * lam[s(ii)]
            dgemv_t_libstr(nx[idxkid], nx[ii], 1.0, &sA[idxkid-1], 0, 0, &qp_out->lam[idxkid], 0, 1.0, &tmp_x, 0, &tmp_x, 0);
            // tmp_u -= B[s(ii)]' * lam[s(ii)]
            dgemv_t_libstr(nx[idxkid], nu[ii], 1.0, &sB[idxkid-1], 0, 0, &qp_out->lam[idxkid], 0, 1.0, &tmp_u, 0, &tmp_u, 0);
        }

        d_cvt_strvec2vec(nx[ii], &tmp_x, 0, &res[idx]);
        idx += nx[ii];
        d_cvt_strvec2vec(nu[ii], &tmp_u, 0, &res[idx]);
        idx += nu[ii];

        d_free_strvec(&tmp_x);
        d_free_strvec(&tmp_u);
    }

    // d_print_e_mat(nz, 1, res, nz);

    return res;
}


real_t max_KKT_residual(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out) {

    int_t nz = number_of_primal_variables(qp_in);
    real_t *res = malloc(nz*sizeof(real_t));
    calculate_KKT_residuals(qp_in, qp_out, res);

    real_t err = ABS(res[0]);
    real_t cur;
    for (int_t ii = 1; ii < nz; ii++) {
        cur = ABS(res[ii]);
        if (cur > err) {
            err = cur;
        }
    }
    free(res);
    return err;
}


int_t number_of_states(tree_ocp_qp_in *qp_in) {
    int_t nx = 0;

    for (int_t ii = 0; ii < qp_in->N; ii++) {
        nx += qp_in->nx[ii];
    }
    return nx;
}


int_t number_of_controls(tree_ocp_qp_in *qp_in) {
    int_t nu = 0;

    for (int_t ii = 0; ii < qp_in->N; ii++) {
        nu += qp_in->nu[ii];
    }
    return nu;
}


int_t number_of_primal_variables(tree_ocp_qp_in *qp_in) {
    // simply return the sum of states and controls
    return number_of_controls(qp_in) + number_of_states(qp_in);
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
        d_print_strmat(qp_in->nx[ii], qp_in->nx[ii], (struct d_strmat *) &qp_in->Q[ii], 0, 0);

        printf("R[%d] = \n", ii);
        d_print_strmat(qp_in->nu[ii], qp_in->nu[ii], (struct d_strmat *) &qp_in->R[ii], 0, 0);

        printf("S[%d] = \n", ii);
        d_print_strmat(qp_in->nu[ii], qp_in->nx[ii], (struct d_strmat *) &qp_in->S[ii], 0, 0);

        printf("q[%d] = \n", ii);
        d_print_tran_strvec(qp_in->nx[ii], (struct d_strvec *) &qp_in->q[ii], 0);
        printf("r[%d] = \n", ii);
        d_print_tran_strvec(qp_in->nu[ii], (struct d_strvec *) &qp_in->r[ii], 0);

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


// NOTE(dimitris): weights are scaled to minimize the average cost over all scenarios
void tree_ocp_qp_in_fill_lti_data_diag_weights(double *A, double *B, double *b,
    double *Q, double *q, double *P, double *p, double *R, double *r,
    double *xmin, double *xmax, double *umin, double *umax, double *x0, tree_ocp_qp_in *qp_in) {

    int_t Nn = qp_in->N;
    struct node *tree = (struct node *) qp_in->tree;
    struct d_strmat *sA = (struct d_strmat *) qp_in->A;
    struct d_strmat *sB = (struct d_strmat *) qp_in->B;
    struct d_strvec *sb = (struct d_strvec *) qp_in->b;
    struct d_strmat *sQ = (struct d_strmat *) qp_in->Q;
    struct d_strmat *sR = (struct d_strmat *) qp_in->R;
    struct d_strvec *sq = (struct d_strvec *) qp_in->q;
    struct d_strvec *sr = (struct d_strvec *) qp_in->r;
    struct d_strvec *sxmin = (struct d_strvec *) qp_in->xmin;
    struct d_strvec *sxmax = (struct d_strvec *) qp_in->xmax;
    struct d_strvec *sumin = (struct d_strvec *) qp_in->umin;
    struct d_strvec *sumax = (struct d_strvec *) qp_in->umax;

    int_t re, nx, nu, nxp, nup;
    real_t scalingFactor;
    int_t currentStage = 0;
    int_t nodesInStage = 0;
    int_t numberOfLeaves = 1;

    struct d_strvec sQvec, sPvec, sRvec;

    nx = qp_in->nx[1];
    nu = qp_in->nu[0];

    d_create_strvec(nx, &sQvec, Q);
    d_create_strvec(nx, &sPvec, P);
    d_create_strvec(nu, &sRvec, R);

    // printf("Q = %d x 1\n", sQvec.m);
    // d_print_tran_strvec(nx, &sQvec, 0);
    // printf("P = %d x 1\n", sPvec.m);
    // d_print_tran_strvec(nx, &sPvec, 0);
    // printf("R = %d x 1\n", sRvec.m);
    // d_print_tran_strvec(nu, &sRvec, 0);

    // detect number of leaves
    for (int_t ii = Nn-1; ii > 0; ii--) {
        if (tree[ii].stage == tree[ii-1].stage) {
            numberOfLeaves++;
        } else {
            break;
        }
    }

    // check if x0 is eliminated
    answer_t eliminatedX0;
    struct d_strmat sA0;
    struct d_strvec sx0;
    if (qp_in->nx[0] == 0) {
        eliminatedX0 = YES;
        // TODO(dimitris): avoid allocating memory here
        d_allocate_strmat(qp_in->nx[1], qp_in->nx[1], &sA0);
        d_allocate_strvec(qp_in->nx[1], &sx0);
        d_cvt_vec2strvec(qp_in->nx[1], x0, &sx0, 0);
    } else {
        eliminatedX0 = NO;
    }

    for (int_t ii = 0; ii < Nn; ii++) {
        nx = qp_in->nx[ii];
        nu = qp_in->nu[ii];
        if (ii > 0) {
            nxp = qp_in->nx[tree[ii].dad];
            nup = qp_in->nu[tree[ii].dad];
            re = tree[ii].real;
            d_cvt_mat2strmat(nx, nxp, &A[re*nx*nxp], nx, &sA[ii-1], 0, 0);
            d_cvt_mat2strmat(nx, nup, &B[re*nx*nup], nx, &sB[ii-1], 0, 0);
            if (tree[ii].dad == 0 && eliminatedX0 == YES) {
                d_cvt_vec2strvec(nx, &b[re*nx], &sb[ii-1], 0);
                d_cvt_mat2strmat(nx, nx, &A[re*nx*nx], nx, &sA0, 0, 0);
                dgemv_n_libstr(sA0.m, sA0.n, 1.0, &sA0, 0, 0, &sx0, 0, 1.0, &sb[ii-1], 0,
                    &sb[ii-1], 0);
            } else {
                d_cvt_vec2strvec(nx, &b[re*nx], &sb[ii-1], 0);
            }
        }
        dgese_libstr(sQ[ii].m, sQ[ii].n, 0.0, &sQ[ii], 0, 0);
        if (tree[ii].nkids > 0) {
            ddiain_libstr(sQ[ii].m, 1.0, &sQvec, 0, &sQ[ii], 0, 0);
            d_cvt_vec2strvec(sq[ii].m, q, &sq[ii], 0);
        } else {
            ddiain_libstr(sQ[ii].m, 1.0, &sPvec, 0, &sQ[ii], 0, 0);
            d_cvt_vec2strvec(sq[ii].m, p, &sq[ii], 0);
        }
        dgese_libstr(sR[ii].m, sR[ii].n, 0.0, &sR[ii], 0, 0);
        ddiain_libstr(sR[ii].m, 1.0, &sRvec, 0, &sR[ii], 0, 0);
        d_cvt_vec2strvec(sr[ii].m, r, &sr[ii], 0);

        // scale objective function with number of nodes per stage
        if (tree[ii].stage > currentStage) {
            scalingFactor = numberOfLeaves/nodesInStage;
            // printf("--- detected %d nodes on stage %d (scaling factor = %f)\n", nodesInStage, currentStage, scalingFactor);
            for (int_t jj = 1; jj <= nodesInStage; jj++) {
                // printf("- scaling node %d with %f\n", ii-jj, scalingFactor);
                dgesc_libstr(sQ[ii-jj].m, sQ[ii-jj].n, scalingFactor, &sQ[ii-jj], 0, 0);
                dgesc_libstr(sR[ii-jj].m, sR[ii-jj].n, scalingFactor, &sR[ii-jj], 0, 0);
                dvecsc_libstr(sq[ii-jj].m, scalingFactor, &sq[ii-jj], 0);
                dvecsc_libstr(sr[ii-jj].m, scalingFactor, &sr[ii-jj], 0);
            }
            // reset counters
            currentStage = tree[ii].stage;
            nodesInStage = 1;
        } else {
            nodesInStage++;
        }
        if (ii == 0 && eliminatedX0 == NO) {
            d_cvt_vec2strvec(sxmin[ii].m, x0, &sxmin[ii], 0);
            d_cvt_vec2strvec(sxmax[ii].m, x0, &sxmax[ii], 0);
        } else {
            d_cvt_vec2strvec(sxmin[ii].m, xmin, &sxmin[ii], 0);
            d_cvt_vec2strvec(sxmax[ii].m, xmax, &sxmax[ii], 0);
        }
        d_cvt_vec2strvec(sumin[ii].m, umin, &sumin[ii], 0);
        d_cvt_vec2strvec(sumax[ii].m, umax, &sumax[ii], 0);
    }

    if (eliminatedX0 == YES) {
        d_free_strmat(&sA0);
        d_free_strvec(&sx0);
    }
}


void tree_ocp_qp_in_read_dynamics_colmajor(real_t *A, real_t *B, real_t *b, tree_ocp_qp_in *qp_in) {

    int_t Nn = qp_in->N;

    struct d_strmat *sA = (struct d_strmat *)qp_in->A;
    struct d_strmat *sB = (struct d_strmat *)qp_in->B;
    struct d_strvec *sb = (struct d_strvec *)qp_in->b;

    int_t idxA = 0;
    int_t idxB = 0;
    int_t idxb = 0;

    for(int_t ii = 0; ii < Nn-1; ii++) {
        d_cvt_mat2strmat(sA[ii].m, sA[ii].n, &A[idxA], sA[ii].m, &sA[ii], 0, 0);
        idxA += sA[ii].m * sA[ii].n;
        assert(sA[ii].m == qp_in->nx[ii+1]);
        assert(sA[ii].n == qp_in->nx[qp_in->tree[ii+1].dad]);

        d_cvt_mat2strmat(sB[ii].m, sB[ii].n, &B[idxB], sB[ii].m, &sB[ii], 0, 0);
        idxB += sB[ii].m * sB[ii].n;
        assert(sB[ii].m == qp_in->nx[ii+1]);
        assert(sB[ii].n == qp_in->nu[qp_in->tree[ii+1].dad]);

        d_cvt_vec2strvec(sb[ii].m, &b[idxb], &sb[ii], 0);
        idxb += sb[ii].m;
        assert(sb[ii].m == qp_in->nx[ii+1]);
    }
}


void tree_ocp_qp_in_read_objective_diag(real_t *Qd, real_t *Rd, real_t *q, real_t *r,
    tree_ocp_qp_in *qp_in) {

    int_t Nn = qp_in->N;

    struct d_strmat *sQ = (struct d_strmat *)qp_in->Q;
    struct d_strmat *sR = (struct d_strmat *)qp_in->R;
    struct d_strmat *sS = (struct d_strmat *)qp_in->S;
    struct d_strvec *sq = (struct d_strvec *)qp_in->q;
    struct d_strvec *sr = (struct d_strvec *)qp_in->r;

    struct d_strvec sQvec, sRvec;

    int_t idxQ = 0;
    int_t idxR = 0;

    for (int_t ii = 0; ii < Nn; ii++) {
        dgese_libstr(sQ[ii].m, sQ[ii].n, 0.0, &sQ[ii], 0, 0);
        d_create_strvec(sQ[ii].m, &sQvec, &Qd[idxQ]);
        ddiain_libstr(sQ[ii].m, 1.0, &sQvec, 0, &sQ[ii], 0, 0);
        d_cvt_vec2strvec(sq[ii].m, &q[idxQ], &sq[ii], 0);

        idxQ += sQ[ii].m;
        assert(sQ[ii].m == qp_in->nx[ii]);
        assert(sQ[ii].n == qp_in->nx[ii]);

        dgese_libstr(sS[ii].m, sS[ii].m, 0.0, &sS[ii], 0, 0);
        assert(sS[ii].m == qp_in->nu[ii]);
        assert(sS[ii].n == qp_in->nx[ii]);

        dgese_libstr(sR[ii].m, sR[ii].n, 0.0, &sR[ii], 0, 0);
        d_create_strvec(sR[ii].m, &sRvec, &Rd[idxR]);
        ddiain_libstr(sR[ii].m, 1.0, &sRvec, 0, &sR[ii], 0, 0);
        d_cvt_vec2strvec(sr[ii].m, &r[idxR], &sr[ii], 0);

        idxR += sR[ii].m;
        assert(sR[ii].m == qp_in->nu[ii]);
        assert(sR[ii].n == qp_in->nu[ii]);
    }
}


void tree_ocp_qp_in_set_inf_bounds(tree_ocp_qp_in *qp_in) {

    real_t inf = 1e12;
    int_t Nn = qp_in->N;

    struct d_strvec *sxmin = (struct d_strvec *)qp_in->xmin;
    struct d_strvec *sxmax = (struct d_strvec *)qp_in->xmax;
    struct d_strvec *sumin = (struct d_strvec *)qp_in->umin;
    struct d_strvec *sumax = (struct d_strvec *)qp_in->umax;

    for (int_t ii = 0; ii < Nn; ii++) {
        dvecse_libstr(sxmin[ii].m, -inf, &sxmin[ii], 0);
        dvecse_libstr(sxmax[ii].m, inf, &sxmax[ii], 0);
        assert(sxmax[ii].m == qp_in->nx[ii]);

        dvecse_libstr(sumin[ii].m, -inf, &sumin[ii], 0);
        dvecse_libstr(sumax[ii].m, inf, &sumax[ii], 0);
        assert(sumax[ii].m == qp_in->nu[ii]);
    }

}


void tree_ocp_qp_in_set_constant_bounds(real_t *xmin, real_t *xmax, real_t *umin, real_t *umax,
    tree_ocp_qp_in *qp_in) {

    int_t Nn = qp_in->N;

    struct d_strvec *sxmin = (struct d_strvec *)qp_in->xmin;
    struct d_strvec *sxmax = (struct d_strvec *)qp_in->xmax;
    struct d_strvec *sumin = (struct d_strvec *)qp_in->umin;
    struct d_strvec *sumax = (struct d_strvec *)qp_in->umax;

    #ifdef RUNTIME_CHECKS
        int_t nx = qp_in->nx[1];
        int_t nu = qp_in->nu[0];
        for (int_t ii = 0; ii < Nn; ii++) {
            assert(qp_in->nx[ii] == nx || qp_in->nx[ii] == 0);
            assert(sxmax[ii].m == qp_in->nx[ii]);
            assert(qp_in->nu[ii] == nu || qp_in->nu[ii] == 0);
            assert(sumax[ii].m == qp_in->nu[ii]);
        }
    #endif

    for (int_t ii = 0; ii < Nn; ii++) {
        d_cvt_vec2strvec(sxmin[ii].m, xmin, &sxmin[ii], 0);
        d_cvt_vec2strvec(sxmax[ii].m, xmax, &sxmax[ii], 0);

        d_cvt_vec2strvec(sumin[ii].m, umin, &sumin[ii], 0);
        d_cvt_vec2strvec(sumax[ii].m, umax, &sumax[ii], 0);
    }

}


// TODO(dimitris): extend to set b instead if nx[0] = 0
void tree_ocp_qp_in_set_x0_bounds(tree_ocp_qp_in *qp_in, real_t *x0) {

    struct d_strvec *sxmin = (struct d_strvec *)qp_in->xmin;
    struct d_strvec *sxmax = (struct d_strvec *)qp_in->xmax;

    d_cvt_vec2strvec(sxmin[0].m, x0, &sxmin[0], 0);
    d_cvt_vec2strvec(sxmax[0].m, x0, &sxmax[0], 0);
}


void write_qp_out_to_txt(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, const char *fpath) {

    int_t Nn = qp_in->N;
    int_t dimx = number_of_states(qp_in);
    int_t dimu = number_of_controls(qp_in);
    int_t iter = qp_out->info.iter;

    // TODO(dimitris): also write multipliers
    struct d_strvec *sx = qp_out->x;
    struct d_strvec *su = qp_out->u;

    real_t *x = malloc(dimx*sizeof(real_t));
    real_t *u = malloc(dimu*sizeof(real_t));

    int_t indx = 0, indu = 0;

    for (int_t kk = 0; kk < Nn; kk++) {
        d_cvt_strvec2vec(sx[kk].m, &sx[kk], 0, &x[indx]);
        indx += sx[kk].m;
        d_cvt_strvec2vec(su[kk].m, &su[kk], 0, &u[indu]);
        indu += su[kk].m;
    }

    char fname[100];
    snprintf(fname, sizeof(fname), "%s/%s.txt", fpath, "xopt");
    write_double_vector_to_txt(x, dimx, fname);
    snprintf(fname, sizeof(fname), "%s/%s.txt", fpath, "uopt");
    write_double_vector_to_txt(u, dimu, fname);
    snprintf(fname, sizeof(fname), "%s/%s.txt", fpath, "iter");
    write_int_vector_to_txt(&iter, 1, fname);

    free(x);
    free(u);
}
