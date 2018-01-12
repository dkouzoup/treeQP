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
    bytes += 2*(Nn-1)*sizeof(struct blasfeo_dmat);  // A, B
    bytes += (Nn-1)*sizeof(struct blasfeo_dvec);  // b

    bytes += 3*Nn*sizeof(struct blasfeo_dmat);  // Q, R, S
    bytes += 6*Nn*sizeof(struct blasfeo_dvec);  // q, r, xmin, xmax, umin, umax

    int_t idx, idxp;
    for (int_t ii = 0; ii < Nn; ii++) {
        idx = ii;
        idxp = tree[idx].dad;

        if (ii > 0) {
            bytes += blasfeo_memsize_dmat(nx[idx], nx[idxp]);  // A
            bytes += blasfeo_memsize_dmat(nx[idx], nu[idxp]);  // B
            bytes += blasfeo_memsize_dvec(nx[idx]);  // b
        }

        bytes += blasfeo_memsize_dmat(nx[idx], nx[idx]);  // Q
        bytes += blasfeo_memsize_dmat(nu[idx], nu[idx]);  // R
        bytes += blasfeo_memsize_dmat(nu[idx], nx[idx]);  // S

        bytes += blasfeo_memsize_dvec(nx[idx]);  // q
        bytes += blasfeo_memsize_dvec(nu[idx]);  // r

        bytes += 2*blasfeo_memsize_dvec(nx[idx]);  // xmin, xmax
        bytes += 2*blasfeo_memsize_dvec(nu[idx]);  // umin, umax
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

    qp_in->A = (struct blasfeo_dmat *) c_ptr;
    c_ptr += (Nn-1)*sizeof(struct blasfeo_dmat);
    qp_in->B = (struct blasfeo_dmat *) c_ptr;
    c_ptr += (Nn-1)*sizeof(struct blasfeo_dmat);
    qp_in->b = (struct blasfeo_dvec *) c_ptr;
    c_ptr += (Nn-1)*sizeof(struct blasfeo_dvec);

    qp_in->Q = (struct blasfeo_dmat *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dmat);
    qp_in->R = (struct blasfeo_dmat *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dmat);
    qp_in->S = (struct blasfeo_dmat *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dmat);

    qp_in->q = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    qp_in->r = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    qp_in->xmin = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    qp_in->xmax = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    qp_in->umin = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    qp_in->umax = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    // align pointer
    long long l_ptr = (long long) c_ptr;
	l_ptr = (l_ptr+63)/64*64;
	c_ptr = (char *) l_ptr;

    int_t idx, idxp;
    for (int_t ii = 0; ii < Nn; ii++) {
        idx = ii;
        idxp = tree[idx].dad;
        if (ii > 0) {
            init_strmat(nx[idx], nx[idxp], (struct blasfeo_dmat *) &qp_in->A[idx-1], &c_ptr);
            init_strmat(nx[idx], nu[idxp], (struct blasfeo_dmat *) &qp_in->B[idx-1], &c_ptr);
            init_strvec(nx[idx], (struct blasfeo_dvec *) &qp_in->b[idx-1], &c_ptr);
        }

        init_strmat(nx[idx], nx[idx], (struct blasfeo_dmat *) &qp_in->Q[idx], &c_ptr);
        init_strmat(nu[idx], nu[idx], (struct blasfeo_dmat *) &qp_in->R[idx], &c_ptr);
        init_strmat(nu[idx], nx[idx], (struct blasfeo_dmat *) &qp_in->S[idx], &c_ptr);

        init_strvec(nx[idx], (struct blasfeo_dvec *) &qp_in->q[idx], &c_ptr);
        init_strvec(nu[idx], (struct blasfeo_dvec *) &qp_in->r[idx], &c_ptr);

        init_strvec(nx[idx], (struct blasfeo_dvec *) &qp_in->xmin[idx], &c_ptr);
        init_strvec(nx[idx], (struct blasfeo_dvec *) &qp_in->xmax[idx], &c_ptr);
        init_strvec(nu[idx], (struct blasfeo_dvec *) &qp_in->umin[idx], &c_ptr);
        init_strvec(nu[idx], (struct blasfeo_dvec *) &qp_in->umax[idx], &c_ptr);
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

    int_t bytes = 5*Nn*sizeof(struct blasfeo_dvec);  // x, u, lam, mu_x, mu_u

    for (int_t kk = 0; kk < Nn; kk++) {
        bytes += 3*blasfeo_memsize_dvec(nx[kk]);  // x, lam, mu_x
        bytes += 2*blasfeo_memsize_dvec(nu[kk]);  // u, mu_u
    }

    bytes = (bytes + 63)/64*64;
    bytes += 64;

    return bytes;
}


void create_tree_ocp_qp_out(int_t Nn, int_t *nx, int_t *nu, tree_ocp_qp_out *qp_out, void *ptr) {

    // char pointer
    char *c_ptr = (char *) ptr;

    qp_out->x = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    qp_out->u = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    qp_out->lam = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    qp_out->mu_x = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    qp_out->mu_u = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

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
    struct blasfeo_dvec tmp;
    blasfeo_allocate_dvec(nxMax, &tmp);

    // calculate maximum error
    int_t idx, idxp;
    real_t error = -1.0;

    for (int ii = 1; ii < qp_in->N; ii++) {
        idx = ii;
        idxp = qp_in->tree[idx].dad;
        // tmp = A[idx-1]*x[p(idx)] + b[idx-1]
        blasfeo_dgemv_n(qp_in->nx[idx], qp_in->nx[idxp], 1.0, (struct blasfeo_dmat*) &qp_in->A[idx-1],
            0, 0, &qp_out->x[idxp], 0, 1.0, (struct blasfeo_dvec*) &qp_in->b[idx-1], 0, &tmp, 0);
        // tmp = tmp + B[idx-1]*u[p(idx)]
        blasfeo_dgemv_n(qp_in->nx[idx], qp_in->nu[idxp], 1.0, (struct blasfeo_dmat*)&qp_in->B[idx-1],
            0, 0, &qp_out->u[idxp], 0, 1.0, &tmp, 0, &tmp, 0);

        // blasfeo_print_tran_dvec(qp_in->nx[idx], &tmp, 0);
        // blasfeo_print_tran_dvec(qp_in->nx[idx], &qp_out->x[idx], 0);

        // tmp = tmp - x[idx], aka error
        blasfeo_daxpy(qp_in->nx[idx], -1.0, &qp_out->x[idx], 0, &tmp, 0, &tmp, 0);

        // printf("error at node %d:\n", ii);
        // blasfeo_print_exp_tran_dvec(qp_in->nx[idx], &tmp, 0);

        for (int_t jj = 0; jj < qp_in->nx[idx]; jj++)
            error = MAX(error, ABS(DVECEL_LIBSTR(&tmp, jj)));
    }

    blasfeo_free_dvec(&tmp);
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

    struct blasfeo_dvec *sx = (struct blasfeo_dvec *)qp_out->x;
    struct blasfeo_dvec *su = (struct blasfeo_dvec *)qp_out->u;
    struct blasfeo_dmat *sQ = (struct blasfeo_dmat *)qp_in->Q;
    struct blasfeo_dmat *sR = (struct blasfeo_dmat *)qp_in->R;
    struct blasfeo_dmat *sS = (struct blasfeo_dmat *)qp_in->S;
    struct blasfeo_dvec *sq = (struct blasfeo_dvec *)qp_in->q;
    struct blasfeo_dvec *sr = (struct blasfeo_dvec *)qp_in->r;
    struct blasfeo_dmat *sA = (struct blasfeo_dmat *)qp_in->A;
    struct blasfeo_dmat *sB = (struct blasfeo_dmat *)qp_in->B;
    struct node *tree = (struct node *) qp_in->tree;

    struct blasfeo_dvec tmp_x, tmp_u;

    int_t idxkid;
    int_t idx = 0;

    for (int_t ii = 0; ii < Nn; ii++) {

        // TODO(dimitris): allocate max dim outside
        blasfeo_allocate_dvec(nx[ii], &tmp_x);
        blasfeo_allocate_dvec(nu[ii], &tmp_u);

        // TODO(dimitris): NOT tested for non-zero S terms

        // tmp_x = Q[ii]*x[ii] + q[ii]
        blasfeo_dgemv_n(nx[ii], nx[ii], 1.0, &sQ[ii], 0, 0, &sx[ii], 0, 1.0, &sq[ii], 0, &tmp_x, 0);
        // tmp_x += S[ii]*u[ii]
        blasfeo_dgemv_n(nx[ii], nu[ii], 1.0, &sS[ii], 0, 0, &su[ii], 0, 1.0, &tmp_x, 0, &tmp_x, 0);
        // tmp_x += mu_x[ii]
        blasfeo_daxpy(nx[ii], 1.0, &qp_out->mu_x[ii], 0, &tmp_x, 0, &tmp_x, 0);
        // tmp_x += lam[ii]
        blasfeo_daxpy(nx[ii], -1.0, &qp_out->lam[ii], 0, &tmp_x, 0, &tmp_x, 0);
        // tmp_u = R[ii]*u[ii] + r[ii]
        blasfeo_dgemv_n(nu[ii], nu[ii], 1.0, &sR[ii], 0, 0, &su[ii], 0, 1.0, &sr[ii], 0, &tmp_u, 0);
        // tmp_u += S[ii]'*x[ii]
        blasfeo_dgemv_t(nu[ii], nx[ii], 1.0, &sS[ii], 0, 0, &sx[ii], 0, 1.0, &tmp_u, 0, &tmp_u, 0);
        // tmp_u += mu_u[ii]
        blasfeo_daxpy(nu[ii], 1.0, &qp_out->mu_u[ii], 0, &tmp_u, 0, &tmp_u, 0);

        for (int_t jj = 0; jj < tree[ii].nkids; jj++) {
            idxkid = tree[ii].kids[jj];
            // tmp_x -= A[s(ii)]' * lam[s(ii)]
            blasfeo_dgemv_t(nx[idxkid], nx[ii], 1.0, &sA[idxkid-1], 0, 0, &qp_out->lam[idxkid], 0, 1.0, &tmp_x, 0, &tmp_x, 0);
            // tmp_u -= B[s(ii)]' * lam[s(ii)]
            blasfeo_dgemv_t(nx[idxkid], nu[ii], 1.0, &sB[idxkid-1], 0, 0, &qp_out->lam[idxkid], 0, 1.0, &tmp_u, 0, &tmp_u, 0);
        }

        blasfeo_unpack_dvec(nx[ii], &tmp_x, 0, &res[idx]);
        idx += nx[ii];
        blasfeo_unpack_dvec(nu[ii], &tmp_u, 0, &res[idx]);
        idx += nu[ii];

        blasfeo_free_dvec(&tmp_x);
        blasfeo_free_dvec(&tmp_u);
    }

    // d_print_e_mat(nz, 1, res, nz);
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
        blasfeo_print_dmat(qp_in->nx[ii], qp_in->nx[ii], (struct blasfeo_dmat *) &qp_in->Q[ii], 0, 0);

        printf("R[%d] = \n", ii);
        blasfeo_print_dmat(qp_in->nu[ii], qp_in->nu[ii], (struct blasfeo_dmat *) &qp_in->R[ii], 0, 0);

        printf("S[%d] = \n", ii);
        blasfeo_print_dmat(qp_in->nu[ii], qp_in->nx[ii], (struct blasfeo_dmat *) &qp_in->S[ii], 0, 0);

        printf("q[%d] = \n", ii);
        blasfeo_print_tran_dvec(qp_in->nx[ii], (struct blasfeo_dvec *) &qp_in->q[ii], 0);
        printf("r[%d] = \n", ii);
        blasfeo_print_tran_dvec(qp_in->nu[ii], (struct blasfeo_dvec *) &qp_in->r[ii], 0);

        // printf("real = %d\n\n", qp_in->tree[ii].real);
        if (ii > 0) {
            // TODO(dimitris): check that .m/.n of structs coincide with nx/nu
            jj = qp_in->tree[ii].dad;
            printf("A[%d] = \n", ii-1);
            blasfeo_print_dmat(qp_in->nx[ii], qp_in->nx[jj], (struct blasfeo_dmat*) &qp_in->A[ii-1], 0, 0);
            printf("B[%d] = \n", ii-1);
            blasfeo_print_dmat(qp_in->nx[ii], qp_in->nu[jj], (struct blasfeo_dmat *) &qp_in->B[ii-1], 0, 0);
            printf("b[%d] = \n", ii-1);
            blasfeo_print_tran_dvec(qp_in->nx[ii], (struct blasfeo_dvec *) &qp_in->b[ii-1], 0);
        }
    }
}


// NOTE(dimitris): weights are scaled to minimize the average cost over all scenarios
void tree_ocp_qp_in_fill_lti_data_diag_weights(double *A, double *B, double *b,
    double *Q, double *q, double *P, double *p, double *R, double *r,
    double *xmin, double *xmax, double *umin, double *umax, double *x0, tree_ocp_qp_in *qp_in) {

    int_t Nn = qp_in->N;
    struct node *tree = (struct node *) qp_in->tree;
    struct blasfeo_dmat *sA = (struct blasfeo_dmat *) qp_in->A;
    struct blasfeo_dmat *sB = (struct blasfeo_dmat *) qp_in->B;
    struct blasfeo_dvec *sb = (struct blasfeo_dvec *) qp_in->b;
    struct blasfeo_dmat *sQ = (struct blasfeo_dmat *) qp_in->Q;
    struct blasfeo_dmat *sR = (struct blasfeo_dmat *) qp_in->R;
    struct blasfeo_dvec *sq = (struct blasfeo_dvec *) qp_in->q;
    struct blasfeo_dvec *sr = (struct blasfeo_dvec *) qp_in->r;
    struct blasfeo_dvec *sxmin = (struct blasfeo_dvec *) qp_in->xmin;
    struct blasfeo_dvec *sxmax = (struct blasfeo_dvec *) qp_in->xmax;
    struct blasfeo_dvec *sumin = (struct blasfeo_dvec *) qp_in->umin;
    struct blasfeo_dvec *sumax = (struct blasfeo_dvec *) qp_in->umax;

    int_t re, nx, nu, nxp, nup;
    real_t scalingFactor;
    int_t currentStage = 0;
    int_t nodesInStage = 0;
    int_t numberOfLeaves = 1;

    struct blasfeo_dvec sQvec, sPvec, sRvec;

    nx = qp_in->nx[1];
    nu = qp_in->nu[0];

    blasfeo_create_dvec(nx, &sQvec, Q);
    blasfeo_create_dvec(nx, &sPvec, P);
    blasfeo_create_dvec(nu, &sRvec, R);

    // printf("Q = %d x 1\n", sQvec.m);
    // blasfeo_print_tran_dvec(nx, &sQvec, 0);
    // printf("P = %d x 1\n", sPvec.m);
    // blasfeo_print_tran_dvec(nx, &sPvec, 0);
    // printf("R = %d x 1\n", sRvec.m);
    // blasfeo_print_tran_dvec(nu, &sRvec, 0);

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
    struct blasfeo_dmat sA0;
    struct blasfeo_dvec sx0;
    if (qp_in->nx[0] == 0) {
        eliminatedX0 = YES;
        // TODO(dimitris): avoid allocating memory here
        blasfeo_allocate_dmat(qp_in->nx[1], qp_in->nx[1], &sA0);
        blasfeo_allocate_dvec(qp_in->nx[1], &sx0);
        blasfeo_pack_dvec(qp_in->nx[1], x0, &sx0, 0);
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
            blasfeo_pack_dmat(nx, nxp, &A[re*nx*nxp], nx, &sA[ii-1], 0, 0);
            blasfeo_pack_dmat(nx, nup, &B[re*nx*nup], nx, &sB[ii-1], 0, 0);
            if (tree[ii].dad == 0 && eliminatedX0 == YES) {
                blasfeo_pack_dvec(nx, &b[re*nx], &sb[ii-1], 0);
                blasfeo_pack_dmat(nx, nx, &A[re*nx*nx], nx, &sA0, 0, 0);
                blasfeo_dgemv_n(sA0.m, sA0.n, 1.0, &sA0, 0, 0, &sx0, 0, 1.0, &sb[ii-1], 0,
                    &sb[ii-1], 0);
            } else {
                blasfeo_pack_dvec(nx, &b[re*nx], &sb[ii-1], 0);
            }
        }
        blasfeo_dgese(sQ[ii].m, sQ[ii].n, 0.0, &sQ[ii], 0, 0);
        if (tree[ii].nkids > 0) {
            blasfeo_ddiain(sQ[ii].m, 1.0, &sQvec, 0, &sQ[ii], 0, 0);
            blasfeo_pack_dvec(sq[ii].m, q, &sq[ii], 0);
        } else {
            blasfeo_ddiain(sQ[ii].m, 1.0, &sPvec, 0, &sQ[ii], 0, 0);
            blasfeo_pack_dvec(sq[ii].m, p, &sq[ii], 0);
        }
        blasfeo_dgese(sR[ii].m, sR[ii].n, 0.0, &sR[ii], 0, 0);
        blasfeo_ddiain(sR[ii].m, 1.0, &sRvec, 0, &sR[ii], 0, 0);
        blasfeo_pack_dvec(sr[ii].m, r, &sr[ii], 0);

        // scale objective function with number of nodes per stage
        if (tree[ii].stage > currentStage) {
            scalingFactor = numberOfLeaves/nodesInStage;
            // printf("--- detected %d nodes on stage %d (scaling factor = %f)\n", nodesInStage, currentStage, scalingFactor);
            for (int_t jj = 1; jj <= nodesInStage; jj++) {
                // printf("- scaling node %d with %f\n", ii-jj, scalingFactor);
                blasfeo_dgesc(sQ[ii-jj].m, sQ[ii-jj].n, scalingFactor, &sQ[ii-jj], 0, 0);
                blasfeo_dgesc(sR[ii-jj].m, sR[ii-jj].n, scalingFactor, &sR[ii-jj], 0, 0);
                blasfeo_dvecsc(sq[ii-jj].m, scalingFactor, &sq[ii-jj], 0);
                blasfeo_dvecsc(sr[ii-jj].m, scalingFactor, &sr[ii-jj], 0);
            }
            // reset counters
            currentStage = tree[ii].stage;
            nodesInStage = 1;
        } else {
            nodesInStage++;
        }
        if (ii == 0 && eliminatedX0 == NO) {
            blasfeo_pack_dvec(sxmin[ii].m, x0, &sxmin[ii], 0);
            blasfeo_pack_dvec(sxmax[ii].m, x0, &sxmax[ii], 0);
        } else {
            blasfeo_pack_dvec(sxmin[ii].m, xmin, &sxmin[ii], 0);
            blasfeo_pack_dvec(sxmax[ii].m, xmax, &sxmax[ii], 0);
        }
        blasfeo_pack_dvec(sumin[ii].m, umin, &sumin[ii], 0);
        blasfeo_pack_dvec(sumax[ii].m, umax, &sumax[ii], 0);
    }

    if (eliminatedX0 == YES) {
        blasfeo_free_dmat(&sA0);
        blasfeo_free_dvec(&sx0);
    }
}


void tree_ocp_qp_in_read_dynamics_colmajor(real_t *A, real_t *B, real_t *b, tree_ocp_qp_in *qp_in) {

    int_t Nn = qp_in->N;

    struct blasfeo_dmat *sA = (struct blasfeo_dmat *)qp_in->A;
    struct blasfeo_dmat *sB = (struct blasfeo_dmat *)qp_in->B;
    struct blasfeo_dvec *sb = (struct blasfeo_dvec *)qp_in->b;

    int_t idxA = 0;
    int_t idxB = 0;
    int_t idxb = 0;

    for(int_t ii = 0; ii < Nn-1; ii++) {
        blasfeo_pack_dmat(sA[ii].m, sA[ii].n, &A[idxA], sA[ii].m, &sA[ii], 0, 0);
        idxA += sA[ii].m * sA[ii].n;
        assert(sA[ii].m == qp_in->nx[ii+1]);
        assert(sA[ii].n == qp_in->nx[qp_in->tree[ii+1].dad]);

        blasfeo_pack_dmat(sB[ii].m, sB[ii].n, &B[idxB], sB[ii].m, &sB[ii], 0, 0);
        idxB += sB[ii].m * sB[ii].n;
        assert(sB[ii].m == qp_in->nx[ii+1]);
        assert(sB[ii].n == qp_in->nu[qp_in->tree[ii+1].dad]);

        blasfeo_pack_dvec(sb[ii].m, &b[idxb], &sb[ii], 0);
        idxb += sb[ii].m;
        assert(sb[ii].m == qp_in->nx[ii+1]);
    }
}


void tree_ocp_qp_in_read_objective_diag(real_t *Qd, real_t *Rd, real_t *q, real_t *r,
    tree_ocp_qp_in *qp_in) {

    int_t Nn = qp_in->N;

    struct blasfeo_dmat *sQ = (struct blasfeo_dmat *)qp_in->Q;
    struct blasfeo_dmat *sR = (struct blasfeo_dmat *)qp_in->R;
    struct blasfeo_dmat *sS = (struct blasfeo_dmat *)qp_in->S;
    struct blasfeo_dvec *sq = (struct blasfeo_dvec *)qp_in->q;
    struct blasfeo_dvec *sr = (struct blasfeo_dvec *)qp_in->r;

    struct blasfeo_dvec sQvec, sRvec;

    int_t idxQ = 0;
    int_t idxR = 0;

    for (int_t ii = 0; ii < Nn; ii++) {
        blasfeo_dgese(sQ[ii].m, sQ[ii].n, 0.0, &sQ[ii], 0, 0);
        blasfeo_create_dvec(sQ[ii].m, &sQvec, &Qd[idxQ]);
        blasfeo_ddiain(sQ[ii].m, 1.0, &sQvec, 0, &sQ[ii], 0, 0);
        blasfeo_pack_dvec(sq[ii].m, &q[idxQ], &sq[ii], 0);

        idxQ += sQ[ii].m;
        assert(sQ[ii].m == qp_in->nx[ii]);
        assert(sQ[ii].n == qp_in->nx[ii]);

        blasfeo_dgese(sS[ii].m, sS[ii].m, 0.0, &sS[ii], 0, 0);
        assert(sS[ii].m == qp_in->nu[ii]);
        assert(sS[ii].n == qp_in->nx[ii]);

        blasfeo_dgese(sR[ii].m, sR[ii].n, 0.0, &sR[ii], 0, 0);
        blasfeo_create_dvec(sR[ii].m, &sRvec, &Rd[idxR]);
        blasfeo_ddiain(sR[ii].m, 1.0, &sRvec, 0, &sR[ii], 0, 0);
        blasfeo_pack_dvec(sr[ii].m, &r[idxR], &sr[ii], 0);

        idxR += sR[ii].m;
        assert(sR[ii].m == qp_in->nu[ii]);
        assert(sR[ii].n == qp_in->nu[ii]);
    }
}


void tree_ocp_qp_in_set_inf_bounds(tree_ocp_qp_in *qp_in) {

    real_t inf = 1e12;
    int_t Nn = qp_in->N;

    struct blasfeo_dvec *sxmin = (struct blasfeo_dvec *)qp_in->xmin;
    struct blasfeo_dvec *sxmax = (struct blasfeo_dvec *)qp_in->xmax;
    struct blasfeo_dvec *sumin = (struct blasfeo_dvec *)qp_in->umin;
    struct blasfeo_dvec *sumax = (struct blasfeo_dvec *)qp_in->umax;

    for (int_t ii = 0; ii < Nn; ii++) {
        blasfeo_dvecse(sxmin[ii].m, -inf, &sxmin[ii], 0);
        blasfeo_dvecse(sxmax[ii].m, inf, &sxmax[ii], 0);
        assert(sxmax[ii].m == qp_in->nx[ii]);

        blasfeo_dvecse(sumin[ii].m, -inf, &sumin[ii], 0);
        blasfeo_dvecse(sumax[ii].m, inf, &sumax[ii], 0);
        assert(sumax[ii].m == qp_in->nu[ii]);
    }

}


void tree_ocp_qp_in_set_constant_bounds(real_t *xmin, real_t *xmax, real_t *umin, real_t *umax,
    tree_ocp_qp_in *qp_in) {

    int_t Nn = qp_in->N;

    struct blasfeo_dvec *sxmin = (struct blasfeo_dvec *)qp_in->xmin;
    struct blasfeo_dvec *sxmax = (struct blasfeo_dvec *)qp_in->xmax;
    struct blasfeo_dvec *sumin = (struct blasfeo_dvec *)qp_in->umin;
    struct blasfeo_dvec *sumax = (struct blasfeo_dvec *)qp_in->umax;

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
        blasfeo_pack_dvec(sxmin[ii].m, xmin, &sxmin[ii], 0);
        blasfeo_pack_dvec(sxmax[ii].m, xmax, &sxmax[ii], 0);

        blasfeo_pack_dvec(sumin[ii].m, umin, &sumin[ii], 0);
        blasfeo_pack_dvec(sumax[ii].m, umax, &sumax[ii], 0);
    }

}


// TODO(dimitris): extend to set b instead if nx[0] = 0
void tree_ocp_qp_in_set_x0_bounds(tree_ocp_qp_in *qp_in, real_t *x0) {

    struct blasfeo_dvec *sxmin = (struct blasfeo_dvec *)qp_in->xmin;
    struct blasfeo_dvec *sxmax = (struct blasfeo_dvec *)qp_in->xmax;

    blasfeo_pack_dvec(sxmin[0].m, x0, &sxmin[0], 0);
    blasfeo_pack_dvec(sxmax[0].m, x0, &sxmax[0], 0);
}


void write_qp_out_to_txt(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, const char *fpath) {

    int_t Nn = qp_in->N;
    int_t dimx = number_of_states(qp_in);
    int_t dimu = number_of_controls(qp_in);
    int_t iter = qp_out->info.iter;

    // TODO(dimitris): also write multipliers
    struct blasfeo_dvec *sx = qp_out->x;
    struct blasfeo_dvec *su = qp_out->u;

    real_t *x = malloc(dimx*sizeof(real_t));
    real_t *u = malloc(dimu*sizeof(real_t));

    int_t indx = 0, indu = 0;

    for (int_t kk = 0; kk < Nn; kk++) {
        blasfeo_unpack_dvec(sx[kk].m, &sx[kk], 0, &x[indx]);
        indx += sx[kk].m;
        blasfeo_unpack_dvec(su[kk].m, &su[kk], 0, &u[indu]);
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
