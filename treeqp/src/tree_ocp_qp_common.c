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
#include "treeqp/utils/memory.h"
#include "treeqp/utils/tree.h"
#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blas.h"


int tree_ocp_qp_in_calculate_size(int Nn, int *nx, int *nu, int *nc, struct node *tree)
{
    int bytes = 0;

    bytes += 3*Nn*sizeof(int);  // nx, nu, nc

    bytes += 2*(Nn-1)*sizeof(struct blasfeo_dmat);  // A, B
    bytes += (Nn-1)*sizeof(struct blasfeo_dvec);  // b

    bytes += 3*Nn*sizeof(struct blasfeo_dmat);  // Q, R, S
    bytes += 2*Nn*sizeof(struct blasfeo_dvec);  // q, r

    bytes += 4*Nn*sizeof(struct blasfeo_dvec);  // xmin, xmax, umin, umax

    bytes += 2*Nn*sizeof(struct blasfeo_dmat);  // C, D
    bytes += 2*Nn*sizeof(struct blasfeo_dvec);  // dmin, dmax

    int idx, idxp, nc_;

    for (idx = 0; idx < Nn; idx++)
    {
        idxp = tree[idx].dad;

        if (nc == NULL)
        {
            nc_ = 0;
        }
        else
        {
            nc_ = nc[idx];
        }

        if (idx > 0)
        {
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

        bytes += blasfeo_memsize_dmat(nc_, nx[idx]);  // C
        bytes += blasfeo_memsize_dmat(nc_, nu[idx]);  // D
        bytes += 2*blasfeo_memsize_dvec(nc_);  // dmin, dmax
    }

    make_int_multiple_of(64, &bytes);
    bytes += 1*64;

    return bytes;
}



void tree_ocp_qp_in_create(int Nn, int *nx, int *nu, int *nc, struct node *tree, tree_ocp_qp_in *qp_in, void *ptr)
{
    char *c_ptr = (char *) ptr;

    qp_in->N = Nn;
    qp_in->tree = tree;

    qp_in->nx = (int *) c_ptr;
    c_ptr += Nn*sizeof(int);
    qp_in->nu = (int *) c_ptr;
    c_ptr += Nn*sizeof(int);
    qp_in->nc = (int *) c_ptr;
    c_ptr += Nn*sizeof(int);

    // copy dimensions to allocated memory
    for (int ii = 0; ii < Nn; ii++)
    {
        qp_in->nx[ii] = nx[ii];
        qp_in->nu[ii] = nu[ii];

        if (nc == NULL)
        {
            qp_in->nc[ii] = 0;
        } else
        {
            qp_in->nc[ii] = nc[ii];
        }
    }

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

    qp_in->C = (struct blasfeo_dmat *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dmat);
    qp_in->D = (struct blasfeo_dmat *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dmat);
    qp_in->dmin = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    qp_in->dmax = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    // align pointer
    align_char_to(64, &c_ptr);

    int idx, idxp;

    // strmats
    for (int idx = 0; idx < Nn; idx++)
    {
        idxp = tree[idx].dad;

        if (idx > 0)
        {
            init_strmat(nx[idx], nx[idxp], &qp_in->A[idx-1], &c_ptr);
            init_strmat(nx[idx], nu[idxp], &qp_in->B[idx-1], &c_ptr);
        }

        init_strmat(nx[idx], nx[idx], &qp_in->Q[idx], &c_ptr);
        init_strmat(nu[idx], nu[idx], &qp_in->R[idx], &c_ptr);
        init_strmat(nu[idx], nx[idx], &qp_in->S[idx], &c_ptr);

        init_strmat(qp_in->nc[idx], nx[idx], &qp_in->C[idx], &c_ptr);
        init_strmat(qp_in->nc[idx], nu[idx], &qp_in->D[idx], &c_ptr);
    }

    // strvecs
    for (idx = 0; idx < Nn; idx++)
    {
        idxp = tree[idx].dad;

        if (idx > 0)
        {
            init_strvec(nx[idx], &qp_in->b[idx-1], &c_ptr);
        }
        init_strvec(nx[idx], &qp_in->q[idx], &c_ptr);
        init_strvec(nu[idx], &qp_in->r[idx], &c_ptr);

        init_strvec(nx[idx], &qp_in->xmin[idx], &c_ptr);
        init_strvec(nx[idx], &qp_in->xmax[idx], &c_ptr);
        init_strvec(nu[idx], &qp_in->umin[idx], &c_ptr);
        init_strvec(nu[idx], &qp_in->umax[idx], &c_ptr);

        init_strvec(qp_in->nc[idx], &qp_in->dmin[idx], &c_ptr);
        init_strvec(qp_in->nc[idx], &qp_in->dmax[idx], &c_ptr);
    }

    assert((char *)ptr + tree_ocp_qp_in_calculate_size(Nn, nx, nu, nc, tree) >= c_ptr);
    // printf("memory starts at\t%p\nmemory ends at  \t%p\ndistance from the end\t%lu bytes\n",
    //     ptr, c_ptr, (char *)ptr + tree_ocp_qp_in_calculate_size(Nn, nx, nu, tree) - c_ptr);
    // exit(1);
}



int tree_ocp_qp_out_calculate_size(int Nn, int *nx, int *nu, int *nc)
{
    int bytes = 6*Nn*sizeof(struct blasfeo_dvec);  // x, u, lam, mu_x, mu_u, mu_d

    int nc_;

    for (int idx = 0; idx < Nn; idx++)
    {
        if (nc == NULL)
        {
            nc_ = 0;
        }
        else
        {
            nc_ = nc[idx];
        }

        bytes += 3*blasfeo_memsize_dvec(nx[idx]);  // x, lam, mu_x
        bytes += 2*blasfeo_memsize_dvec(nu[idx]);  // u, mu_u
        bytes += 1*blasfeo_memsize_dvec(nc_);  // mu_d
    }

    make_int_multiple_of(64, &bytes);
    bytes += 1*64;

    return bytes;
}



void tree_ocp_qp_out_create(int Nn, int *nx, int *nu, int *nc, tree_ocp_qp_out *qp_out, void *ptr)
{
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
    qp_out->mu_d = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);

    align_char_to(64, &c_ptr);

    int nc_;

    for (int kk = 0; kk < Nn; kk++)
    {
        if (nc == NULL)
        {
            nc_ = 0;
        }
        else
        {
            nc_ = nc[kk];
        }

        init_strvec(nx[kk], &qp_out->x[kk], &c_ptr);
        init_strvec(nu[kk], &qp_out->u[kk], &c_ptr);
        init_strvec(nx[kk], &qp_out->lam[kk], &c_ptr);
        init_strvec(nx[kk], &qp_out->mu_x[kk], &c_ptr);
        init_strvec(nu[kk], &qp_out->mu_u[kk], &c_ptr);
        init_strvec(nc_, &qp_out->mu_d[kk], &c_ptr);
    }

    assert((char *)ptr + tree_ocp_qp_out_calculate_size(Nn, nx, nu, nc) >= c_ptr);
    // printf("memory starts at\t%p\nmemory ends at  \t%p\ndistance from the end\t%lu bytes\n",
    //     ptr, c_ptr, (char *)ptr + tree_ocp_qp_out_calculate_size(Nn, nx, nu) - c_ptr);
    // exit(1);
}



// TODO(dimitris): add general constraints
void calculate_KKT_residuals(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, double *res)
{
    int Nn = qp_in->N;
    int nz = number_of_primal_variables(qp_in);
    int ne = number_of_dynamic_constraints(qp_in);
    int nKKT = 2*nz + ne;

    // initialize to NaN
    for (int ii = 0; ii < nKKT; ii++)
    {
        res[ii] = 0.0/0.0;
    }

    int *nx = (int *)qp_in->nx;
    int *nu = (int *)qp_in->nu;

    struct blasfeo_dvec *sx = qp_out->x;
    struct blasfeo_dvec *su = qp_out->u;
    struct blasfeo_dmat *sQ = qp_in->Q;
    struct blasfeo_dmat *sR = qp_in->R;
    struct blasfeo_dmat *sS = qp_in->S;
    struct blasfeo_dvec *sq = qp_in->q;
    struct blasfeo_dvec *sr = qp_in->r;
    struct blasfeo_dmat *sA = qp_in->A;
    struct blasfeo_dmat *sB = qp_in->B;
    struct node *tree = (struct node *) qp_in->tree;

    struct blasfeo_dvec tmp_x, tmp_u;

    int idxkid;
    int idxdad;
    int pos = 0;

    double mu;

    int dimx = max_number_of_states(qp_in);
    int dimu = max_number_of_controls(qp_in);
    blasfeo_allocate_dvec(dimx, &tmp_x);
    blasfeo_allocate_dvec(dimu, &tmp_u);

    for (int ii = 0; ii < Nn; ii++)
    {

        // --- stationarity (nz x 1)

        // tmp_x = Q[ii]*x[ii] + q[ii]
        blasfeo_dgemv_n(nx[ii], nx[ii], 1.0, &sQ[ii], 0, 0, &sx[ii], 0, 1.0, &sq[ii], 0, &tmp_x, 0);
        // tmp_x += S[ii]*u[ii]
        blasfeo_dgemv_t(nx[ii], nu[ii], 1.0, &sS[ii], 0, 0, &su[ii], 0, 1.0, &tmp_x, 0, &tmp_x, 0);
        // tmp_x += mu_x[ii]
        blasfeo_daxpy(nx[ii], 1.0, &qp_out->mu_x[ii], 0, &tmp_x, 0, &tmp_x, 0);
        // tmp_x += lam[ii]
        blasfeo_daxpy(nx[ii], -1.0, &qp_out->lam[ii], 0, &tmp_x, 0, &tmp_x, 0);
        // tmp_u = R[ii]*u[ii] + r[ii]
        blasfeo_dgemv_n(nu[ii], nu[ii], 1.0, &sR[ii], 0, 0, &su[ii], 0, 1.0, &sr[ii], 0, &tmp_u, 0);
        // tmp_u += S[ii]'*x[ii]
        blasfeo_dgemv_n(nu[ii], nx[ii], 1.0, &sS[ii], 0, 0, &sx[ii], 0, 1.0, &tmp_u, 0, &tmp_u, 0);
        // tmp_u += mu_u[ii]
        blasfeo_daxpy(nu[ii], 1.0, &qp_out->mu_u[ii], 0, &tmp_u, 0, &tmp_u, 0);

        for (int jj = 0; jj < tree[ii].nkids; jj++)
        {
            idxkid = tree[ii].kids[jj];
            // tmp_x -= A[s(ii)]' * lam[s(ii)]
            blasfeo_dgemv_t(nx[idxkid], nx[ii], 1.0, &sA[idxkid-1], 0, 0, &qp_out->lam[idxkid], 0, 1.0, &tmp_x, 0, &tmp_x, 0);
            // tmp_u -= B[s(ii)]' * lam[s(ii)]
            blasfeo_dgemv_t(nx[idxkid], nu[ii], 1.0, &sB[idxkid-1], 0, 0, &qp_out->lam[idxkid], 0, 1.0, &tmp_u, 0, &tmp_u, 0);
        }

        blasfeo_unpack_dvec(nx[ii], &tmp_x, 0, &res[pos]);
        pos += nx[ii];
        blasfeo_unpack_dvec(nu[ii], &tmp_u, 0, &res[pos]);
        pos += nu[ii];

        // --- primal feasibility (dynamics, ne x 1)

        if (ii > 0)
        {
            idxdad = qp_in->tree[ii].dad;

            // tmp_x = A[ii-1]*x[p(ii)] + b[ii-1]
            blasfeo_dgemv_n(nx[ii], nx[idxdad], 1.0, &qp_in->A[ii-1], 0, 0,
                &qp_out->x[idxdad], 0, 1.0, &qp_in->b[ii-1], 0, &tmp_x, 0);

            // tmp_x = tmp_x + B[ii-1]*u[p(iii)]
            blasfeo_dgemv_n(nx[ii], nu[idxdad], 1.0, &qp_in->B[ii-1], 0, 0,
                &qp_out->u[idxdad], 0, 1.0, &tmp_x, 0, &tmp_x, 0);

            // tmp_x = tmp_x - x[idx]
            blasfeo_daxpy(nx[ii], -1.0, &qp_out->x[ii], 0, &tmp_x, 0, &tmp_x, 0);

            blasfeo_unpack_dvec(nx[ii], &tmp_x, 0, &res[pos]);
            pos += nx[ii];
        }

        // --- complementarity (nz x 1)

        for (int jj = 0; jj < nx[ii]; jj++)
        {
            mu = BLASFEO_DVECEL(&qp_out->mu_x[ii], jj);
            if ( mu > 0)
            {
                res[pos+jj] = mu*(BLASFEO_DVECEL(&qp_out->x[ii], jj) - BLASFEO_DVECEL(&qp_in->xmax[ii], jj));
            }
            else
            {
                res[pos+jj] = mu*(-BLASFEO_DVECEL(&qp_out->x[ii], jj) + BLASFEO_DVECEL(&qp_in->xmin[ii], jj));
            }
        }

        for (int jj = 0; jj < nu[ii]; jj++)
        {
            mu = BLASFEO_DVECEL(&qp_out->mu_u[ii], jj);
            if ( mu > 0)
            {
                res[pos+jj] = mu*(BLASFEO_DVECEL(&qp_out->u[ii], jj) - BLASFEO_DVECEL(&qp_in->umax[ii], jj));
            }
            else
            {
                res[pos+jj] = mu*(-BLASFEO_DVECEL(&qp_out->u[ii], jj) + BLASFEO_DVECEL(&qp_in->umin[ii], jj));
            }
        }

        pos += nx[ii];
        pos += nu[ii];
    }

    blasfeo_free_dvec(&tmp_x);
    blasfeo_free_dvec(&tmp_u);

    assert(nKKT == pos && "incorrect size of KKT residuals");
    // d_print_e_tran_mat(nKKT, 1, res, 1);
}



double max_KKT_residual(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out)
{
    int nz = number_of_primal_variables(qp_in);
    int ne = number_of_dynamic_constraints(qp_in);
    int nKKT = 2*nz + ne;

    double *res = malloc(nKKT*sizeof(double));
    calculate_KKT_residuals(qp_in, qp_out, res);

    double err = ABS(res[0]);
    double cur;
    for (int ii = 1; ii < nKKT; ii++)
    {
        cur = ABS(res[ii]);
        if (cur > err) err = cur;
    }
    free(res);
    return err;
}



int number_of_states(tree_ocp_qp_in *qp_in)
{
    int nx = 0;

    for (int ii = 0; ii < qp_in->N; ii++) nx += qp_in->nx[ii];

    return nx;
}



int max_number_of_states(tree_ocp_qp_in *qp_in)
{
    int nx_max = 0;

    for (int ii = 0; ii < qp_in->N; ii++) nx_max = MAX(nx_max, qp_in->nx[ii]);

    return nx_max;
}



int number_of_controls(tree_ocp_qp_in *qp_in)
{
    int nu = 0;

    for (int ii = 0; ii < qp_in->N; ii++) nu += qp_in->nu[ii];

    return nu;
}



int max_number_of_controls(tree_ocp_qp_in *qp_in)
{
    int nu_max = 0;

    for (int ii = 0; ii < qp_in->N; ii++) nu_max = MAX(nu_max, qp_in->nu[ii]);

    return nu_max;
}



int number_of_primal_variables(tree_ocp_qp_in *qp_in)
{
    return number_of_controls(qp_in) + number_of_states(qp_in);
}



int number_of_dynamic_constraints(tree_ocp_qp_in *qp_in)
{
    int ne = 0;

    for (int ii = 1; ii < qp_in->N; ii++) ne += qp_in->nx[ii];
    return ne;
}



void tree_ocp_qp_in_print(tree_ocp_qp_in *qp_in)
{
    int Nn = qp_in->N;
    double min, max;
    int idxdad;

    for (int ii = 0; ii < Nn; ii++)
    {
        printf("* Node %d/%d (nx = %d, nu = %d) ---------------------------------\n\n",
            ii, Nn-1, qp_in->nx[ii],  qp_in->nu[ii]);

        // print bounds on x
        for (int jj = 0; jj < qp_in->nx[ii]; jj++)
        {
            min = BLASFEO_DVECEL(&qp_in->xmin[ii], jj);
            if (min > -1e10)
            {  // TODO(dimitris): check opts->inf instead
                printf("%5.2f  ", min);
            }
            else
            {
                printf("-INF   ");
            }
            printf("<=  x_%d  <=  ", jj);
            max = BLASFEO_DVECEL(&qp_in->xmax[ii], jj);
            if (max < 1e10)
            {
                printf("%5.2f\n", max);
            } else
            {
                printf("  INF\n");
            }
        }
        printf("\n");

        // print bounds on u
        for (int jj = 0; jj < qp_in->nu[ii]; jj++)
        {
            min = BLASFEO_DVECEL(&qp_in->umin[ii], jj);
            if (min > -1e10)
            {
                printf("%5.2f  ", min);
            }
            else
            {
                printf("-INF   ");
            }
            printf("<=  u_%d  <=  ", jj);
            max = BLASFEO_DVECEL(&qp_in->umax[ii], jj);
            if (max < 1e10)
            {
                printf("%5.2f\n", max);
            }
            else
            {
                printf("  INF\n");
            }
        }
        printf("\n\n");

        printf("Q[%d] = \n", ii);
        blasfeo_print_dmat(qp_in->nx[ii], qp_in->nx[ii], &qp_in->Q[ii], 0, 0);

        printf("R[%d] = \n", ii);
        blasfeo_print_dmat(qp_in->nu[ii], qp_in->nu[ii], &qp_in->R[ii], 0, 0);

        printf("S[%d] = \n", ii);
        blasfeo_print_dmat(qp_in->nu[ii], qp_in->nx[ii], &qp_in->S[ii], 0, 0);

        printf("q[%d] = \n", ii);
        blasfeo_print_tran_dvec(qp_in->nx[ii], &qp_in->q[ii], 0);
        printf("r[%d] = \n", ii);
        blasfeo_print_tran_dvec(qp_in->nu[ii], &qp_in->r[ii], 0);

        // printf("real = %d\n\n", qp_in->tree[ii].real);
        if (ii > 0)
        {
            // TODO(dimitris): check that .m/.n of structs coincide with nx/nu
            idxdad = qp_in->tree[ii].dad;
            printf("A[%d] = \n", ii-1);
            blasfeo_print_dmat(qp_in->nx[ii], qp_in->nx[idxdad], &qp_in->A[ii-1], 0, 0);
            printf("B[%d] = \n", ii-1);
            blasfeo_print_dmat(qp_in->nx[ii], qp_in->nu[idxdad], &qp_in->B[ii-1], 0, 0);
            printf("b[%d] = \n", ii-1);
            blasfeo_print_tran_dvec(qp_in->nx[ii], &qp_in->b[ii-1], 0);
        }
    }
}



// TODO(dimitris): move prints to utils
void tree_ocp_qp_out_print(int Nn, tree_ocp_qp_out *qp_out)
{
    int nx, nu;

    printf("\nProblem solved in %d iterations (%f ms)\n\n",
        qp_out->info.iter, qp_out->info.solver_time+qp_out->info.interface_time);

    for (int ii = 0; ii < Nn; ii++)
    {
        nx = qp_out->x[ii].m;
        nu = qp_out->u[ii].m;

        printf("* Node %d/%d (nx = %d, nu = %d) ---------------------------------\n\n",
            ii, Nn-1, nx,  nu);

        printf("x[%d] = \n", ii);
        blasfeo_print_tran_dvec(nx, &qp_out->x[ii], 0);

        printf("u[%d] = \n", ii);
        blasfeo_print_tran_dvec(nu, &qp_out->u[ii], 0);

        // NOTE(dimitris): always zero at root node
        printf("lam[%d] = \n", ii);
        blasfeo_print_tran_dvec(qp_out->lam[ii].m, &qp_out->lam[ii], 0);

        printf("mu_x[%d] = \n", ii);
        blasfeo_print_tran_dvec(nx, &qp_out->mu_x[ii], 0);

        printf("mu_u[%d] = \n", ii);
        blasfeo_print_tran_dvec(nu, &qp_out->mu_u[ii], 0);
    }
}



// NOTE(dimitris): weights are scaled to minimize the average cost over all scenarios
void tree_ocp_qp_in_fill_lti_data_diag_weights(double *A, double *B, double *b,
    double *Q, double *q, double *P, double *p, double *R, double *r,
    double *xmin, double *xmax, double *umin, double *umax, double *x0,
    double *C, double *D, double *dmin, double *dmax, tree_ocp_qp_in *qp_in)
{
    int Nn = qp_in->N;
    struct node *tree = (struct node *) qp_in->tree;
    struct blasfeo_dmat *sA = qp_in->A;
    struct blasfeo_dmat *sB = qp_in->B;
    struct blasfeo_dvec *sb = qp_in->b;
    struct blasfeo_dmat *sQ = qp_in->Q;
    struct blasfeo_dmat *sR = qp_in->R;
    struct blasfeo_dvec *sq = qp_in->q;
    struct blasfeo_dvec *sr = qp_in->r;
    struct blasfeo_dvec *sxmin = qp_in->xmin;
    struct blasfeo_dvec *sxmax = qp_in->xmax;
    struct blasfeo_dvec *sumin = qp_in->umin;
    struct blasfeo_dvec *sumax = qp_in->umax;
    struct blasfeo_dmat *sC = qp_in->C;
    struct blasfeo_dmat *sD = qp_in->D;
    struct blasfeo_dvec *sdmin = qp_in->dmin;
    struct blasfeo_dvec *sdmax = qp_in->dmax;

    int re, nx, nu, nc, nxp, nup;
    double scalingFactor;
    int currentStage = 0;
    int nodesInStage = 0;
    int numberOfLeaves = 1;

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
    for (int ii = Nn-1; ii > 0; ii--)
    {
        if (tree[ii].stage == tree[ii-1].stage)
        {
            numberOfLeaves++;
        }
        else
        {
            break;
        }
    }

    // check if x0 is eliminated
    answer_t eliminatedX0;
    struct blasfeo_dmat sA0;
    struct blasfeo_dvec sx0;
    if (qp_in->nx[0] == 0)
    {
        eliminatedX0 = YES;
        // TODO(dimitris): avoid allocating memory here
        blasfeo_allocate_dmat(qp_in->nx[1], qp_in->nx[1], &sA0);
        blasfeo_allocate_dvec(qp_in->nx[1], &sx0);
        blasfeo_pack_dvec(qp_in->nx[1], x0, &sx0, 0);
    }
    else
    {
        eliminatedX0 = NO;
    }

    for (int ii = 0; ii < Nn; ii++)
    {
        nx = qp_in->nx[ii];
        nu = qp_in->nu[ii];
        nc = qp_in->nc[ii];

        if (ii > 0)
        {
            nxp = qp_in->nx[tree[ii].dad];
            nup = qp_in->nu[tree[ii].dad];
            re = tree[ii].real;
            blasfeo_pack_dmat(nx, nxp, &A[re*nx*nxp], nx, &sA[ii-1], 0, 0);
            blasfeo_pack_dmat(nx, nup, &B[re*nx*nup], nx, &sB[ii-1], 0, 0);
            if (tree[ii].dad == 0 && eliminatedX0 == YES)
            {
                blasfeo_pack_dvec(nx, &b[re*nx], &sb[ii-1], 0);
                blasfeo_pack_dmat(nx, nx, &A[re*nx*nx], nx, &sA0, 0, 0);
                blasfeo_dgemv_n(sA0.m, sA0.n, 1.0, &sA0, 0, 0, &sx0, 0, 1.0, &sb[ii-1], 0, &sb[ii-1], 0);
            }
            else
            {
                blasfeo_pack_dvec(nx, &b[re*nx], &sb[ii-1], 0);
            }
        }
        blasfeo_dgese(sQ[ii].m, sQ[ii].n, 0.0, &sQ[ii], 0, 0);
        if (tree[ii].nkids > 0)
        {
            blasfeo_ddiain(sQ[ii].m, 1.0, &sQvec, 0, &sQ[ii], 0, 0);
            blasfeo_pack_dvec(sq[ii].m, q, &sq[ii], 0);
        }
        else
        {
            blasfeo_ddiain(sQ[ii].m, 1.0, &sPvec, 0, &sQ[ii], 0, 0);
            blasfeo_pack_dvec(sq[ii].m, p, &sq[ii], 0);
        }
        blasfeo_dgese(sR[ii].m, sR[ii].n, 0.0, &sR[ii], 0, 0);
        blasfeo_ddiain(sR[ii].m, 1.0, &sRvec, 0, &sR[ii], 0, 0);
        blasfeo_pack_dvec(sr[ii].m, r, &sr[ii], 0);

        // scale objective function with number of nodes per stage
        if (tree[ii].stage > currentStage)
        {
            scalingFactor = numberOfLeaves/nodesInStage;
            // printf("--- detected %d nodes on stage %d (scaling factor = %f)\n", nodesInStage, currentStage, scalingFactor);
            for (int jj = 1; jj <= nodesInStage; jj++)
            {
                // printf("- scaling node %d with %f\n", ii-jj, scalingFactor);
                blasfeo_dgesc(sQ[ii-jj].m, sQ[ii-jj].n, scalingFactor, &sQ[ii-jj], 0, 0);
                blasfeo_dgesc(sR[ii-jj].m, sR[ii-jj].n, scalingFactor, &sR[ii-jj], 0, 0);
                blasfeo_dvecsc(sq[ii-jj].m, scalingFactor, &sq[ii-jj], 0);
                blasfeo_dvecsc(sr[ii-jj].m, scalingFactor, &sr[ii-jj], 0);
            }
            // reset counters
            currentStage = tree[ii].stage;
            nodesInStage = 1;
        }
        else
        {
            nodesInStage++;
        }
        if (ii == 0 && eliminatedX0 == NO)
        {
            blasfeo_pack_dvec(sxmin[ii].m, x0, &sxmin[ii], 0);
            blasfeo_pack_dvec(sxmax[ii].m, x0, &sxmax[ii], 0);
        }
        else
        {
            blasfeo_pack_dvec(sxmin[ii].m, xmin, &sxmin[ii], 0);
            blasfeo_pack_dvec(sxmax[ii].m, xmax, &sxmax[ii], 0);
        }
        blasfeo_pack_dvec(sumin[ii].m, umin, &sumin[ii], 0);
        blasfeo_pack_dvec(sumax[ii].m, umax, &sumax[ii], 0);

        if (C != NULL && D != NULL && dmin != NULL && dmax != NULL)
        {
            blasfeo_pack_dmat(nc, nx, C, nc, &sC[ii], 0, 0);
            blasfeo_pack_dmat(nc, nu, D, nc, &sD[ii], 0, 0);
            blasfeo_pack_dvec(sdmin[ii].m, dmin, &sdmin[ii], 0);
            blasfeo_pack_dvec(sdmax[ii].m, dmax, &sdmax[ii], 0);
        }
    }

    if (eliminatedX0 == YES)
    {
        blasfeo_free_dmat(&sA0);
        blasfeo_free_dvec(&sx0);
    }
}



void tree_ocp_qp_in_set_ltv_dynamics_colmajor(double *A, double *B, double *b, tree_ocp_qp_in *qp_in)
{
    int Nn = qp_in->N;

    struct blasfeo_dmat *sA = qp_in->A;
    struct blasfeo_dmat *sB = qp_in->B;
    struct blasfeo_dvec *sb = qp_in->b;

    int idxA = 0;
    int idxB = 0;
    int idxb = 0;

    for(int ii = 0; ii < Nn-1; ii++)
    {
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



void tree_ocp_qp_in_set_ltv_objective_diag(double *Qd, double *Rd, double *q, double *r,
    tree_ocp_qp_in *qp_in)
    {
    int Nn = qp_in->N;

    struct blasfeo_dmat *sQ = qp_in->Q;
    struct blasfeo_dmat *sR = qp_in->R;
    struct blasfeo_dmat *sS = qp_in->S;
    struct blasfeo_dvec *sq = qp_in->q;
    struct blasfeo_dvec *sr = qp_in->r;

    struct blasfeo_dvec sQvec, sRvec;

    int idxQ = 0;
    int idxR = 0;

    for (int ii = 0; ii < Nn; ii++)
    {
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



void tree_ocp_qp_in_set_ltv_objective_colmajor(double *Q, double *R, double *S, double *q, double *r,
    tree_ocp_qp_in *qp_in)
{
    int Nn = qp_in->N;

    struct blasfeo_dmat *sQ = qp_in->Q;
    struct blasfeo_dmat *sR = qp_in->R;
    struct blasfeo_dmat *sS = qp_in->S;
    struct blasfeo_dvec *sq = qp_in->q;
    struct blasfeo_dvec *sr = qp_in->r;

    int idxQ = 0;
    int idxR = 0;
    int idxS = 0;
    int idxq = 0;
    int idxr = 0;

    for(int ii = 0; ii < Nn; ii++)
    {
        blasfeo_pack_dmat(sQ[ii].m, sQ[ii].n, &Q[idxQ], sQ[ii].m, &sQ[ii], 0, 0);
        idxQ += sQ[ii].m * sQ[ii].n;
        assert(sQ[ii].m == qp_in->nx[ii]);
        assert(sQ[ii].n == qp_in->nx[ii]);
        // TODO(dimitris): assert is_Q_symmetric, is_Q_pos_def

        blasfeo_pack_dmat(sR[ii].m, sR[ii].n, &R[idxR], sR[ii].m, &sR[ii], 0, 0);
        idxR += sR[ii].m * sR[ii].n;
        assert(sR[ii].m == qp_in->nu[ii]);
        assert(sR[ii].n == qp_in->nu[ii]);

        blasfeo_pack_dmat(sS[ii].m, sS[ii].n, &S[idxS], sS[ii].m, &sS[ii], 0, 0);
        idxS += sS[ii].m * sS[ii].n;
        assert(sS[ii].m == qp_in->nu[ii]);
        assert(sS[ii].n == qp_in->nx[ii]);

        blasfeo_pack_dvec(sq[ii].m, &q[idxq], &sq[ii], 0);
        idxq += sq[ii].m;
        assert(sq[ii].m == qp_in->nx[ii]);

        blasfeo_pack_dvec(sr[ii].m, &r[idxr], &sr[ii], 0);
        idxr += sr[ii].m;
        assert(sr[ii].m == qp_in->nu[ii]);
    }
}



void tree_ocp_qp_in_set_inf_bounds(tree_ocp_qp_in *qp_in)
{
    double inf = 1e12;
    int Nn = qp_in->N;

    struct blasfeo_dvec *sxmin = qp_in->xmin;
    struct blasfeo_dvec *sxmax = qp_in->xmax;
    struct blasfeo_dvec *sumin = qp_in->umin;
    struct blasfeo_dvec *sumax = qp_in->umax;

    for (int ii = 0; ii < Nn; ii++)
    {
        blasfeo_dvecse(sxmin[ii].m, -inf, &sxmin[ii], 0);
        blasfeo_dvecse(sxmax[ii].m, inf, &sxmax[ii], 0);
        assert(sxmax[ii].m == qp_in->nx[ii]);

        blasfeo_dvecse(sumin[ii].m, -inf, &sumin[ii], 0);
        blasfeo_dvecse(sumax[ii].m, inf, &sumax[ii], 0);
        assert(sumax[ii].m == qp_in->nu[ii]);
    }

}



void tree_ocp_qp_in_set_const_bounds(double *xmin, double *xmax, double *umin, double *umax,
    tree_ocp_qp_in *qp_in)
{
    int Nn = qp_in->N;
    int nx = qp_in->nx[1];
    int nu = qp_in->nu[0];

    struct blasfeo_dvec *sxmin = qp_in->xmin;
    struct blasfeo_dvec *sxmax = qp_in->xmax;
    struct blasfeo_dvec *sumin = qp_in->umin;
    struct blasfeo_dvec *sumax = qp_in->umax;

    for (int ii = 0; ii < Nn; ii++)
    {
        assert(qp_in->nx[ii] == nx || qp_in->nx[ii] == 0);
        assert(sxmax[ii].m == qp_in->nx[ii]);
        assert(qp_in->nu[ii] == nu || qp_in->nu[ii] == 0);
        assert(sumax[ii].m == qp_in->nu[ii]);
    }

    for (int ii = 0; ii < Nn; ii++)
    {
        blasfeo_pack_dvec(sxmin[ii].m, xmin, &sxmin[ii], 0);
        blasfeo_pack_dvec(sxmax[ii].m, xmax, &sxmax[ii], 0);

        blasfeo_pack_dvec(sumin[ii].m, umin, &sumin[ii], 0);
        blasfeo_pack_dvec(sumax[ii].m, umax, &sumax[ii], 0);
    }

}



// TODO(dimitris): extend to set b instead if nx[0] = 0
void tree_ocp_qp_in_set_x0_bounds(tree_ocp_qp_in *qp_in, double *x0)
{
    struct blasfeo_dvec *sxmin = qp_in->xmin;
    struct blasfeo_dvec *sxmax = qp_in->xmax;

    blasfeo_pack_dvec(sxmin[0].m, x0, &sxmin[0], 0);
    blasfeo_pack_dvec(sxmax[0].m, x0, &sxmax[0], 0);
}



void tree_ocp_qp_out_write_to_txt(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, const char *fpath)
{
    int Nn = qp_in->N;
    int dimx = number_of_states(qp_in);
    int dimu = number_of_controls(qp_in);
    int iter = qp_out->info.iter;

    // TODO(dimitris): also write multipliers
    struct blasfeo_dvec *sx = qp_out->x;
    struct blasfeo_dvec *su = qp_out->u;

    double *x = malloc(dimx*sizeof(double));
    double *u = malloc(dimu*sizeof(double));

    int indx = 0, indu = 0;

    for (int kk = 0; kk < Nn; kk++)
    {
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
