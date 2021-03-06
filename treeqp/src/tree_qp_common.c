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

#include "treeqp/src/tree_qp_common.h"
#include "treeqp/utils/blasfeo.h"
#include "treeqp/utils/memory.h"
#include "treeqp/utils/tree.h"
#include "treeqp/utils/types.h"

#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_aux_ext_dep.h>  // blasfeo_allocate_dvec
#include <blasfeo_d_blas.h>



static int find_parent(int idx, const int *nk)
{
    if (idx == 0) return -1;  // root

    int acc = 0;
    for (int ii = 0; ii < idx; ii++)
    {
        acc += nk[ii];
        if (acc >= idx) return ii;
    }
}



int tree_qp_in_calculate_size(int Nn, const int * nx, const int * nu, const int * nc, const int * nk)
{
    int bytes = 0;

    bytes += Nn*sizeof(struct node);
    bytes += tree_calculate_size(nk);

    bytes += 3*Nn*sizeof(int);  // nx, nu, nc

    bytes += 2*(Nn-1)*sizeof(struct blasfeo_dmat);  // A, B
    bytes += (Nn-1)*sizeof(struct blasfeo_dvec);  // b

    bytes += 3*Nn*sizeof(struct blasfeo_dmat);  // Q, R, S
    bytes += 2*Nn*sizeof(struct blasfeo_dvec);  // q, r

    bytes += 4*Nn*sizeof(struct blasfeo_dvec);  // xmin, xmax, umin, umax

    bytes += 2*Nn*sizeof(struct blasfeo_dmat);  // C, D
    bytes += 2*Nn*sizeof(struct blasfeo_dvec);  // dmin, dmax

    bytes += nk[0]*sizeof(int);  // internal_memory.is_A_initialized
    bytes += nk[0]*sizeof(int);  // internal_memory.is_b_initialized
    bytes += nk[0]*sizeof(struct blasfeo_dmat);  // internal_memory.A0
    bytes += nk[0]*sizeof(struct blasfeo_dvec);  // internal_memory.b0

    int idx, nc_;

    int idxp;

    for (idx = 0; idx < Nn; idx++)
    {
        idxp = find_parent(idx, nk);

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

            if (idx <= nk[0])  // children of root
            {
                bytes += blasfeo_memsize_dmat(nx[idx], nx[idxp]);  // internal_memory.A0
                bytes += blasfeo_memsize_dvec(nx[idx]);  // internal_memory.b0
            }
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

        if (idx == 0)
        {
            bytes += blasfeo_memsize_dvec(nx[0]);  // internal_memory.x0
            bytes += blasfeo_memsize_dmat(nc_, nx[0]);  // internal_memory.C0
            bytes += 2*blasfeo_memsize_dvec(nc_);  // internal_memory.dmin0, internal_memory.dmax0

            bytes += blasfeo_memsize_dmat(nu[0], nx[0]);  // internal_memory.S0
            bytes += blasfeo_memsize_dvec(nu[0]);  // internal_memory.r0
        }
    }

    make_int_multiple_of(64, &bytes);
    bytes += 1*64;

    return bytes;
}



void tree_qp_in_create(int Nn, const int * nx, const int * nu, const int * nc,  const int * nk,
    tree_qp_in * qp_in, void *ptr)
{
    char *c_ptr = (char *) ptr;

    qp_in->N = Nn;

    qp_in->tree = (struct node *) c_ptr;
    c_ptr += Nn*sizeof(struct node);
    tree_create(nk, qp_in->tree, (void *)c_ptr);
    c_ptr += tree_calculate_size(nk);

    assert (Nn == number_of_nodes_from_tree(qp_in->tree) && "Detected number of nodes different than given one");

    qp_in->nx = (int *) c_ptr;
    c_ptr += Nn*sizeof(int);
    qp_in->nu = (int *) c_ptr;
    c_ptr += Nn*sizeof(int);
    qp_in->nc = (int *) c_ptr;
    c_ptr += Nn*sizeof(int);

    qp_in->internal_memory.is_A_initialized = (int *) c_ptr;
    c_ptr += qp_in->tree[0].nkids*sizeof(int);
    qp_in->internal_memory.is_b_initialized = (int *) c_ptr;
    c_ptr += qp_in->tree[0].nkids*sizeof(int);

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
        qp_in->internal_memory.is_A_initialized[ii] = 0;
        qp_in->internal_memory.is_b_initialized[ii] = 0;
    }

    qp_in->internal_memory.is_C_initialized = 0;
    qp_in->internal_memory.is_dmin_initialized = 0;
    qp_in->internal_memory.is_dmax_initialized = 0;
    qp_in->internal_memory.is_S_initialized = 0;
    qp_in->internal_memory.is_r_initialized = 0;

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

    qp_in->internal_memory.A0 = (struct blasfeo_dmat *) c_ptr;
    c_ptr += qp_in->tree[0].nkids*sizeof(struct blasfeo_dmat);
    qp_in->internal_memory.b0 = (struct blasfeo_dvec *) c_ptr;
    c_ptr += qp_in->tree[0].nkids*sizeof(struct blasfeo_dvec);

    // align pointer
    align_char_to(64, &c_ptr);

    int idx, idxp;

    // strmats
    for (int idx = 0; idx < Nn; idx++)
    {
        idxp = qp_in->tree[idx].dad;

        if (idx > 0)
        {
            init_strmat(nx[idx], nx[idxp], &qp_in->A[idx-1], &c_ptr);
            init_strmat(nx[idx], nu[idxp], &qp_in->B[idx-1], &c_ptr);
            if (idx <= qp_in->tree[0].nkids)
            {
                init_strmat(nx[idx], nx[idxp], &qp_in->internal_memory.A0[idx-1], &c_ptr);
            }
        }

        init_strmat(nx[idx], nx[idx], &qp_in->Q[idx], &c_ptr);
        init_strmat(nu[idx], nu[idx], &qp_in->R[idx], &c_ptr);
        init_strmat(nu[idx], nx[idx], &qp_in->S[idx], &c_ptr);

        init_strmat(qp_in->nc[idx], nx[idx], &qp_in->C[idx], &c_ptr);
        init_strmat(qp_in->nc[idx], nu[idx], &qp_in->D[idx], &c_ptr);
    }

    init_strmat(qp_in->nc[0], nx[0], &qp_in->internal_memory.C0, &c_ptr);
    init_strmat(nu[0], nx[0], &qp_in->internal_memory.S0, &c_ptr);

    // strvecs
    for (idx = 0; idx < Nn; idx++)
    {
        idxp = qp_in->tree[idx].dad;

        if (idx > 0)
        {
            init_strvec(nx[idx], &qp_in->b[idx-1], &c_ptr);
            if (idx <= qp_in->tree[0].nkids)
            {
                init_strvec(nx[idx], &qp_in->internal_memory.b0[idx-1], &c_ptr);
            }
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

    init_strvec(qp_in->nx[0], &qp_in->internal_memory.x0, &c_ptr);
    init_strvec(qp_in->nc[0], &qp_in->internal_memory.dmin0, &c_ptr);
    init_strvec(qp_in->nc[0], &qp_in->internal_memory.dmax0, &c_ptr);
    init_strvec(qp_in->nu[0], &qp_in->internal_memory.r0, &c_ptr);

    tree_qp_in_set_inf_bounds(qp_in);

    assert((char *)ptr + tree_qp_in_calculate_size(Nn, nx, nu, nc, nk) >= c_ptr);
    // printf("memory starts at\t%p\nmemory ends at  \t%p\ndistance from the end\t%lu bytes\n",
    //     ptr, c_ptr, (char *)ptr + tree_qp_in_calculate_size(Nn, nx, nu, nc,nk) - c_ptr);
    // exit(1);
}



int tree_qp_out_calculate_size(const int Nn, const int * const nx, const int * const nu, const int * const nc)
{
    int bytes = 0;

    bytes += 5*Nn*sizeof(struct blasfeo_dvec);  // x, u, mu_x, mu_u, mu_d
    bytes += (Nn-1)*sizeof(struct blasfeo_dvec);  // lam

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

        bytes += 2*blasfeo_memsize_dvec(nx[idx]);  // x, mu_x
        bytes += 2*blasfeo_memsize_dvec(nu[idx]);  // u, mu_u
        bytes += 1*blasfeo_memsize_dvec(nc_);  // mu_d

        if (idx > 0)
        {
            bytes += 1*blasfeo_memsize_dvec(nx[idx]);  // lam
        }
    }

    make_int_multiple_of(64, &bytes);
    bytes += 1*64;

    return bytes;
}



void tree_qp_out_create(const int Nn, const int * const nx, const int * const nu,
    const int * const nc, tree_qp_out * const qp_out, void *ptr)
{
    // char pointer
    char *c_ptr = (char *) ptr;

    qp_out->x = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    qp_out->u = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    qp_out->mu_x = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    qp_out->mu_u = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    qp_out->mu_d = (struct blasfeo_dvec *) c_ptr;
    c_ptr += Nn*sizeof(struct blasfeo_dvec);
    qp_out->lam = (struct blasfeo_dvec *) c_ptr;
    c_ptr += (Nn-1)*sizeof(struct blasfeo_dvec);

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
        init_strvec(nx[kk], &qp_out->mu_x[kk], &c_ptr);
        init_strvec(nu[kk], &qp_out->mu_u[kk], &c_ptr);
        init_strvec(nc_, &qp_out->mu_d[kk], &c_ptr);

        if (kk > 0)
        {
            init_strvec(nx[kk], &qp_out->lam[kk-1], &c_ptr);
        }
    }

    qp_out->info.Nn = Nn;

    assert((char *)ptr + tree_qp_out_calculate_size(Nn, nx, nu, nc) >= c_ptr);
    // printf("memory starts at\t%p\nmemory ends at  \t%p\ndistance from the end\t%lu bytes\n",
    //     ptr, c_ptr, (char *)ptr + tree_qp_out_calculate_size(Nn, nx, nu) - c_ptr);
    // exit(1);
}



void tree_qp_in_eliminate_x0(tree_qp_in * const qp_in)
{
    int nx0 = qp_in->nx[0];
    int nc0 = qp_in->nc[0];

    if (nx0 == 0)
    {
        return;
    }

    struct node *tree = qp_in->tree;

    struct blasfeo_dmat *sA;
    struct blasfeo_dvec *sb;
    struct blasfeo_dmat *sA0;
    struct blasfeo_dvec *sb0;

    struct blasfeo_dmat *sC = &qp_in->C[0];
    struct blasfeo_dmat *sC0 = &qp_in->internal_memory.C0;
    struct blasfeo_dvec *sdmin = &qp_in->dmin[0];
    struct blasfeo_dvec *sdmax = &qp_in->dmax[0];
    struct blasfeo_dvec *sdmin0 = &qp_in->internal_memory.dmin0;
    struct blasfeo_dvec *sdmax0 = &qp_in->internal_memory.dmax0;

    struct blasfeo_dmat *sS = &qp_in->S[0];
    struct blasfeo_dmat *sS0 = &qp_in->internal_memory.S0;
    struct blasfeo_dvec *sr = &qp_in->r[0];
    struct blasfeo_dvec *sr0 = &qp_in->internal_memory.r0;

    // copy data to internal memory (to always be able to update x0)

    if (qp_in->internal_memory.is_C_initialized == 0 && nc0 > 0)
    {
        blasfeo_dgecp(sC->m, sC->n, sC, 0, 0, sC0, 0, 0);
        assert(sC0->m == sC->m);
        assert(sC0->n == sC->n);

        qp_in->internal_memory.is_C_initialized = 1;

    }

    if (qp_in->internal_memory.is_dmin_initialized == 0 && nc0 > 0)
    {
        blasfeo_dveccp(sdmin->m, sdmin, 0, sdmin0, 0);
        assert(sdmin0->m == sdmin->m);
    }

    if (qp_in->internal_memory.is_dmax_initialized == 0 && nc0 > 0)
    {
        blasfeo_dveccp(sdmax->m, sdmax, 0, sdmax0, 0);
        assert(sdmax0->m == sdmax->m);
    }

    sC->pA = NULL;
    sC->n = 0;

    if (qp_in->internal_memory.is_S_initialized == 0)
    {
        blasfeo_dgecp(sS->m, sS->n, sS, 0, 0, sS0, 0, 0);
        assert(sS0->m == sS->m);
        assert(sS0->n == sS->n);

        qp_in->internal_memory.is_S_initialized = 1;
    }

    if (qp_in->internal_memory.is_r_initialized == 0)
    {
        blasfeo_dveccp(sr->m, sr, 0, sr0, 0);
        assert(sr0->m == sr->m);

        qp_in->internal_memory.is_r_initialized = 1;
    }

    sS->pA = NULL;
    sS->n = 0;

    for (int ii = 0; ii < tree[0].nkids; ii++)
    {
        sA = &qp_in->A[ii];
        sb = &qp_in->b[ii];

        sA0 = &qp_in->internal_memory.A0[ii];
        sb0 = &qp_in->internal_memory.b0[ii];

        if (qp_in->internal_memory.is_A_initialized[ii] == 0)
        {
            blasfeo_dgecp(sA->m, sA->n, sA, 0, 0, sA0, 0, 0);

            qp_in->internal_memory.is_A_initialized[ii] = 1;

            assert(sA0->m == sA->m);
            assert(sA0->n == sA->n);
        }
        if (qp_in->internal_memory.is_b_initialized[ii] == 0)
        {
            blasfeo_dveccp(sb->m, sb, 0, sb0, 0);

            qp_in->internal_memory.is_b_initialized[ii] = 1;

            assert(sb0->m == sb->m);
        }
        sA->n = 0;
        sA->pA = NULL;
    }

    assert(check_error_strvec(&qp_in->xmin[0], &qp_in->xmax[0]) < 1e-10);

    qp_in->nx[0] = 0;

    tree_qp_in_set_x0_strvec(qp_in, &qp_in->xmin[0]);

    qp_in->Q[0].pA = NULL;
    qp_in->Q[0].m = 0;
    qp_in->Q[0].n = 0;
    qp_in->q[0].pa = NULL;
    qp_in->q[0].m = 0;

    qp_in->xmin[0].pa = NULL;
    qp_in->xmin[0].m = 0;
    qp_in->xmax[0].pa = NULL;
    qp_in->xmax[0].m = 0;
}



void tree_qp_out_eliminate_x0(tree_qp_out * const qp_out)
{
    qp_out->x[0].pa = NULL;
    qp_out->x[0].m = 0;

    qp_out->mu_x[0].pa = NULL;
    qp_out->mu_x[0].m = 0;
}



void tree_qp_out_calculate_KKT_res(const tree_qp_in * const qp_in,
    const tree_qp_out * const qp_out, double *res)
{
    int Nn = qp_in->N;
    int nz = total_number_of_primal_variables(qp_in);
    int ne = total_number_of_dynamic_constraints(qp_in);
    int ng = total_number_of_general_constraints(qp_in);
    int nKKT = 3*nz + ne + 2*ng;

    for (int ii = 0; ii < nKKT; ii++)
    {
        res[ii] = 1e12;
    }

    int *nx = (int *)qp_in->nx;
    int *nu = (int *)qp_in->nu;
    int *nc = (int *)qp_in->nc;

    struct blasfeo_dvec *sx = qp_out->x;
    struct blasfeo_dvec *su = qp_out->u;
    struct blasfeo_dmat *sQ = qp_in->Q;
    struct blasfeo_dmat *sR = qp_in->R;
    struct blasfeo_dmat *sS = qp_in->S;
    struct blasfeo_dvec *sq = qp_in->q;
    struct blasfeo_dvec *sr = qp_in->r;
    struct blasfeo_dmat *sA = qp_in->A;
    struct blasfeo_dmat *sB = qp_in->B;
    struct blasfeo_dmat *sC = qp_in->C;
    struct blasfeo_dmat *sD = qp_in->D;

    struct node *tree = qp_in->tree;

    struct blasfeo_dvec tmp_x, tmp_u, tmp_g;

    int idxkid;
    int idxdad;
    int pos = 0;

    double mu;

    int dimx = max_number_of_states(qp_in);
    int dimu = max_number_of_controls(qp_in);
    int dimg = max_number_of_general_constraints(qp_in);
    blasfeo_allocate_dvec(dimx, &tmp_x);
    blasfeo_allocate_dvec(dimu, &tmp_u);
    blasfeo_allocate_dvec(dimg, &tmp_g);

    for (int ii = 0; ii < Nn; ii++)
    {
        // --- stationarity (nz x 1)

        // tmp_x = Q[ii]*x[ii] + q[ii]
        blasfeo_dgemv_n(nx[ii], nx[ii], 1.0, &sQ[ii], 0, 0, &sx[ii], 0, 1.0, &sq[ii], 0, &tmp_x, 0);
        // tmp_x += S[ii]'*u[ii]
        blasfeo_dgemv_t(nu[ii], nx[ii], 1.0, &sS[ii], 0, 0, &su[ii], 0, 1.0, &tmp_x, 0, &tmp_x, 0);
        // tmp_x += mu_x[ii]
        blasfeo_daxpy(nx[ii], 1.0, &qp_out->mu_x[ii], 0, &tmp_x, 0, &tmp_x, 0);
        // tmp_x += C[ii]'*mu_d[ii]
        blasfeo_dgemv_t(nc[ii], nx[ii], 1.0, &sC[ii], 0, 0, &qp_out->mu_d[ii], 0, 1.0, &tmp_x, 0, &tmp_x, 0);
        if (ii > 0)
        {
            // tmp_x += lam[ii]
            blasfeo_daxpy(nx[ii], -1.0, &qp_out->lam[ii-1], 0, &tmp_x, 0, &tmp_x, 0);
        }
        // tmp_u = R[ii]*u[ii] + r[ii]
        blasfeo_dgemv_n(nu[ii], nu[ii], 1.0, &sR[ii], 0, 0, &su[ii], 0, 1.0, &sr[ii], 0, &tmp_u, 0);
        // tmp_u += S[ii]*x[ii]
        blasfeo_dgemv_n(nu[ii], nx[ii], 1.0, &sS[ii], 0, 0, &sx[ii], 0, 1.0, &tmp_u, 0, &tmp_u, 0);
        // tmp_u += mu_u[ii]
        blasfeo_daxpy(nu[ii], 1.0, &qp_out->mu_u[ii], 0, &tmp_u, 0, &tmp_u, 0);
        // tmp_u += D[ii]'*mu_d[ii]
        blasfeo_dgemv_t(nc[ii], nu[ii], 1.0, &sD[ii], 0, 0, &qp_out->mu_d[ii], 0, 1.0, &tmp_u, 0, &tmp_u, 0);

        for (int jj = 0; jj < tree[ii].nkids; jj++)
        {
            idxkid = tree[ii].kids[jj];
            // tmp_x -= A[s(ii)]' * lam[s(ii)]
            blasfeo_dgemv_t(nx[idxkid], nx[ii], 1.0, &sA[idxkid-1], 0, 0, &qp_out->lam[idxkid-1], 0, 1.0, &tmp_x, 0, &tmp_x, 0);
            // tmp_u -= B[s(ii)]' * lam[s(ii)]
            blasfeo_dgemv_t(nx[idxkid], nu[ii], 1.0, &sB[idxkid-1], 0, 0, &qp_out->lam[idxkid-1], 0, 1.0, &tmp_u, 0, &tmp_u, 0);
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

            // tmp_x += B[ii-1]*u[p(iii)]
            blasfeo_dgemv_n(nx[ii], nu[idxdad], 1.0, &qp_in->B[ii-1], 0, 0,
                &qp_out->u[idxdad], 0, 1.0, &tmp_x, 0, &tmp_x, 0);

            // tmp_x -= x[idx]
            blasfeo_daxpy(nx[ii], -1.0, &qp_out->x[ii], 0, &tmp_x, 0, &tmp_x, 0);

            blasfeo_unpack_dvec(nx[ii], &tmp_x, 0, &res[pos]);
            pos += nx[ii];
        }


        // --- primal feasibility (bounds, nz x 1)

        for (int jj = 0; jj < nx[ii]; jj++)
        {
            if (BLASFEO_DVECEL(&qp_out->x[ii], jj) > BLASFEO_DVECEL(&qp_in->xmax[ii], jj))
            {
                res[pos+jj] = BLASFEO_DVECEL(&qp_out->x[ii], jj) - BLASFEO_DVECEL(&qp_in->xmax[ii], jj);
            }
            else if (BLASFEO_DVECEL(&qp_out->x[ii], jj) < BLASFEO_DVECEL(&qp_in->xmin[ii], jj))
            {
                res[pos+jj] = BLASFEO_DVECEL(&qp_in->xmin[ii], jj) - BLASFEO_DVECEL(&qp_out->x[ii], jj);
            }
            else
            {
                res[pos+jj] = 0.0;
            }
        }
        pos += nx[ii];

        for (int jj = 0; jj < nu[ii]; jj++)
        {
            if (BLASFEO_DVECEL(&qp_out->u[ii], jj) > BLASFEO_DVECEL(&qp_in->umax[ii], jj))
            {
                res[pos+jj] = BLASFEO_DVECEL(&qp_out->u[ii], jj) - BLASFEO_DVECEL(&qp_in->umax[ii], jj);
            }
            else if (BLASFEO_DVECEL(&qp_out->u[ii], jj) < BLASFEO_DVECEL(&qp_in->umin[ii], jj))
            {
                res[pos+jj] = BLASFEO_DVECEL(&qp_in->umin[ii], jj) - BLASFEO_DVECEL(&qp_out->u[ii], jj);
            }
            else
            {
                res[pos+jj] = 0.0;
            }
        }
        pos += nu[ii];


        // --- complementarity (bounds, nz x 1)

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
        pos += nx[ii];

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
        pos += nu[ii];


        // --- primal feasibility (general constraints, ng x 1)

        // tmp_g = C[ii]*x[ii]
        blasfeo_dgemv_n(nc[ii], nx[ii], 1.0, &qp_in->C[ii], 0, 0, &qp_out->x[ii], 0, 0.0, &tmp_g, 0, &tmp_g, 0);

        // tmp_g += D[ii]*u[ii]
        blasfeo_dgemv_n(nc[ii], nu[ii], 1.0, &qp_in->D[ii], 0, 0, &qp_out->u[ii], 0, 1.0, &tmp_g, 0, &tmp_g, 0);

        for (int jj = 0; jj < nc[ii]; jj++)
        {
            if (BLASFEO_DVECEL(&tmp_g, jj) > BLASFEO_DVECEL(&qp_in->dmax[ii], jj))
            {
                res[pos+jj] = BLASFEO_DVECEL(&tmp_g, jj) - BLASFEO_DVECEL(&qp_in->dmax[ii], jj);
            }
            else if (BLASFEO_DVECEL(&tmp_g, jj) < BLASFEO_DVECEL(&qp_in->dmin[ii], jj))
            {
                res[pos+jj] = BLASFEO_DVECEL(&qp_in->dmin[ii], jj) - BLASFEO_DVECEL(&tmp_g, jj);
            }
            else
            {
                res[pos+jj] = 0.0;
            }
        }
        pos += nc[ii];

        // --- complementarity (general constraints, ng x 1)

        for (int jj = 0; jj < nc[ii]; jj++)
        {
            mu = BLASFEO_DVECEL(&qp_out->mu_d[ii], jj);
            if ( mu > 0)
            {
                res[pos+jj] = mu*(BLASFEO_DVECEL(&tmp_g, jj) - BLASFEO_DVECEL(&qp_in->dmax[ii], jj));
            }
            else
            {
                res[pos+jj] = mu*(-BLASFEO_DVECEL(&tmp_g, jj) + BLASFEO_DVECEL(&qp_in->dmin[ii], jj));
            }
        }
        pos += nc[ii];
    }

    blasfeo_free_dvec(&tmp_x);
    blasfeo_free_dvec(&tmp_u);
    blasfeo_free_dvec(&tmp_g);

    assert(nKKT == pos && "incorrect size of KKT residuals");
    // d_print_e_mat(nKKT, 1, res, 1);
}



double tree_qp_out_max_KKT_res(const tree_qp_in * const qp_in, const tree_qp_out * const qp_out)
{
    int nz = total_number_of_primal_variables(qp_in);
    int ne = total_number_of_dynamic_constraints(qp_in);
    int ng = total_number_of_general_constraints(qp_in);
    int nKKT = 3*nz + ne + 2*ng;

    double *res = malloc(nKKT*sizeof(double));
    tree_qp_out_calculate_KKT_res(qp_in, qp_out, res);

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



int total_number_of_states(const tree_qp_in * const qp_in)
{
    int nx = 0;

    for (int ii = 0; ii < qp_in->N; ii++) nx += qp_in->nx[ii];

    return nx;
}



int max_number_of_states(const tree_qp_in * const qp_in)
{
    int nx_max = 0;

    for (int ii = 0; ii < qp_in->N; ii++) nx_max = MAX(nx_max, qp_in->nx[ii]);

    return nx_max;
}



int total_number_of_controls(const tree_qp_in * const qp_in)
{
    int nu = 0;

    for (int ii = 0; ii < qp_in->N; ii++) nu += qp_in->nu[ii];

    return nu;
}



int max_number_of_controls(const tree_qp_in * const qp_in)
{
    int nu_max = 0;

    for (int ii = 0; ii < qp_in->N; ii++) nu_max = MAX(nu_max, qp_in->nu[ii]);

    return nu_max;
}



int total_number_of_primal_variables(const tree_qp_in * const qp_in)
{
    return total_number_of_controls(qp_in) + total_number_of_states(qp_in);
}



int total_number_of_dynamic_constraints(const tree_qp_in * const qp_in)
{
    int ne = 0;

    for (int ii = 1; ii < qp_in->N; ii++) ne += qp_in->nx[ii];
    return ne;
}



int total_number_of_general_constraints(const tree_qp_in * const qp_in)
{
    int ng = 0;

    for (int ii = 0; ii < qp_in->N; ii++) ng += qp_in->nc[ii];
    return ng;
}



int max_number_of_general_constraints(const tree_qp_in * const qp_in)
{
    int ng_max = 0;

    for (int ii = 0; ii < qp_in->N; ii++) ng_max = MAX(ng_max, qp_in->nc[ii]);

    return ng_max;
}



void tree_qp_in_set_edge_A_colmajor(const double * const A, const int lda, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;
    int lda_mod;

    assert(indx >= 0);
    assert(indx < Nn-1);

    int node_indx = indx + 1;

    struct node * const tree = qp_in->tree;

    int nxp = qp_in->nx[tree[node_indx].dad];
    int nx = qp_in->nx[node_indx];

    if (lda <= 0)  // infer lda
    {
        lda_mod = nx;
    }
    else  // use lda from user (for padded matrices or submatrices)
    {
        lda_mod = lda;
    }

    struct blasfeo_dmat *sA = &qp_in->A[indx];

    blasfeo_pack_dmat(nx, nxp, (double *)A, lda_mod, sA, 0, 0);

    assert(sA->m == nx);
    assert(sA->n == nxp);

    if (tree[node_indx].dad == 0 && nxp > 0)  // cannot initialize internal_memory if x0 is eliminated
    {
        struct blasfeo_dmat *sA0 = &qp_in->internal_memory.A0[indx];

        nxp = sA0->n;

        blasfeo_dgecp(nx, nxp, sA, 0, 0, sA0, 0, 0);
        qp_in->internal_memory.is_A_initialized[indx] = 1;

        assert(sA0->m == nx);
        assert(sA0->n == nxp);
    }
}



void tree_qp_in_get_edge_A_colmajor(double * const A, const int lda, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;
    int lda_mod;

    assert(indx >= 0);
    assert(indx < Nn-1);

    int node_indx = indx + 1;

    struct node * const tree = qp_in->tree;

    int nxp = qp_in->nx[tree[node_indx].dad];
    int nx = qp_in->nx[node_indx];

    if (lda <= 0)  // infer lda
    {
        lda_mod = nx;
    }
    else  // use lda from user (for padded matrices or submatrices)
    {
        lda_mod = lda;
    }

    struct blasfeo_dmat *sA = &qp_in->A[indx];

    blasfeo_unpack_dmat(nx, nxp, sA, 0, 0, A, lda_mod);
}



void tree_qp_in_set_edge_B_colmajor(const double * const B, const int lda, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;
    int lda_mod;

    assert(indx >= 0);
    assert(indx < Nn-1);

    int node_indx = indx + 1;

    struct node * const tree = qp_in->tree;

    int nup = qp_in->nu[tree[node_indx].dad];
    int nx = qp_in->nx[node_indx];

    if (lda <= 0)  // infer lda
    {
        lda_mod = nx;
    }
    else  // use lda from user (for padded matrices or submatrices)
    {
        lda_mod = lda;
    }

    struct blasfeo_dmat *sB = &qp_in->B[indx];

    blasfeo_pack_dmat(nx, nup, (double *)B, lda_mod, sB, 0, 0);

    assert(sB->m == nx);
    assert(sB->n == nup);
}



void tree_qp_in_get_edge_B_colmajor(double * const B, const int lda, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;
    int lda_mod;

    assert(indx >= 0);
    assert(indx < Nn-1);

    int node_indx = indx + 1;

    struct node * const tree = qp_in->tree;

    int nup = qp_in->nu[tree[node_indx].dad];
    int nx = qp_in->nx[node_indx];

    if (lda <= 0)  // infer lda
    {
        lda_mod = nx;
    }
    else  // use lda from user (for padded matrices or submatrices)
    {
        lda_mod = lda;
    }

    struct blasfeo_dmat *sB = &qp_in->B[indx];

    blasfeo_unpack_dmat(nx, nup, sB, 0, 0, B, lda_mod);
}



void tree_qp_in_set_edge_b(const double * const b, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn-1);

    int node_indx = indx + 1;

    struct node * const tree = qp_in->tree;

    int nxp = qp_in->nx[tree[node_indx].dad];
    int nup = qp_in->nu[tree[node_indx].dad];
    int nx = qp_in->nx[node_indx];

    struct blasfeo_dvec *sb = &qp_in->b[indx];

    blasfeo_pack_dvec(nx, (double *)b, sb, 0);

    assert(sb->m == nx);

    if (tree[node_indx].dad == 0 && nxp > 0)
    {
        struct blasfeo_dvec *sb0 = &qp_in->internal_memory.b0[indx];

        blasfeo_dveccp(nx, sb, 0, sb0, 0);
        qp_in->internal_memory.is_b_initialized[indx] = 1;

        assert(sb0->m == nx);
    }
}



void tree_qp_in_get_edge_b(double * const b, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn-1);

    int node_indx = indx + 1;

    struct node * const tree = qp_in->tree;

    int nxp = qp_in->nx[tree[node_indx].dad];
    int nup = qp_in->nu[tree[node_indx].dad];
    int nx = qp_in->nx[node_indx];

    struct blasfeo_dvec *sb = &qp_in->b[indx];

    blasfeo_unpack_dvec(nx, sb, 0, b);
}



void tree_qp_in_set_edge_dynamics_colmajor(const double * const A, const double * const B,
    const double * const b, tree_qp_in * const qp_in, const int indx)
{
    tree_qp_in_set_edge_A_colmajor(A, -1, qp_in, indx);
    tree_qp_in_set_edge_B_colmajor(B, -1, qp_in, indx);
    tree_qp_in_set_edge_b(b, qp_in, indx);
}



void tree_qp_in_get_edge_dynamics_colmajor(double * const A, double * const B,
    double * const b, const tree_qp_in * const qp_in, const int indx)
{
    tree_qp_in_get_edge_A_colmajor(A, -1, qp_in, indx);
    tree_qp_in_get_edge_B_colmajor(B, -1, qp_in, indx);
    tree_qp_in_get_edge_b(b, qp_in, indx);
}



void tree_qp_in_set_node_Q_colmajor(const double * const Q, const int lda, tree_qp_in * const qp_in, const int indx)
{
    // TODO(dimitris): assert is_Q_symmetric, is_Q_pos_def

    int Nn = qp_in->N;
    int lda_mod;

    assert(indx >= 0);
    assert(indx < Nn);

    int nx = qp_in->nx[indx];

    if (lda <= 0)  // infer lda
    {
        lda_mod = nx;
    }
    else  // use lda from user (for padded matrices or submatrices)
    {
        lda_mod = lda;
    }

    struct blasfeo_dmat *sQ = &qp_in->Q[indx];

    blasfeo_pack_dmat(nx, nx, (double *)Q, lda_mod, sQ, 0, 0);

    assert(sQ->m == nx);
    assert(sQ->n == nx);
}



void tree_qp_in_get_node_Q_colmajor(double * const Q, const int lda, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;
    int lda_mod;

    assert(indx >= 0);
    assert(indx < Nn);

    int nx = qp_in->nx[indx];

    if (lda <= 0)  // infer lda
    {
        lda_mod = nx;
    }
    else  // use lda from user (for padded matrices or submatrices)
    {
        lda_mod = lda;
    }

    struct blasfeo_dmat *sQ = &qp_in->Q[indx];

    blasfeo_unpack_dmat(nx, nx, sQ, 0, 0, Q, lda_mod);
}



void tree_qp_in_set_node_R_colmajor(const double * const R, const int lda, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;
    int lda_mod;

    assert(indx >= 0);
    assert(indx < Nn);

    int nu = qp_in->nu[indx];

    if (lda <= 0)  // infer lda
    {
        lda_mod = nu;
    }
    else  // use lda from user (for padded matrices or submatrices)
    {
        lda_mod = lda;
    }

    struct blasfeo_dmat *sR = &qp_in->R[indx];

    blasfeo_pack_dmat(nu, nu, (double *)R, lda_mod, sR, 0, 0);

    assert(sR->m == nu);
    assert(sR->n == nu);
}



void tree_qp_in_get_node_R_colmajor(double * const R, const int lda, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;
    int lda_mod;

    assert(indx >= 0);
    assert(indx < Nn);

    int nu = qp_in->nu[indx];

    if (lda <= 0)  // infer lda
    {
        lda_mod = nu;
    }
    else  // use lda from user (for padded matrices or submatrices)
    {
        lda_mod = lda;
    }

    struct blasfeo_dmat *sR = &qp_in->R[indx];

    blasfeo_unpack_dmat(nu, nu, sR, 0, 0, R, lda_mod);
}



void tree_qp_in_set_node_S_colmajor(const double * const S, const int lda, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;
    int lda_mod;

    assert(indx >= 0);
    assert(indx < Nn);

    int nx = qp_in->nx[indx];
    int nu = qp_in->nu[indx];

    if (lda <= 0)  // infer lda
    {
        lda_mod = nu;
    }
    else  // use lda from user (for padded matrices or submatrices)
    {
        lda_mod = lda;
    }

    struct blasfeo_dmat *sS = &qp_in->S[indx];

    blasfeo_pack_dmat(nu, nx, (double *)S, lda_mod, sS, 0, 0);

    assert(sS->m == nu);
    assert(sS->n == nx);

    if (indx == 0 && nx > 0)
    {
        struct blasfeo_dmat *sS0 = &qp_in->internal_memory.S0;

        blasfeo_dgecp(nu, nx, sS, 0, 0, sS0, 0, 0);
        qp_in->internal_memory.is_S_initialized = 1;

        assert(sS0->m == nu);
        assert(sS0->n == nx);
    }
}



void tree_qp_in_get_node_S_colmajor(double * const S, const int lda, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;
    int lda_mod;

    assert(indx >= 0);
    assert(indx < Nn);

    int nx = qp_in->nx[indx];
    int nu = qp_in->nu[indx];

    if (lda <= 0)  // infer lda
    {
        lda_mod = nu;
    }
    else  // use lda from user (for padded matrices or submatrices)
    {
        lda_mod = lda;
    }

    struct blasfeo_dmat *sS = &qp_in->S[indx];

    blasfeo_unpack_dmat(nu, nx, sS, 0, 0, S, lda_mod);
}



void tree_qp_in_set_node_q(const double * const q, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nx = qp_in->nx[indx];

    struct blasfeo_dvec *sq = &qp_in->q[indx];

    blasfeo_pack_dvec(nx, (double *)q, sq, 0);

    assert(sq->m == nx);
}



void tree_qp_in_get_node_q(double * const q, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nx = qp_in->nx[indx];

    struct blasfeo_dvec *sq = &qp_in->q[indx];

    blasfeo_unpack_dvec(nx, sq, 0, q);
}



void tree_qp_in_set_node_r(const double * const r, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nu = qp_in->nu[indx];
    int nx = qp_in->nx[indx];

    struct blasfeo_dvec *sr = &qp_in->r[indx];

    blasfeo_pack_dvec(nu, (double *)r, sr, 0);

    assert(sr->m == nu);

    if (indx == 0 && nx > 0)
    {
        struct blasfeo_dvec *sr0 = &qp_in->internal_memory.r0;

        qp_in->internal_memory.is_r_initialized = 1;

        blasfeo_dveccp(nu, sr, 0, sr0, 0);
        assert(sr0->m == nu);
    }
}



void tree_qp_in_get_node_r(double * const r, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nu = qp_in->nu[indx];
    int nx = qp_in->nx[indx];

    struct blasfeo_dvec *sr = &qp_in->r[indx];

    blasfeo_unpack_dvec(nu, sr, 0, r);
}



void tree_qp_in_set_node_objective_colmajor(const double * const Q, const double * const R, const double * const S,
    const double * const q, const double * const r, tree_qp_in * const qp_in, const int indx)
{
    tree_qp_in_set_node_Q_colmajor(Q, -1, qp_in, indx);
    tree_qp_in_set_node_R_colmajor(R, -1, qp_in, indx);
    tree_qp_in_set_node_S_colmajor(S, -1, qp_in, indx);
    tree_qp_in_set_node_q(q, qp_in, indx);
    tree_qp_in_set_node_r(r, qp_in, indx);
}



void tree_qp_in_get_node_objective_colmajor(double * const Q, double * const R, double * const S,
    double * const q, double * const r, const tree_qp_in * const qp_in, const int indx)
{
    tree_qp_in_get_node_Q_colmajor(Q, -1, qp_in, indx);
    tree_qp_in_get_node_R_colmajor(R, -1, qp_in, indx);
    tree_qp_in_get_node_S_colmajor(S, -1, qp_in, indx);
    tree_qp_in_get_node_q(q, qp_in, indx);
    tree_qp_in_get_node_r(r, qp_in, indx);
}



void tree_qp_in_set_node_objective_diag(const double * const Qd, const double * const Rd,
    const double * const q, const double * const r, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nx = qp_in->nx[indx];
    int nu = qp_in->nu[indx];

    struct blasfeo_dmat *sQ = &qp_in->Q[indx];
    struct blasfeo_dmat *sR = &qp_in->R[indx];
    struct blasfeo_dmat *sS = &qp_in->S[indx];
    struct blasfeo_dvec *sq = &qp_in->q[indx];
    struct blasfeo_dvec *sr = &qp_in->r[indx];

    // set cross-term to zero
    if (nx > 0 && nu > 0)
    {
        blasfeo_dgese(nu, nx, 0.0, sS, 0, 0);
        assert(sS->m == nu);
        assert(sS->n == nx);
    }

    // temporarily pack Qd to q and then copy it to the diagonal of Q
    if (nx > 0)
    {
        blasfeo_dgese(nx, nx, 0.0, sQ, 0, 0);
        blasfeo_pack_dvec(nx, (double *)Qd, sq, 0);
        blasfeo_ddiain(nx, 1.0, sq, 0, sQ, 0, 0);

        blasfeo_pack_dvec(nx, (double *)q, sq, 0);

        assert(sQ->m == nx);
        assert(sQ->n == nx);
        assert(sq->m == nx);
    }

    // do the same with Rd
    if (nu > 0)
    {
        blasfeo_dgese(nu, nu, 0.0, sR, 0, 0);
        blasfeo_pack_dvec(nu, (double *)Rd, sr, 0);
        blasfeo_ddiain(nu, 1.0, sr, 0, sR, 0, 0);

        blasfeo_pack_dvec(nu, (double *)r, sr, 0);

        assert(sR->m == nu);
        assert(sR->n == nu);
        assert(sr->m == nu);
    }
    // TODO(dimitris): assert Q,R pos. semi-definite
}



void tree_qp_in_set_node_xmin(const double * const xmin, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nx = qp_in->nx[indx];

    struct blasfeo_dvec *sxmin = &qp_in->xmin[indx];

    blasfeo_pack_dvec(nx, (double *)xmin, sxmin, 0);

    assert(sxmin->m == nx);
}



void tree_qp_in_get_node_xmin(double * const xmin, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nx = qp_in->nx[indx];

    struct blasfeo_dvec *sxmin = &qp_in->xmin[indx];

    blasfeo_unpack_dvec(nx, sxmin, 0, xmin);
}



void tree_qp_in_set_node_xmax(const double * const xmax, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nx = qp_in->nx[indx];

    struct blasfeo_dvec *sxmax = &qp_in->xmax[indx];

    blasfeo_pack_dvec(nx, (double *)xmax, sxmax, 0);

    assert(sxmax->m == nx);
}



void tree_qp_in_get_node_xmax(double * const xmax, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nx = qp_in->nx[indx];

    struct blasfeo_dvec *sxmax = &qp_in->xmax[indx];

    blasfeo_unpack_dvec(nx, sxmax, 0, xmax);
}



void tree_qp_in_set_node_umin(const double * const umin, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nu = qp_in->nu[indx];

    struct blasfeo_dvec *sumin = &qp_in->umin[indx];

    blasfeo_pack_dvec(nu, (double *)umin, sumin, 0);

    assert(sumin->m == nu);
}



void tree_qp_in_get_node_umin(double * const umin, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nu = qp_in->nu[indx];

    struct blasfeo_dvec *sumin = &qp_in->umin[indx];

    blasfeo_unpack_dvec(nu, sumin, 0, umin);
}



void tree_qp_in_set_node_umax(const double * const umax, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nu = qp_in->nu[indx];

    struct blasfeo_dvec *sumax = &qp_in->umax[indx];

    blasfeo_pack_dvec(nu, (double *)umax, sumax, 0);

    assert(sumax->m == nu);
}



void tree_qp_in_get_node_umax(double * const umax, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nu = qp_in->nu[indx];

    struct blasfeo_dvec *sumax = &qp_in->umax[indx];

    blasfeo_unpack_dvec(nu, sumax, 0, umax);
}



void tree_qp_in_set_node_bounds(const double * const xmin, const double * const xmax,
    const double * const umin, const double * const umax, tree_qp_in * const qp_in, const int indx)
{
    tree_qp_in_set_node_xmin(xmin, qp_in, indx);
    tree_qp_in_set_node_xmax(xmax, qp_in, indx);
    tree_qp_in_set_node_umin(umin, qp_in, indx);
    tree_qp_in_set_node_umax(umax, qp_in, indx);

    // TODO(dimitris): assert lower bounds <= upper bounds
}



void tree_qp_in_get_node_bounds(double * const xmin, double * const xmax,
    double * const umin, double * const umax, const tree_qp_in * const qp_in, const int indx)
{
    tree_qp_in_get_node_xmin(xmin, qp_in, indx);
    tree_qp_in_get_node_xmax(xmax, qp_in, indx);
    tree_qp_in_get_node_umin(umin, qp_in, indx);
    tree_qp_in_get_node_umax(umax, qp_in, indx);
}



void tree_qp_in_set_node_C_colmajor(const double * const C, const int lda, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;
    int lda_mod;

    assert(indx >= 0);
    assert(indx < Nn);

    int nc = qp_in->nc[indx];
    int nx = qp_in->nx[indx];

    if (lda <= 0)  // infer lda
    {
        lda_mod = nc;
    }
    else  // use lda from user (for padded matrices or submatrices)
    {
        lda_mod = lda;
    }

    struct blasfeo_dmat *sC = &qp_in->C[indx];

    blasfeo_pack_dmat(nc, nx, (double *)C, lda_mod, sC, 0, 0);

    assert(sC->m == nc);
    assert(sC->n == nx);

    if (indx == 0 && nx > 0)
    {
        struct blasfeo_dmat *sC0 = &qp_in->internal_memory.C0;

        blasfeo_dgecp(nc, nx, sC, 0, 0, sC0, 0, 0);
        qp_in->internal_memory.is_C_initialized = 1;

        assert(sC0->m == nc);
        assert(sC0->n == nx);
    }
}



void tree_qp_in_get_node_C_colmajor(double * const C, const int lda, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;
    int lda_mod;

    assert(indx >= 0);
    assert(indx < Nn);

    int nc = qp_in->nc[indx];
    int nx = qp_in->nx[indx];

    if (lda <= 0)  // infer lda
    {
        lda_mod = nc;
    }
    else  // use lda from user (for padded matrices or submatrices)
    {
        lda_mod = lda;
    }

    struct blasfeo_dmat *sC = &qp_in->C[indx];

    blasfeo_unpack_dmat(nc, nx, sC, 0, 0, C, lda_mod);
}



void tree_qp_in_set_node_D_colmajor(const double * const D, const int lda, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;
    int lda_mod;

    assert(indx >= 0);
    assert(indx < Nn);

    int nc = qp_in->nc[indx];
    int nu = qp_in->nu[indx];

    if (lda <= 0)  // infer lda
    {
        lda_mod = nc;
    }
    else  // use lda from user (for padded matrices or submatrices)
    {
        lda_mod = lda;
    }

    struct blasfeo_dmat *sD = &qp_in->D[indx];

    blasfeo_pack_dmat(nc, nu, (double *)D, lda_mod, sD, 0, 0);

    assert(sD->m == nc);
    assert(sD->n == nu);

}



void tree_qp_in_get_node_D_colmajor(double * const D, const int lda, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;
    int lda_mod;

    assert(indx >= 0);
    assert(indx < Nn);

    int nc = qp_in->nc[indx];
    int nu = qp_in->nu[indx];

    if (lda <= 0)  // infer lda
    {
        lda_mod = nc;
    }
    else  // use lda from user (for padded matrices or submatrices)
    {
        lda_mod = lda;
    }

    struct blasfeo_dmat *sD = &qp_in->D[indx];

    blasfeo_unpack_dmat(nc, nu, sD, 0, 0, D, lda_mod);
}



void tree_qp_in_set_node_dmin(const double * const dmin, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nc = qp_in->nc[indx];
    int nx = qp_in->nx[indx];

    struct blasfeo_dvec *sdmin = &qp_in->dmin[indx];

    blasfeo_pack_dvec(nc, (double *)dmin, sdmin, 0);

    assert(sdmin->m == nc);

    if (indx == 0 && nx > 0)
    {
        struct blasfeo_dvec *sdmin0 = &qp_in->internal_memory.dmin0;

        qp_in->internal_memory.is_dmin_initialized = 1;

        blasfeo_dveccp(nc, sdmin, 0, sdmin0, 0);
        assert(sdmin0->m == nc);
    }
}



void tree_qp_in_get_node_dmin(double * const dmin, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nc = qp_in->nc[indx];
    int nx = qp_in->nx[indx];

    struct blasfeo_dvec *sdmin = &qp_in->dmin[indx];

    blasfeo_unpack_dvec(nc, sdmin, 0, dmin);
}



void tree_qp_in_set_node_dmax(const double * const dmax, tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nc = qp_in->nc[indx];
    int nx = qp_in->nx[indx];

    struct blasfeo_dvec *sdmax = &qp_in->dmax[indx];

    blasfeo_pack_dvec(nc, (double *)dmax, sdmax, 0);

    assert(sdmax->m == nc);

    if (indx == 0 && nx > 0)
    {
        struct blasfeo_dvec *sdmax0 = &qp_in->internal_memory.dmax0;

        qp_in->internal_memory.is_dmax_initialized = 1;

        blasfeo_dveccp(nc, sdmax, 0, sdmax0, 0);
        assert(sdmax0->m == nc);
    }
}



void tree_qp_in_get_node_dmax(double * const dmax, const tree_qp_in * const qp_in, const int indx)
{
    int Nn = qp_in->N;

    assert(indx >= 0);
    assert(indx < Nn);

    int nc = qp_in->nc[indx];
    int nx = qp_in->nx[indx];

    struct blasfeo_dvec *sdmax = &qp_in->dmax[indx];

    blasfeo_unpack_dvec(nc, sdmax, 0, dmax);
}



void tree_qp_in_set_node_general_constraints(const double * const C, const double * const D,
    const double * const dmin, const double * const dmax, tree_qp_in * const qp_in, const int indx)
{
    tree_qp_in_set_node_C_colmajor(C, -1, qp_in, indx);
    tree_qp_in_set_node_D_colmajor(D, -1, qp_in, indx);
    tree_qp_in_set_node_dmin(dmin, qp_in, indx);
    tree_qp_in_set_node_dmax(dmax, qp_in, indx);

    // TODO(dimitris): assert lower bounds <= upper bounds
}



void tree_qp_in_get_node_general_constraints(double * const C, double * const D,
    double * const dmin, double * const dmax, const tree_qp_in * const qp_in, const int indx)
{
    tree_qp_in_get_node_C_colmajor(C, -1, qp_in, indx);
    tree_qp_in_get_node_D_colmajor(D, -1, qp_in, indx);
    tree_qp_in_get_node_dmin(dmin, qp_in, indx);
    tree_qp_in_get_node_dmax(dmax, qp_in, indx);
}



// NOTE(dimitris): weights are scaled to minimize the average cost over all scenarios
void tree_qp_in_fill_lti_data_diag_weights(double *A, double *B, double *b,
    double *Q, double *q, double *P, double *p, double *R, double *r,
    double *xmin, double *xmax, double *umin, double *umax, double *x0,
    double *C, double *CN, double *D, double *dmin, double *dmax, tree_qp_in *qp_in)
{
    int Nn = qp_in->N;
    struct node *tree = qp_in->tree;
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

    nx = qp_in->nx[1];
    nu = qp_in->nu[0];

    assert(qp_in->nx[0] > 0 && "Use eliminate_x0 functions instead of passing nx[0] = 0 here!");

    // count number of leaves
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

    for (int ii = 0; ii < Nn; ii++)
    {
        nx = qp_in->nx[ii];
        nu = qp_in->nu[ii];
        nc = qp_in->nc[ii];

        if (ii > 0)
        {
            re = tree[ii].real;
            nxp = qp_in->nx[tree[ii].dad];
            nup = qp_in->nu[tree[ii].dad];

            tree_qp_in_set_edge_dynamics_colmajor(&A[re*nx*nxp], &B[re*nx*nup], &b[re*nx], qp_in, ii-1);
        }

        if (tree[ii].nkids > 0)
        {
            tree_qp_in_set_node_objective_diag(Q, R, q, r, qp_in, ii);
        }
        else
        {
            tree_qp_in_set_node_objective_diag(P, NULL, p, NULL, qp_in, ii);
        }

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

        if (ii == 0)
        {
            tree_qp_in_set_node_bounds(x0, x0, umin, umax, qp_in, ii);
        }
        else
        {
            tree_qp_in_set_node_bounds(xmin, xmax, umin, umax, qp_in, ii);
        }

        if (tree[ii].nkids > 0)
        {
            tree_qp_in_set_node_general_constraints(C, D, dmin, dmax, qp_in, ii);
        }
        else
        {
            // TODO: maybe we need dNmin, dNmax?
            tree_qp_in_set_node_general_constraints(CN, NULL, dmin, dmax, qp_in, ii);
        }
    }
}



void tree_qp_in_set_ltv_dynamics_colmajor(double *A, double *B, double *b, tree_qp_in *qp_in)
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
        #if 1
        tree_qp_in_set_edge_dynamics_colmajor(&A[idxA], &B[idxB], &b[idxb], qp_in, ii);
        idxA += sA[ii].m * sA[ii].n;
        idxB += sB[ii].m * sB[ii].n;
        idxb += sb[ii].m;
        #else
        blasfeo_pack_dmat(sA[ii].m, sA[ii].n, &A[idxA], sA[ii].m, &sA[ii], 0, 0);
        idxA += sA[ii].m * sA[ii].n;

        blasfeo_pack_dmat(sB[ii].m, sB[ii].n, &B[idxB], sB[ii].m, &sB[ii], 0, 0);
        idxB += sB[ii].m * sB[ii].n;

        blasfeo_pack_dvec(sb[ii].m, &b[idxb], &sb[ii], 0);
        idxb += sb[ii].m;
        #endif
    }
}



void tree_qp_in_set_ltv_objective_diag(double *Qd, double *Rd, double *q, double *r, tree_qp_in *qp_in)
{
    int Nn = qp_in->N;

    struct blasfeo_dmat *sQ = qp_in->Q;
    struct blasfeo_dmat *sR = qp_in->R;
    struct blasfeo_dmat *sS = qp_in->S;
    struct blasfeo_dvec *sq = qp_in->q;
    struct blasfeo_dvec *sr = qp_in->r;

    // struct blasfeo_dvec sQvec, sRvec;

    int idxQ = 0;
    int idxR = 0;

    for (int ii = 0; ii < Nn; ii++)
    {
        #if 1
        tree_qp_in_set_node_objective_diag(&Qd[idxQ], &Rd[idxR], &q[idxQ], &r[idxR], qp_in, ii);
        idxQ += sQ[ii].m;
        idxR += sR[ii].m;
        #else
        blasfeo_dgese(sQ[ii].m, sQ[ii].n, 0.0, &sQ[ii], 0, 0);
        blasfeo_create_dvec(sQ[ii].m, &sQvec, &Qd[idxQ]);
        blasfeo_ddiain(sQ[ii].m, 1.0, &sQvec, 0, &sQ[ii], 0, 0);
        blasfeo_pack_dvec(sq[ii].m, &q[idxQ], &sq[ii], 0);

        idxQ += sQ[ii].m;

        blasfeo_dgese(sS[ii].m, sS[ii].m, 0.0, &sS[ii], 0, 0);

        blasfeo_dgese(sR[ii].m, sR[ii].n, 0.0, &sR[ii], 0, 0);
        blasfeo_create_dvec(sR[ii].m, &sRvec, &Rd[idxR]);
        blasfeo_ddiain(sR[ii].m, 1.0, &sRvec, 0, &sR[ii], 0, 0);
        blasfeo_pack_dvec(sr[ii].m, &r[idxR], &sr[ii], 0);

        idxR += sR[ii].m;
        #endif
    }
}



void tree_qp_in_set_ltv_objective_colmajor(double *Q, double *R, double *S, double *q, double *r,
    tree_qp_in *qp_in)
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
        #if 1
        tree_qp_in_set_node_objective_colmajor(&Q[idxQ], &R[idxR], &S[idxS], &q[idxq], &r[idxr], qp_in, ii);
        idxQ += sQ[ii].m * sQ[ii].n;
        idxR += sR[ii].m * sR[ii].n;
        idxS += sS[ii].m * sS[ii].n;
        idxq += sq[ii].m;
        idxr += sr[ii].m;
        #else
        blasfeo_pack_dmat(sQ[ii].m, sQ[ii].n, &Q[idxQ], sQ[ii].m, &sQ[ii], 0, 0);
        idxQ += sQ[ii].m * sQ[ii].n;

        blasfeo_pack_dmat(sR[ii].m, sR[ii].n, &R[idxR], sR[ii].m, &sR[ii], 0, 0);
        idxR += sR[ii].m * sR[ii].n;

        blasfeo_pack_dmat(sS[ii].m, sS[ii].n, &S[idxS], sS[ii].m, &sS[ii], 0, 0);
        idxS += sS[ii].m * sS[ii].n;

        blasfeo_pack_dvec(sq[ii].m, &q[idxq], &sq[ii], 0);
        idxq += sq[ii].m;

        blasfeo_pack_dvec(sr[ii].m, &r[idxr], &sr[ii], 0);
        idxr += sr[ii].m;
        #endif
    }
}



void tree_qp_in_set_ltv_bounds(double *xmin, double *xmax, double *umin, double *umax,
    tree_qp_in *qp_in)
{
    int Nn = qp_in->N;

    int idxx = 0;
    int idxu = 0;

    struct blasfeo_dvec *sxmin = qp_in->xmin;
    struct blasfeo_dvec *sxmax = qp_in->xmax;
    struct blasfeo_dvec *sumin = qp_in->umin;
    struct blasfeo_dvec *sumax = qp_in->umax;

    for (int ii = 0; ii < Nn; ii++)
    {
        tree_qp_in_set_node_bounds(&xmin[idxx], &xmax[idxx], &umin[idxu], &umax[idxu], qp_in, ii);
        idxx += sxmin[ii].m;
        idxu += sumin[ii].m;
    }
    assert(idxx == total_number_of_states(qp_in));
    assert(idxu == total_number_of_controls(qp_in));
}



void tree_qp_in_set_const_bounds(double *xmin, double *xmax, double *umin, double *umax,
    tree_qp_in *qp_in)
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
        assert(qp_in->nu[ii] == nu || qp_in->nu[ii] == 0);

        tree_qp_in_set_node_bounds(xmin, xmax, umin, umax, qp_in, ii);
    }
}



void tree_qp_in_set_inf_bounds(tree_qp_in *qp_in)
{
    int Nn = qp_in->N;
    int *nx = qp_in->nx;
    int *nu = qp_in->nu;

    double inf = TREEQP_INF;

    struct blasfeo_dvec *sxmin = qp_in->xmin;
    struct blasfeo_dvec *sxmax = qp_in->xmax;
    struct blasfeo_dvec *sumin = qp_in->umin;
    struct blasfeo_dvec *sumax = qp_in->umax;

    for (int ii = 0; ii < Nn; ii++)
    {
        blasfeo_dvecse(sxmin[ii].m, -inf, &sxmin[ii], 0);
        blasfeo_dvecse(sxmax[ii].m, inf, &sxmax[ii], 0);
        assert(sxmin[ii].m == nx[ii]);
        assert(sxmax[ii].m == nx[ii]);

        blasfeo_dvecse(sumin[ii].m, -inf, &sumin[ii], 0);
        blasfeo_dvecse(sumax[ii].m, inf, &sumax[ii], 0);
        assert(sumin[ii].m == nu[ii]);
        assert(sumax[ii].m == nu[ii]);
    }
}



void tree_qp_in_set_x0_strvec(tree_qp_in *qp_in, struct blasfeo_dvec *sx0)
{
    int Nn = qp_in->N;
    int nx0 = qp_in->nx[0];

    struct node *tree = qp_in->tree;

    if (nx0 > 0)
    {
        struct blasfeo_dvec *sxmin0 = &qp_in->xmin[0];
        struct blasfeo_dvec *sxmax0 = &qp_in->xmax[0];

        blasfeo_dveccp(sx0->m, sx0, 0, sxmin0, 0);
        blasfeo_dveccp(sx0->m, sx0, 0, sxmax0, 0);

        assert(sxmin0->m == sx0->m);
        assert(sxmax0->m == sx0->m);
    }
    else
    {
        struct blasfeo_dvec *sx0_mem = &qp_in->internal_memory.x0;

        struct blasfeo_dmat *sA0;
        struct blasfeo_dvec *sb0;

        struct blasfeo_dmat *sC0 = &qp_in->internal_memory.C0;
        struct blasfeo_dvec *sdmin0 = &qp_in->internal_memory.dmin0;
        struct blasfeo_dvec *sdmax0  = &qp_in->internal_memory.dmax0;

        struct blasfeo_dmat *sS0 = &qp_in->internal_memory.S0;
        struct blasfeo_dvec *sr0 = &qp_in->internal_memory.r0;

        int nx0 = sx0->m;
        int nc0 = sC0->m;
        int nu0 = sS0->m;

        assert(sx0_mem->m == nx0);
        assert(qp_in->nu[0] == nu0);
        assert(qp_in->nc[0] == nc0);

        blasfeo_dveccp(sx0->m, sx0, 0, sx0_mem, 0);

        for (int ii = 1; ii <= tree[0].nkids; ii++)
        {
            sA0 = &qp_in->internal_memory.A0[ii-1];
            sb0 = &qp_in->internal_memory.b0[ii-1];

            assert(qp_in->internal_memory.is_A_initialized[ii-1] == 1);
            assert(qp_in->internal_memory.is_b_initialized[ii-1] == 1);

            blasfeo_dgemv_n(sA0->m, nx0, 1.0, sA0, 0, 0, sx0, 0, 1.0, sb0, 0, &qp_in->b[ii-1], 0);

            assert(sA0->m == sb0->m);
        }

        if (nc0 > 0)
        {
            assert(qp_in->internal_memory.is_C_initialized == 1);
            assert(qp_in->internal_memory.is_dmin_initialized == 1);
            assert(qp_in->internal_memory.is_dmax_initialized == 1);

            blasfeo_dgemv_n(nc0, nx0, -1.0, sC0, 0, 0, sx0, 0, 1.0, sdmin0, 0, &qp_in->dmin[0], 0);

            blasfeo_dgemv_n(nc0, nx0, -1.0, sC0, 0, 0, sx0, 0, 1.0, sdmax0, 0, &qp_in->dmax[0], 0);
        }

        assert(qp_in->internal_memory.is_S_initialized == 1);
        assert(qp_in->internal_memory.is_r_initialized == 1);

        blasfeo_dgemv_n(nu0, nx0, 1.0, sS0, 0, 0, sx0, 0, 1.0, sr0, 0, &qp_in->r[0], 0);
    }
}



void tree_qp_in_set_x0_colmaj(tree_qp_in *qp_in, double *x0)
{
    struct blasfeo_dvec *sx0 = &qp_in->internal_memory.x0;

    blasfeo_pack_dvec(sx0->m, x0, sx0, 0);
    tree_qp_in_set_x0_strvec(qp_in, sx0);
}



void tree_qp_out_set_node_x(const double * const x, tree_qp_out * const qp_out, const int indx)
{
    int Nn = qp_out->info.Nn;

    assert(indx >= 0);
    assert(indx < Nn);

    int nx = qp_out->x[indx].m;

    struct blasfeo_dvec *sx = &qp_out->x[indx];

    blasfeo_pack_dvec(nx, (double*)x, sx, 0);
}



void tree_qp_out_get_node_x(double * const x, const tree_qp_out * const qp_out, const int indx)
{
    int Nn = qp_out->info.Nn;  // TODO(dimitris): use the fact that Nn is stored here in other functions too

    assert(indx >= 0);
    assert(indx < Nn);

    int nx = qp_out->x[indx].m;

    struct blasfeo_dvec *sx = &qp_out->x[indx];

    blasfeo_unpack_dvec(nx, sx, 0, x);
}



void tree_qp_out_set_node_u(const double * const u, tree_qp_out * const qp_out, const int indx)
{
    int Nn = qp_out->info.Nn;

    assert(indx >= 0);
    assert(indx < Nn);

    int nu = qp_out->u[indx].m;

    struct blasfeo_dvec *su = &qp_out->u[indx];

    blasfeo_pack_dvec(nu, (double*)u, su, 0);
}



void tree_qp_out_get_node_u(double * const u, const tree_qp_out * const qp_out, const int indx)
{
    int Nn = qp_out->info.Nn;

    assert(indx >= 0);
    assert(indx < Nn);

    int nu = qp_out->u[indx].m;

    struct blasfeo_dvec *su = &qp_out->u[indx];

    blasfeo_unpack_dvec(nu, su, 0, u);
}



void tree_qp_out_set_edge_lam(const double * const lam, tree_qp_out * const qp_out, const int indx)
{
    int Nn = qp_out->info.Nn;

    assert(indx >= 0);
    assert(indx < Nn-1);

    int nx = qp_out->lam[indx].m;

    struct blasfeo_dvec *slam = &qp_out->lam[indx];

    blasfeo_pack_dvec(nx, (double*)lam, slam, 0);
}



void tree_qp_out_get_edge_lam(double * const lam, const tree_qp_out * const qp_out, const int indx)
{
    int Nn = qp_out->info.Nn;

    assert(indx >= 0);
    assert(indx < Nn-1);

    int nx = qp_out->lam[indx].m;

    struct blasfeo_dvec *slam = &qp_out->lam[indx];

    blasfeo_unpack_dvec(nx, slam, 0, lam);
}



void tree_qp_out_set_node_mu_u(const double * const mu_u, tree_qp_out * const qp_out, const int indx)
{
    int Nn = qp_out->info.Nn;

    assert(indx >= 0);
    assert(indx < Nn);

    int nu = qp_out->u[indx].m;

    struct blasfeo_dvec *smu_u = &qp_out->mu_u[indx];

    blasfeo_pack_dvec(nu, (double*)mu_u, smu_u, 0);
}



void tree_qp_out_get_node_mu_u(double * const mu_u, const tree_qp_out * const qp_out, const int indx)
{
    int Nn = qp_out->info.Nn;

    assert(indx >= 0);
    assert(indx < Nn);

    int nu = qp_out->u[indx].m;

    struct blasfeo_dvec *smu_u = &qp_out->mu_u[indx];

    blasfeo_unpack_dvec(nu, smu_u, 0, mu_u);
}



void tree_qp_out_set_node_mu_x(const double * const mu_x, tree_qp_out * const qp_out, const int indx)
{
    int Nn = qp_out->info.Nn;

    assert(indx >= 0);
    assert(indx < Nn);

    int nx = qp_out->x[indx].m;

    struct blasfeo_dvec *smu_x = &qp_out->mu_x[indx];

    blasfeo_pack_dvec(nx, (double*)mu_x, smu_x, 0);
}



void tree_qp_out_get_node_mu_x(double * const mu_x, const tree_qp_out * const qp_out, const int indx)
{
    int Nn = qp_out->info.Nn;

    assert(indx >= 0);
    assert(indx < Nn);

    int nx = qp_out->x[indx].m;

    struct blasfeo_dvec *smu_x = &qp_out->mu_x[indx];

    blasfeo_unpack_dvec(nx, smu_x, 0, mu_x);
}



void tree_qp_out_set_node_mu_d(const double * const mu_d, tree_qp_out * const qp_out, const int indx)
{
    int Nn = qp_out->info.Nn;

    assert(indx >= 0);
    assert(indx < Nn);

    int nc = qp_out->mu_d[indx].m;

    struct blasfeo_dvec *smu_d = &qp_out->mu_d[indx];

    blasfeo_pack_dvec(nc, (double *)mu_d, smu_d, 0);
}



void tree_qp_out_get_node_mu_d(double * const mu_d, const tree_qp_out * const qp_out, const int indx)
{
    int Nn = qp_out->info.Nn;

    assert(indx >= 0);
    assert(indx < Nn);

    int nc = qp_out->mu_d[indx].m;

    struct blasfeo_dvec *smu_d = &qp_out->mu_d[indx];

    blasfeo_unpack_dvec(nc, smu_d, 0, mu_d);
}
