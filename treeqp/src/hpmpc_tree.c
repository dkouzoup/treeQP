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
#include "treeqp/utils/memory.h"
#include "treeqp/utils/timing.h"
#include "treeqp/utils/types.h"

#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_blas.h>

#include "hpmpc/include/target.h"
#include "hpmpc/include/tree.h"
#include "hpmpc/include/mpc_solvers.h"



int treeqp_hpmpc_opts_calculate_size(int Nn)
{
    int bytes =  Nn*sizeof(int);  // scrap.nb
    return bytes;
}



void treeqp_hpmpc_opts_create(int Nn, treeqp_hpmpc_opts_t *opts, void *ptr)
{
    // char pointer
    char *c_ptr = (char *) ptr;

    opts->scrap.nb = (int *)c_ptr;
    c_ptr += Nn*sizeof(int);

    assert((char *)ptr + treeqp_hpmpc_opts_calculate_size(Nn) == c_ptr);
}



void treeqp_hpmpc_opts_set_default(treeqp_hpmpc_opts_t *opts)
{
    opts->maxIter = 20;
	opts->mu0 = 2.0;
	opts->mu_tol = 1e-12;
	opts->alpha_min = 1e-8;
	opts->warm_start = 0;
    opts->compute_mult = 1;
}



int number_of_bounds(const struct blasfeo_dvec *vmin, const struct blasfeo_dvec *vmax)
{
    int nb = 0;
    int n = vmin->m;
    assert(vmin->m == vmax->m);

    for (int ii = 0; ii < n; ii++)
    {
        if (BLASFEO_DVECEL(vmin, ii) > -TREEQP_INF || BLASFEO_DVECEL(vmax, ii) < TREEQP_INF)
        {
            nb += 1;
        }
    }
    return nb;
}



int get_size_idxb(tree_ocp_qp_in *qp_in)
{
    int size = 0;
    int Nn = qp_in->N;

    for (int ii = 0; ii < Nn; ii++)
    {
        size += number_of_bounds(&qp_in->umin[ii], &qp_in->umax[ii]);
        size += number_of_bounds(&qp_in->xmin[ii], &qp_in->xmax[ii]);
    }

    return size;
}



void setup_nb(tree_ocp_qp_in *qp_in, int *nb)
{
    int Nn = qp_in->N;

    for (int ii = 0; ii < Nn; ii++)
    {
        nb[ii] = 0;
        nb[ii] += number_of_bounds(&qp_in->umin[ii], &qp_in->umax[ii]);
        nb[ii] += number_of_bounds(&qp_in->xmin[ii], &qp_in->xmax[ii]);
    }
}



void setup_nb_idxb(tree_ocp_qp_in *qp_in, int *nb, int **idxb)
{
    int Nn = qp_in->N;
    int kk;

    struct blasfeo_dvec *sxmin = (struct blasfeo_dvec *) qp_in->xmin;
    struct blasfeo_dvec *sxmax = (struct blasfeo_dvec *) qp_in->xmax;
    struct blasfeo_dvec *sumin = (struct blasfeo_dvec *) qp_in->umin;
    struct blasfeo_dvec *sumax = (struct blasfeo_dvec *) qp_in->umax;

    for (int ii = 0; ii < Nn; ii++)
    {
        nb[ii] = 0;
        kk = 0;
        for (int jj = 0; jj < qp_in->nu[ii]; jj++)
        {
            if (BLASFEO_DVECEL(&sumin[ii], jj) > -TREEQP_INF || BLASFEO_DVECEL(&sumax[ii], jj) < TREEQP_INF)
            {
                nb[ii] += 1;
                idxb[ii][kk++] = jj;
            }
        }
        for (int jj = 0; jj < qp_in->nx[ii]; jj++)
        {
            if (BLASFEO_DVECEL(&sxmin[ii], jj) > -TREEQP_INF || BLASFEO_DVECEL(&sxmax[ii], jj) < TREEQP_INF)
            {
                nb[ii] += 1;
                idxb[ii][kk++] = jj + qp_in->nu[ii];
            }

        }
    }
}



int treeqp_hpmpc_calculate_size(tree_ocp_qp_in *qp_in, treeqp_hpmpc_opts_t *opts)
{
    int bytes = 0;
    int idxp;
    int Nn = qp_in->N;
    int *nx = qp_in->nx;
    int *nu = qp_in->nu;
    int *nc = qp_in->nc;

    int *nb = opts->scrap.nb;
    setup_nb(qp_in, nb);

    bytes += Nn*sizeof(int);  // nb

    bytes += Nn*sizeof(int*);  // idxb
    bytes += get_size_idxb(qp_in)*sizeof(int);

    bytes += 3*Nn*sizeof(struct blasfeo_dvec);  // sux, slam, sst

    bytes += Nn*sizeof(struct blasfeo_dmat);  // sRSQrq
    bytes += (Nn-1)*sizeof(struct blasfeo_dmat);  // sBAbt

    bytes += Nn*sizeof(struct blasfeo_dmat);  // sDCt
    bytes += Nn*sizeof(struct blasfeo_dvec);  // sd

    for (int ii = 0; ii < Nn; ii++)
    {
        bytes += blasfeo_memsize_dvec(nx[ii] + nu[ii]);  // sux
        bytes += 2*blasfeo_memsize_dvec(2*nb[ii] + 2*nc[ii]);  // slam, sst

        bytes += blasfeo_memsize_dmat(nx[ii] + nu[ii] + 1, nx[ii] + nu[ii]);  // sRSQrq

        if (ii > 0)
        {
            idxp = qp_in->tree[ii].dad;
            bytes += blasfeo_memsize_dmat(nx[idxp] + nu[idxp] + 1, nx[ii]);  // sBAbt
        }
        bytes += blasfeo_memsize_dmat(nx[ii]+nu[ii], nc[ii]);  // sDCt
        bytes += blasfeo_memsize_dvec(2*nb[ii]+2*nc[ii]);  // sd
    }

    bytes += 5*opts->maxIter*sizeof(double);  // status

    bytes += d_tree_ip2_res_mpc_hard_work_space_size_bytes_libstr(Nn, qp_in->tree, nx, nu, nb, nc);

    make_int_multiple_of(64, &bytes);
    bytes += 2*64;

    return bytes;
}



void treeqp_hpmpc_create(tree_ocp_qp_in *qp_in, treeqp_hpmpc_opts_t *opts,
    treeqp_hpmpc_workspace *work, void *ptr)
{
    int idxp;
    struct node *tree = qp_in->tree;
    int Nn = qp_in->N;
    int *nx = qp_in->nx;
    int *nu = qp_in->nu;
    int *nc = qp_in->nc;

    // char pointer
    char *c_ptr = (char *) ptr;

    // double pointers
    work->idxb = (int **) c_ptr;
    c_ptr += Nn*sizeof(int *);
    for (int ii = 0; ii < Nn; ii++)
    {
        work->idxb[ii] = (int *) c_ptr;
        c_ptr += number_of_bounds(&qp_in->umin[ii], &qp_in->umax[ii])*sizeof(int);
        c_ptr += number_of_bounds(&qp_in->xmin[ii], &qp_in->xmax[ii])*sizeof(int);
    }

    // pointers
    work->nb = (int *) c_ptr;
    c_ptr += Nn*sizeof(int);

    setup_nb_idxb(qp_in, work->nb, work->idxb);

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

    // strmats
    for (int ii = 0; ii < Nn; ii++)
    {
        init_strmat(nx[ii]+nu[ii]+1, nx[ii]+nu[ii], &work->sRSQrq[ii], &c_ptr);

        if (ii > 0)
        {
            idxp = tree[ii].dad;
            init_strmat(nx[idxp]+nu[idxp]+1, nx[ii], &work->sBAbt[ii-1], &c_ptr);
        }
        init_strmat(nx[ii]+nu[ii], nc[ii], &work->sDCt[ii], &c_ptr);
    }

    // strvecs
    for (int ii = 0; ii < Nn; ii++)
    {
        init_strvec(nx[ii] + nu[ii], &work->sux[ii], &c_ptr);
        init_strvec(2*work->nb[ii]+2*nc[ii], &work->slam[ii], &c_ptr);
        init_strvec(2*work->nb[ii]+2*nc[ii], &work->sst[ii], &c_ptr);
        init_strvec(2*work->nb[ii]+2*nc[ii], &work->sd[ii], &c_ptr);
    }

    work->status = (double *) c_ptr;
    c_ptr += 5*opts->maxIter*sizeof(double);

    // TODO(dimitris): Maybe realign not needed
    align_char_to(64, &c_ptr);

    work->internal = (void *) c_ptr;
    c_ptr += d_tree_ip2_res_mpc_hard_work_space_size_bytes_libstr(Nn, tree, nx, nu, work->nb, nc);


    assert((char *)ptr + treeqp_hpmpc_calculate_size(qp_in, opts) >= c_ptr);
    // printf("memory starts at\t%p\nmemory ends at  \t%p\ndistance from the end\t%lu bytes\n",
    //     ptr, c_ptr, (char *)ptr + treeqp_hpmpc_calculate_size(qp_in, opts) - c_ptr);
    // exit(1);
}



int treeqp_hpmpc_solve(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, treeqp_hpmpc_opts_t *opts,
    treeqp_hpmpc_workspace *work)
{
    struct node *tree = qp_in->tree;
    int Nn = qp_in->N;
    int *nx = qp_in->nx;
    int *nu = qp_in->nu;
    int *nc = qp_in->nc;
    int *nb = work->nb;

    struct blasfeo_dmat *sBAbt = work->sBAbt;
    struct blasfeo_dmat *sRSQrq = work->sRSQrq;
    struct blasfeo_dmat *sDCt = work->sDCt;

    treeqp_timer solver_tmr, interface_tmr;

    int idxp, idxb;
    double mu_lb, mu_ub;

    // convert input to HPMPC format

    treeqp_tic(&interface_tmr);

    for (int ii = 0; ii < Nn; ii++)
    {
        blasfeo_dgecp(nu[ii], nu[ii], &qp_in->R[ii], 0, 0, &sRSQrq[ii], 0, 0);
        blasfeo_dgecp(nx[ii], nx[ii], &qp_in->Q[ii], 0, 0, &sRSQrq[ii], nu[ii], nu[ii]);
        blasfeo_dgetr(nu[ii], nx[ii], &qp_in->S[ii], 0, 0, &sRSQrq[ii], nu[ii], 0);

        blasfeo_drowin(nu[ii], 1.0, &qp_in->r[ii], 0, &sRSQrq[ii], nu[ii] + nx[ii], 0);
        blasfeo_drowin(nx[ii], 1.0, &qp_in->q[ii], 0, &sRSQrq[ii], nu[ii] + nx[ii], nu[ii]);

        if (ii > 0)
        {
            idxp = tree[ii].dad;
            blasfeo_dgetr(nx[ii], nu[idxp], &qp_in->B[ii-1], 0, 0, &sBAbt[ii-1], 0, 0);
            blasfeo_dgetr(nx[ii], nx[idxp], &qp_in->A[ii-1], 0, 0, &sBAbt[ii-1], nu[idxp], 0);
            blasfeo_drowin(nx[ii], 1.0, &qp_in->b[ii-1], 0, &sBAbt[ii-1], nx[idxp] + nu[idxp], 0);
        }

        for (int jj = 0; jj < nb[ii]; jj++)
        {
            idxb = work->idxb[ii][jj];
            if (idxb < nu[ii])
            {
                BLASFEO_DVECEL(&work->sd[ii], jj) = BLASFEO_DVECEL(&qp_in->umin[ii], idxb);
                BLASFEO_DVECEL(&work->sd[ii], jj+nb[ii]+nc[ii]) = BLASFEO_DVECEL(&qp_in->umax[ii], idxb);
            }
            else
            {
                BLASFEO_DVECEL(&work->sd[ii], jj) = BLASFEO_DVECEL(&qp_in->xmin[ii], idxb - nu[ii]);
                BLASFEO_DVECEL(&work->sd[ii], jj+nb[ii]+nc[ii]) = BLASFEO_DVECEL(&qp_in->xmax[ii], idxb - nu[ii]);
            }
        }
        blasfeo_dgetr(nc[ii], nu[ii], &qp_in->D[ii], 0, 0, &sDCt[ii], 0, 0);
        blasfeo_dgetr(nc[ii], nx[ii], &qp_in->C[ii], 0, 0, &sDCt[ii], nu[ii], 0);
        blasfeo_dveccp(nc[ii], &qp_in->dmin[ii], 0, &work->sd[ii], nb[ii]);
        blasfeo_dveccp(nc[ii], &qp_in->dmax[ii], 0, &work->sd[ii], 2*nb[ii]+nc[ii]);
        // blasfeo_dvecse(nc[ii], -1000.0, &qp_in->dmin[ii], 0);
        // blasfeo_dvecse(nc[ii], +1000.0, &qp_in->dmax[ii], 0);
    }
    qp_out->info.interface_time = treeqp_toc(&interface_tmr);
    treeqp_tic(&solver_tmr);


    // // TODO(dimitris): fix bug in general constraints and remove those prints
    // for (int ii = 0; ii < qp_in->N; ii++)
    // {
    //     printf("ii = %d / %d (nx = %d, nu = %d, nb = %d, nc = %d)\n", ii, qp_in->N, nx[ii], nu[ii], nb[ii], nc[ii]);
    //     printf("idxb = \n");
    //     int_print_mat(1, nb[ii], work->idxb[ii], 1);
    //     printf("d = \n");
    //     blasfeo_print_tran_dvec(2*nb[ii]+2*nc[ii], &work->sd[ii], 0);
    //     printf("DCt = \n");
    //     blasfeo_print_dmat(nx[ii]+nu[ii], nc[ii], &sDCt[ii], 0, 0);
    //     // if (ii == 10) exit(1);
    // }
    // exit(1);

    // solve QP
    int status = d_tree_ip2_res_mpc_hard_libstr(&qp_out->info.iter, opts->maxIter, opts->mu0,
            opts->mu_tol, opts->alpha_min, opts->warm_start, work->status, qp_in->N, tree,
            nx, nu, nb, work->idxb, nc, sBAbt, sRSQrq, sDCt, work->sd,
            work->sux, opts->compute_mult, qp_out->lam, work->slam, work->sst, work->internal);

    qp_out->info.solver_time = treeqp_toc(&solver_tmr);

    // copy results to qp_out struct
    treeqp_tic(&interface_tmr);

    for (int ii = 0; ii < Nn; ii++)
    {
        blasfeo_dveccp(nu[ii], &work->sux[ii], 0, &qp_out->u[ii], 0);
        blasfeo_dveccp(nx[ii], &work->sux[ii], nu[ii], &qp_out->x[ii], 0);

        blasfeo_dvecse(nx[ii], 0.0, &qp_out->mu_x[ii], 0);
        blasfeo_dvecse(nu[ii], 0.0, &qp_out->mu_u[ii], 0);
        for (int jj = 0; jj < nb[ii]; jj++)
        {
            idxb = work->idxb[ii][jj];
            mu_lb = BLASFEO_DVECEL(&work->slam[ii], jj);
            mu_ub = BLASFEO_DVECEL(&work->slam[ii], jj+nb[ii]+nc[ii]);
            if (idxb < nu[ii])
            {
                BLASFEO_DVECEL(&qp_out->mu_u[ii], idxb) += mu_ub - mu_lb;
            }
            else //if (idxb < nu[ii]+nx[ii])
            {
                BLASFEO_DVECEL(&qp_out->mu_x[ii], idxb-nu[ii]) +=  mu_ub -mu_lb;
            }
        }
        blasfeo_daxpy(nc[ii], -1.0, &work->slam[ii], nb[ii], &work->slam[ii], 2*nb[ii]+nc[ii], &qp_out->mu_d[ii], 0);
    }

    qp_out->info.interface_time += treeqp_toc(&interface_tmr);

    return status;
}
