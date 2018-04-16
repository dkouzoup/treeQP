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

#include "treeqp/src/hpmpc_tree.h"  // for helper functions
#include "treeqp/src/hpipm_tree.h"
#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/utils/memory.h"
#include "treeqp/utils/timing.h"
#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_blas.h"

// #include "hpipm/include/hpipm_target.h" // WHERE IS THIS????
#include "hpipm/include/hpipm_tree.h"

#define INF 1e10 // TODO(dimitris): option instead of hardcoded here?


treeqp_hpipm_options_t treeqp_hpipm_default_options()
{
    treeqp_hpipm_options_t opts;

    opts.maxIter = 20;

	// opts.mu0 = 2.0;
	// opts.mu_tol = 1e-12;
	// opts.alpha_min = 1e-8;
	// opts.warm_start = 0;
    // opts.compute_mult = 1;

    return opts;
}



void setup_nkids(tree_ocp_qp_in *qp_in, int *nkids)
{
    int Nn = qp_in->N;
    struct node *tree = qp_in->tree;

    int kk = 0;

    for (int ii = 0; ii < Nn; ii++)
    {
        for (int jj = 0; jj < tree[ii].nkids; jj++)
        {
            nkids[kk++] = tree[ii].kids[jj];
        }
    }
}



void setup_nbx(tree_ocp_qp_in *qp_in, int *nbx)
{
    int Nn = qp_in->N;

    for (int ii = 0; ii < Nn; ii++)
    {
        nbx[ii] = number_of_bounds(&qp_in->xmin[ii], &qp_in->xmax[ii]);
    }
}



void setup_nbu(tree_ocp_qp_in *qp_in, int *nbu)
{
    int Nn = qp_in->N;

    for (int ii = 0; ii < Nn; ii++)
    {
        nbu[ii] = number_of_bounds(&qp_in->umin[ii], &qp_in->umax[ii]);
    }
}



void setup_ns(tree_ocp_qp_in *qp_in, int *ns)
{
    int Nn = qp_in->N;

    for (int ii = 0; ii < Nn; ii++)
    {
        ns[ii] = 0;
    }
}



// int number_of_bounds(const struct blasfeo_dvec *vmin, const struct blasfeo_dvec *vmax)
// {
//     int nb = 0;
//     int n = vmin->m;
//     assert(vmin->m == vmax->m);

//     for (int ii = 0; ii < n; ii++)
//     {
//         if (BLASFEO_DVECEL(vmin, ii) > -INF || BLASFEO_DVECEL(vmax, ii) < INF)
//         {
//             nb += 1;
//         }
//     }
//     return nb;
// }



// int get_size_idxb(tree_ocp_qp_in *qp_in)
// {
//     int size = 0;
//     int Nn = qp_in->N;

//     for (int ii = 0; ii < Nn; ii++)
//     {
//         size += number_of_bounds(&qp_in->umin[ii], &qp_in->umax[ii]);
//         size += number_of_bounds(&qp_in->xmin[ii], &qp_in->xmax[ii]);
//     }

//     return size;
// }



// void setup_nb(tree_ocp_qp_in *qp_in, int *nb)
// {
//     int Nn = qp_in->N;

//     for (int ii = 0; ii < Nn; ii++)
//     {
//         nb[ii] = 0;
//         nb[ii] += number_of_bounds(&qp_in->umin[ii], &qp_in->umax[ii]);
//         nb[ii] += number_of_bounds(&qp_in->xmin[ii], &qp_in->xmax[ii]);
//     }
// }



// void setup_nb_idxb(tree_ocp_qp_in *qp_in, int *nb, int **idxb)
// {
//     int Nn = qp_in->N;
//     int kk;

//     struct blasfeo_dvec *sxmin = (struct blasfeo_dvec *) qp_in->xmin;
//     struct blasfeo_dvec *sxmax = (struct blasfeo_dvec *) qp_in->xmax;
//     struct blasfeo_dvec *sumin = (struct blasfeo_dvec *) qp_in->umin;
//     struct blasfeo_dvec *sumax = (struct blasfeo_dvec *) qp_in->umax;

//     for (int ii = 0; ii < Nn; ii++)
//     {
//         nb[ii] = 0;
//         kk = 0;
//         for (int jj = 0; jj < qp_in->nu[ii]; jj++)
//         {
//             if (BLASFEO_DVECEL(&sumin[ii], jj) > -INF || BLASFEO_DVECEL(&sumax[ii], jj) < INF)
//             {
//                 nb[ii] += 1;
//                 idxb[ii][kk++] = jj;
//             }
//         }
//         for (int jj = 0; jj < qp_in->nx[ii]; jj++)
//         {
//             if (BLASFEO_DVECEL(&sxmin[ii], jj) > -INF || BLASFEO_DVECEL(&sxmax[ii], jj) < INF)
//             {
//                 nb[ii] += 1;
//                 idxb[ii][kk++] = jj + qp_in->nu[ii];
//             }

//         }
//     }
// }



int treeqp_hpipm_calculate_size(tree_ocp_qp_in *qp_in, treeqp_hpipm_options_t *opts)
{
    int bytes = 0;
    int Nn = qp_in->N;
    int *nx = qp_in->nx;
    int *nu = qp_in->nu;
    int *nc = qp_in->nc;

    // TODO(dimitris): can we avoid memory allocation in here?
    int *nkids = (int *)malloc(Nn*sizeof(int));
    int *nb = (int *)malloc(Nn*sizeof(int));
    int *nbx = (int *)malloc(Nn*sizeof(int));
    int *nbu = (int *)malloc(Nn*sizeof(int));
    int *ns = (int *)malloc(Nn*sizeof(int));

    setup_nkids(qp_in, nkids);
    setup_nb(qp_in, nb);
    setup_nbx(qp_in, nbx);
    setup_nbu(qp_in, nbu);
    setup_nb(qp_in, nb);
    setup_ns(qp_in, ns);

    // set up temporary hpipm tree
    struct tree hpipm_tree;
    hpipm_tree.Nn = Nn;
    hpipm_tree.memsize = -1;
    hpipm_tree.root = &qp_in->tree[0];
    hpipm_tree.kids = nkids;

    // set up temporary hpipm dimensions
	struct d_tree_ocp_qp_dim dim;

    dim.Nn = Nn;
    dim.ttree = &hpipm_tree;
    dim.nx = nx;
    dim.nu = nu;
    dim.nb = nb;
    dim.nbx = nbx;
    dim.nbu = nbu;
    dim.ng = nc;
    dim.ns = ns;
    dim.memsize = -1;

    // set up dummy qp in (only dims matter in calculate size)
    struct d_tree_ocp_qp qp;
    qp.dim = &dim;

    // set up dummy args (only stat_max matters in calculate size)
    struct d_tree_ocp_qp_ipm_arg arg;
    arg.stat_max = opts->maxIter; // TODO(dimitris): IS THIS CORRECT???

    // calculate memory size

    bytes += Nn*sizeof(int);  // hpipm_tree.nkids

	bytes += d_memsize_tree_ocp_qp_dim(Nn);  // hpipm_qp_dim

    bytes += d_memsize_tree_ocp_qp(&dim);  // hpipm_qp_in

    bytes += d_memsize_tree_ocp_qp_ipm_arg(NULL);  // hpipm_arg

    bytes += d_memsize_tree_ocp_qp_sol(&dim);  //hpipm_qp_out

	bytes += d_memsize_tree_ocp_qp_ipm(&qp, &arg);  // hpipm_memory

    bytes += 1*64;

    free(nkids);
    free(nb);
    free(nbx);
    free(nbu);
    free(ns);

    return bytes;
}



void create_treeqp_hpipm(tree_ocp_qp_in *qp_in, treeqp_hpipm_options_t *opts,
    treeqp_hpipm_workspace *work, void *ptr)
{
    struct node *tree = qp_in->tree;
    int Nn = qp_in->N;
    int *nx = qp_in->nx;
    int *nu = qp_in->nu;
    int *nc = qp_in->nc;

    // char pointer
    char *c_ptr = (char *) ptr;

    // pointers
    work->nkids = (int *) c_ptr;
    c_ptr += Nn*sizeof(int);
    setup_nkids(qp_in, work->nkids);  // TODO(dimitris): CHECK THAT THIS IS CORRECTLY SET!!!!!!!!!!

    // move pointer for proper alignment of doubles and blasfeo matrices/vectors
    align_char_to(64, &c_ptr);

    // set up tree
    work->hpipm_tree.Nn = Nn;
    work->hpipm_tree.memsize = -1;
    work->hpipm_tree.root = &qp_in->tree[0];
    work->hpipm_tree.kids = work->nkids;

    // set up dimensions
	d_create_tree_ocp_qp_dim(Nn, &work->hpipm_qp_dim, c_ptr);
    c_ptr += work->hpipm_qp_dim.memsize;

    work->hpipm_qp_dim.Nn = Nn;
    work->hpipm_qp_dim.ttree = &work->hpipm_tree;
    for (int ii = 0; ii < Nn; ii++)
    {
        work->hpipm_qp_dim.nx[ii] = nx[ii];
        work->hpipm_qp_dim.nu[ii] = nu[ii];
    }
    setup_nb(qp_in, work->hpipm_qp_dim.nb);
    setup_nbx(qp_in, work->hpipm_qp_dim.nbx);
    setup_nbu(qp_in, work->hpipm_qp_dim.nbu);
    setup_nb(qp_in, work->hpipm_qp_dim.nb);
    setup_ns(qp_in, work->hpipm_qp_dim.ns);

    // set up qp
	d_create_tree_ocp_qp(&work->hpipm_qp_dim, &work->hpipm_qp_in, c_ptr);
    c_ptr += work->hpipm_qp_in.memsize;

    // set up args
    d_create_tree_ocp_qp_ipm_arg(NULL, &work->arg, c_ptr);
    c_ptr += work->arg.memsize;

    work->arg.stat_max = opts->maxIter; // TODO TEMP!!!!!!!!!!!!!!!!!!!!!!!!!!

    // set up qp sol
	d_create_tree_ocp_qp_sol(&work->hpipm_qp_dim, &work->hpipm_qp_out, c_ptr);
    c_ptr += work->hpipm_qp_out.memsize;

    // set up hpipm memory
    d_create_tree_ocp_qp_ipm(&work->hpipm_qp_in, &work->arg, &work->hpipm_memory, c_ptr);
    c_ptr += work->hpipm_memory.memsize;

    assert((char *)ptr + treeqp_hpipm_calculate_size(qp_in, opts) >= c_ptr);
    // printf("memory starts at\t%p\nmemory ends at  \t%p\ndistance from the end\t%lu bytes\n",
    //     ptr, c_ptr, (char *)ptr + treeqp_hpipm_calculate_size(qp_in, opts) - c_ptr);
    // exit(1);
}



int treeqp_hpipm_solve(tree_ocp_qp_in *qp_in, tree_ocp_qp_out *qp_out, treeqp_hpipm_options_t *opts,
    treeqp_hpipm_workspace *work)
{
    struct node *tree = qp_in->tree;
    int Nn = qp_in->N;
    int *nx = qp_in->nx;
    int *nu = qp_in->nu;
    int *nc = qp_in->nc;

    // struct blasfeo_dmat *sBAbt = work->sBAbt;
    // struct blasfeo_dmat *sRSQrq = work->sRSQrq;
    // struct blasfeo_dmat *sDCt = work->sDCt;

    // treeqp_timer solver_tmr, interface_tmr;

    // int idxp, idxb;
    // double mu_lb, mu_ub;

    // // convert input to HPIPM format

    // treeqp_tic(&interface_tmr);

    // for (int ii = 0; ii < Nn; ii++)
    // {
    //     blasfeo_dgecp(nu[ii], nu[ii], &qp_in->R[ii], 0, 0, &sRSQrq[ii], 0, 0);
    //     blasfeo_dgecp(nx[ii], nx[ii], &qp_in->Q[ii], 0, 0, &sRSQrq[ii], nu[ii], nu[ii]);
    //     blasfeo_dgetr(nu[ii], nx[ii], &qp_in->S[ii], 0, 0, &sRSQrq[ii], nu[ii], 0);

    //     blasfeo_drowin(nu[ii], 1.0, &qp_in->r[ii], 0, &sRSQrq[ii], nu[ii] + nx[ii], 0);
    //     blasfeo_drowin(nx[ii], 1.0, &qp_in->q[ii], 0, &sRSQrq[ii], nu[ii] + nx[ii], nu[ii]);

    //     if (ii > 0)
    //     {
    //         idxp = tree[ii].dad;
    //         blasfeo_dgetr(nx[ii], nu[idxp], &qp_in->B[ii-1], 0, 0, &sBAbt[ii-1], 0, 0);
    //         blasfeo_dgetr(nx[ii], nx[idxp], &qp_in->A[ii-1], 0, 0, &sBAbt[ii-1], nu[idxp], 0);
    //         blasfeo_drowin(nx[ii], 1.0, &qp_in->b[ii-1], 0, &sBAbt[ii-1], nx[idxp] + nu[idxp], 0);
    //     }

    //     for (int jj = 0; jj < nb[ii]; jj++)
    //     {
    //         idxb = work->idxb[ii][jj];
    //         if (idxb < nu[ii])
    //         {
    //             BLASFEO_DVECEL(&work->sd[ii], jj) = BLASFEO_DVECEL(&qp_in->umin[ii], idxb);
    //             BLASFEO_DVECEL(&work->sd[ii], jj+nb[ii]+nc[ii]) = BLASFEO_DVECEL(&qp_in->umax[ii], idxb);
    //         }
    //         else
    //         {
    //             BLASFEO_DVECEL(&work->sd[ii], jj) = BLASFEO_DVECEL(&qp_in->xmin[ii], idxb - nu[ii]);
    //             BLASFEO_DVECEL(&work->sd[ii], jj+nb[ii]+nc[ii]) = BLASFEO_DVECEL(&qp_in->xmax[ii], idxb - nu[ii]);
    //         }
    //     }
    //     blasfeo_dgetr(nc[ii], nu[ii], &qp_in->D[ii], 0, 0, &sDCt[ii], 0, 0);
    //     blasfeo_dgetr(nc[ii], nx[ii], &qp_in->C[ii], 0, 0, &sDCt[ii], nu[ii], 0);
    //     blasfeo_dveccp(nc[ii], &qp_in->dmin[ii], 0, &work->sd[ii], nb[ii]);
    //     blasfeo_dveccp(nc[ii], &qp_in->dmax[ii], 0, &work->sd[ii], 2*nb[ii]+nc[ii]);
    //     // blasfeo_dvecse(nc[ii], -1000.0, &qp_in->dmin[ii], 0);
    //     // blasfeo_dvecse(nc[ii], +1000.0, &qp_in->dmax[ii], 0);
    // }
    // qp_out->info.interface_time = treeqp_toc(&interface_tmr);
    // treeqp_tic(&solver_tmr);


    // // solve QP
    // int status = d_tree_ip2_res_mpc_hard_libstr(&qp_out->info.iter, opts->maxIter, opts->mu0,
    //         opts->mu_tol, opts->alpha_min, opts->warm_start, work->status, qp_in->N, tree,
    //         nx, nu, nb, work->idxb, nc, sBAbt, sRSQrq, sDCt, work->sd,
    //         work->sux, opts->compute_mult, qp_out->lam, work->slam, work->sst, work->internal);

    // qp_out->info.solver_time = treeqp_toc(&solver_tmr);

    // // copy results to qp_out struct
    // treeqp_tic(&interface_tmr);

    // for (int ii = 0; ii < Nn; ii++)
    // {
    //     blasfeo_dveccp(nu[ii], &work->sux[ii], 0, &qp_out->u[ii], 0);
    //     blasfeo_dveccp(nx[ii], &work->sux[ii], nu[ii], &qp_out->x[ii], 0);

    //     blasfeo_dvecse(nx[ii], 0.0, &qp_out->mu_x[ii], 0);
    //     blasfeo_dvecse(nu[ii], 0.0, &qp_out->mu_u[ii], 0);
    //     for (int jj = 0; jj < nb[ii]; jj++)
    //     {
    //         idxb = work->idxb[ii][jj];
    //         mu_lb = BLASFEO_DVECEL(&work->slam[ii], jj);
    //         mu_ub = BLASFEO_DVECEL(&work->slam[ii], jj+nb[ii]+nc[ii]);
    //         if (idxb < nu[ii])
    //         {
    //             BLASFEO_DVECEL(&qp_out->mu_u[ii], idxb) += mu_ub - mu_lb;
    //         }
    //         else //if (idxb < nu[ii]+nx[ii])
    //         {
    //             BLASFEO_DVECEL(&qp_out->mu_x[ii], idxb-nu[ii]) +=  mu_ub -mu_lb;
    //         }
    //     }
    //     blasfeo_daxpy(nc[ii], -1.0, &work->slam[ii], nb[ii], &work->slam[ii], 2*nb[ii]+nc[ii], &qp_out->mu_d[ii], 0);
    // }

    // qp_out->info.interface_time += treeqp_toc(&interface_tmr);

    // return status;
}
