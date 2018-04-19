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

    // // TEMP
    // int dim_size = d_memsize_tree_ocp_qp_dim(Nn);
	// void *dim_mem = malloc(dim_size);
	// struct d_tree_ocp_qp_dim dim_new;
	// d_create_tree_ocp_qp_dim(Nn, &dim_new, dim_mem);
	// d_cvt_int_to_tree_ocp_qp_dim(&hpipm_tree, nx, nu, nbx, nbu, nc, ns, &dim_new);

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

    // TODO(dimitris): pass dims insteam in HPIPM

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
    setup_nkids(qp_in, work->nkids);

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
    setup_nbx(qp_in, work->hpipm_qp_dim.nbx);
    setup_nbu(qp_in, work->hpipm_qp_dim.nbu);
    setup_nb(qp_in, work->hpipm_qp_dim.nb);
    setup_ns(qp_in, work->hpipm_qp_dim.ns);

    // set up qp
	d_create_tree_ocp_qp(&work->hpipm_qp_dim, &work->hpipm_qp_in, c_ptr);
    c_ptr += work->hpipm_qp_in.memsize;
    setup_nb_idxb(qp_in, work->hpipm_qp_dim.nb, work->hpipm_qp_in.idxb);

    // set up args
    d_create_tree_ocp_qp_ipm_arg(NULL, &work->arg, c_ptr);
	d_set_default_tree_ocp_qp_ipm_arg(&work->arg);
    work->arg.stat_max = opts->maxIter; // TODO CAST OPTIONS PROPERLY!!!!!!!!!!!!!!!!!!!!!!!!!!
    c_ptr += work->arg.memsize;

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
    int *nb = work->hpipm_qp_in.dim->nb;

    struct blasfeo_dmat *sBAbt = work->hpipm_qp_in.BAbt;
    struct blasfeo_dmat *sRSQrq = work->hpipm_qp_in.RSQrq;
    struct blasfeo_dmat *sDCt = work->hpipm_qp_in.DCt;

    struct blasfeo_dvec *sb = work->hpipm_qp_in.b;
	struct blasfeo_dvec *srq = work->hpipm_qp_in.rqz;

    struct blasfeo_dvec *sd = work->hpipm_qp_in.d;

    struct blasfeo_dvec *sux = work->hpipm_qp_out.ux;

    treeqp_timer solver_tmr, interface_tmr;

    int idxp, idxb;
    double mu_lb, mu_ub;

    // convert input to HPIPM format

    treeqp_tic(&interface_tmr);

    // TODO: PUT IN FUNCTIONS AND USE THE SAME IN HPMPC AND HPMPC (FLIP SIGN OF D)!!!!!!!
    for (int ii = 0; ii < Nn; ii++)
    {
        blasfeo_dgecp(nu[ii], nu[ii], &qp_in->R[ii], 0, 0, &sRSQrq[ii], 0, 0);
        blasfeo_dgecp(nx[ii], nx[ii], &qp_in->Q[ii], 0, 0, &sRSQrq[ii], nu[ii], nu[ii]);
        blasfeo_dgetr(nu[ii], nx[ii], &qp_in->S[ii], 0, 0, &sRSQrq[ii], nu[ii], 0);

        blasfeo_dveccp(nu[ii], &qp_in->r[ii], 0, &srq[ii], 0);
        blasfeo_dveccp(nx[ii], &qp_in->q[ii], 0, &srq[ii], nu[ii]);

        // TODO ARE THOSE NEEDED??
        // blasfeo_drowin(nu[ii], 1.0, &qp_in->r[ii], 0, &sRSQrq[ii], nu[ii] + nx[ii], 0);
        // blasfeo_drowin(nx[ii], 1.0, &qp_in->q[ii], 0, &sRSQrq[ii], nu[ii] + nx[ii], nu[ii]);

        if (ii > 0)
        {
            idxp = tree[ii].dad;
            blasfeo_dgetr(nx[ii], nu[idxp], &qp_in->B[ii-1], 0, 0, &sBAbt[ii-1], 0, 0);
            blasfeo_dgetr(nx[ii], nx[idxp], &qp_in->A[ii-1], 0, 0, &sBAbt[ii-1], nu[idxp], 0);

            blasfeo_dveccp(nx[ii], &qp_in->b[ii-1], 0, &sb[ii-1], 0);
            // TODO IS THIS NEEDED??
            // blasfeo_drowin(nx[ii], 1.0, &qp_in->b[ii-1], 0, &sBAbt[ii-1], nx[idxp] + nu[idxp], 0);
        }

        for (int jj = 0; jj < nb[ii]; jj++)
        {
            idxb = work->hpipm_qp_in.idxb[ii][jj];
            if (idxb < nu[ii])
            {
                BLASFEO_DVECEL(&sd[ii], jj) = BLASFEO_DVECEL(&qp_in->umin[ii], idxb);
                BLASFEO_DVECEL(&sd[ii], jj+nb[ii]+nc[ii]) = -BLASFEO_DVECEL(&qp_in->umax[ii], idxb);
            }
            else
            {
                BLASFEO_DVECEL(&sd[ii], jj) = BLASFEO_DVECEL(&qp_in->xmin[ii], idxb - nu[ii]);
                BLASFEO_DVECEL(&sd[ii], jj+nb[ii]+nc[ii]) = -BLASFEO_DVECEL(&qp_in->xmax[ii], idxb - nu[ii]);
            }
        }
        blasfeo_dgetr(nc[ii], nu[ii], &qp_in->D[ii], 0, 0, &sDCt[ii], 0, 0);
        blasfeo_dgetr(nc[ii], nx[ii], &qp_in->C[ii], 0, 0, &sDCt[ii], nu[ii], 0);
        blasfeo_dveccp(nc[ii], &qp_in->dmin[ii], 0, &sd[ii], nb[ii]);
        blasfeo_dveccpsc(nc[ii], -1.0, &qp_in->dmax[ii], 0, &sd[ii], 2*nb[ii]+nc[ii]);
    }
    qp_out->info.interface_time = treeqp_toc(&interface_tmr);
    treeqp_tic(&solver_tmr);

#if 0

    struct d_tree_ocp_qp *qp = &work->hpipm_qp_in;

	int N2 = qp->dim->Nn;
	int *nx2 = qp->dim->nx;
	int *nu2 = qp->dim->nu;
	int *nb2 = qp->dim->nb;
	int *ng2 = qp->dim->ng;
	int *ns2 = qp->dim->ns;

	int ii;

	printf("\nnb\n");
	int_print_mat(1, N2, nb2, 1);
	printf("\nng\n");
	int_print_mat(1, N2, ng2, 1);
	printf("\nns\n");
	int_print_mat(1, N2, ns2, 1);

    printf("\nd\n");
	for(ii=0; ii<N2; ii++)
    {
        printf("ii = %d\n", ii);
		blasfeo_print_tran_dvec(2*nb[ii]+2*ng2[ii]+2*ns2[ii], qp->d+ii, 0);
        assert((qp->d+ii)->m == 2*nb[ii]+2*ng2[ii]+2*ns2[ii]);
    }

    printf("\nlb\n");
	for(ii=0; ii<N2; ii++)
    {
        printf("ii = %d\n", ii);
		blasfeo_print_tran_dvec(nb2[ii], qp->d+ii, 0);
    }

	printf("\nlg\n");
	for(ii=0; ii<N2; ii++)
    {
        printf("ii = %d\n", ii);
    	blasfeo_print_tran_dvec(ng2[ii], qp->d+ii, nb2[ii]);
    }

    printf("\nub\n");
	for(ii=0; ii<N2; ii++)
    {
        printf("ii = %d\n", ii);
    	blasfeo_print_tran_dvec(nb2[ii], qp->d+ii, nb2[ii]+ng2[ii]);
    }

    printf("\nug\n");
	for(ii=0; ii<N2; ii++)
    {
        printf("ii = %d\n", ii);
    	blasfeo_print_tran_dvec(ng2[ii], qp->d+ii, 2*nb2[ii]+ng2[ii]);
    }

    printf("\nls\n");
	for(ii=0; ii<N2; ii++)
    {
        printf("ii = %d\n", ii);
    	blasfeo_print_tran_dvec(ns2[ii], qp->d+ii, 2*nb2[ii]+2*ng2[ii]);
    }

    printf("\nus\n");
	for(ii=0; ii<N2; ii++)
    {
        printf("ii = %d\n", ii);
    	blasfeo_print_tran_dvec(ns2[ii], qp->d+ii, 2*nb2[ii]+2*ng2[ii]+ns2[ii]);
    }

	printf("\nidxb\n");
	for(ii=0; ii<N2; ii++)
    {
        printf("ii = %d\n", ii);
    	int_print_mat(1, nb2[ii], qp->idxb[ii], 1);
    }

	printf("\nidxs\n");
	for(ii=0; ii<N2; ii++)
    {
		int_print_mat(1, ns2[ii], qp->idxs[ii], 1);
        assert(ns2[ii] == 0);
    }

	printf("\nZ\n");
	for(ii=0; ii<N2; ii++)
		blasfeo_print_tran_dvec(2*ns2[ii], qp->Z+ii, 0);

    printf("\nrqz\n");
	for(ii=0; ii<N2; ii++)
		blasfeo_print_tran_dvec(nu2[ii]+nx2[ii]+2*ns2[ii], qp->rqz+ii, 0);

	printf("\nRSQ\n");
	for(ii=0; ii<N2; ii++)
		blasfeo_print_dmat(nu2[ii]+nx2[ii]+2*ns2[ii], nu2[ii]+nx2[ii]+2*ns2[ii], qp->RSQrq+ii, 0, 0);

	printf("\nBAt\n");
	for(ii=0; ii<N2-1; ii++)
		blasfeo_print_dmat(nu2[ii]+nx2[ii], nx2[ii+1], qp->BAbt+ii, 0, 0);

#endif

    // solve QP
    // TODO(dimitris): rename arg to hpipm_arg
    // work->hpipm_qp_in.dim->ttree->kids = NULL;  // TODO is this even needed?
    int status = d_solve_tree_ocp_qp_ipm(&work->hpipm_qp_in, &work->hpipm_qp_out, &work->arg, &work->hpipm_memory);


#if 0

    struct d_tree_ocp_qp_sol *sol = &work->hpipm_qp_out;

    printf("\nux\n");
	for(ii=0; ii<N2; ii++)
		blasfeo_print_tran_dvec(nu2[ii]+nx2[ii], sol->ux+ii, 0);
    exit(1);

    printf("\nSTATUS = %d!!!!", status);

#endif

    qp_out->info.solver_time = treeqp_toc(&solver_tmr);

    // copy results to qp_out struct
    treeqp_tic(&interface_tmr);

    for (int ii = 0; ii < Nn; ii++)
    {
        blasfeo_dveccp(nu[ii], &sux[ii], 0, &qp_out->u[ii], 0);
        blasfeo_dveccp(nx[ii], &sux[ii], nu[ii], &qp_out->x[ii], 0);

        blasfeo_dvecse(nx[ii], 0.0, &qp_out->mu_x[ii], 0);
        blasfeo_dvecse(nu[ii], 0.0, &qp_out->mu_u[ii], 0);
        for (int jj = 0; jj < nb[ii]; jj++)
        {
            idxb = work->hpipm_qp_in.idxb[ii][jj];
            mu_lb = BLASFEO_DVECEL(&work->hpipm_qp_out.lam[ii], jj);
            mu_ub = BLASFEO_DVECEL(&work->hpipm_qp_out.lam[ii], jj+nb[ii]+nc[ii]);
            if (idxb < nu[ii])
            {
                BLASFEO_DVECEL(&qp_out->mu_u[ii], idxb) += mu_ub - mu_lb;
            }
            else //if (idxb < nu[ii]+nx[ii])
            {
                BLASFEO_DVECEL(&qp_out->mu_x[ii], idxb-nu[ii]) +=  mu_ub -mu_lb;
            }
        }
        blasfeo_daxpy(nc[ii], -1.0, &work->hpipm_qp_out.lam[ii], nb[ii],
            &work->hpipm_qp_out.lam[ii], 2*nb[ii]+nc[ii], &qp_out->mu_d[ii], 0);
    }

    qp_out->info.interface_time += treeqp_toc(&interface_tmr);

    return status;
}
