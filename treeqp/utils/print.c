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

#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_aux_ext_dep.h>

#include "treeqp/src/dual_Newton_common.h"
#include "treeqp/src/tree_qp_common.h"
#include "treeqp/utils/print.h"
#include "treeqp/utils/tree.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/utils.h"
#include "treeqp/utils/profiling.h"



void node_print(const struct node *tree)
{
    printf("\n");
    printf("idx    = \t%d\n", tree[0].idx);
    printf("dad    = \t%d\n", tree[0].dad);
    printf("nkids  = \t%d\n", tree[0].nkids);
    printf("kids   = \t");
    for (int ii = 0; ii < tree[0].nkids; ii++)
    {
        printf("%d\t", tree[0].kids[ii]);
    }
    printf("\n");
    printf("stage  = \t%d\n", tree[0].stage);
    printf("real   = \t%d\n", tree[0].real);
    printf("idxkid = \t%d\n", tree[0].idxkid);
    printf("\n");
    return;
}



void tree_qp_in_print_dims(const tree_qp_in *qp_in)
{
    int N = qp_in->N;
    int *nx = qp_in->nx;
    int *nu = qp_in->nu;
    int *nc = qp_in->nc;

    // printf("k\tnx\tnu\tnb\tnbx\tnbu\tng\tns\n");
    printf("k\tnx\tnu\tnc\n");

    for (int ii = 0; ii < N; ii++)
    {
        printf("%d\t%d\t%d\t%d\n", ii, nx[ii], nu[ii], nc[ii]);
    }
}



void tree_qp_in_print(const tree_qp_in *qp_in)
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
            if (min >= -TREEQP_INF)
            {
                printf("%5.2f  ", min);
            }
            else
            {
                printf("-INF   ");
            }
            printf("<=  x_%d  <=  ", jj);
            max = BLASFEO_DVECEL(&qp_in->xmax[ii], jj);
            if (max <= TREEQP_INF)
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
            if (min >= -TREEQP_INF)
            {
                printf("%5.2f  ", min);
            }
            else
            {
                printf("-INF   ");
            }
            printf("<=  u_%d  <=  ", jj);
            max = BLASFEO_DVECEL(&qp_in->umax[ii], jj);
            if (max <= TREEQP_INF)
            {
                printf("%5.2f\n", max);
            }
            else
            {
                printf("  INF\n");
            }
        }
        printf("\n\n");

        printf("C[%d] = \n", ii);
        blasfeo_print_dmat(qp_in->nc[ii], qp_in->nx[ii], &qp_in->C[ii], 0, 0);
        printf("D[%d] = \n", ii);
        blasfeo_print_dmat(qp_in->nc[ii], qp_in->nu[ii], &qp_in->D[ii], 0, 0);
        for (int jj = 0; jj < qp_in->nc[ii]; jj++)
        {
            min = BLASFEO_DVECEL(&qp_in->dmin[ii], jj);
            if (min >= -TREEQP_INF)
            {
                printf("%5.2f  ", min);
            }
            else
            {
                printf("-INF   ");
            }
            printf("<=  C[%d, :]*x + D[%d, :]*u  <=  ", jj, jj);
            max = BLASFEO_DVECEL(&qp_in->dmax[ii], jj);
            if (max <= TREEQP_INF)
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



void tree_qp_out_print(int Nn, const tree_qp_out *qp_out)
{
    int nx, nu, nc;

    printf("\nSolver performed %d iterations in %f ms\n\n",
        qp_out->info.iter, 1e3*(qp_out->info.solver_time+qp_out->info.interface_time));

    for (int ii = 0; ii < Nn; ii++)
    {
        nx = qp_out->x[ii].m;
        nu = qp_out->u[ii].m;
        nc = qp_out->mu_d[ii].m;

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

        printf("mu_d[%d] = \n", ii);
        blasfeo_print_tran_dvec(nc, &qp_out->mu_d[ii], 0);
    }
}



void tree_qp_out_write_to_txt(const tree_qp_in *qp_in, const tree_qp_out *qp_out, const char *fpath)
{
    int Nn = qp_in->N;
    int dimx = total_number_of_states(qp_in);
    int dimu = total_number_of_controls(qp_in);
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



void regularization_print_status(regType_t reg_type, reg_result_t reg_res)
{
    switch (reg_type)
    {
        case TREEQP_NO_REGULARIZATION:
            printf("NO REGULARIZATION: Hessian block never regularized\n");
            break;
        case TREEQP_ALWAYS_LEVENBERG_MARQUARDT:
            printf("ALWAYS REGULARIZATION: Hessian block always regularized\n");
            break;
        case TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT:
            // check result
            switch (reg_res)
            {
                case TREEQP_NO_REGULARIZATION_ADDED:
                    printf("ON-THE-FLY REGULARIZATION: Hessian block not regularized\n");
                    break;
                case TREEQP_REGULARIZATION_ADDED:
                    printf("ON-THE-FLY REGULARIZATION: Hessian block regularized\n");
                    break;
            }
            break;
    }
}



#if PROFILE > 0

void timers_write_to_txt(treeqp_profiling_t *timings)
{
    char fname[256];
    char prefix[] = "examples/spring_mass_utils";

    #if PROFILE > 1
    snprintf(fname, sizeof(fname), "%s/%s.txt", prefix, "ls_iters");
    write_int_vector_to_txt(timings->ls_iters, timings->num_iter, fname);
    #endif

    // NOTE(dimitris): do not save cpu time if PROFILE is too high (inaccurate results)
    #if PROFILE < 3

    snprintf(fname, sizeof(fname), "%s/%s.txt", prefix, "cputime");
    write_double_vector_to_txt(&timings->min_total_time, 1, fname);

    #if PROFILE > 1
    snprintf(fname, sizeof(fname), "%s/%s.txt", prefix, "iter_times");
    write_double_vector_to_txt(timings->min_iter_times, timings->num_iter, fname);
    #endif

    #endif  /* PROFILE < 3 */

    #if PROFILE > 2
    snprintf(fname, sizeof(fname), "%s/%s.txt", prefix, "stage_qps_times");
    write_double_vector_to_txt(timings->min_stage_qps_times, timings->num_iter, fname);
    snprintf(fname, sizeof(fname), "%s/%s.txt", prefix, "build_dual_times");
    write_double_vector_to_txt(timings->min_build_dual_times, timings->num_iter, fname);
    snprintf(fname, sizeof(fname), "%s/%s.txt", prefix, "newton_direction_times");
    write_double_vector_to_txt(timings->min_newton_direction_times, timings->num_iter, fname);
    snprintf(fname, sizeof(fname), "%s/%s.txt", prefix, "line_search_times");
    write_double_vector_to_txt(timings->min_line_search_times, timings->num_iter, fname);
    #endif
}

#endif  /* PROFILE > 0 */



void blasfeo_print_target(void)
{
    printf("\n");
    #if defined(LA_HIGH_PERFORMANCE)
    printf("blasfeo compiled with LA = HIGH_PERFORMANCE\n");
    #elif defined(LA_REFERENCE)
    printf("blasfeo compiled with LA = REFERENCE\n");
    #elif defined(LA_BLAS)
    printf("blasfeo compiled with LA = BLAS\n");
    #endif
    printf("\n");
}
