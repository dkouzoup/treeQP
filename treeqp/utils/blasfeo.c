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

#include "treeqp/utils/blasfeo.h"
#include "treeqp/utils/types.h"

#include <blasfeo_d_aux.h>
#include <blasfeo_d_aux_ext_dep.h>
#include <blasfeo_v_aux_ext_dep.h>


void convert_strvecs_to_single_vec(int n, const struct blasfeo_dvec *sv, double *v)
{
    int ind = 0;
    for (int i = 0; i < n; i++)
    {
        blasfeo_unpack_dvec(sv[i].m, (struct blasfeo_dvec *)&sv[i], 0, &v[ind]);
        ind += sv[i].m;
    }
}



void convert_strmats_to_single_vec(int n, const struct blasfeo_dmat *sM, double *M)
{
    int ind = 0;
    for (int i = 0; i < n; i++)
    {
        blasfeo_unpack_dmat(sM[i].m, sM[i].n, (struct blasfeo_dmat *) &sM[i], 0, 0, &M[ind], sM[i].m);
        ind += sM[i].m*sM[i].n;
    }
}



void convert_strmats_tran_to_single_vec(int n, const struct blasfeo_dmat *sM, double *M)
{
    int ind = 0;
    for (int i = 0; i < n; i++)
    {
        blasfeo_unpack_tran_dmat(sM[i].m, sM[i].n, (struct blasfeo_dmat *)&sM[i], 0, 0, &M[ind], sM[i].n);
        ind += sM[i].m*sM[i].n;
    }
}



double check_error_strmat(const struct blasfeo_dmat *M1, const struct blasfeo_dmat *M2)
{
    double err = 0;

    if ((M1->m != M2->m) || (M1->n != M2->n))
    {
        // TODO(dimitris): proper error handling and remove stdio header
        printf("[TREEQP]: Error! Matrices do not have the same dimensions ");
        printf("(%d x %d) vs (%d x %d)\n", M1->m, M1->n, M2->m, M2->n);
        exit(1);
    }
    for (int ii = 0; ii < M1->m; ii++)
    {
        for (int jj = 0; jj < M1->n; jj++)
        {
            err = MAX(ABS(BLASFEO_DMATEL(M1, ii, jj) - BLASFEO_DMATEL(M2, ii, jj)), err);
        }
    }
    return err;
}



double check_error_strvec(const struct blasfeo_dvec *v1, const struct blasfeo_dvec *v2)
{
    double err = 0;

    if (v1->m != v2->m)
    {
        // TODO(dimitris): proper error handling and remove stdio header
        printf("[TREEQP]: Error! Vectors do not have the same dimensions ");
        printf("(%d x 1) vs (%d x 1)\n", v1->m, v2->m);
        exit(1);
    }
    for (int ii = 0; ii < v1->m; ii++)
    {
        err = MAX(ABS(BLASFEO_DVECEL(v1, ii) - BLASFEO_DVECEL(v2, ii)), err);
    }
    return err;
}



double check_error_strvec_double(const struct blasfeo_dvec *v1, const double *v2)
{
    double err = 0;

    for (int ii = 0; ii < v1->m; ii++)
    {
        err = MAX(ABS(BLASFEO_DVECEL(v1, ii) - v2[ii]), err);
    }
    return err;
}



answer_t is_strmat_symmetric(const struct blasfeo_dmat *M)
{
    double tol = 1e-8;
    answer_t ans = YES;
    assert(M->m == M->n);

    for (int ii = 0; ii < M->m; ii++)
    {
        for (int jj = 0; jj < ii; jj++)
        {
            // printf("error(%d, %d) = %f\n", ii, jj, BLASFEO_DMATEL(M, ii, jj) - BLASFEO_DMATEL(M, jj, ii));
            if (BLASFEO_DMATEL(M, ii, jj) - BLASFEO_DMATEL(M, jj, ii) > tol)
            {
                ans = NO;
            }

        }
    }
    return ans;
}



answer_t is_strmat_diagonal(const struct blasfeo_dmat *M)
{
    answer_t ans = YES;
    assert(M->m == M->n);
    for (int ii = 0; ii < M->m; ii++)
    {
        for (int jj = 0; jj < M->n; jj++)
        {
            if (ii != jj)
            {
                if (BLASFEO_DMATEL(M, ii, jj) != 0)
                {
                    ans = NO;
                }
            }
        }
    }
    return ans;
}



answer_t is_strmat_zero(const struct blasfeo_dmat *M)
{
    answer_t ans = YES;
    for (int ii = 0; ii < M->m; ii++)
    {
        for (int jj = 0; jj < M->n; jj++)
        {
            if (BLASFEO_DMATEL(M, ii, jj) != 0)
            {
                ans = NO;
            }
        }
    }
    return ans;
}
