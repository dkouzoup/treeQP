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

#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_v_aux_ext_dep.h"


void convert_strvecs_to_single_vec(int n, struct blasfeo_dvec sv[], double *v)
{
    int ind = 0;
    for (int i = 0; i < n; i++)
    {
        blasfeo_unpack_dvec(sv[i].m, &sv[i], 0, &v[ind]);
        ind += sv[i].m;
    }
}



void convert_strmats_to_single_vec(int n, struct blasfeo_dmat sMat[], double *mat)
{
    int ind = 0;
    for (int i = 0; i < n; i++)
    {
        blasfeo_unpack_dmat(sMat[i].m, sMat[i].n, &sMat[i], 0, 0, &mat[ind], sMat[i].m);
        ind += sMat[i].m*sMat[i].n;
    }
}



void convert_strmats_tran_to_single_vec(int n, struct blasfeo_dmat sMat[], double *mat)
{
    int ind = 0;
    for (int i = 0; i < n; i++)
    {
        blasfeo_unpack_tran_dmat(sMat[i].m, sMat[i].n, &sMat[i], 0, 0, &mat[ind], sMat[i].n);
        ind += sMat[i].m*sMat[i].n;
    }
}



double check_error_strmat(struct blasfeo_dmat *M1, struct blasfeo_dmat *M2)
{
    double err = 0;

    if ((M1->m != M2->m) || (M1->n != M2->n))
    {
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
    if (err > 0)
    {
        printf("[TREEQP]: Error! Matrices are different (error = %2.2e)\n", err);
        exit(1);
    }
    return err;
}



double check_error_strvec(struct blasfeo_dvec *V1, struct blasfeo_dvec *V2)
{
    double err = 0;

    if (V1->m != V2->m)
    {
        printf("[TREEQP]: Error! Vectors do not have the same dimensions ");
        printf("(%d x 1) vs (%d x 1)\n", V1->m, V2->m);
        exit(1);
    }
    for (int ii = 0; ii < V1->m; ii++)
    {
        err = MAX(ABS(BLASFEO_DVECEL(V1, ii) - BLASFEO_DVECEL(V2, ii)), err);
    }
    if (err > 0)
    {
        printf("[TREEQP]: Error! Vectors are different (error = %2.2e)\n", err);
        blasfeo_print_tran_dvec(V1->m, V1, 0);
        blasfeo_print_tran_dvec(V2->m, V2, 0);
        exit(1);
    }
    return err;
}



// TODO(dimitris): weird name, probably edited by change_name.sh?
answer_t is_strmat_diagonal(struct blasfeo_dmat *M)
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



answer_t is_strmat_zero(struct blasfeo_dmat *M)
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



void print_blasfeo_target()
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
