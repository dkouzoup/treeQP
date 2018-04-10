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

#include "treeqp/src/dual_Newton_common.h"

#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_blas.h"

// Cholesky factorization with regularization options
reg_result_t factorize_with_reg_opts(struct blasfeo_dmat *M, struct blasfeo_dmat *CholM,
    struct blasfeo_dvec *regMat, regType_t reg_type, double reg_tol)
{
    if (reg_type == TREEQP_NO_REGULARIZATION)
    {
        // factorize
        blasfeo_dpotrf_l(M->m, M, 0, 0, CholM, 0, 0);
    }
    else if (reg_type == TREEQP_ALWAYS_LEVENBERG_MARQUARDT)
    {
        // add regularization to diagonal elements and the factorize
        blasfeo_ddiaad(M->m, 1.0, regMat, 0, M, 0, 0);
        blasfeo_dpotrf_l(M->m, M, 0, 0, CholM, 0, 0);
    }
    else if (reg_type == TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT)
    {
        // factorize
        blasfeo_dpotrf_l(M->m, M, 0, 0, CholM, 0, 0);

        // check diagonal elements
        for (int jj = 0; jj < M->m; jj++)
        {
            if (BLASFEO_DMATEL(CholM, jj, jj) <= reg_tol)
            {
                // if small diagonal element is detected, regularize
                blasfeo_ddiaad(M->m, 1.0, regMat, 0, M, 0, 0);

                // re-factorize
                blasfeo_dpotrf_l(M->m, M, 0, 0, CholM, 0, 0);
                // printf("regularized Lambda[%d][%d]\n", ii, kk);
                // exit(1);
                break;
            }
        }
    }
}



// Cholesky factorization with regularization options
reg_result_t treeqp_dpotrf_l_with_reg_opts(struct blasfeo_dmat *M, struct blasfeo_dmat *CholM,
    regType_t reg_type, double reg_tol, double reg_val)
{
    reg_result_t res = TREEQP_NO_REGULARIZATION_ADDED;

    if (reg_type == TREEQP_NO_REGULARIZATION)
    {
        // factorize
        blasfeo_dpotrf_l(M->m, M, 0, 0, CholM, 0, 0);
    }
    else if (reg_type == TREEQP_ALWAYS_LEVENBERG_MARQUARDT)
    {
        // add regularization to diagonal elements and the factorize
        blasfeo_ddiare(M->m, reg_val, M, 0, 0);

        blasfeo_dpotrf_l(M->m, M, 0, 0, CholM, 0, 0);
        res = TREEQP_REGULARIZATION_ADDED;
    }
    else if (reg_type == TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT)
    {
        // factorize
        blasfeo_dpotrf_l(M->m, M, 0, 0, CholM, 0, 0);

        // check diagonal elements
        for (int jj = 0; jj < M->m; jj++)
        {
            if (BLASFEO_DMATEL(CholM, jj, jj) <= reg_tol)
            {
                // if small diagonal element is detected, regularize
                blasfeo_ddiare(M->m, reg_val, M, 0, 0);

                // re-factorize
                blasfeo_dpotrf_l(M->m, M, 0, 0, CholM, 0, 0);
                // printf("regularized Lambda[%d][%d]\n", ii, kk);
                // exit(1);
                res = TREEQP_REGULARIZATION_ADDED;
                break;
            }
        }
    }
    return res;
}



reg_result_t treeqp_dpotrf_l_mn_with_reg_opts(struct blasfeo_dmat *M, struct blasfeo_dmat *CholM,
    regType_t reg_type, double reg_tol, double reg_val)
{
    reg_result_t res = TREEQP_NO_REGULARIZATION_ADDED;

    if (reg_type == TREEQP_NO_REGULARIZATION)
    {
        // factorize
        blasfeo_dpotrf_l_mn(M->m, M->n, M, 0, 0, CholM, 0, 0);
    }
    else if (reg_type == TREEQP_ALWAYS_LEVENBERG_MARQUARDT)
    {
        // add regularization to diagonal elements and the factorize
        blasfeo_ddiare(M->m, reg_val, M, 0, 0);

        blasfeo_dpotrf_l_mn(M->m, M->n, M, 0, 0, CholM, 0, 0);
        res = TREEQP_REGULARIZATION_ADDED;
    }
    else if (reg_type == TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT)
    {
        // factorize
        blasfeo_dpotrf_l_mn(M->m, M->n, M, 0, 0, CholM, 0, 0);

        // check diagonal elements
        for (int jj = 0; jj < M->m; jj++)
        {
            if (BLASFEO_DMATEL(CholM, jj, jj) <= reg_tol)
            {
                // if small diagonal element is detected, regularize
                blasfeo_ddiare(M->m, reg_val, M, 0, 0);

                // re-factorize
                blasfeo_dpotrf_l_mn(M->m, M->n, M, 0, 0, CholM, 0, 0);
                // printf("regularized Lambda[%d][%d]\n", ii, kk);
                // exit(1);
                res = TREEQP_REGULARIZATION_ADDED;
                break;
            }
        }
    }
    return res;
}
