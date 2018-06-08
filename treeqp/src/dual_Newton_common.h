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


#ifndef DUAL_NEWTON_COMMON_H_
#define DUAL_NEWTON_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/utils/types.h"

#include <blasfeo_target.h>
#include <blasfeo_common.h>

// regularization options
typedef enum {
    TREEQP_NO_REGULARIZATION = 0,  // never regularize (solver may fail)
    TREEQP_ALWAYS_LEVENBERG_MARQUARDT,  // always use LM regularization (regValue in tree_options)
    TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT,  // regularize when diag. elements too small
} regType_t;


typedef enum {
    TREEQP_NO_REGULARIZATION_ADDED = 0,
    TREEQP_REGULARIZATION_ADDED,
} reg_result_t;

reg_result_t treeqp_dpotrf_l_with_reg_opts(struct blasfeo_dmat *M, struct blasfeo_dmat *CholM,
    regType_t reg_type, double reg_tol, double reg_val);

reg_result_t treeqp_dpotrf_l_mn_with_reg_opts(struct blasfeo_dmat *M, struct blasfeo_dmat *CholM,
    regType_t reg_type, double reg_tol, double reg_val);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* DUAL_NEWTON_COMMON_H_ */
