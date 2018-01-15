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


#ifndef TREEQP_UTILS_TYPES_H_
#define TREEQP_UTILS_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned int uint;

// TODO(dimitris): remove those definitions (exist in code generated files of fault tol. example)
typedef double real_t;
typedef int int_t;

// Boolean answer
typedef enum {
    YES,
    NO
} answer_t;

// Stopping criteria
typedef enum {
    TREEQP_SUMSQUAREDERRORS = 0,  // sum of squares
    TREEQP_TWONORM,               // 2-norm (square root of previous option)
    TREEQP_INFNORM,               // infinity norm
} termination_t;

// Exit codes
typedef enum {
    TREEQP_OK = 0,

    // exit status of QP solver
    TREEQP_SUCC_OPTIMAL_SOLUTION_FOUND,
    TREEQP_ERR_MAXIMUM_ITERATIONS_REACHED,

    // reading/writing to txt files
    TREEQP_ERR_ERROR_OPENING_FILE,

    TREEQP_ERR_UNKNOWN_ERROR,
} return_t;

// Stage QP solvers
typedef enum {
    TREEQP_CLIPPING_SOLVER = 0,
    TREEQP_DENSE_SOLVER,  // TODO(dimitris): NOT IMPLEMENTED YET
} stage_qp_t;

// Regularization of dual Hessian
typedef enum {
    TREEQP_NO_REGULARIZATION = 0,  // never regularize (solver may fail)
    TREEQP_ALWAYS_LEVENBERG_MARQUARDT,  // always use LM regularization (regValue in tree_options)
    TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT,  // regularize when diag. elements too small
} regType_t;


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  // TREEQP_UTILS_TYPES_H_
