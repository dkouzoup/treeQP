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

#define TREEQP_INF 1e12

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

    // exit status of QP solvers (common)

    TREEQP_OPTIMAL_SOLUTION_FOUND,
    TREEQP_MAXIMUM_ITERATIONS_REACHED,

    // exit status of QP solvers (specific)

    TREEQP_DN_NOT_DESCENT_DIRECTION,  // typically NaN due to insufficient regularization
    TREEQP_DN_STAGE_QP_INIT_FAILED,  // when using qpOASES
    TREEQP_DN_STAGE_QP_SOLVE_FAILED,  // when using qpOASES
    TREEQP_IP_MIN_STEP,
    TREEQP_IP_UNKNOWN_FLAG,

    // misc

    TREEQP_OK,
    TREEQP_INVALID_OPTION,
    TREEQP_ERROR_OPENING_FILE,
    TREEQP_UNKNOWN_ERROR,

} return_t;


// Stage QP solvers
typedef enum {
    TREEQP_CLIPPING_SOLVER = 0,
    TREEQP_QPOASES_SOLVER,
} stage_qp_t;

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREEQP_UTILS_TYPES_H_ */
