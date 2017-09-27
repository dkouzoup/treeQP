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

#ifndef EXAMPLES_FAULT_TOLERANCE_UTILS_LOAD_DATA_H_
#define EXAMPLES_FAULT_TOLERANCE_UTILS_LOAD_DATA_H_

#ifdef __cplusplus
extern "C" {
#endif

#define NOMINAL_MPC

#include "treeqp/utils/types.h"

// data of one pruned tree
typedef struct input_data_ {
    int_t Nn;
    int_t *nc;
    int_t *nx;
    int_t *nu;

    real_t *A;
    real_t *B;
    real_t *b;

    real_t *Qd;
    real_t *Rd;
    real_t *q;
    real_t *r;
} input_data;

// simulation data of one spring configuration
typedef struct sim_data_ {
    real_t *A;
    real_t *B;
    real_t *b;
} sim_data;

input_data *load_data( );

input_data *load_nominal_data( );

sim_data *load_sim_data( );

int_t get_number_of_realizations( );
int_t get_number_of_trees( );
int_t get_nx( );
int_t get_nu( );
real_t *get_ptr_transition_matrix( );

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  // EXAMPLES_FAULT_TOLERANCE_UTILS_LOAD_DATA_H_