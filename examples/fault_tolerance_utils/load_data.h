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

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/utils/types.h"

// data of one pruned tree
typedef struct input_data_ {
    int Nn;
    int *nc;
    int *nx;
    int *nu;

    double *A;
    double *B;
    double *b;

    double *Qd;
    double *Rd;
    double *q;
    double *r;
} input_data;

// simulation data of one spring configuration
typedef struct sim_data_ {
    double *A;
    double *B;
    double *b;
} sim_data;

input_data *load_data( );

input_data *load_ms_data( );

input_data *load_nominal_data( );

sim_data *load_sim_data( );

int get_number_of_realizations( );
int get_number_of_trees( );
int get_number_of_ms_trees( );
bool pruned_tree_exists(int config);
bool ms_tree_exists(int config);
int get_nx( );
int get_nu( );
double *get_ptr_transition_matrix( );

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  // EXAMPLES_FAULT_TOLERANCE_UTILS_LOAD_DATA_H_