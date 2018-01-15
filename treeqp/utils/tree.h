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


#ifndef TREEQP_UTILS_TREE_UTILS_H_
#define TREEQP_UTILS_TREE_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/utils/types.h"

// TODO(dimitris): MAKE INDEPENDENT OF ORDER (now HPMPC header must come first)
#ifndef TREE_MPC

struct node {
    int *kids;   // 64 bits
    int idx;     // 32 bits
    int dad;     // 32 bits
    int nkids;   // 32 bits
    int stage;   // 32 bits
    int real;    // 32 bits
    int idxkid;  // 32 bits
    // total       256 bits
};

#endif

int calculate_number_of_nodes(int md, int Nr, int Nh);
int get_number_of_parent_nodes(int Nn, struct node *tree);
int get_robust_horizon(int Nn, struct node *tree);
void print_node(struct node *tree);
void setup_multistage_tree(int md, int Nr, int Nh, int Nn, struct node *tree);
void setup_tree(int Nn, int *nkids, struct node *tree);
void free_tree(int Nn, struct node *tree);


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  // TREEQP_UTILS_TREE_UTILS_H_
