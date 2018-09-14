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


#ifndef TREEQP_UTILS_TREE_H_
#define TREEQP_UTILS_TREE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/utils/types.h"

// TODO(dimitris): MAKE INDEPENDENT OF ORDER (now HPIPM header must come first)
#ifndef TREE_MPC
#ifndef HPIPM_TREE_H_

struct node
{
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
#endif

int calculate_number_of_nodes(int md, int Nr, int Nh);

int get_number_of_parent_nodes(const int Nn, const struct node * const tree);

int get_robust_horizon(const int Nn, const struct node * const tree);

int tree_calculate_size(const int *nk);

return_t tree_create(const int *nk, struct node * tree, void *ptr);

void setup_multistage_tree(const int md, const int Nr, const int Nh, const int Nn, struct node * const tree);

void setup_multistage_tree_new(int md, int Nr, int Nh, int * nk);

return_t setup_tree(const int * const nkids, struct node * const tree);

return_t free_tree(struct node * const tree);

int number_of_nodes_from_nkids(const int * const nkids);

// TODO(dimitris): use this to eliminate Nn from input arguments in several (non time critical) functions
int number_of_nodes_from_tree(const struct node * const tree);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREEQP_UTILS_TREE_H_ */
