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


#ifndef TREEQP_UTILS_PRINT_H_
#define TREEQP_UTILS_PRINT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/src/dual_Newton_common.h"
#include "treeqp/src/tree_qp_common.h"
#include "treeqp/utils/tree.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/profiling.h"



void node_print(const struct node *tree);

void tree_qp_in_print_dims(const tree_qp_in *qp_in);

void tree_qp_in_print(const tree_qp_in *qp_in);

void tree_qp_out_print(int Nn, const tree_qp_out *qp_out);

void tree_qp_out_write_to_txt(const tree_qp_in *qp_in, const tree_qp_out *qp_out, const char *fpath);

// TODO(dimitris): return return_t
void timers_write_to_txt(treeqp_profiling_t *timings);

void regularization_print_status(regType_t reg_type, reg_result_t reg_res);

void blasfeo_print_target(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREEQP_UTILS_PRINT_H_ */
