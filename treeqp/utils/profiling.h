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


#ifndef TREEQP_UTILS_PROFILING_H_
#define TREEQP_UTILS_PROFILING_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/utils/types.h"
#include "treeqp/utils/timing.h"

typedef struct treeqp_profiling_t_
{
    int num_iter;
    int run_indx;
// total cpu time and ls iterations
#if PROFILE > 0
    double total_time;
    double min_total_time;
    int total_ls_iter;
#endif

// + cputime and ls iterations per iteration
#if PROFILE > 1
    double *iter_times;
    double *min_iter_times;
    int *ls_iters;
#endif

// + time per key operation per iteration
#if PROFILE > 2
    double *stage_qps_times;
    double *min_stage_qps_times;
    double *build_dual_times;
    double *min_build_dual_times;
    double *newton_direction_times;
    double *min_newton_direction_times;
    double *line_search_times;
    double *min_line_search_times;
#endif

} treeqp_profiling_t;

// TODO(dimitris): no need for global variables
#if PROFILE > 1
treeqp_timer iter_tmr;
#endif

#if PROFILE > 2
treeqp_timer tmr;
#endif

int timers_calculate_size(int num_iter);

void timers_create(int num_iter, treeqp_profiling_t *timings, void *ptr);

void timers_initialize(treeqp_profiling_t *timings);

void timers_update(treeqp_profiling_t *timings);

void timers_print(treeqp_profiling_t *timings);

// TODO(dimitris): return return_t
void timers_write_to_txt(treeqp_profiling_t *timings);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREEQP_UTILS_PROFILING_H_ */
