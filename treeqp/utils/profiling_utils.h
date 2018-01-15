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


#ifndef TREEQP_UTILS_PROFILING_UTILS_H_
#define TREEQP_UTILS_PROFILING_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/flags.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/timing.h"

// TODO(dimitris): this should be maxit
#define kMax 100

// total cpu time and ls iterations
#if PROFILE > 0
treeqp_timer tot_tmr;
double total_time;
double min_total_time;
int total_ls_iter;
#endif

// + cputime and ls iterations per iteration
#if PROFILE > 1
treeqp_timer iter_tmr;
double iter_times[kMax];
double min_iter_times[kMax];
int ls_iters[kMax];
#endif

// + time per key operation per iteration
#if PROFILE > 2
treeqp_timer tmr;
double stage_qps_times[kMax];
double min_stage_qps_times[kMax];
double build_dual_times[kMax];
double min_build_dual_times[kMax];
double newton_direction_times[kMax];
double min_newton_direction_times[kMax];
double line_search_times[kMax];
double min_line_search_times[kMax];
#endif

#if ALG == TREEQP_DUAL_NEWTON_SCENARIOS

// + finer profiling per key operation per iteration
#if PROFILE > 3
treeqp_timer sub_tmr;
double xopt_times[kMax];
double min_xopt_times[kMax];
double uopt_times[kMax];
double min_uopt_times[kMax];
double Zbar_times[kMax];
double min_Zbar_times[kMax];
double Lambda_blocks_times[kMax];
double min_Lambda_blocks_times[kMax];
#endif

#endif  // TREEQP_DUAL_NEWTON_SCENARIOS

// + temporary profiling for debugging purposes
#if PROFILE >  4
treeqp_timer tmp_tmr;
double tmp_time;
#endif

void initialize_timers(void);
#if PROFILE > 3
void reset_accumulative_timers(int iter);
#endif
void update_min_timers(int iter);
void print_timers(int newtonIter);
void write_timers_to_txt(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  // TREEQP_UTILS_PROFILING_UTILS_H_
