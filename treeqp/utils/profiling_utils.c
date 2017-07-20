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


#include <stdio.h>

#include "treeqp/flags.h"
#include "treeqp/utils/profiling_utils.h"
#include "treeqp/utils/utils.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/timing.h"

#if PROFILE > 0

// assign NaNs everywhere
void initialize_timers(void) {
    total_time = 0.0/0.0;
    #if PROFILE > 1
    int_t ii;
    for (ii = 0; ii < kMax; ii++) {
        iter_times[ii] = 0.0/0.0;
        ls_iters[ii] = 0;
        }
    #endif
    #if PROFILE > 2
    for (ii = 0; ii < kMax; ii++) {
        stage_qps_times[ii] = 0.0/0.0;
        build_dual_times[ii] = 0.0/0.0;
        newton_direction_times[ii] = 0.0/0.0;
        line_search_times[ii] = 0.0/0.0;
    }
    #endif
    #if PROFILE > 3
    #if ALG == TREEQP_DUAL_NEWTON_SCENARIOS
    for (ii = 0; ii < kMax; ii++) {
        xopt_times[ii] = 0.0/0.0;
        uopt_times[ii] = 0.0/0.0;
        Zbar_times[ii] = 0.0/0.0;
        Lambda_blocks_times[ii] = 0.0/0.0;
    }
    #endif
    #endif
}

#if PROFILE > 3
void reset_accumulative_timers(int_t iter) {
    #if ALG == TREEQP_DUAL_NEWTON_SCENARIOS
    xopt_times[iter] = 0.0;
    uopt_times[iter] = 0.0;
    Zbar_times[iter] = 0.0;
    Lambda_blocks_times[iter] = 0.0;
    #endif
}
#endif


void update_min_timers(int_t iter) {
    int_t ii;
    if (iter == 0) {
        min_total_time = total_time;
    } else {
        min_total_time = MIN(min_total_time, total_time);
    }
    #if PROFILE > 1
    if (iter == 0) {
        for (ii = 0; ii < kMax; ii++) min_iter_times[ii] = iter_times[ii];
    } else {
        for (ii = 0; ii < kMax; ii++) min_iter_times[ii] =
            MIN(min_iter_times[ii], iter_times[ii]);
    }
    if (iter == 0) {
        for (ii = 0; ii < kMax; ii++) {
            total_ls_iter += ls_iters[ii];
        }
    }
    #endif
    #if PROFILE > 2
    if (iter == 0) {
        for (ii = 0; ii < kMax; ii++) {
            min_stage_qps_times[ii] = stage_qps_times[ii];
            min_build_dual_times[ii] = build_dual_times[ii];
            min_newton_direction_times[ii] = newton_direction_times[ii];
            min_line_search_times[ii] = line_search_times[ii];
        }
    } else {
        for (ii = 0; ii < kMax; ii++) {
            min_stage_qps_times[ii] = MIN(min_stage_qps_times[ii], stage_qps_times[ii]);
            min_build_dual_times[ii] = MIN(min_build_dual_times[ii], build_dual_times[ii]);
            min_newton_direction_times[ii] =
                MIN(min_newton_direction_times[ii], newton_direction_times[ii]);
            min_line_search_times[ii] = MIN(min_line_search_times[ii], line_search_times[ii]);
        }
    }
    #endif
    #if PROFILE > 3
    #if ALG == TREEQP_DUAL_NEWTON_SCENARIOS
    if (iter == 0) {
        for (ii = 0; ii < kMax; ii++) {
            min_xopt_times[ii] = xopt_times[ii];
            min_uopt_times[ii] = uopt_times[ii];
            min_Zbar_times[ii] = Zbar_times[ii];
            min_Lambda_blocks_times[ii] = Lambda_blocks_times[ii];
        }
    } else {
        for (ii = 0; ii < kMax; ii++) {
            min_xopt_times[ii] = MIN(min_xopt_times[ii], xopt_times[ii]);
            min_uopt_times[ii] = MIN(min_uopt_times[ii], uopt_times[ii]);
            min_Zbar_times[ii] = MIN(min_Zbar_times[ii], Zbar_times[ii]);
            min_Lambda_blocks_times[ii] = MIN(min_Lambda_blocks_times[ii], Lambda_blocks_times[ii]);
        }
    }
    #endif  // TREEQP_DUAL_NEWTON_SCENARIOS
    #ifdef QPDUNES_TREE
    // TODO(dimitris): PROFILE > 3 timings for tree
    #endif
    #endif
}


void print_timers(int_t newtonIter) {
    int_t jj;

    #if DEBUG == 1
    printf("\n!!! WARNING: detailed DEBUG is on, timings are inaccurate !!!\n\n");
    #endif
    #if NRUNS < 10
    printf("\n!!! WARNING: algorithm run less than 10 times, timings may be inaccurate !!!\n\n");
    #endif
    #if PRINT_LEVEL > 1
    printf("\n!!! WARNING: print level is too high, timings may be inaccurate !!!\n\n");
    #endif
    #if PROFILE > 3
    printf("\n!!! WARNING: profile level too high, total and iteration timings inaccurate !!!\n\n");
    #endif

    #if PROFILE > 0
    printf("\nTotal time:\n\n");
    printf("> > > algorithm converged in (%d it):\t %10.4f ms\n\n", newtonIter, min_total_time*1e3);
    #endif
    #if PROFILE > 1
    printf("\nTimings per iteration:\n\n");
    for (jj = 0; jj < newtonIter; jj++) {
        printf("Iteration #%3d - %7.3f ms  (%3d ls iters. )\n", jj+1,
            min_iter_times[jj]*1e3, ls_iters[jj]);
    }
    #endif

    #if PROFILE > 2
    printf("\nTimings per operation:\n\n");

    real_t sum_stage_qps_times = 0.;
    real_t sum_build_dual_times = 0.;
    real_t sum_newton_direction_times = 0.;
    real_t sum_line_search_times = 0.;
    real_t sum_all = 0.;
    for (jj = 0; jj < newtonIter; jj++) {
        sum_stage_qps_times += min_stage_qps_times[jj];
        sum_build_dual_times += min_build_dual_times[jj];
        sum_newton_direction_times += min_newton_direction_times[jj];
        sum_line_search_times += min_line_search_times[jj];
    }
    sum_all = sum_stage_qps_times + sum_build_dual_times + sum_newton_direction_times
        + sum_line_search_times;

    printf("> > > solved stage QPs in:\t\t %10.4f ms (%5.2f %%)\n",
        sum_stage_qps_times*1e3, 100*sum_stage_qps_times/sum_all);
    printf("> > > built dual problem in:\t\t %10.4f ms (%5.2f %%)\n",
        sum_build_dual_times*1e3, 100*sum_build_dual_times/sum_all);
    printf("> > > calculated Newton direction in: \t %10.4f ms (%5.2f %%)\n",
        sum_newton_direction_times*1e3, 100*sum_newton_direction_times/sum_all);
    printf("> > > performed line-search (%d it) in:\t %10.4f ms (%5.2f %%)\n",
        total_ls_iter, sum_line_search_times*1e3, 100*sum_line_search_times/sum_all);
    // NOTE(dimitris): this sum would be equal to min_total_time if
    // a) the run where min_total_time occured coincides with the run where all min_times occured
    // b) the time for the calculation of the termination condition was zero
    printf("> > > sum all of the above:\t\t %10.4f ms\n", sum_all*1e3);

    #if ALG == TREEQP_DUAL_NEWTON_SCENARIOS
    printf("\nNOTE: solve stage QPs timings contain building the dual Hessian blocks too!\n\n");
    #endif
    #endif

    #if PROFILE > 3
    #if ALG == TREEQP_DUAL_NEWTON_SCENARIOS
    printf("\nTimings per sub-operation:\n\n");
    real_t sum_xopt_times = 0.;
    real_t sum_uopt_times = 0.;
    real_t sum_Zbar_times = 0.;
    real_t sum_Lambda_blocks_times = 0.;
    real_t sum_stage_qps_detailed_times = 0.;
    for (jj = 0; jj < newtonIter; jj++) {
        sum_xopt_times += min_xopt_times[jj];
        sum_uopt_times += min_uopt_times[jj];
        sum_Zbar_times += min_Zbar_times[jj];
        sum_Lambda_blocks_times += min_Lambda_blocks_times[jj];
    }
    sum_stage_qps_detailed_times = sum_xopt_times + sum_uopt_times + sum_Zbar_times
        + sum_Lambda_blocks_times;

    printf("1) Solve stage QPs \n\n");
    printf("> > > calculated xopt in:\t\t %10.4f ms (%5.2f %%)\n",
        sum_xopt_times*1e3, 100*sum_xopt_times/sum_stage_qps_detailed_times);
    printf("> > > calculated uopt in:\t\t %10.4f ms (%5.2f %%)\n",
        sum_uopt_times*1e3, 100*sum_uopt_times/sum_stage_qps_detailed_times);
    printf("> > > calculated Zbar in:\t\t %10.4f ms (%5.2f %%)\n",
    sum_Zbar_times*1e3, 100*sum_Zbar_times/sum_stage_qps_detailed_times);
    printf("> > > calculated Lambda blocks in:\t %10.4f ms (%5.2f %%)\n",
        sum_Lambda_blocks_times*1e3, 100*sum_Lambda_blocks_times/sum_stage_qps_detailed_times);
    printf("> > > sum all of the above:\t\t %10.4f ms\n", sum_stage_qps_detailed_times*1e3);
    printf("> > > instead of:\t\t\t %10.4f ms\n", sum_stage_qps_times*1e3);
    printf("\n");
    #endif
    #ifdef QPDUNES_TREE
    printf("PROFILE > 2 not implemented yet.\n");
    #endif
    #endif
}


void write_timers_to_txt(void) {
    char fname[256];
    #ifdef QPDUNES_TREE
    char prefix[] = "data_tree";
    #endif
    #if ALG == TREEQP_DUAL_NEWTON_SCENARIOS
    char prefix[] = "examples/data_spring_mass";
    #endif

    #if PROFILE > 1
    snprintf(fname, sizeof(fname), "%s/%s.txt", prefix, "ls_iters");
    write_int_vector_to_txt(ls_iters, kMax, fname);
    #endif

    // NOTE(dimitris): do not save cpu time if PROFILE is too high (inaccurate results)
    #if PROFILE < 3

    snprintf(fname, sizeof(fname), "%s/%s.txt", prefix, "cputime");
    write_double_vector_to_txt(&min_total_time, 1, fname);

    #if PROFILE > 1
    snprintf(fname, sizeof(fname), "%s/%s.txt", prefix, "iter_times");
    write_double_vector_to_txt(min_iter_times, kMax, fname);
    #endif

    #endif  /* PROFILE < 3 */

    #if PROFILE > 2
    snprintf(fname, sizeof(fname), "%s/%s.txt", prefix, "stage_qps_times");
    write_double_vector_to_txt(min_stage_qps_times, kMax, fname);
    snprintf(fname, sizeof(fname), "%s/%s.txt", prefix, "build_dual_times");
    write_double_vector_to_txt(min_build_dual_times, kMax, fname);
    snprintf(fname, sizeof(fname), "%s/%s.txt", prefix, "newton_direction_times");
    write_double_vector_to_txt(min_newton_direction_times, kMax, fname);
    snprintf(fname, sizeof(fname), "%s/%s.txt", prefix, "line_search_times");
    write_double_vector_to_txt(min_line_search_times, kMax, fname);
    #endif
}

#endif  /* PROFILE > 0 */
