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
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "treeqp/utils/profiling.h"
#include "treeqp/utils/utils.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/timing.h"

#if PROFILE > 0

int timers_calculate_size(int num_iter)
{
    int bytes = 0;

    #if PROFILE > 1
    bytes += 2*num_iter*sizeof(double);  // iter_times, min_iter_times
    bytes += 1*num_iter*sizeof(int);  // ls_iters
    #endif

    #if PROFILE > 2
    bytes += 8*num_iter*sizeof(double);  // stage_qps_times, ..., min_line_search_times
    #endif

    return bytes;
}



void timers_create(int num_iter, treeqp_profiling_t *timings, void *ptr)
{
    // char pointer
    char *c_ptr = (char *) ptr;

    timings->num_iter = num_iter;

    // doubles
    #if PROFILE > 1
    timings->iter_times = (double *)c_ptr;
    c_ptr += num_iter*sizeof(double);
    timings->min_iter_times = (double *)c_ptr;
    c_ptr += num_iter*sizeof(double);
    #endif

    #if PROFILE > 2
    timings->stage_qps_times = (double *)c_ptr;
    c_ptr += num_iter*sizeof(double);
    timings->build_dual_times = (double *)c_ptr;
    c_ptr += num_iter*sizeof(double);
    timings->newton_direction_times = (double *)c_ptr;
    c_ptr += num_iter*sizeof(double);
    timings->line_search_times = (double *)c_ptr;
    c_ptr += num_iter*sizeof(double);
    timings->min_stage_qps_times = (double *)c_ptr;
    c_ptr += num_iter*sizeof(double);
    timings->min_build_dual_times = (double *)c_ptr;
    c_ptr += num_iter*sizeof(double);
    timings->min_newton_direction_times = (double *)c_ptr;
    c_ptr += num_iter*sizeof(double);
    timings->min_line_search_times = (double *)c_ptr;
    c_ptr += num_iter*sizeof(double);
    #endif

    // ints
    #if PROFILE > 1
    timings->ls_iters = (int *)c_ptr;
    c_ptr += num_iter*sizeof(int);
    #endif

    assert((char *)ptr + timers_calculate_size(num_iter) >= c_ptr);
}



// assign NaNs everywhere
void timers_initialize(treeqp_profiling_t *timings)
{
    int num_iter = timings->num_iter;

    timings->total_time = 0.0/0.0;
    timings->run_indx = 0;

    #if PROFILE > 1
    for (int ii = 0; ii < num_iter; ii++)
    {
        timings->iter_times[ii] = 0.0/0.0;
        timings->ls_iters[ii] = 0;
    }
    #endif

    #if PROFILE > 2
    for (int ii = 0; ii < num_iter; ii++)
    {
        timings->stage_qps_times[ii] = 0.0/0.0;
        timings->build_dual_times[ii] = 0.0/0.0;
        timings->newton_direction_times[ii] = 0.0/0.0;
        timings->line_search_times[ii] = 0.0/0.0;
    }
    #endif
}



// update min timers
void timers_update(treeqp_profiling_t *timings)
{
    int run_indx = timings->run_indx;
    int num_iter = timings->num_iter;

    if (run_indx == 0)
    {
        timings->min_total_time = timings->total_time;
    }
    else
    {
        timings->min_total_time = MIN(timings->min_total_time, timings->total_time);
    }

    #if PROFILE > 1
    if (run_indx == 0)
    {
        for (int ii = 0; ii < num_iter; ii++)
        {
            timings->min_iter_times[ii] = timings->iter_times[ii];
        }
    }
    else
    {
        for (int ii = 0; ii < num_iter; ii++)
        {
            timings->min_iter_times[ii] = MIN(timings->min_iter_times[ii], timings->iter_times[ii]);
        }
    }
    if (run_indx == 0)
    {
        for (int ii = 0; ii < num_iter; ii++)
        {
            timings->total_ls_iter += timings->ls_iters[ii];
        }
    }
    #endif

    #if PROFILE > 2
    if (run_indx == 0)
    {
        for (int ii = 0; ii < num_iter; ii++)
        {
            timings->min_stage_qps_times[ii] = timings->stage_qps_times[ii];
            timings->min_build_dual_times[ii] = timings->build_dual_times[ii];
            timings->min_newton_direction_times[ii] = timings->newton_direction_times[ii];
            timings->min_line_search_times[ii] = timings->line_search_times[ii];
        }
    }
    else
    {
        for (int ii = 0; ii < num_iter; ii++)
        {
            timings->min_stage_qps_times[ii] =
                MIN(timings->min_stage_qps_times[ii], timings->stage_qps_times[ii]);
            timings->min_build_dual_times[ii] =
                MIN(timings->min_build_dual_times[ii], timings->build_dual_times[ii]);
            timings->min_newton_direction_times[ii] =
                MIN(timings->min_newton_direction_times[ii], timings->newton_direction_times[ii]);
            timings->min_line_search_times[ii] =
                MIN(timings->min_line_search_times[ii], timings->line_search_times[ii]);
        }
    }
    #endif
    timings->run_indx++;
}



void timers_print(treeqp_profiling_t *timings)
{
    #if PROFILE > 1
    int num_iter;
    for (num_iter = 0; num_iter < timings->num_iter; num_iter++)
    {
        if (isnan(timings->iter_times[num_iter])) break;
    }
    #endif

    #ifdef SAVE_DATA
    printf("\n!!! WARNING: detailed logging is on, timings are inaccurate !!!\n\n");
    #endif
    #if NREP < 10
    printf("\n!!! WARNING: algorithm run less than 10 times, timings may be inaccurate !!!\n\n");
    #endif
    #if PRINT_LEVEL > 1
    printf("\n!!! WARNING: print level is too high, timings may be inaccurate !!!\n\n");
    #endif

    #if PROFILE > 0
    printf("\nTotal time:\n\n");
    #if PROFILE > 1
    printf("> > > algorithm converged in (%d it):\t %10.4f ms\n\n", num_iter, timings->min_total_time*1e3);
    #else
    printf("> > > algorithm converged in:\t %10.4f ms\n\n", timings->min_total_time*1e3);
    #endif
    #endif

    #if PROFILE > 1
    printf("\nTimings per iteration:\n\n");
    for (int jj = 0; jj < num_iter; jj++)
    {
        printf("Iteration #%3d - %7.3f ms  (%3d ls iters. )\n",
            jj+1, timings->min_iter_times[jj]*1e3, timings->ls_iters[jj]);
    }
    #endif

    #if PROFILE > 2
    printf("\nTimings per operation:\n\n");

    double sum_stage_qps_times = 0.;
    double sum_build_dual_times = 0.;
    double sum_newton_direction_times = 0.;
    double sum_line_search_times = 0.;
    double sum_all = 0.;
    for (int jj = 0; jj < num_iter; jj++)
    {
        sum_stage_qps_times += timings->min_stage_qps_times[jj];
        sum_build_dual_times += timings->min_build_dual_times[jj];
        sum_newton_direction_times += timings->min_newton_direction_times[jj];
        sum_line_search_times += timings->min_line_search_times[jj];
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
        timings->total_ls_iter, sum_line_search_times*1e3, 100*sum_line_search_times/sum_all);
    // NOTE(dimitris): this sum would be equal to min_total_time if
    // a) the run where min_total_time occured coincides with the run where all min_times occured
    // b) the time for the calculation of the termination condition was zero
    printf("> > > sum all of the above:\t\t %10.4f ms\n", sum_all*1e3);

    // NOTE(dimitris): for the scenario formulation, stage_qps_time contains building Hessian blocks
    #endif
}

#endif  /* PROFILE > 0 */
