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


#ifndef TREEQP_UTILS_TIMING_H_
#define TREEQP_UTILS_TIMING_H_

#if defined(__APPLE__)
#include <mach/mach_time.h>
#else
#include <sys/stat.h>
#include <sys/time.h>
#endif

#include "treeqp/utils/types.h"

#if defined(__APPLE__)

typedef struct treeqp_timer_
{
    uint64_t tic;
    uint64_t toc;
    mach_timebase_info_data_t tinfo;
} treeqp_timer;

#else

typedef struct treeqp_timer_
{
    struct timeval tic;
    struct timeval toc;
} treeqp_timer;

#endif

void treeqp_tic(treeqp_timer* t);

double treeqp_toc(treeqp_timer* t);

#endif  /* TREEQP_UTILS_TIMING_H_ */
