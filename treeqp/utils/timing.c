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

#include "treeqp/utils/timing.h"

#if defined(__APPLE__)

void treeqp_tic(treeqp_timer* t)
{
    /* read current clock cycles */
    t->tic = mach_absolute_time();
}



double treeqp_toc(treeqp_timer* t)
{
    uint64_t duration; /* elapsed time in clock cycles*/

    t->toc = mach_absolute_time();
    duration = t->toc - t->tic;

    /*conversion from clock cycles to nanoseconds*/
    mach_timebase_info(&(t->tinfo));
    duration *= t->tinfo.numer;
    duration /= t->tinfo.denom;

    return (double)duration / 1e9;
}

#else

void treeqp_tic(treeqp_timer* t)
{
    /* read current clock cycles */
    gettimeofday(&t->tic, 0);
}



double treeqp_toc(treeqp_timer* t)
{
    struct timeval temp;

    gettimeofday(&t->toc, 0);

    if ((t->toc.tv_usec - t->tic.tv_usec) < 0)
    {
        temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec - 1;
        temp.tv_usec = 1000000 + t->toc.tv_usec - t->tic.tv_usec;
    } else
    {
        temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec;
        temp.tv_usec = t->toc.tv_usec - t->tic.tv_usec;
    }

    return (double)temp.tv_sec + (double)temp.tv_usec / 1e6;
}

#endif
