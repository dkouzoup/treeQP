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


#ifndef TREEQP_CPP_INTERFACE_HPP_
#define TREEQP_CPP_INTERFACE_HPP_

// typedef enum {
//     TREEQP_TDUNES,
//     TREEQP_SDUNES,
// #ifdef TREEQP_WITH_HPMPC
//     TREEQP_HPMPC,
// #endif
// #ifdef TREEQP_WITH_HPIPM
//     TREEQP_HPIPM,
// #endif
// } treeqp_solver_t;

#include <vector>

// int tree_qp_in_calculate_size(int Nn, const int * nx, const int * nu, const int * nc, const int * nk);

struct TreeQp
{
public:
    TreeQp(std::vector<int> nx, std::vector<int> nu, std::vector<int> nc, std::vector<int> nk);

private:

};

#endif  /* TREEQP_CPP_INTERFACE_HPP_ */
