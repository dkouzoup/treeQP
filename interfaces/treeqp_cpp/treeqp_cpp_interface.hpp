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

// TODO(dimitris): move to Makefile.rule
#define TREEQP_WITH_HPMPC

#include <vector>
#include <string>

#include "treeqp/src/tree_qp_common.h"
#include "treeqp/src/dual_newton_tree.h"

#if defined(TREEQP_WITH_HPMPC)
#include "treeqp/src/hpmpc_tree.h"
#endif

struct Solver
{
public:

    Solver();

    ~Solver();

    int Set(int N, std::string SolverName);

    int Create(tree_qp_in *QpIn);

    int Solve(tree_qp_in *QpIn, tree_qp_out *QpOut);

private:

    std::string SolverName;

    bool OptsCreated;
    bool WorkCreated;
    void *OptsMem;
    void *WorkMem;

    treeqp_tdunes_opts_t TdunesOpts;
    treeqp_tdunes_workspace TdunesWork;
#if defined(TREEQP_WITH_HPMPC)
    treeqp_hpmpc_opts_t HpmpcOpts;
    treeqp_hpmpc_workspace HpmpcWork;
#endif
};


struct TreeQp
{
public:

    TreeQp(std::vector<int> nx, std::vector<int> nu, std::vector<int> nc, std::vector<int> nk);

    ~TreeQp();

    // choose solver and create default options
    int SetSolver(std::string SolverName);

    // initialize solver (NOTE: options _cannot_ be changed upon initialization)
    int CreateSolver();

    // set fields of QpIn
    void SetVector(std::string FieldName, std::vector<double> v, int indx);

    void SetMatrixColMajor(std::string FieldName, std::vector<double> v, int indx);

    void SetMatrixColMajor(std::string FieldName, std::vector<double> v, int lda, int indx);

    // solve QP
    void Solve();

    // utils
    void PrintInput();

    void PrintOutput();

    void PrintOutput(int NumNodes);

private:

    int NumNodes;

    tree_qp_in QpIn;
    void *QpInMem;

    tree_qp_out QpOut;
    void *QpOutMem;

    Solver QpSolver;
};

#endif  /* TREEQP_CPP_INTERFACE_HPP_ */
