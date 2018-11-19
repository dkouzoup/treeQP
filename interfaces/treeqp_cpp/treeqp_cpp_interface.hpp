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
#include "treeqp/src/hpmpc_tree.h"


// Abstract solver base class.
class QpSolver
{
public:

    virtual int Solve(struct TreeQp *Qp) = 0;

    virtual void CreateOptions() = 0;

    virtual void CreateWorkspace() = 0;

    void FreeWorkspace();

    // destroy and re-create solver based on current options
    virtual int SetOption(std::string field, std::string val) = 0;

    virtual int SetOption(std::string field, bool val) = 0;

    virtual int SetOption(std::string field, int val) = 0;

    virtual int SetOption(std::string field, double val) = 0;

    // common destructor for all solvers
    ~QpSolver();

protected:

    // dummy QP to store dimensions of created solver (these dimensions cannot be changed)
    tree_qp_in DummyQpIn;
    void *DummyQpInMem;

    // pointers to allocated memory for options and workspace
    void *OptsMem;
    void *WorkMem;
};



class TdunesSolver
:    public QpSolver
{
public:

    TdunesSolver(struct TreeQp *Qp);

    void CreateOptions() override;

    void CreateWorkspace() override;

    int SetOption(std::string field, std::string val) override;

    int SetOption(std::string field, bool val) override;

    int SetOption(std::string field, int val) override;

    int SetOption(std::string field, double val) override;

    int Solve(struct TreeQp *Qp) override;

private:

    treeqp_tdunes_opts_t TdunesOpts;

    treeqp_tdunes_workspace TdunesWork;
};



class HpmpcSolver
:    public QpSolver
{
public:

    HpmpcSolver(struct TreeQp *Qp);

    void CreateOptions() override;

    void CreateWorkspace() override;

    int SetOption(std::string field, std::string val) override;

    int SetOption(std::string field, bool val) override;

    int SetOption(std::string field, int val) override;

    int SetOption(std::string field, double val) override;

    int Solve(struct TreeQp *Qp) override;

private:

    treeqp_hpmpc_opts_t HpmpcOpts;

    treeqp_hpmpc_workspace HpmpcWork;
};



struct TreeQp
{
public:

    TreeQp(std::vector<int> nx, std::vector<int> nu, std::vector<int> nc, std::vector<int> nk);

    ~TreeQp();

    // set fields of QpIn
    void SetVector(std::string FieldName, std::vector<double> v, int indx);

    void SetMatrixColMajor(std::string FieldName, std::vector<double> v, int indx);

    void SetMatrixColMajor(std::string FieldName, std::vector<double> v, int lda, int indx);

    // utils
    void PrintInput();

    void PrintOutput();

    void PrintOutput(int NumNodes);

    // TODO(dimitris): other way?
    tree_qp_in *GetQpInPtr();

    tree_qp_out *GetQpOutPtr();

private:

    int NumNodes;

    tree_qp_in QpIn;
    void *QpInMem;

    tree_qp_out QpOut;
    void *QpOutMem;
};

#endif  /* TREEQP_CPP_INTERFACE_HPP_ */
