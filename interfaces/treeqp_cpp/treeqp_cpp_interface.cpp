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

#include "interfaces/treeqp_cpp/treeqp_cpp_interface.hpp"

#include <vector>
#include <string>
#include <iostream>

#include "treeqp/utils/print.h"

Solver::Solver()
{
    OptsCreated = false;
    WorkCreated = false;
}


Solver::~Solver()
{
    if (OptsCreated == true)
    {
        free(OptsMem);
    }
    if (WorkCreated == true)
    {
        free(WorkMem);
    }
}



int Solver::Create(std::string SolverName, tree_qp_in *QpIn)
{
    int status;

    status = CreateOptions(QpIn->N, SolverName);
    if (status == -1) return status;

    status = CreateWorkspace(QpIn);
    if (status == -1) return status;

    return 0;
}



int Solver::CreateOptions(int N, std::string SolverName)
{
    NumNodes = N;

    // free memory of previous solver (if applicable)
    if (OptsCreated == true)
    {
        free(OptsMem);
        OptsCreated = false;
    }

    // create default options of selected solver
    int size;

    if (SolverName == "tdunes")
    {
        size = treeqp_tdunes_opts_calculate_size(NumNodes);
        OptsMem = malloc(size);
        treeqp_tdunes_opts_create(NumNodes, &TdunesOpts, OptsMem);
        treeqp_tdunes_opts_set_default(NumNodes, &TdunesOpts);
    }
#if defined (TREEQP_WITH_HPMPC)
    else if (SolverName == "hpmpc")
    {
        size = treeqp_hpmpc_opts_calculate_size(NumNodes);
        OptsMem = malloc(size);
        treeqp_hpmpc_opts_create(NumNodes, &HpmpcOpts, OptsMem);
        treeqp_hpmpc_opts_set_default(NumNodes, &HpmpcOpts);
    }
#endif
    else
    {
        return -1;
    }

    this->SolverName = SolverName;

    OptsCreated = true;
    return 0;
}



int Solver::CreateWorkspace(tree_qp_in *QpIn)
{
    if (OptsCreated == false)
    {
        return -1;
    }
    if (WorkCreated == true)
    {
        free(WorkMem);
        WorkCreated = false;
    }

    int size;

    if (SolverName == "tdunes")
    {
        size = treeqp_tdunes_calculate_size(QpIn, &TdunesOpts);
        WorkMem = malloc(size);
        treeqp_tdunes_create(QpIn, &TdunesOpts, &TdunesWork, WorkMem);
    }
#if defined (TREEQP_WITH_HPMPC)
    else if (SolverName == "hpmpc")
    {
        size = treeqp_hpmpc_calculate_size(QpIn, &HpmpcOpts);
        WorkMem = malloc(size);
        treeqp_hpmpc_create(QpIn, &HpmpcOpts, &HpmpcWork, WorkMem);
    }
#endif
    else
    {
        return -1;
    }

    WorkCreated = true;
    return 0;
}



int Solver::Solve(tree_qp_in *QpIn, tree_qp_out *QpOut)
{
    int status = -1;

    if (SolverName == "tdunes")
    {
        status = treeqp_tdunes_solve(QpIn, QpOut, &TdunesOpts, &TdunesWork);
    }
#if defined (TREEQP_WITH_HPMPC)
    else if (SolverName == "hpmpc")
    {
        status = treeqp_hpmpc_solve(QpIn, QpOut, &HpmpcOpts, &HpmpcWork);
    }
#endif
    return status;
}



int Solver::ChangeOption(tree_qp_in *QpIn, std::string field, bool val)
{
    if (SolverName == "tdunes")
    {
        if (field == "clipping")
        {
            for (int ii = 0; ii < NumNodes; ii++)
            {
                if (val == true)
                {
                    TdunesOpts.qp_solver[ii] = TREEQP_CLIPPING_SOLVER;
                }
                else
                {
                    TdunesOpts.qp_solver[ii] = TREEQP_QPOASES_SOLVER;
                }
            }
        }
        else
        {
            return -1;
        }
    }
#if defined (TREEQP_WITH_HPMPC)
    else if (SolverName == "hpmpc")
    {

    }
#endif
    else
    {
        return -1;
    }

    CreateWorkspace(QpIn);

    return 0;
}



int Solver::ChangeOption(tree_qp_in *QpIn, std::string field, int val)
{
    if (SolverName == "tdunes")
    {
        if (field == "maxIter")
        {
            TdunesOpts.maxIter = val;
        }
        else
        {
            return -1;
        }
    }
#if defined (TREEQP_WITH_HPMPC)
    else if (SolverName == "hpmpc")
    {
        if (field == "maxIter")
        {
            HpmpcOpts.maxIter = val;
        }
        else
        {
            return -1;
        }
    }
#endif
    else
    {
        return -1;
    }

    CreateWorkspace(QpIn);

    return 0;
}



int Solver::ChangeOption(tree_qp_in *QpIn, std::string field, double val)
{
    if (SolverName == "tdunes")
    {
        if (field == "stationarityTolerance")
        {
            TdunesOpts.stationarityTolerance = val;
        }
        else
        {
            return -1;
        }
    }
#if defined (TREEQP_WITH_HPMPC)
    else if (SolverName == "hpmpc")
    {
        if (field == "alpha_min")
        {
            HpmpcOpts.alpha_min = val;
        }
        else
        {
            return -1;
        }
    }
#endif
    else
    {
        return -1;
    }

    CreateWorkspace(QpIn);

    return 0;
}



// TODO(dimitris): throw error if vectors are not of the same size
TreeQp::TreeQp(std::vector<int> nx, std::vector<int> nu, std::vector<int> nc, std::vector<int> nk)
{
    int *nc_ptr;
    if (nc.empty())
    {
        nc_ptr = NULL;
    }
    else
    {
        nc_ptr = nc.data();
    }

    NumNodes = nx.size();

    // create qp_in
    int in_size = tree_qp_in_calculate_size(NumNodes, nx.data(), nu.data(), nc_ptr, nk.data());
    QpInMem = malloc(in_size);
    tree_qp_in_create(NumNodes, nx.data(), nu.data(), nc_ptr, nk.data(), &QpIn, QpInMem);

    // create qp_out
    int out_size = tree_qp_out_calculate_size(NumNodes, nx.data(), nu.data(), nc_ptr);
    QpOutMem = malloc(out_size);
    tree_qp_out_create(NumNodes, nx.data(), nu.data(), nc_ptr, &QpOut, QpOutMem);

}



TreeQp::~TreeQp()
{
    free(QpInMem);
    free(QpOutMem);
}



void TreeQp::SolverName(std::string SolverName)
{
    int status = QpSolver.Create(SolverName, &QpIn);
}


// TODO(dimitris): proper error handling

void TreeQp::SetVector(std::string FieldName, std::vector<double> v, int indx)
{
    if (FieldName == "q")
    {
        tree_qp_in_set_node_q(v.data(), &QpIn, indx);
    }
    else if (FieldName == "r")
    {
        tree_qp_in_set_node_r(v.data(), &QpIn, indx);
    }
    else if (FieldName == "xmin")
    {
        tree_qp_in_set_node_xmin(v.data(), &QpIn, indx);
    }
    else if (FieldName == "xmax")
    {
        tree_qp_in_set_node_xmax(v.data(), &QpIn, indx);
    }
    else if (FieldName == "umin")
    {
        tree_qp_in_set_node_umin(v.data(), &QpIn, indx);
    }
    else if (FieldName == "umax")
    {
        tree_qp_in_set_node_umax(v.data(), &QpIn, indx);
    }
    else if (FieldName == "b")
    {
        tree_qp_in_set_edge_b(v.data(), &QpIn, indx);
    }
}



void TreeQp::SetMatrixColMajor(std::string FieldName, std::vector<double> v, int indx)
{
    SetMatrixColMajor(FieldName, v, -1, indx);
}



void TreeQp::SetMatrixColMajor(std::string FieldName, std::vector<double> v, int lda, int indx)
{
    if (FieldName == "Q")
    {
        tree_qp_in_set_node_Q_colmajor(v.data(), lda, &QpIn, indx);
    }
    else if (FieldName == "R")
    {
        tree_qp_in_set_node_R_colmajor(v.data(), lda, &QpIn, indx);
    }
    else if (FieldName == "S")
    {
        tree_qp_in_set_node_S_colmajor(v.data(), lda, &QpIn, indx);
    }
    else if (FieldName == "A")
    {
        tree_qp_in_set_edge_A_colmajor(v.data(), lda, &QpIn, indx);
    }
    else if (FieldName == "B")
    {
        tree_qp_in_set_edge_B_colmajor(v.data(), lda, &QpIn, indx);
    }
}



void TreeQp::SetOption(std::string field, bool val)
{
    QpSolver.ChangeOption(&QpIn, field, val);
}



void TreeQp::SetOption(std::string field, int val)
{
    QpSolver.ChangeOption(&QpIn, field, val);
}



void TreeQp::SetOption(std::string field, double val)
{
    QpSolver.ChangeOption(&QpIn, field, val);
}



void TreeQp::Solve()
{
    QpSolver.Solve(&QpIn, &QpOut);
}



void TreeQp::PrintInput()
{
    tree_qp_in_print(&QpIn);
}



void TreeQp::PrintOutput()
{
    tree_qp_out_print(NumNodes, &QpOut);
}



void TreeQp::PrintOutput(int NumNodes)
{
    tree_qp_out_print(NumNodes, &QpOut);
}