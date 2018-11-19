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

#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_d_aux.h>

#include "treeqp/utils/print.h"


// TODO(dimitris): code duplicated in solve_qp_json
static regType_t string_to_reg_type(const std::string& str)
{
    if (str == "TREEQP_NO_REGULARIZATION")
    {
        return TREEQP_NO_REGULARIZATION;
    }
    else if (str == "TREEQP_ALWAYS_LEVENBERG_MARQUARDT")
    {
        return TREEQP_ALWAYS_LEVENBERG_MARQUARDT;
    }
    else if (str == "TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT")
    {
        return TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT;
    }
    else
    {
        return TREEQP_UNKNOWN_REGULARIZATION;
    }
}



static void create_qp_in(tree_qp_in *QpIn, void **QpInMem,
    std::vector<int> nx, std::vector<int> nu, std::vector<int> nc, std::vector<int> nk)
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

    int NumNodes = nx.size();

    int in_size = tree_qp_in_calculate_size(NumNodes, nx.data(), nu.data(), nc_ptr, nk.data());
    *QpInMem = malloc(in_size);
    tree_qp_in_create(NumNodes, nx.data(), nu.data(), nc_ptr, nk.data(), QpIn, *QpInMem);
}



static void create_qp_in(tree_qp_in *QpIn, void **QpInMem, int NumNodes, const int * nx,
    const int * nu, const int * nc, const int * nk)
{
    int in_size = tree_qp_in_calculate_size(NumNodes, nx, nu, nc, nk);
    *QpInMem = malloc(in_size);
    tree_qp_in_create(NumNodes, nx, nu, nc, nk, QpIn, *QpInMem);
}



static void create_qp_out(tree_qp_out *QpOut, void **QpOutMem,
    std::vector<int> nx, std::vector<int> nu, std::vector<int> nc)
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

    int NumNodes = nx.size();

    int out_size = tree_qp_out_calculate_size(NumNodes, nx.data(), nu.data(), nc_ptr);
    *QpOutMem = malloc(out_size);
    tree_qp_out_create(NumNodes, nx.data(), nu.data(), nc_ptr, QpOut, *QpOutMem);
}



QpSolver::~QpSolver()
{
    free(OptsMem);
    free(WorkMem);
    free(DummyQpInMem);
}



void QpSolver::FreeWorkspace()
{
    free(WorkMem);
}



TdunesSolver::TdunesSolver(struct TreeQp *Qp)
{
    // TODO: use friend instead
    tree_qp_in *QpIn = Qp->GetQpInPtr();

    // create dummy qp_in to store dimensions
    // TODO: move to function (or write create_qp_in based on struct TreeQp)
    int *nk = (int *) malloc(QpIn->N*sizeof(int));
    for (int ii = 0; ii < QpIn->N; ii++)
    {
        nk[ii] = QpIn->tree[ii].nkids;
    }
    create_qp_in(&DummyQpIn, &DummyQpInMem, QpIn->N, QpIn->nx, QpIn->nu, QpIn->nc, nk);

    CreateOptions();

    CreateWorkspace();
}



void TdunesSolver::CreateOptions()
{
    int NumNodes = DummyQpIn.N;

    // create default options
    int size = treeqp_tdunes_opts_calculate_size(NumNodes);
    OptsMem = malloc(size);
    treeqp_tdunes_opts_create(NumNodes, &TdunesOpts, OptsMem);
    treeqp_tdunes_opts_set_default(NumNodes, &TdunesOpts);
}



void TdunesSolver::CreateWorkspace()
{
    // create default options
    int size = treeqp_tdunes_calculate_size(&DummyQpIn, &TdunesOpts);
    WorkMem = malloc(size);
    treeqp_tdunes_create(&DummyQpIn, &TdunesOpts, &TdunesWork, WorkMem);
}



int TdunesSolver::SetOption(std::string field, std::string val)
{
    if (field == "regType")
    {
        TdunesOpts.regType = string_to_reg_type(val);
    }
    else
    {
        return -1;
    }

    FreeWorkspace();
    CreateWorkspace();

    return 0;
}



int TdunesSolver::SetOption(std::string field, bool val)
{
    int NumNodes = DummyQpIn.N;

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

    FreeWorkspace();
    CreateWorkspace();

    return 0;
}



int TdunesSolver::SetOption(std::string field, int val)
{
    if (field == "maxIter")
    {
        TdunesOpts.maxIter = val;
    }
    else
    {
        return -1;
    }

    FreeWorkspace();
    CreateWorkspace();

    return 0;
}



int TdunesSolver::SetOption(std::string field, double val)
{
    if (field == "stationarityTolerance")
    {
        TdunesOpts.stationarityTolerance = val;
    }
    else if (field == "regValue")
    {
        TdunesOpts.regValue = val;
    }
    else if (field == "regTol")
    {
        TdunesOpts.regTol = val;
    }
    else
    {
        return -1;
    }

    FreeWorkspace();
    CreateWorkspace();

    return 0;
}



int TdunesSolver::Solve(struct TreeQp *Qp)
{
    int status = -1;

    tree_qp_in *QpIn = Qp->GetQpInPtr();
    tree_qp_out *QpOut = Qp->GetQpOutPtr();

    status = treeqp_tdunes_solve(QpIn, QpOut, &TdunesOpts, &TdunesWork);

    return status;
}


// TODO: CheckDims fun that throws error if QP of solve has different dims

HpmpcSolver::HpmpcSolver(struct TreeQp *Qp)
{
    tree_qp_in *QpIn = Qp->GetQpInPtr();

    // create dummy qp_in to store dimensions
    int *nk = (int *) malloc(QpIn->N*sizeof(int));
    for (int ii = 0; ii < QpIn->N; ii++)
    {
        nk[ii] = QpIn->tree[ii].nkids;
    }
    create_qp_in(&DummyQpIn, &DummyQpInMem, QpIn->N, QpIn->nx, QpIn->nu, QpIn->nc, nk);

    // copy constraints (needed to infer number of bounds in hpmpc)
    for (int ii = 0; ii < QpIn->N; ii++)
    {
        blasfeo_dveccp(QpIn->xmin[ii].m, &QpIn->xmin[ii], 0, &DummyQpIn.xmin[ii], 0);
        blasfeo_dveccp(QpIn->xmax[ii].m, &QpIn->xmax[ii], 0, &DummyQpIn.xmax[ii], 0);
        blasfeo_dveccp(QpIn->umin[ii].m, &QpIn->umin[ii], 0, &DummyQpIn.umin[ii], 0);
        blasfeo_dveccp(QpIn->umax[ii].m, &QpIn->umax[ii], 0, &DummyQpIn.umax[ii], 0);
    }

    CreateOptions();

    CreateWorkspace();

    free(nk);
}



void HpmpcSolver::CreateOptions()
{
    int NumNodes = DummyQpIn.N;

    // create default options
    int size = treeqp_hpmpc_opts_calculate_size(NumNodes);
    OptsMem = malloc(size);
    treeqp_hpmpc_opts_create(NumNodes, &HpmpcOpts, OptsMem);
    treeqp_hpmpc_opts_set_default(NumNodes, &HpmpcOpts);
}



void HpmpcSolver::CreateWorkspace()
{
    int size = treeqp_hpmpc_calculate_size(&DummyQpIn, &HpmpcOpts);
    WorkMem = malloc(size);
    treeqp_hpmpc_create(&DummyQpIn, &HpmpcOpts, &HpmpcWork, WorkMem);
}



int HpmpcSolver::Solve(struct TreeQp *Qp)
{
    int status = -1;

    tree_qp_in *QpIn = Qp->GetQpInPtr();
    tree_qp_out *QpOut = Qp->GetQpOutPtr();

    status = treeqp_hpmpc_solve(QpIn, QpOut, &HpmpcOpts, &HpmpcWork);

    return status;
}



int HpmpcSolver::SetOption(std::string field, std::string val)
{
    return -1;

    // FreeWorkspace();
    // CreateWorkspace();

    // return 0;
}



int HpmpcSolver::SetOption(std::string field, bool val)
{
    return -1;

    // int NumNodes = DummyQpIn.N;

    // FreeWorkspace();
    // CreateWorkspace();

    // return 0;
}



int HpmpcSolver::SetOption(std::string field, int val)
{
    if (field == "maxIter")
    {
        HpmpcOpts.maxIter = val;
    }
    else
    {
        return -1;
    }

    FreeWorkspace();
    CreateWorkspace();

    return 0;
}



int HpmpcSolver::SetOption(std::string field, double val)
{
    if (field == "alpha_min")
    {
        HpmpcOpts.alpha_min = val;
    }
    else
    {
        return -1;
    }

    FreeWorkspace();
    CreateWorkspace();

    return 0;
}



// TODO(dimitris): throw error if vectors are not of the same size
TreeQp::TreeQp(std::vector<int> nx, std::vector<int> nu, std::vector<int> nc, std::vector<int> nk)
{
    NumNodes = nx.size();

    // create qp_in
    create_qp_in(&QpIn, &QpInMem, nx, nu, nc, nk);

    // create qp_out
    create_qp_out(&QpOut, &QpOutMem, nx, nu, nc);
}



TreeQp::~TreeQp()
{
    free(QpInMem);
    free(QpOutMem);
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



tree_qp_in *TreeQp::GetQpInPtr()
{
    return &QpIn;
}



tree_qp_out *TreeQp::GetQpOutPtr()
{
    return &QpOut;
}