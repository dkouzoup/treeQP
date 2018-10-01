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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "treeqp/src/tree_qp_common.h"
#include "treeqp/src/dual_Newton_tree.h"
#include "treeqp/src/hpmpc_tree.h"

#include "treeqp/utils/types.h"
#include "treeqp/utils/tree.h"
#include "treeqp/utils/profiling.h"
#include "treeqp/utils/print.h"
#include "treeqp/utils/blasfeo.h"

#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_v_aux_ext_dep.h>
#include <blasfeo_d_aux_ext_dep.h>
#include <blasfeo_d_blas.h>

#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>

// #define USE_HPMPC


std::vector<double> readColMajorMatrix(nlohmann::json const& js, size_t M, size_t N)
{
    std::vector<double> v(M * N);

    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            v[i + j * M] = js.at(i).at(j);

    return v;
}


std::vector<double> readVector(nlohmann::json const& js, size_t N)
{
    std::vector<double> v(N);

    for (size_t i = 0; i < N; ++i)
        v[i] = js.at(i);

    return v;
}


int main(int argc, char * argv[])
{
    printf("hello hangover\n");

    nlohmann::json j;

    if (argc > 1)
        // If a file name argument is specified, read from a file
        std::ifstream(argv[1]) >> j;
    else
        // Otherwise, read from the stdin.
        std::cin >> j;

    auto const& nodes = j.at("nodes");
    auto const& edges = j.at("edges");

    // Fill nx, nu, nc
    std::vector<int> nx, nu, nc;
    nx.reserve(nodes.size());
    nu.reserve(nodes.size());
    nc.reserve(nodes.size());
    for (auto const& node : nodes)
    {
        nx.push_back(node.at("q").size());
        nu.push_back(node.at("r").size());

        auto const ld = node.find("ld");
        nc.push_back(ld != node.end() ? ld.value().size() : 0);
    }

    // Fill nk
    std::vector<int> nk(nodes.size(), 0);

    for (auto const& edge : edges)
        ++nk.at(edge.at("from"));

    std::cout << "nk=\t";
    for (auto n : nk)
        std::cout << n << "\t";
    std::cout << std::endl;

    std::cout << "nx=\t";
    for (auto n : nx)
        std::cout << n << "\t";
    std::cout << std::endl;

    std::cout << "nu=\t";
    for (auto n : nu)
        std::cout << n << "\t";
    std::cout << std::endl;

    std::cout << "nc=\t";
    for (auto n : nc)
        std::cout << n << "\t";
    std::cout << std::endl;


    // set up QP data
    tree_qp_in qp_in;

    int qp_in_size = tree_qp_in_calculate_size(nodes.size(), nx.data(), nu.data(), nc.data(), nk.data());
    void *qp_in_memory = malloc(qp_in_size);
    tree_qp_in_create(nodes.size(), nx.data(), nu.data(), nc.data(), nk.data(), &qp_in, qp_in_memory);

    for (auto const& edge : edges)
    {
        int const to = edge.at("to");
        int const from = edge.at("from");

        std::vector<double> const A = readColMajorMatrix(edge.at("A"), nx.at(to), nx.at(from));
        std::vector<double> const B = readColMajorMatrix(edge.at("B"), nx.at(to), nu.at(from));
        std::vector<double> const b = readVector(edge.at("b"), nx.at(to));

        tree_qp_in_set_edge_A_colmajor(A.data(), -1, &qp_in, to - 1);
        tree_qp_in_set_edge_B_colmajor(B.data(), -1, &qp_in, to - 1);
        tree_qp_in_set_edge_b(b.data(), &qp_in, to - 1);
    }

    for (size_t i = 0; i < nodes.size(); ++i)
    {
        auto const node = nodes.at(i);

        std::vector<double> const Q = readColMajorMatrix(node.at("Q"), nx.at(i), nx.at(i));
        std::vector<double> const R = readColMajorMatrix(node.at("R"), nu.at(i), nu.at(i));
        std::vector<double> const S = readColMajorMatrix(node.at("S"), nu.at(i), nx.at(i));

        std::vector<double> const q = readVector(node.at("q"), nx.at(i));
        std::vector<double> const r = readVector(node.at("r"), nu.at(i));

        std::vector<double> const lx = readVector(node.at("lx"), nx.at(i));
        std::vector<double> const lu = readVector(node.at("lu"), nu.at(i));
        std::vector<double> const ux = readVector(node.at("ux"), nx.at(i));
        std::vector<double> const uu = readVector(node.at("uu"), nu.at(i));

        tree_qp_in_set_node_Q_colmajor(Q.data(), -1, &qp_in, i);
        tree_qp_in_set_node_R_colmajor(R.data(), -1, &qp_in, i);
        tree_qp_in_set_node_S_colmajor(S.data(), -1, &qp_in, i);
        tree_qp_in_set_node_q(q.data(), &qp_in, i);
        tree_qp_in_set_node_r(r.data(), &qp_in, i);

        tree_qp_in_set_node_xmin(lx.data(), &qp_in, i);
        tree_qp_in_set_node_umin(lu.data(), &qp_in, i);
        tree_qp_in_set_node_xmax(ux.data(), &qp_in, i);
        tree_qp_in_set_node_umax(uu.data(), &qp_in, i);

    }

    // tree_qp_in_print(&qp_in);

    // set up QP solution
    tree_qp_out qp_out;

    int qp_out_size = tree_qp_out_calculate_size(nodes.size(), nx.data(), nu.data(), nc.data());
    void *qp_out_memory = malloc(qp_out_size);
    tree_qp_out_create(nodes.size(), nx.data(), nu.data(), nc.data(), &qp_out, qp_out_memory);

#if 0
    // eliminate x0 variable
    tree_qp_in_eliminate_x0(&qp_in);
    tree_qp_out_eliminate_x0(&qp_out);
#endif

    // set up QP solver
#ifndef USE_HPMPC

    treeqp_tdunes_opts_t opts;
    int tdunes_opts_size = treeqp_tdunes_opts_calculate_size(nodes.size());
    void *opts_memory = malloc(tdunes_opts_size);
    treeqp_tdunes_opts_create(nodes.size(), &opts, opts_memory);
    treeqp_tdunes_opts_set_default(nodes.size(), &opts);

    opts.maxIter = 1000;
    opts.stationarityTolerance = 1.0e-6;
    opts.lineSearchMaxIter = 100;  
    // opts.regType  = TREEQP_NO_REGULARIZATION;

    for (int ii = 0; ii < nodes.size(); ii++) opts.qp_solver[ii] = TREEQP_QPOASES_SOLVER;

    treeqp_tdunes_workspace work;

    int treeqp_size = treeqp_tdunes_calculate_size(&qp_in, &opts);
    void *qp_solver_memory = malloc(treeqp_size);
    treeqp_tdunes_create(&qp_in, &opts, &work, qp_solver_memory);

    // tree_qp_in_print(&qp_in);
    // return 0;

#else
    treeqp_hpmpc_opts_t opts;
    int hpmpc_opts_size = treeqp_hpmpc_opts_calculate_size(nodes.size());
    void *opts_memory = malloc(hpmpc_opts_size);
    treeqp_hpmpc_opts_create(nodes.size(), &opts, opts_memory);
    treeqp_hpmpc_opts_set_default(nodes.size(), &opts);

    treeqp_hpmpc_workspace work;

    int treeqp_size = treeqp_hpmpc_calculate_size(&qp_in, &opts);
    void *qp_solver_memory = malloc(treeqp_size);
    treeqp_hpmpc_create(&qp_in, &opts, &work, qp_solver_memory);
#endif  // USE_HPMPC

    // solve QP
#ifndef USE_HPMPC
    treeqp_tdunes_solve(&qp_in, &qp_out, &opts, &work);
#else
    treeqp_hpmpc_solve(&qp_in, &qp_out, &opts, &work);
#endif

#ifndef DATA
    tree_qp_out_print(nodes.size(), &qp_out);
#endif

#ifndef USE_HPMPC
    printf("SOLVER:\ttdunes\n");
#else
    printf("SOLVER:\thpmpc\n");
#endif

    printf("ITERS:\t%d\n", qp_out.info.iter);

    double kkt_err = tree_qp_out_max_KKT_res(&qp_in, &qp_out);
    printf("KKT:\t%2.2e\n", kkt_err);

    free(qp_solver_memory);
    free(opts_memory);
    free(qp_out_memory);
    free(qp_in_memory);

    assert(kkt_err < 1e-6 && "maximum KKT residual too high!");

    return 0;
}
