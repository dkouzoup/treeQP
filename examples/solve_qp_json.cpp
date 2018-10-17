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
#include <iomanip>
#include <string>

// #define USE_HPMPC

using nlohmann::json;


std::vector<double> readColMajorMatrix(json const& js, size_t M, size_t N)
{
    std::vector<double> v(M * N);

    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            v[i + j * M] = js.at(i).at(j);

    return v;
}


std::vector<double> readVector(json const& js, size_t N)
{
    std::vector<double> v(N);

    for (size_t i = 0; i < N; ++i)
        v[i] = js.at(i);

    return v;
}


json qpSolutionToJson(tree_qp_out const& qp_out, std::vector<int> const& nx,
    std::vector<int> const& nu, std::vector<int> const& nc)
{
    size_t const n_nodes = nx.size();
    assert(nu.size() == n_nodes);
    assert(nc.size() == n_nodes);

    json j_sol;

    // process nodes
    for (size_t i = 0; i < n_nodes; ++i)
    {
        auto& j_node = j_sol["nodes"][i];

        {
            std::vector<double> buf(nx[i]);

            tree_qp_out_get_node_x(buf.data(), &qp_out, i);
            j_node["x"] = buf;

            tree_qp_out_get_node_mu_x(buf.data(), &qp_out, i);
            j_node["mu_x"] = buf;
        }

        {
            std::vector<double> buf(nu[i]);

            tree_qp_out_get_node_u(buf.data(), &qp_out, i);
            j_node["u"] = buf;

            tree_qp_out_get_node_mu_u(buf.data(), &qp_out, i);
            j_node["mu_u"] = buf;
        }

        {
            std::vector<double> buf(nc[i]);
            tree_qp_out_get_node_mu_d(buf.data(), &qp_out, i);
            j_node["mu_d"] = buf;
        }
    }

    // process edges
    for (size_t i = 0; i + 1 < n_nodes; ++i)
    {
        auto& j_edge = j_sol["edges"][i];

        std::vector<double> buf(nx[i + 1]);
        tree_qp_out_get_edge_lam(buf.data(), &qp_out, i);
        j_edge["lam"] = buf;
    }

    return j_sol;
}


int main(int argc, char * argv[])
{
    json j_in;

    if (argc > 1)
        // If a file name argument is specified, read from a file
        std::ifstream(argv[1]) >> j_in;
    else
        // Otherwise, read from the stdin.
        std::cin >> j_in;

    auto const& nodes = j_in.at("nodes");
    auto const& edges = j_in.at("edges");
    size_t const num_nodes = nodes.size();

    // Fill nx, nu, nc
    std::vector<int> nx, nu, nc;
    nx.reserve(num_nodes);
    nu.reserve(num_nodes);
    nc.reserve(num_nodes);
    for (auto const& node : nodes)
    {
        nx.push_back(node.at("q").size());
        nu.push_back(node.at("r").size());

        auto const ld = node.find("ld");
        nc.push_back(ld != node.end() ? ld.value().size() : 0);
    }

    // Fill nk
    std::vector<int> nk(num_nodes, 0);

    for (auto const& edge : edges)
        ++nk.at(edge.at("from"));

    // set up QP data
    tree_qp_in qp_in;

    int qp_in_size = tree_qp_in_calculate_size(num_nodes, nx.data(), nu.data(), nc.data(), nk.data());
    void *qp_in_memory = malloc(qp_in_size);
    tree_qp_in_create(num_nodes, nx.data(), nu.data(), nc.data(), nk.data(), &qp_in, qp_in_memory);

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

    for (size_t i = 0; i < num_nodes; ++i)
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

    int qp_out_size = tree_qp_out_calculate_size(num_nodes, nx.data(), nu.data(), nc.data());
    void *qp_out_memory = malloc(qp_out_size);
    tree_qp_out_create(num_nodes, nx.data(), nu.data(), nc.data(), &qp_out, qp_out_memory);

#if 0
    // eliminate x0 variable
    tree_qp_in_eliminate_x0(&qp_in);
    tree_qp_out_eliminate_x0(&qp_out);
#endif

    // read options
    int maxit;
    std::string solver;

    if (j_in.count("options"))
    {
        auto const& options = j_in.at("options");

        // common options
        maxit = options["maxit"];

        // solver
        solver = options["solver"];
    }
    else
    {
        // TODO: set default if opts don't exist
        maxit = 1000;
        solver = "tdunes";
    }

    int status;

    void *opts_memory;
    void *solver_memory;

    // set up QP solver and solve QP
    if (solver.compare("tdunes") == 0)
    {
        treeqp_tdunes_opts_t tdunes_opts;
        int tdunes_opts_size = treeqp_tdunes_opts_calculate_size(num_nodes);
        void *opts_memory = malloc(tdunes_opts_size);
        treeqp_tdunes_opts_create(num_nodes, &tdunes_opts, opts_memory);
        treeqp_tdunes_opts_set_default(num_nodes, &tdunes_opts);

        tdunes_opts.maxIter = maxit;
        tdunes_opts.stationarityTolerance = 1.0e-6;
        tdunes_opts.lineSearchMaxIter = 100;
        // tdunes_opts.regType  = TREEQP_NO_REGULARIZATION;

        for (int ii = 0; ii < num_nodes; ii++)
            tdunes_opts.qp_solver[ii] = TREEQP_QPOASES_SOLVER;

        treeqp_tdunes_workspace tdunes_work;

        int tdunes_solver_size = treeqp_tdunes_calculate_size(&qp_in, &tdunes_opts);
        solver_memory = malloc(tdunes_solver_size);
        treeqp_tdunes_create(&qp_in, &tdunes_opts, &tdunes_work, solver_memory);

        status = treeqp_tdunes_solve(&qp_in, &qp_out, &tdunes_opts, &tdunes_work);

    }
    else if (solver.compare("hpmpc") == 0)
    {
        treeqp_hpmpc_opts_t hpmpc_opts;
        int hpmpc_opts_size = treeqp_hpmpc_opts_calculate_size(num_nodes);
        void *opts_memory = malloc(hpmpc_opts_size);
        treeqp_hpmpc_opts_create(num_nodes, &hpmpc_opts, opts_memory);
        treeqp_hpmpc_opts_set_default(num_nodes, &hpmpc_opts);

        hpmpc_opts.maxIter = maxit;

        treeqp_hpmpc_workspace hpmpc_work;

        int hpmpc_solver_size = treeqp_hpmpc_calculate_size(&qp_in, &hpmpc_opts);
        solver_memory = malloc(hpmpc_solver_size);
        treeqp_hpmpc_create(&qp_in, &hpmpc_opts, &hpmpc_work, solver_memory);

        status = treeqp_hpmpc_solve(&qp_in, &qp_out, &hpmpc_opts, &hpmpc_work);
    }
    else
    {
        return -1;
    }

    // write output to json file
    json j_out;
    j_out["solution"] = qpSolutionToJson(qp_out, nx, nu, nc);
    if (solver.compare("tdunes") == 0)
    {
        j_out["info"]["solver"] = "tdunes";
        j_out["info"]["cpu_time"] = qp_out.info.total_time;
    }
    else if (solver.compare("hpmpc") == 0)
    {
        j_out["info"]["solver"] = "hpmpc";
        // NOTE(dimitris): do not take into account interface overhead for HPMPC
        j_out["info"]["cputime"] = qp_out.info.solver_time;
    }

    j_out["info"]["status"] = status;
    j_out["info"]["num_iter"] = qp_out.info.iter;

    double const kkt_err = tree_qp_out_max_KKT_res(&qp_in, &qp_out);
    j_out["info"]["kkt_tol"] = kkt_err;


    free(solver_memory);
    free(opts_memory);
    free(qp_out_memory);
    free(qp_in_memory);

    if (kkt_err >= 1e-6)
        std::cerr << "maximum KKT residual too high!" << std::endl;

    // Write json to stdout
    std::cout << std::setw(4) << j_out << std::endl;

    return 0;
}
