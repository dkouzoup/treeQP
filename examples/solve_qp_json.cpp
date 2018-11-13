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
#include "treeqp/src/dual_Newton_scenarios.h"
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
#include <boost/range/iterator_range.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

using nlohmann::json;

regType_t convert_reg_type(const std::string& str)
{
    if (str == "TREEQP_NO_REGULARIZATION")
        return TREEQP_NO_REGULARIZATION;
    else if (str == "TREEQP_ALWAYS_LEVENBERG_MARQUARDT")
        return TREEQP_ALWAYS_LEVENBERG_MARQUARDT;
    else if (str == "TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT")
        return TREEQP_ON_THE_FLY_LEVENBERG_MARQUARDT;
    else
        return TREEQP_UNKNOWN_REGULARIZATION;
}



std::vector<double> readVector(json const& js, size_t N)
{
    std::vector<double> v(N);

    if (N == 1)
    {
        v[0] = js;
    }
    else
    {
        for (size_t i = 0; i < N; ++i)
            v[i] = js.at(i);
    }
    return v;
}



std::vector<double> readColMajorMatrix(json const& js, size_t M, size_t N)
{
    std::vector<double> v(M * N);

    if (M == 1)
        return readVector(js, N);
    if (N == 1)
        return readVector(js, M);

    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            v[i + j * M] = js.at(i).at(j);

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

        std::vector<double> buf(nx[i+1]);
        tree_qp_out_get_edge_lam(buf.data(), &qp_out, i);
        j_edge["lam"] = buf;
    }

    return j_sol;
}



void sdunes_update_multipliers(double *lam_scen, double *mu_scen, treeqp_sdunes_workspace *work)
{
    int ind_mu = 0;
    int ind_lam = 0;

    int nu = work->su[0][0].m;
    int nx = work->sx[0][0].m;
    int Ns = work->Ns;
    int Nh = work->Nh;

    for (int ii = 0; ii < Ns; ii++)
    {
        for (int kk = 0; kk < Nh; kk++)
        {
            blasfeo_unpack_dvec(nx, &work->smu[ii][kk], 0, &mu_scen[ind_mu]);
            ind_mu += nx;
        }
        if (ii < Ns-1)
        {
            blasfeo_unpack_dvec(work->slambda[ii].m, &work->slambda[ii], 0, &lam_scen[ind_lam]);
            ind_lam += work->slambda[ii].m;
        }
    }
}



void tdunes_update_multipliers(double *lam_tree, tree_qp_out *qp_out)
{
    int num_nodes = qp_out->info.Nn;
    int idx = 0;

    for (int ii = 0; ii < num_nodes-1; ii++)
    {
        tree_qp_out_get_edge_lam(&lam_tree[idx], qp_out, ii);
        idx += qp_out->lam[ii].m;
    }
}



int main(int argc, char * argv[])
{
    json j_in, j_out;

    // optional json file to overwrite constraint on x0 and initialization of lam0
    bool overwrite = false;
    json j_init;

    if (argc > 2)
    {
        std::ifstream(argv[2]) >> j_init;
        overwrite = true;
    }

    if (argc > 1)
    {
        // If a file name argument is specified, read from a file
        std::ifstream(argv[1]) >> j_in;
    }
    else
    {
        throw std::invalid_argument("no input files");
    }

    auto const& nodes = j_in.at("nodes");
    auto & edges = j_in.at("edges");
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

    if (overwrite)
    {
        if (nx.at(0) > 0)
        {
            std::vector<double> const x0 = readVector(j_init.at("x0"), nx.at(0));

            tree_qp_in_set_node_xmin(x0.data(), &qp_in, 0);
            tree_qp_in_set_node_xmax(x0.data(), &qp_in, 0);
        }
        else
        {
            // TODO(dimitris): should instead change b0
        }
    }

    // tree_qp_in_print(&qp_in);

    // set up QP solution
    tree_qp_out qp_out;

    int qp_out_size = tree_qp_out_calculate_size(num_nodes, nx.data(), nu.data(), nc.data());
    void *qp_out_memory = malloc(qp_out_size);
    tree_qp_out_create(num_nodes, nx.data(), nu.data(), nc.data(), &qp_out, qp_out_memory);

    // read solver name from json file
    std::string solver;

    if (j_in.count("options"))
    {
        solver = j_in.at("options").at("solver");
    }
    else
    {
        solver = "tdunes";
    }

    int status, prev_status, num_iter;
    double min_time;

    void *opts_memory;
    void *solver_memory;

    treeqp_tdunes_workspace tdunes_work;
    treeqp_sdunes_workspace sdunes_work;
    treeqp_hpmpc_workspace hpmpc_work;

    // eliminate x0
    std::vector<double> x0_bkp(qp_in.nx[0]);
    tree_qp_in_get_node_xmin(x0_bkp.data(), &qp_in, 0);
    tree_qp_in_eliminate_x0(&qp_in);
    // tree_qp_out_eliminate_x0(&qp_out);

    // set up QP solver and solve QP
    if (solver == "tdunes")
    {
        treeqp_tdunes_opts_t tdunes_opts;
        int tdunes_opts_size = treeqp_tdunes_opts_calculate_size(num_nodes);
        opts_memory = malloc(tdunes_opts_size);
        treeqp_tdunes_opts_create(num_nodes, &tdunes_opts, opts_memory);
        treeqp_tdunes_opts_set_default(num_nodes, &tdunes_opts);

        for (int ii = 0; ii < num_nodes; ii++)
        {
            tdunes_opts.qp_solver[ii] = TREEQP_QPOASES_SOLVER;
        }

        // read solver-specific options from json file
        if (j_in.count("options"))
        {
            auto const& options = j_in.at("options");

            tdunes_opts.maxIter = options["maxit"];
            tdunes_opts.stationarityTolerance = options["stationarityTolerance"];

            tdunes_opts.lineSearchMaxIter = options["lineSearchMaxIter"];
            tdunes_opts.lineSearchBeta = options["lineSearchBeta"];
            tdunes_opts.lineSearchGamma = options["lineSearchGamma"];

            tdunes_opts.checkLastActiveSet = options["checkLastActiveSet"];

            for (int ii = 0; ii < num_nodes; ii++)
            {
                if (options["clipping"])
                {
                    tdunes_opts.qp_solver[ii] = TREEQP_CLIPPING_SOLVER;
                }
                else
                {
                    tdunes_opts.qp_solver[ii] = TREEQP_QPOASES_SOLVER;
                }
            }

            tdunes_opts.regType = convert_reg_type(options["regType"]);
            tdunes_opts.regTol = options["regTol"];
            tdunes_opts.regValue = options["regValue"];
        }

        int tdunes_solver_size = treeqp_tdunes_calculate_size(&qp_in, &tdunes_opts);
        solver_memory = malloc(tdunes_solver_size);
        treeqp_tdunes_create(&qp_in, &tdunes_opts, &tdunes_work, solver_memory);

        int dim_lam = total_number_of_dynamic_constraints(&qp_in);
        std::vector<double> lam0_tree(dim_lam, 0.0);

        // TODO(dimitris): check that size in json is the same as calc. here
        if (overwrite)
        {
            lam0_tree = readVector(j_init.at("lam0_tree"), dim_lam);
        }

        for (int ii = 0; ii < NREP; ii++) // TODO(dimitris): NREP in options instead
        {
            treeqp_tdunes_set_dual_initialization(lam0_tree.data(), &tdunes_work);

            status = treeqp_tdunes_solve(&qp_in, &qp_out, &tdunes_opts, &tdunes_work);

            if (ii == 0)
            {
                min_time = qp_out.info.total_time;
                num_iter = qp_out.info.iter;
            }
            else
            {
                min_time = MIN(min_time, qp_out.info.total_time);
                assert(status == prev_status);
                assert(num_iter == qp_out.info.iter);
            }
            prev_status = status;
        }
        // min_time = tdunes_work.timings.min_total_time;
        tdunes_update_multipliers(lam0_tree.data(), &qp_out);
        j_out["init"]["lam0_tree"] = lam0_tree;
    }
    else if (solver == "sdunes")
    {
        treeqp_sdunes_opts_t sdunes_opts;
        int sdunes_opts_size = treeqp_sdunes_opts_calculate_size(num_nodes);
        opts_memory = malloc(sdunes_opts_size);
        treeqp_sdunes_opts_create(num_nodes, &sdunes_opts, opts_memory);
        treeqp_sdunes_opts_set_default(num_nodes, &sdunes_opts);

        // read solver-specific options from json file
        if (j_in.count("options"))
        {
            auto const& options = j_in.at("options");

            sdunes_opts.maxIter = options["maxit"];
            sdunes_opts.stationarityTolerance = options["stationarityTolerance"];

            sdunes_opts.lineSearchMaxIter = options["lineSearchMaxIter"];
            sdunes_opts.lineSearchBeta = options["lineSearchBeta"];
            sdunes_opts.lineSearchGamma = options["lineSearchGamma"];

            sdunes_opts.checkLastActiveSet = options["checkLastActiveSet"];

            sdunes_opts.regType = convert_reg_type(options["regType"]);
            sdunes_opts.regTol = options["regTol"];
            sdunes_opts.regValue = options["regValue"];
        }

        int sdunes_solver_size = treeqp_sdunes_calculate_size(&qp_in, &sdunes_opts);
        solver_memory = malloc(sdunes_solver_size);
        treeqp_sdunes_create(&qp_in, &sdunes_opts, &sdunes_work, solver_memory);

        int dim_lam = treeqp_sdunes_calculate_dual_dimension(sdunes_work.Nr, sdunes_work.md, sdunes_work.su[0][0].m);
        int dim_mu  = sdunes_work.Ns*sdunes_work.Nh*sdunes_work.sx[0][0].m;

        std::vector<double> lam0_scen(dim_lam, 0.0);
        std::vector<double> mu0_scen(dim_mu, 0.0);

        // TODO(dimitris): check that size in json is the same as calc. here
        if (overwrite)
        {
            lam0_scen = readVector(j_init.at("lam0_scen"), dim_lam);
            mu0_scen  = readVector(j_init.at("mu0_scen"), dim_mu);
        }

        for (int ii = 0; ii < NREP; ii++)
        {
            treeqp_sdunes_set_dual_initialization(lam0_scen.data(), mu0_scen.data(), &sdunes_work);

            status = treeqp_sdunes_solve(&qp_in, &qp_out, &sdunes_opts, &sdunes_work);

            if (ii == 0)
            {
                min_time = qp_out.info.total_time;
                num_iter = qp_out.info.iter;
            }
            else
            {
                min_time = MIN(min_time, qp_out.info.total_time);
                assert(status == prev_status);
                assert(num_iter == qp_out.info.iter);
            }
            prev_status = status;
        }
        sdunes_update_multipliers(lam0_scen.data(), mu0_scen.data(), &sdunes_work);
        j_out["init"]["lam0_scen"] = lam0_scen;
        j_out["init"]["mu0_scen"] = mu0_scen;
    }
    else if (solver == "hpmpc")
    {
        treeqp_hpmpc_opts_t hpmpc_opts;
        int hpmpc_opts_size = treeqp_hpmpc_opts_calculate_size(num_nodes);
        opts_memory = malloc(hpmpc_opts_size);
        treeqp_hpmpc_opts_create(num_nodes, &hpmpc_opts, opts_memory);
        treeqp_hpmpc_opts_set_default(num_nodes, &hpmpc_opts);

        int hpmpc_solver_size = treeqp_hpmpc_calculate_size(&qp_in, &hpmpc_opts);
        solver_memory = malloc(hpmpc_solver_size);
        treeqp_hpmpc_create(&qp_in, &hpmpc_opts, &hpmpc_work, solver_memory);

        // read solver-specific options from json file
        if (j_in.count("options"))
        {
            auto const& options = j_in.at("options");

            hpmpc_opts.maxIter = options.at("maxit");

            // TODO(dimitris): do this check also in tdunes/sdunes
            if (options.count("mu0"))
            {
                hpmpc_opts.mu0 = options["mu0"];
            }
            if (options.count("mu_tol"))
            {
                hpmpc_opts.mu_tol = options["mu_tol"];
            }
            if (options.count("alpha_min"))
            {
                hpmpc_opts.alpha_min = options["alpha_min"];
            }
        }

        for (int ii = 0; ii < NREP; ii++)
        {
            status = treeqp_hpmpc_solve(&qp_in, &qp_out, &hpmpc_opts, &hpmpc_work);

            if (ii == 0)
            {
                // NOTE(dimitris): do not take into account interface overhead for HPMPC
                min_time = qp_out.info.solver_time;
                num_iter = qp_out.info.iter;
            }
            else
            {
                min_time = MIN(min_time, qp_out.info.solver_time);
                assert(status == prev_status);
                assert(num_iter == qp_out.info.iter);
            }
            prev_status = status;
        }
    }
    else
    {
        return -1;
    }

    // write output to json file
    j_out["solution"] = qpSolutionToJson(qp_out, nx, nu, nc);

    // restore x0
    j_out["solution"]["nodes"][0]["x"] = x0_bkp;

    double const kkt_err = tree_qp_out_max_KKT_res(&qp_in, &qp_out);

    j_out["info"]["solver"] = solver;
    j_out["info"]["cpu_time"] = min_time;
    j_out["info"]["status"] = status;
    j_out["info"]["num_iter"] = qp_out.info.iter;
    j_out["info"]["kkt_tol"] = kkt_err;

    if (solver == "tdunes")
    {
        #if PROFILE > 1
        std::vector<double> buf(num_iter);
        for (int jj = 0; jj < num_iter; jj++) buf[jj] = tdunes_work.timings.ls_iters[jj];
        j_out["info"]["ls_iters"] = buf;
        #endif
        #if PROFILE > 2
            // TODO: do the same for the rest
            j_out["info"]["cpu_times_stage_qps"] = boost::make_iterator_range(
                tdunes_work.timings.min_stage_qps_times, tdunes_work.timings.min_stage_qps_times + num_iter);

            for (int jj = 0; jj < num_iter; jj++) buf[jj] = tdunes_work.timings.min_stage_qps_times[jj];
            j_out["info"]["cpu_times_stage_qps"] = buf;
            for (int jj = 0; jj < num_iter; jj++) buf[jj] = tdunes_work.timings.min_build_dual_times[jj];
            j_out["info"]["cpu_times_dual_system"] = buf;
            for (int jj = 0; jj < num_iter; jj++) buf[jj] = tdunes_work.timings.min_newton_direction_times[jj];
            j_out["info"]["cpu_times_newton_direction"] = buf;
            for (int jj = 0; jj < num_iter; jj++) buf[jj] = tdunes_work.timings.min_line_search_times[jj];
            j_out["info"]["cpu_times_line_search"] = buf;
        #endif
    }

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
