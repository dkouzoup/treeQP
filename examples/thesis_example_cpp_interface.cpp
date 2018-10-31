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

#include <vector>
#include <iostream>

#include "interfaces/treeqp_cpp/treeqp_cpp_interface.hpp"

int main(int argc, char ** argv)
{
    // set up dimensions and create QP instance

    std::vector<int> const nk = {2, 2, 1, 0, 0, 0};
    std::vector<int> const nx = {2, 2, 2, 2, 2, 2};
    std::vector<int> const nu = {1, 1, 1, 0, 0, 0};
    std::vector<int> const nc;

    TreeQp QP(nx, nu, nc, nk);


    // set up dynamics (NOTE: column major for matrices)

    std::vector<double>  A1 = {1.1, 3.3, 2.2, 4.4};
    std::vector<double>  A2 = {5.5, 7.7, 6.6, 8.8};
    std::vector<double>  B1 = {1.0, 2.0};
    std::vector<double>  B2 = {3.0, 4.0};
    std::vector<double>  b1 = {0.0, 0.0};
    std::vector<double>  b2 = {1.0, 1.0};

    for (int ii = 0; ii < 5; ii++)
    {
        if (ii == 0 || ii == 2)
        {
            QP.SetMatrixColMajor("A", A1, ii);
            QP.SetMatrixColMajor("B", B1, ii);
            QP.SetVector("b", b1, ii);
        }
        else
        {
            QP.SetMatrixColMajor("A", A2, ii);
            QP.SetMatrixColMajor("B", B2, ii);
            QP.SetVector("b", b2, ii);
        }
    }


    // set up objective

    std::vector<double> Q = {2.0, 0.0, 0.0, 2.0};
    std::vector<double> R = {1.0};
    std::vector<double> S = {0.0, 0.0};
    std::vector<double> q = {0.0, 0.0};
    std::vector<double> r = {0.0};

    int N = nk.size();

    for (int ii = 0; ii < N; ii++)
    {
        QP.SetMatrixColMajor("Q", Q, ii);
        QP.SetMatrixColMajor("R", R, ii);
        QP.SetMatrixColMajor("S", S, ii);
        QP.SetVector("q", q, ii);
        QP.SetVector("r", r, ii);
    }

    // set up constraints

    std::vector<double> x0 = {2.1, 2.1};
    std::vector<double> umin = {-1};
    std::vector<double> umax = {1};

    QP.SetVector("xmin", x0, 0);
    QP.SetVector("xmax", x0, 0);

    for (int ii = 0; ii < 3; ii++)
    {
        QP.SetVector("umin", umin, ii);
        QP.SetVector("umax", umax, ii);
    }


    // set up solver and adapt options

    QP.SolverName("tdunes");
    QP.SetOption("clipping", true);

    // QP.CreateSolver("hpmpc");

    // solve QP and print solution

    QP.Solve();

    // TODO(dimitris): fix valgrind errors in printing when I use hpmpc
    QP.PrintOutput();

    return 0;
}
