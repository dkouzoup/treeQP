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


#include <stdio.h>

#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/utils/tree_utils.h"
#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"


void print_tree_ocp_qp_in(tree_ocp_qp_in *qp_in) {
    int_t ii, jj;
    int_t Nn = qp_in->N;
    real_t min, max;

    for (ii = 0; ii < Nn; ii++) {
        printf("* Node %d/%d (nx = %d, nu = %d) ---------------------------------\n\n",
            ii, Nn-1, qp_in->nx[ii],  qp_in->nu[ii]);

        // print bounds on x
        for (jj = 0; jj < qp_in->nx[ii]; jj++) {
            min = DVECEL_LIBSTR(&qp_in->xmin[ii], jj);
            if (min > -1e10) {  // TODO(dimitris): check opts->inf instead
                printf("%5.2f  ", min);
            } else {
                printf("-INF   ");
            }
            printf("<=  x_%d  <=  ", jj);
            max = DVECEL_LIBSTR(&qp_in->xmax[ii], jj);
            if (max < 1e10) {
                printf("%5.2f\n", max);
            } else {
                printf("  INF\n");
            }
        }
        printf("\n");

        // print bounds on u
        for (jj = 0; jj < qp_in->nu[ii]; jj++) {
            min = DVECEL_LIBSTR(&qp_in->umin[ii], jj);
            if (min > -1e10) {
                printf("%5.2f  ", min);
            } else {
                printf("-INF   ");
            }
            printf("<=  u_%d  <=  ", jj);
            max = DVECEL_LIBSTR(&qp_in->umax[ii], jj);
            if (max < 1e10) {
                printf("%5.2f\n", max);
            } else {
                printf("  INF\n");
            }
        }
        printf("\n\n");

        printf("Q[%d] = \n", ii);
        d_print_tran_strvec(qp_in->nx[ii], (struct d_strvec *) &qp_in->Q[ii], 0);
        printf("q[%d] = \n", ii);
        d_print_tran_strvec(qp_in->nx[ii], (struct d_strvec *) &qp_in->q[ii], 0);

        printf("R[%d] = \n", ii);
        d_print_tran_strvec(qp_in->nu[ii], (struct d_strvec *) &qp_in->R[ii], 0);
        printf("r[%d] = \n", ii);
        d_print_tran_strvec(qp_in->nu[ii], (struct d_strvec *) &qp_in->r[ii], 0);

        // TODO(dimitris): add scaling factor for weights
        // printf("real = %d\n\n", qp_in->tree[ii].real);
        if (ii > 0) {
            // TODO(dimitris): check that .m/.n of structs coincide with nx/nu
            jj = qp_in->tree[ii].dad;
            printf("A[%d] = \n", ii-1, qp_in->A[ii-1]);
            d_print_strmat(qp_in->nx[ii], qp_in->nx[jj], (struct d_strmat*) &qp_in->A[ii-1], 0, 0);
            printf("B[%d] = \n", ii-1);
            d_print_strmat(qp_in->nx[ii], qp_in->nu[jj], (struct d_strmat *) &qp_in->B[ii-1], 0, 0);
            printf("b[%d] = \n", ii-1);
            d_print_tran_strvec(qp_in->nx[ii], (struct d_strvec *) &qp_in->b[ii-1], 0);
        }
    }
}
