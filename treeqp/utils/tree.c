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
*    Authors: Dimitris Kouzoupis, Gianluca Frison, name.surname (at) imtek.uni-freiburg.de         *
*                                                                                                  *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "treeqp/utils/tree.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/utils.h"

int calculate_number_of_nodes(int md, int Nr, int Nh)
{
    int n_nodes;
    if (md == 1)  // i.e. standard block-banded structure
    {
        n_nodes = Nh+1;
    }
    else
    {
        n_nodes = (Nh-Nr)*ipow(md, Nr) + (ipow(md, Nr+1)-1)/(md-1);
    }
    return n_nodes;
}



int get_number_of_parent_nodes(int Nn, const struct node *tree)
{
    int Np = 0;

    for (int kk = 0; kk < Nn; kk++)
    {
        if (tree[kk].nkids > 0) Np++;
    }
    return Np;
}



int get_robust_horizon(int Nn, const struct node *tree)
{
    int Nr = 0;

    for (int kk = 0; kk < Nn; kk++)
    {
        if (tree[kk].nkids > 1)
        {
            Nr = tree[kk].stage+1;
        }
        else
        {
            break;
        }
    }
    return Nr;
}



// TODO(dimitris): check that result is correct for nominal MPC tree
int get_prediction_horizon(int Nn, const struct node *tree)
{
    int Nh = 0;
    int ii = Nn-1;

    for (int jj = 0; jj < Nn; jj++)
    {
        if (ii == 0)
        {
            return Nh;
        }
        ii = tree[ii].dad;
        Nh++;
    }
    return -1;
}



int number_of_nodes_from_nkids(const int *nkids)
{
    int indx = 0;
    int nodes_in_stage = 1;
    int nodes_in_next_stage;

    while (1)
    {
        nodes_in_next_stage = 0;
        for (int ii = 0; ii < nodes_in_stage; ii++)
        {
            if (nkids[indx+ii] < 0) return -1;
            if (nkids[indx+ii] == 0) break;  // reached a leaf
            nodes_in_next_stage += nkids[indx+ii];
        }
        indx += nodes_in_stage;
        if (nodes_in_next_stage == 0) break;
        if (nodes_in_next_stage < nodes_in_stage) return -1; // inconsistent tree data
        nodes_in_stage = nodes_in_next_stage;
    }
    return indx;
}



int number_of_nodes_from_tree(const struct node *tree)
{
    int indx = 0;
    int nodes_in_stage = 1;
    int nodes_in_next_stage;

    while (1)
    {
        nodes_in_next_stage = 0;
        for (int ii = 0; ii < nodes_in_stage; ii++)
        {
            if (tree[indx+ii].nkids < 0) return -1;
            if (tree[indx+ii].nkids == 0) break;
            nodes_in_next_stage += tree[indx+ii].nkids;
        }
        indx += nodes_in_stage;
        if (nodes_in_next_stage == 0) break;
        if (nodes_in_next_stage < nodes_in_stage) return -1;
        nodes_in_stage = nodes_in_next_stage;
    }
    return indx;
}



int tree_calculate_size(const int *nk)
{
    int Nn = number_of_nodes_from_nkids(nk);

    int bytes = 0;

    for (int ii = 0; ii < Nn; ii++)
    {
        bytes += nk[ii]*sizeof(int);
    }

    return bytes;
}



return_t tree_create(const int *nk, struct node *tree, void *ptr)
{
    int Nn = number_of_nodes_from_nkids(nk);
    if (Nn < 0) return TREEQP_FAILURE;

    int realization;  // NOTE(dimitris): used in the LTI case

    char *c_ptr = (char *) ptr;

    // initialize nodes to 'unassigned'
    for (int ii = 0; ii < Nn; ii++)
    {
        tree[ii].stage = -1;
        tree[ii].real = -1;
    }

    // initialize root
    tree[0].idx = 0;
    tree[0].dad = -1;
    tree[0].stage = 0;
    tree[0].idxkid = 0;

    // set up tree
    int idxkids;
    for (int ii = 0; ii < Nn; ii++)
    {
        tree[ii].nkids = nk[ii];
        if (nk[ii] > 0)
        {
            tree[ii].kids = (int *) c_ptr;
            c_ptr += nk[ii]*sizeof(int);
        }

        // identify where children nodes start
        idxkids = 0;
        for (int jj = ii; jj < Nn; jj++)
        {
            if (tree[jj].stage == -1)
            {
                idxkids = jj;
                break;
            }
        }

        // assign data to children nodes
        realization = 0;
        for (int jj = idxkids; jj < idxkids + nk[ii]; jj++)
        {
            tree[ii].kids[jj - idxkids] = jj;
            tree[jj].idx = jj;
            tree[jj].dad = ii;
            tree[jj].stage = tree[ii].stage +1;
            tree[jj].idxkid = jj - idxkids;
            if (tree[ii].nkids > 1)
            {
                tree[jj].real = realization++;
            }
            else
            {
                if (ii > 0)
                {
                    tree[jj].real = tree[ii].real;
                }
                else  // treat nominal case (linear topology) separately
                {
                    tree[jj].real = 0;
                }
            }
        }
    }
    assert((char *)ptr + tree_calculate_size(nk) == c_ptr);
    return_t TREEQP_OK;
}



void setup_multistage_tree(int md, int Nr, int Nh, int *nk)
{
    int num_scenarios = ipow(md, Nr);
    int num_nodes = calculate_number_of_nodes(md, Nr, Nh);

    int nodes_in_next_stage;
    int nodes_in_stage = 1;
    int idx = 0;

    for (int kk = 0; kk < Nh; kk++)
    {
        // printf("nodes in stage %d: %d\n", kk, nodes_in_stage);
        nodes_in_next_stage = 0;
        for (int ii = 0; ii < nodes_in_stage; ii++)
        {
            if (kk < Nr)
            {
                nk[idx+ii] = md;
            }
            else
            {
                nk[idx+ii] = 1;
            }
            nodes_in_next_stage += nk[idx+ii];
        }
        idx += nodes_in_stage;
        nodes_in_stage = nodes_in_next_stage;
    }
    // printf("nodes in stage %d: %d\n", Nh, nodes_in_stage);
    for (int ii = 0; ii < nodes_in_stage; ii++)
    {
        nk[idx+ii] = 0;
    }
}
