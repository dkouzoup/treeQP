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

#include "treeqp/utils/tree.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/utils.h"

int calculate_number_of_nodes(const int md, const int Nr, const int Nh)
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



int get_number_of_parent_nodes(const int Nn, const struct node * const tree)
{
    int Np = 0;

    for (int kk = 0; kk < Nn; kk++)
    {
        if (tree[kk].nkids > 0) Np++;
    }
    return Np;
}



int get_robust_horizon(const int Nn, const struct node * const tree)
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



// NOTE(dimitris): courtasy of Misha
static int number_of_nodes_from_nkids(const int * const nkids)
{
    int const * od = nkids;
    int u = 0;
    int v = 1;

    while (u < v)
    {
        v += *od;
        ++u;
        ++od;
    }
    return u;
}



int number_of_nodes_from_tree(const struct node * const tree)
{
    int i = 0;
    int u = 0;
    int v = 1;

    while (u < v)
    {
        v += tree[i].nkids;
        ++u;
        ++i;
    }

    return u;
}



void setup_tree(const int * const nkids, struct node * const tree)
{
    int Nn = number_of_nodes_from_nkids(nkids);

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
        tree[ii].nkids = nkids[ii];
        if (nkids[ii] > 0) {
            tree[ii].kids = (int *) malloc(nkids[ii]*sizeof(int));
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
        for (int jj = idxkids; jj < idxkids + nkids[ii]; jj++)
        {
            tree[ii].kids[jj - idxkids] = jj;
            tree[jj].idx = jj;
            tree[jj].dad = ii;
            tree[jj].stage = tree[ii].stage +1;
            tree[jj].idxkid = jj - idxkids;
        }
    }
}



void setup_multistage_tree(const int md, const int Nr, const int Nh, const int Nn, struct node * const tree)
{
    int idx, dad, stage, real, nkids, idxkid;

    // root
    idx = 0;
    dad = -1;
    stage = 0;
    real = -1;
    if (stage < Nr)
    {
        nkids = md;
    }
    else if (stage < Nh)
    {
        nkids = 1;
    }
    else
    {
        nkids = 0;
    }
    tree[idx].idx = idx;
    tree[idx].dad = dad;
    tree[idx].stage = stage;
    tree[idx].real = real;
    tree[idx].nkids = nkids;
    tree[idx].idxkid = 0;

    if (nkids > 0)
    {
        tree[idx].kids = (int *) malloc(nkids*sizeof(int));
        if (nkids > 1)
        {
            for (int ii = 0; ii < nkids; ii++)
            {
                idxkid = ii+1;
                tree[idx].kids[ii] = idxkid;
                tree[idxkid].dad = idx;
                tree[idxkid].real = ii;
                tree[idxkid].idxkid = ii;
            }
        }
        else  // nkids == 1
        {
            idxkid = 1;
            tree[idx].kids[0] = idxkid;
            tree[idxkid].dad = idx;
            tree[idxkid].real = 0;
            tree[idxkid].idxkid = 0;
        }
    }
    // kids
    for (idx = 1; idx < Nn; idx++)
    {
        stage = tree[tree[idx].dad].stage+1;
        if (stage < Nr)
        {
            nkids = md;
        }
        else if (stage < Nh)
        {
            nkids = 1;
        }
        else
        {
            nkids = 0;
        }
        tree[idx].idx = idx;
        tree[idx].stage = stage;
        tree[idx].nkids = nkids;
        if (nkids > 0)
        {
            tree[idx].kids = (int *) malloc(nkids*sizeof(int));
            if (nkids > 1)
            {
                for (int ii = 0; ii < nkids; ii++)
                {
                    idxkid = tree[idx-1].kids[tree[idx-1].nkids-1]+ii+1;
                    tree[idx].kids[ii] = idxkid;
                    tree[idxkid].dad = idx;
                    tree[idxkid].real = ii;
                    tree[idxkid].idxkid = ii;
                }
            }
            else  // nkids==1
            {
                idxkid = tree[idx-1].kids[tree[idx-1].nkids-1]+1;
                tree[idx].kids[0] = idxkid;
                tree[idxkid].dad = idx;
                tree[idxkid].real = tree[idx].real;
                tree[idxkid].idxkid = 0;
            }
        }
    }
}



void free_tree(struct node * const tree)
{
    int Nn = number_of_nodes_from_tree(tree);

    for (int ii = 0; ii < Nn; ii++)
    {
        if (tree[ii].nkids > 0)
        {
            free(tree[ii].kids);
        }
    }
}
