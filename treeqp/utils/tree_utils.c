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
#include <stdlib.h>

#include "treeqp/utils/tree_utils.h"
#include "treeqp/utils/types.h"
#include "treeqp/utils/utils.h"

int_t calculate_number_of_nodes(int_t md, int_t Nr, int_t Nh) {
    int_t n_nodes;
    if (md == 1)  // i.e. standard block-banded structure
        n_nodes = Nh+1;
    else
        n_nodes = (Nh-Nr)*ipow(md, Nr) + (ipow(md, Nr+1)-1)/(md-1);
    return n_nodes;
}


int_t get_number_of_parent_nodes(int_t Nn, struct node *tree) {
    int_t kk;
    int_t Np = 0;

    for (kk = 0; kk < Nn; kk++) {
        if (tree[kk].nkids > 0) Np++;
    }
    return Np;
}


int_t get_robust_horizon(int_t Nn, struct node *tree) {
    int_t kk;
    int_t Nr = 0;

    for (kk = 0; kk < Nn; kk++) {
        if (tree[kk].nkids > 1) {
            Nr = tree[kk].stage+1;
        } else {
            break;
        }
    }

    return Nr;
}


void print_node(struct node *tree) {
    int_t ii;
    printf("\n");
    printf("idx = \t\t%d\n", tree[0].idx);
    printf("dad = \t\t%d\n", tree[0].dad);
    printf("nkids = \t%d\n", tree[0].nkids);
    printf("kids = \t\t");
    for (ii = 0; ii < tree[0].nkids; ii++)
        printf("%d\t", tree[0].kids[ii]);
    printf("\n");
    printf("stage = \t%d\n", tree[0].stage);
    printf("realization = \t%d\n", tree[0].real);
    printf("index as a kid = \t%d\n", tree[0].idxkid);
    printf("\n");
    return;
}


void setup_tree(int_t md, int_t Nr, int_t Nh, int_t Nn, struct node *tree) {
    int_t ii;
    int_t idx, dad, stage, real, nkids, idxkid;
    // root
    idx = 0;
    dad = -1;
    stage = 0;
    real = -1;
    if (stage < Nr)
        nkids = md;
    else if (stage < Nh)
        nkids = 1;
    else
        nkids = 0;
    tree[idx].idx = idx;
    tree[idx].dad = dad;
    tree[idx].stage = stage;
    tree[idx].real = real;
    tree[idx].nkids = nkids;
    tree[idx].idxkid = 0;
    if (nkids > 0) {
        tree[idx].kids = (int_t *) malloc(nkids*sizeof(int_t));
        if (nkids > 1) {
            for (ii = 0; ii < nkids; ii++) {
                idxkid = ii+1;
                tree[idx].kids[ii] = idxkid;
                tree[idxkid].dad = idx;
                tree[idxkid].real = ii;
                tree[idxkid].idxkid = ii;
            }
        } else {  // nkids==1
            idxkid = 1;
            tree[idx].kids[0] = idxkid;
            tree[idxkid].dad = idx;
            tree[idxkid].real = 0;
            tree[idxkid].idxkid = 0;
        }
    }
    // kids
    for (idx = 1; idx < Nn; idx++) {
        stage = tree[tree[idx].dad].stage+1;
        if (stage < Nr)
            nkids = md;
        else if (stage < Nh)
            nkids = 1;
        else
            nkids = 0;
        tree[idx].idx = idx;
        tree[idx].stage = stage;
        tree[idx].nkids = nkids;
        if (nkids > 0) {
            tree[idx].kids = (int_t *) malloc(nkids*sizeof(int_t));
            if (nkids > 1) {
                for (ii = 0; ii < nkids; ii++) {
                    idxkid = tree[idx-1].kids[tree[idx-1].nkids-1]+ii+1;
                    tree[idx].kids[ii] = idxkid;
                    tree[idxkid].dad = idx;
                    tree[idxkid].real = ii;
                    tree[idxkid].idxkid = ii;
                }
            } else {  // nkids==1
                idxkid = tree[idx-1].kids[tree[idx-1].nkids-1]+1;
                tree[idx].kids[0] = idxkid;
                tree[idxkid].dad = idx;
                tree[idxkid].real = tree[idx].real;
                tree[idxkid].idxkid = 0;
            }
        }
    }
    // return
    return;
}


void free_tree(int_t md, int_t Nr, int_t Nh, int_t Nn, struct node *tree) {
    int_t ii;
    int_t idx, dad, stage, real, nkids, idxkid;
    // root
    idx = 0;
    dad = -1;
    stage = 0;
    real = -1;
    if (stage < Nr)
        nkids = md;
    else if (stage < Nh)
        nkids = 1;
    else
        nkids = 0;
    if (nkids > 0) {
        free(tree[idx].kids);
    }
    // kids
    for (idx = 1; idx < Nn; idx++) {
        stage = tree[tree[idx].dad].stage+1;
        if (stage < Nr)
            nkids = md;
        else if (stage < Nh)
            nkids = 1;
        else
            nkids = 0;
        if (nkids > 0) {
            free(tree[idx].kids);
        }
    }
    // return
    return;
}
