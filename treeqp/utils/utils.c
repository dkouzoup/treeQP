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

#include "treeqp/utils/utils.h"
#include "treeqp/utils/types.h"

int ipow(int base, int exp) {
    int result = 1;
    while (exp) {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }
    return result;
}


return_t read_int_vector_from_txt(int *vec, int n, const char *filename) {
    int i, c;
    FILE *myFile;
    myFile = fopen(filename, "r");

    if (myFile == NULL) {
        printf("Error Reading File (%s)\n", filename);
        return TREEQP_ERR_ERROR_OPENING_FILE;
    }

    for (i = 0; i < n; i++) {
        c = fscanf(myFile, "%d,", &vec[i]);
    }

    fclose(myFile);

    return TREEQP_OK;
}


return_t read_double_vector_from_txt(double *vec, int n, const char *filename) {
    int i, c;
    FILE *myFile;
    myFile = fopen(filename, "r");

    if (myFile == NULL) {
        printf("Error Reading File (%s)\n", filename);
        return TREEQP_ERR_ERROR_OPENING_FILE;
    }

    for (i = 0; i < n; i++) {
         c = fscanf(myFile, "%lf,", &vec[i]);
    }

    fclose(myFile);

    return TREEQP_OK;
}


return_t write_double_vector_to_txt(double *vec, int n, const char *filename) {
    int i, c;
    FILE *myFile;
    myFile = fopen(filename, "wr");

    if (myFile == NULL) {
        printf("Error opening file (%s)\n", filename);
        return TREEQP_ERR_ERROR_OPENING_FILE;
    }

    for (i = 0; i < n; i++) {
        c = fprintf(myFile, "%.16e\n", vec[i]);
    }

    fclose(myFile);

    return TREEQP_OK;
}


return_t write_int_vector_to_txt(int *vec, int n, const char *filename) {
    int i, c;
    FILE *myFile;
    myFile = fopen(filename, "wr");

    if (myFile == NULL) {
        printf("Error opening file (%s)\n", filename);
        return TREEQP_ERR_ERROR_OPENING_FILE;
    }

    for (i = 0; i < n; i++) {
        c = fprintf(myFile, "%d\n", vec[i]);
    }

    fclose(myFile);

    return TREEQP_OK;
}
