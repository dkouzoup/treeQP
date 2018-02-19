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
#include <dlfcn.h>

#include "examples/fault_tolerance_utils/load_data.h"


void load_dimensions(int *n_config_ptr, int *nx_ptr, int *nu_ptr, char *lib_string)
{
    void *lib = dlopen(lib_string, RTLD_NOW);
    if (lib == NULL) {
        printf("dlopen failed: %s\n", dlerror());
        exit(1);
    }

    void *fun;
    int (*fun_ptr)();

    // load n_config
    fun = dlsym(lib, "get_number_of_realizations");
    if (fun == NULL) {
        printf("dlsym failed: %s\n", dlerror());
        exit(1);
    }
    fun_ptr = fun;
    *n_config_ptr = fun_ptr();

    // load nx
    fun = dlsym(lib, "get_nx");
    if (fun == NULL) {
        printf("dlsym failed: %s\n", dlerror());
        exit(1);
    }
    fun_ptr = fun;
    *nx_ptr = fun_ptr();

    // load nu
    fun = dlsym(lib, "get_nu");
    if (fun == NULL) {
        printf("dlsym failed: %s\n", dlerror());
        exit(1);
    }
    fun_ptr = fun;
    *nu_ptr = fun_ptr();

    // printf("NC = %d\n", *n_config_ptr);
    // printf("NX = %d\n", *nx_ptr);
    // printf("NU = %d\n", *nu_ptr);
}



void load_ptr(void *lib, char *data_string, void **ptr)
{
    *ptr = dlsym(lib, data_string);
    if (ptr == NULL)
    {
        printf("dlsym failed: %s\n", dlerror());
        exit(1);
    }
}



sim_data *load_sim_data(int n_config, char *lib_string)
{
    sim_data *data = malloc(n_config*sizeof(sim_data));

    char data_string[256];

    void *lib = dlopen(lib_string, RTLD_NOW);
    if (lib == NULL) {
        printf("dlopen failed: %s\n", dlerror());
        exit(1);
    }


    for (int ii = 0; ii < n_config; ii++)
    {
        // load A
        snprintf(data_string, sizeof(data_string), "Asim_%d", ii);
        load_ptr(lib, data_string, (void **)&data[ii].A);

        // load B
        snprintf(data_string, sizeof(data_string), "Bsim_%d", ii);
        load_ptr(lib, data_string, (void **)&data[ii].B);

        // load b
        snprintf(data_string, sizeof(data_string), "bsim_%d", ii);
        load_ptr(lib, data_string, (void **)&data[ii].b);
    }
    return data;
}



input_data *load_nominal_data(int n_config, char *lib_string)
{
    input_data *data = malloc(n_config*sizeof(input_data));

    char data_string[256];

    void *lib = dlopen(lib_string, RTLD_NOW);
    if (lib == NULL) {
        printf("dlopen failed: %s\n", dlerror());
        exit(1);
    }

    int *Nn_ptr;
    snprintf(data_string, sizeof(data_string), "Nn_nom");
    load_ptr(lib, data_string, (void **)&Nn_ptr);

    int *nc_ptr;
    snprintf(data_string, sizeof(data_string), "nc_nom");
    load_ptr(lib, data_string, (void **)&nc_ptr);

    int *nx_ptr;
    snprintf(data_string, sizeof(data_string), "nx_nom");
    load_ptr(lib, data_string, (void **)&nx_ptr);

    int *nu_ptr;
    snprintf(data_string, sizeof(data_string), "nu_nom");
    load_ptr(lib, data_string, (void **)&nu_ptr);

    double *Qnom_ptr;
    snprintf(data_string, sizeof(data_string), "Qnom");
    load_ptr(lib, data_string, (void **)&Qnom_ptr);

    double *Rnom_ptr;
    snprintf(data_string, sizeof(data_string), "Rnom");
    load_ptr(lib, data_string, (void **)&Rnom_ptr);

    double *qnom_ptr;
    snprintf(data_string, sizeof(data_string), "qnom");
    load_ptr(lib, data_string, (void **)&qnom_ptr);

    double *rnom_ptr;
    snprintf(data_string, sizeof(data_string), "rnom");
    load_ptr(lib, data_string, (void **)&rnom_ptr);

    for (int ii = 0; ii < n_config; ii++)
    {
        // set dimensions (same for all trees)
        data[ii].Nn = *Nn_ptr;
        data[ii].nc = nc_ptr;
        data[ii].nx = nx_ptr;
        data[ii].nu = nu_ptr;

        // load A
        snprintf(data_string, sizeof(data_string), "Anom_%d", ii);
        load_ptr(lib, data_string, (void **)&data[ii].A);

        // load B
        snprintf(data_string, sizeof(data_string), "Bnom_%d", ii);
        load_ptr(lib, data_string, (void **)&data[ii].B);

        // load b
        snprintf(data_string, sizeof(data_string), "bnom_%d", ii);
        load_ptr(lib, data_string, (void **)&data[ii].b);

        // set objective (same for all trees)
        data[ii].Qd = Qnom_ptr;
        data[ii].Rd = Rnom_ptr;
        data[ii].q = qnom_ptr;
        data[ii].r = rnom_ptr;

        // printf("%d nodes at tree %d\n", data[ii].Nn, ii);
    }
    return data;
}



input_data *load_tree_data(int n_config, char *lib_string)
{
    input_data *data = malloc(n_config*sizeof(input_data));

    char data_string[256];

    void *lib = dlopen(lib_string, RTLD_NOW);
    if (lib == NULL) {
        printf("dlopen failed: %s\n", dlerror());
        exit(1);
    }

    int Nn;
    int *Nn_ptr = &Nn;

    // initialization
    for (int ii = 0; ii < n_config; ii++)
    {
        // try to load number of nodes
        snprintf(data_string, sizeof(data_string), "Nn_%d", ii);
        load_ptr(lib, data_string, (void **)&Nn_ptr);

        if (Nn_ptr != NULL)  // tree configuration exists
        {
            data[ii].Nn = *Nn_ptr;
            printf("\nNn[%d] = %d\n", ii, data[ii].Nn);

            snprintf(data_string, sizeof(data_string), "nc_%d", ii);
            load_ptr(lib, data_string, (void **)&data[ii].nc);

            snprintf(data_string, sizeof(data_string), "nx_%d", ii);
            load_ptr(lib, data_string, (void **)&data[ii].nx);

            snprintf(data_string, sizeof(data_string), "nu_%d", ii);
            load_ptr(lib, data_string, (void **)&data[ii].nu);

            snprintf(data_string, sizeof(data_string), "Qd_%d", ii);
            load_ptr(lib, data_string, (void **)&data[ii].Qd);

            snprintf(data_string, sizeof(data_string), "Rd_%d", ii);
            load_ptr(lib, data_string, (void **)&data[ii].Rd);

            snprintf(data_string, sizeof(data_string), "q_%d", ii);
            load_ptr(lib, data_string, (void **)&data[ii].q);

            snprintf(data_string, sizeof(data_string), "r_%d", ii);
            load_ptr(lib, data_string, (void **)&data[ii].r);

            snprintf(data_string, sizeof(data_string), "A_%d", ii);
            load_ptr(lib, data_string, (void **)&data[ii].A);

            snprintf(data_string, sizeof(data_string), "B_%d", ii);
            load_ptr(lib, data_string, (void **)&data[ii].B);

            snprintf(data_string, sizeof(data_string), "b_%d", ii);
            load_ptr(lib, data_string, (void **)&data[ii].b);
        }
        else
        {
            data[ii].Nn = -1;
            data[ii].nc = NULL;
            data[ii].nx = NULL;
            data[ii].nu = NULL;

            data[ii].A = NULL;
            data[ii].B = NULL;
            data[ii].b = NULL;

            data[ii].Qd = NULL;
            data[ii].Rd = NULL;
            data[ii].q = NULL;
            data[ii].r = NULL;
        }
    }

    return data;
}
