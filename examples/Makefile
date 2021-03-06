# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#    This file is part of treeQP.                                                                  #
#                                                                                                  #
#    treeQP -- A toolbox of tree-sparse Quadratic Programming solvers.                             #
#    Copyright (C) 2017 by Dimitris Kouzoupis.                                                     #
#    Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.            #
#    All rights reserved.                                                                          #
#                                                                                                  #
#    treeQP is free software; you can redistribute it and/or                                       #
#    modify it under the terms of the GNU Lesser General Public                                    #
#    License as published by the Free Software Foundation; either                                  #
#    version 3 of the License, or (at your option) any later version.                              #
#                                                                                                  #
#    treeQP is distributed in the hope that it will be useful,                                     #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of                                #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU                             #
#    Lesser General Public License for more details.                                               #
#                                                                                                  #
#    You should have received a copy of the GNU Lesser General Public                              #
#    License along with treeQP; if not, write to the Free Software Foundation,                     #
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                            #
#                                                                                                  #
#    Author: Dimitris Kouzoupis, dimitris.kouzoupis (at) imtek.uni-freiburg.de                     #
#                                                                                                  #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


include ../Makefile.rule

FAULT_TOLERANCE_OBJS =
FAULT_TOLERANCE_OBJS += fault_tolerance.o
FAULT_TOLERANCE_OBJS += fault_tolerance_utils/load_data.o

LIBS = -ltreeqp -lhpmpc -lhpipm -lblasfeo -lqpoases -lm -ldl

unit_tests:
	$(CC) -o unit_test_0_tdunes.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=0
	$(CC) -o unit_test_0_hpmpc.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=0 -DUSE_HPMPC
	$(CC) -o unit_test_0_hpipm.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=0 -DUSE_HPIPM
	$(CC) -o unit_test_1_tdunes.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=1
	$(CC) -o unit_test_1_hpmpc.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=1 -DUSE_HPMPC
	$(CC) -o unit_test_1_hpipm.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=1 -DUSE_HPIPM
	$(CC) -o unit_test_2_tdunes.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=2
	$(CC) -o unit_test_2_hpmpc.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=2 -DUSE_HPMPC
	$(CC) -o unit_test_2_hpipm.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=2 -DUSE_HPIPM
	$(CC) -o unit_test_3_tdunes.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=3
	$(CC) -o unit_test_3_hpmpc.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=3 -DUSE_HPMPC
	$(CC) -o unit_test_3_hpipm.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=3 -DUSE_HPIPM
	$(CC) -o unit_test_4_tdunes.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=4
	$(CC) -o unit_test_4_hpmpc.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=4 -DUSE_HPMPC
	$(CC) -o unit_test_4_hpipm.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=4 -DUSE_HPIPM
	$(CC) -o unit_test_5_tdunes.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=5
	$(CC) -o unit_test_5_hpmpc.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=5 -DUSE_HPMPC
	$(CC) -o unit_test_5_hpipm.out random_qp.c $(CFLAGS) $(LDFLAGS) $(LIBS) -DDATA=5 -DUSE_HPIPM

thesis_example: thesis_example.o
	$(CC) -o thesis_example.out thesis_example.o $(CFLAGS) $(LDFLAGS) $(LIBS)

thesis_example_cpp_interface: thesis_example_cpp_interface.o
	$(CXX) -o thesis_example_cpp_interface.out thesis_example_cpp_interface.o $(CXXFLAGS) $(LDFLAGS) $(LIBS) -ltreeqp_cpp_interface

random_qp_example: random_qp.o
	$(CC) -o random_qp.out random_qp.o $(CFLAGS) $(LDFLAGS) $(LIBS)

spring_mass_tdunes_example: spring_mass_dual_newton_tree.o
	$(CC) -o spring_mass_tdunes.out spring_mass_dual_newton_tree.o $(CFLAGS) $(LDFLAGS) $(LIBS)

spring_mass_sdunes_example: spring_mass_dual_newton_scenarios.o
	$(CC) -o spring_mass_sdunes.out spring_mass_dual_newton_scenarios.o $(CFLAGS) $(LDFLAGS) $(LIBS)

spring_mass_example: spring_mass.o
	$(CC) -o spring_mass.out spring_mass.o $(CFLAGS) $(LDFLAGS) $(LIBS)

spring_mass_debug_example: spring_mass_debug.o
	$(CC) -o spring_mass_debug.out spring_mass_debug.o $(CFLAGS) $(LDFLAGS) $(LIBS)
	./spring_mass_debug.out

fault_tolerance_example: $(FAULT_TOLERANCE_OBJS)
	$(CC) -o fault_tolerance.out fault_tolerance.o fault_tolerance_utils/transition_matrix.c fault_tolerance_utils/load_data.o $(CFLAGS) $(LDFLAGS) $(LIBS)
	$(CC) -o fault_tolerance.so fault_tolerance.o fault_tolerance_utils/transition_matrix.c fault_tolerance_utils/load_data.o $(CFLAGS) $(LDFLAGS) $(LIBS) -shared

solve_qp_json: solve_qp_json.o
	$(CXX) -o solve_qp_json.out solve_qp_json.o $(CXXFLAGS) $(LDFLAGS) $(LIBS)

examples:
	$(MAKE) random_qp_example
	$(MAKE) thesis_example
	$(MAKE) spring_mass_tdunes_example
	$(MAKE) spring_mass_sdunes_example
	$(MAKE) spring_mass_example

clean:
	rm -f fault_tolerance_utils/*.o fault_tolerance_utils/*.exe fault_tolerance_utils/*.out
	rm -f *.o *.exe *.out *.so
	rm -rf *.dSYM
