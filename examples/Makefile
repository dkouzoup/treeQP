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

OBJS = spring_mass_dual_newton_scenarios.o spring_mass_dual_newton_tree.o spring_mass.o random_qp.o fault_tolerance.o

obj: $(OBJS)
	$(CC) -o spring_mass_sdunes.out spring_mass_dual_newton_scenarios.o -ltreeqp -lblasfeo -lm $(CFLAGS) $(LDFLAGS)
	$(CC) -o spring_mass_tdunes.out spring_mass_dual_newton_tree.o -ltreeqp -lblasfeo -lm $(CFLAGS) $(LDFLAGS)
	$(CC) -o spring_mass.out spring_mass.o -ltreeqp -lhpmpc -lblasfeo -lm $(CFLAGS) $(LDFLAGS)
	$(CC) -o random_qp.out random_qp.o -ltreeqp -lblasfeo -lm $(CFLAGS) $(LDFLAGS)
	$(CC) -o fault_tolerance.out fault_tolerance.o fault_tolerance_utils/load_data.c -ltreeqp -lblasfeo -lm $(CFLAGS) $(LDFLAGS)

clean:
	rm -f *.o *.exe *.out
	rm -r .dSYM
