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


include ../../Makefile.rule

OBJS = tree_ocp_qp_common.o dual_Newton_scenarios.o dual_Newton_tree.o hpmpc_tree.o

obj: $(OBJS)

clean:
	rm -f *.o