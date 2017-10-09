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


OBJS=
OBJS+=./treeqp/src/tree_ocp_qp_common.o ./treeqp/src/dual_Newton_scenarios.o ./treeqp/src/dual_Newton_tree.o ./treeqp/src/hpmpc_tree.o
OBJS+=./treeqp/utils/blasfeo_utils.o ./treeqp/utils/profiling_utils.o ./treeqp/utils/tree_utils.o
OBJS+=./treeqp/utils/utils.o ./treeqp/utils/timing.o

static_library:
	make -C treeqp/src obj
	make -C treeqp/utils obj
	ar rcs libtreeqp.a $(OBJS)
	@echo
	@echo " libtreeqp.a static library build complete."
	@echo

examples: static_library
	make -C examples obj
	#./examples/test.out

clean:
	make -C treeqp/src clean
	make -C treeqp/utils clean
	make -C examples clean
	rm -f *.a

lint: # TODO(dimitris): fix for Linux, currently works only on mac
	# Generate list of files and pass them to lint
	find . \( -not -path "./tmp/*" -not -path "./external/*" \( -name "*.c" -o -name "*.h" -o -name "*.cpp" \) ! -name "*blasfeo_d_aux_tmp.h" ! -name  "dims.h" ! -name  "old_*.h" ! -name  "timing.h" ! -name "*data.c" \) > project.lnt
	./cpplint.py --filter=-legal/copyright,-readability/casting --counting=detailed --linelength=100 --extensions=c,h,cpp $$(<project.lnt)
	rm project.lnt

.PHONY: all static_library examples clean lint
