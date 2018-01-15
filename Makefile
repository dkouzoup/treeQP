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
#    Authors: Dimitris Kouzoupis, Gianluca Frison, name.surname (at) imtek.uni-freiburg.de         #
#                                                                                                  #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

include ./Makefile.rule

OBJS =
OBJS += treeqp/src/tree_ocp_qp_common.o
OBJS += treeqp/src/dual_Newton_scenarios.o
OBJS += treeqp/src/dual_Newton_tree.o
OBJS += treeqp/src/hpmpc_tree.o
OBJS += treeqp/utils/blasfeo_utils.o
OBJS += treeqp/utils/profiling.o
OBJS += treeqp/utils/tree.o
OBJS+= treeqp/utils/utils.o
OBJS += treeqp/utils/timing.o

DEPS = blasfeo_static hpmpc_static

treeqp_static: $(DEPS)
	( cd treeqp/src; $(MAKE) obj TOP=$(TOP) )
	( cd treeqp/utils; $(MAKE) obj TOP=$(TOP) )
	ar rcs libtreeqp.a $(OBJS)
	mkdir -p lib
	mv libtreeqp.a lib
	@echo
	@echo " libtreeqp.a static library build complete."
	@echo

blasfeo_static:
	( cd external/blasfeo; $(MAKE) static_library CC=$(CC) LA=$(BLASFEO_VERSION) TARGET=$(BLASFEO_TARGET) )
	#mkdir -p include/blasfeo
	mkdir -p lib
	#cp external/blasfeo/include/*.h include/blasfeo
	cp external/blasfeo/lib/libblasfeo.a lib

hpmpc_static: blasfeo_static
	( cd external/hpmpc; $(MAKE) static_library CC=$(CC) TARGET=$(HPMPC_TARGET) BLASFEO_PATH=$(TOP)/external/blasfeo )
	#mkdir -p include/hpmpc
	mkdir -p lib
	#cp external/hpmpc/include/*.h include/hpmpc
	cp external/hpmpc/libhpmpc.a lib

examples: treeqp_static
	( cd examples; $(MAKE) examples TOP=$(TOP) )

run_examples:
	./examples/random_qp.out
	./examples/spring_mass_tdunes.out
	./examples/spring_mass_sdunes.out
	./examples/fault_tolerance.out
	./examples/spring_mass.out

clean:
	( cd treeqp/src; $(MAKE) clean )
	( cd treeqp/utils; $(MAKE) clean )
	( cd examples; $(MAKE) clean )
	rm -f lib/libtreeqp.a

deep_clean: clean
	( cd external/blasfeo; $(MAKE) deep_clean )
	( cd external/hpmpc; $(MAKE) clean )
	rm -rf include
	rm -rf lib

lint: # TODO(dimitris): fix for Linux, currently works only on mac
	# Generate list of files and pass them to lint
	find . \( -not -path "./tmp/*" -not -path "./external/*" \( -name "*.c" -o -name "*.h" -o -name "*.cpp" \) ! -name "*blasfeo_d_aux_tmp.h" ! -name  "dims.h" ! -name  "old_*.h" ! -name  "timing.h" ! -name "*data.c" \) > project.lnt
	./cpplint.py --filter=-legal/copyright,-readability/casting --counting=detailed --linelength=100 --extensions=c,h,cpp $$(<project.lnt)
	rm project.lnt

.PHONY: all static_library examples clean lint
