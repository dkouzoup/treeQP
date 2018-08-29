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
OBJS += treeqp/src/dual_Newton_common.o
OBJS += treeqp/src/dual_Newton_scenarios.o
OBJS += treeqp/src/dual_Newton_tree.o
OBJS += treeqp/src/dual_Newton_tree_clipping.o
OBJS += treeqp/src/dual_Newton_tree_qpoases.o
OBJS += treeqp/src/hpmpc_tree.o
OBJS += treeqp/src/hpipm_tree.o
OBJS += treeqp/utils/blasfeo.o
OBJS += treeqp/utils/memory.o
OBJS += treeqp/utils/profiling.o
OBJS += treeqp/utils/print.o
OBJS += treeqp/utils/timing.o
OBJS += treeqp/utils/tree.o
OBJS+= treeqp/utils/utils.o

ifeq ($(SKIP_BLASFEO_COMPILATION), ON)
BLASFEO_DEP =
else
BLASFEO_DEP = blasfeo_static
endif

DEPS =
DEPS += $(BLASFEO_DEP) hpmpc_static hpipm_static qpoases_static

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
	( cd $(BLASFEO_PATH); $(MAKE) static_library CC=$(CC) LA=$(BLASFEO_VERSION) TARGET=$(BLASFEO_TARGET) )
	#mkdir -p include/blasfeo
	mkdir -p lib
	#cp external/blasfeo/include/*.h include/blasfeo
	cp $(BLASFEO_PATH)/lib/libblasfeo.a lib

hpmpc_static: $(BLASFEO_DEP)
	( cd external/hpmpc; $(MAKE) static_library CC=$(CC) TARGET=$(HPMPC_TARGET) BLASFEO_PATH=$(BLASFEO_PATH) )
	#mkdir -p include/hpmpc
	mkdir -p lib
	#cp external/hpmpc/include/*.h include/hpmpc
	cp external/hpmpc/libhpmpc.a lib

hpipm_static: $(BLASFEO_DEP)
	( cd external/hpipm; $(MAKE) static_library CC=$(CC) TARGET=$(HPIPM_TARGET) BLASFEO_PATH=$(BLASFEO_PATH) )
	#mkdir -p include/hpipm
	mkdir -p lib
	#cp external/hpipm/include/*.h include/hpipm
	cp external/hpipm/libhpipm.a lib

qpoases_static:
	( cd external/qpoases; $(MAKE) src CC=$(CC) )
	#mkdir -p include/qpoases
	mkdir -p lib
	#cp -r external/qpoases/include/* include/qpoases
	cp external/qpoases/bin/libqpOASES_e.a lib/libqpoases.a

spring_mass_tdunes_example: treeqp_static
	( cd examples; $(MAKE) spring_mass_tdunes_example TOP=$(TOP) )

spring_mass_sdunes_example: treeqp_static
	( cd examples; $(MAKE) spring_mass_sdunes_example TOP=$(TOP) )

spring_mass_debug_example: treeqp_static
	( cd examples; $(MAKE) spring_mass_debug_example TOP=$(TOP) )

fault_tolerance_example: treeqp_static # code-generate data in python first first
	( cd examples; $(MAKE) fault_tolerance_example TOP=$(TOP) )

run_fault_tolerance_example: fault_tolerance_example
	./examples/fault_tolerance.out

examples: treeqp_static
	( cd examples; $(MAKE) examples TOP=$(TOP) )

unit_tests: treeqp_static
	( cd examples; $(MAKE) unit_tests TOP=$(TOP) )

run_examples: examples
	./examples/random_qp.out
	./examples/spring_mass_tdunes.out
	./examples/spring_mass_sdunes.out
	./examples/spring_mass.out
	@echo
	@echo " All examples were executed succesfully!"
	@echo

run_unit_tests: unit_tests
	./examples/unit_test_0_tdunes.out
	./examples/unit_test_0_hpmpc.out
	./examples/unit_test_1_tdunes.out
	./examples/unit_test_1_hpmpc.out
	./examples/unit_test_2_tdunes.out
	./examples/unit_test_2_hpmpc.out
	./examples/unit_test_3_tdunes.out
	./examples/unit_test_3_hpmpc.out
	./examples/unit_test_4_tdunes.out
	./examples/unit_test_4_hpmpc.out
	./examples/unit_test_5_tdunes.out
	./examples/unit_test_5_hpmpc.out

clean:
	( cd treeqp/src; $(MAKE) clean )
	( cd treeqp/utils; $(MAKE) clean )
	( cd examples; $(MAKE) clean )
	rm -f lib/libtreeqp.a

ifeq ($(SKIP_BLASFEO_COMPILATION), ON)

clean_blasfeo:
	# BLASFEO, must be deep_cleaned manually!

else

clean_blasfeo:
	( cd $(BLASFEO_PATH); $(MAKE) deep_clean )

endif

deep_clean: clean clean_blasfeo
	( cd external/hpmpc; $(MAKE) clean )
	( cd external/hpipm; $(MAKE) clean )
	( cd external/qpoases; $(MAKE) clean )
	rm -rf include
	rm -rf lib

lint: # TODO(dimitris): fix for Linux, currently works only on mac
	# Generate list of files and pass them to lint
	find . \( -not -path "./tmp/*" -not -path "./external/*" \( -name "*.c" -o -name "*.h" -o -name "*.cpp" \) ! -name "*blasfeo_d_aux_tmp.h" ! -name  "dims.h" ! -name  "old_*.h" ! -name  "timing.h" ! -name "*data.c" \) > project.lnt
	./cpplint.py --filter=-legal/copyright,-readability/casting --counting=detailed --linelength=100 --extensions=c,h,cpp $$(<project.lnt)
	rm project.lnt

.PHONY: all static_library examples clean lint
