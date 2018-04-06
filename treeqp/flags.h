#ifndef TREEQP_FLAGS_H_
#define TREEQP_FLAGS_H_

// DEBUG = 0: do not save debugging info
// DEBUG = 1: store intermediate results in txt files to be loaded and checked in matlab
#define DEBUG 0

// Run algorithm NRUNS times and take minimum CPU time
#define NRUNS 10

// PRINT_LEVEL = 0: supress all printing
// PRINT_LEVEL = 1: print profiling information at the end
// PRINT_LEVEL = 2: print also some iteration stats
// PRINT_LEVEL = 3: print also debugging info (nothing for scenarios atm)
#define PRINT_LEVEL 1

// PROFILE = 0: no profiling
// PROFILE = 1: save only (min) total time
// PROFILE = 2: additionally save (min) time per iteration (and LS iterations)
// PROFILE = 3: additionally save (min) time per key operation per iteration
#define PROFILE 3

#endif  // TREEQP_FLAGS_H_
