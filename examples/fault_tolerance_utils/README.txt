
Instructions for fault_tolerance example (WIP):

- clone "prunest" repo

    > git clone ...
    > git checkout results

- run python script to code generate data for your problem

    > open spring_packets_example.py
        - adapt: tree_depth (horizon length), N (n_masses+1), m (springs per packet),
          p (fault probability), k_nom (nominal spring constant), nu (controlled masses),
          Qd/Rd (weights for tracking), prob_coverage (desired coverage), max_scenarios (maximum
          number of scenarios), min_scenario_prob ('adaptive' means cut when new prob. too small).

    > execute "python2 spring_packets_example.py"

- copy .c generated functions to examples/fault_tolerance_utils

    >markov_sim.c:        dynamics of simulation model for all possible spring configurations
    >markov_tree_XY.c:    QP data of pruned robust controller for spring configuration XY
    >markov_nominal.c:    QP data of nominal controller for all possible spring configurations
    >markov_ms.c:         multi-stgae MPC tree (WIP)
    >transition_matrix.c: transition matrix to update simulation (and controller) config. online
    >load_data.c:         data file that imports all code generated data in C

- make examples (from treeQP root directory)

- ./examples/fault_tolerance.out

- options/parameters:

    > comment in/out "#define NOMINAL_MPC" in load_data.h to enable/disable nominal MPC controller
    > allow online changes of MPC config. by ...


Interesting simulations so far:

- less oscillations with treeMPC (2 masses, 1 control):

    real_t Pmin = -5;
    real_t Pmax = 2.7;
    real_t Vmin = -10;
    real_t Vmax = 10;
    real_t Fmin = -100;
    real_t Fmax = 100;

    x0[0] = 0;
    x0[1] = 2;
    x0[2] = -8;
    x0[3] = 4;
