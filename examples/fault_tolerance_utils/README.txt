
Instructions for fault_tolerance example (WIP):

- clone "prunest" repo

    > git clone ...
    > git submodule update (to fetch treeQP)
    > cd code/treeQP
    > git submodule update (to fetch blafeo, hpmpc, etc)
    > make fault_tolerance_example (to build treeQP lib and example with shared library for python)

- run python scripts to code generate data and run simulations

    > adapt simulation options in spring_packets_example.py
    > execute "python3 spring_packets_example.py" to code generate shared libraries with data
    > execute "python3 run_simulation.py" to run a closed loop simulation
