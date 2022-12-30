# Shot_allocation_projects

Code to allocate shots _optimally_ (in some sense) during the McLachlan time evolution of a state. Here we use the global phase explicitly and therefore do not use the quantum metric in the definition of the M matrix.

The run_allocateShots.py file uses MPI (mpi4py) to run multiple instances of allocateShots.py in parallel.

The allocateShotsKoczor.py file runs a different algorithm from the literature as a comparisson to our idea. (PRX Quantum 2, 030324 (2021))
