#!/usr/bin/env python
# coding: utf-8

from mpi4py import MPI
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("-n", "--n", type=int, default=4, help="System size.")
    parser.add_argument("-p", "--p", type=int, default=1, help="Number of layers.")
    parser.add_argument("-t", "--t", type=float, default=1, help="Final time")
    parser.add_argument("-d", "--d", type=float, default=0.001, help="McLachlan step size")
    parser.add_argument("-D", "--D", type=float, default=0.0001, help="Tikhonov regulator")
    parser.add_argument("-f", "--f", type=float, default=1, help="1 is uniform dist. 0 is maximally non-uniform dist.")
    parser.add_argument("-l", "--l", type=float, default=1, help="l-norm value.")
    parser.add_argument("-s", "--s", type=int, default=20000, help="Average number of shots per matrix element.")
    # parser.add_argument("-S", "--S", type=int, default=0, help="Random seed initialization.")
    parser.add_argument("-r", "--r", type=int, default=1, help="Run number to start at.")
    # parser.add_argument("-R", "--R", type=int, default=1, help="Run number to end at.")
    args = parser.parse_args()
    Nl = args.n
    p = args.p
    tf = args.t
    dt = args.d
    de = args.D
    frac = args.f
    l = args.l
    avg = args.s
    # seed = args.S
    run = args.r

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # size = comm.Get_size()

    runs = [run+i for i in range(4)]

    if rank == 0:

        if not os.path.exists(f'../shot_allocation/l_eq_{l}/'):
            os.mkdir(f'../shot_allocation/l_eq_{l}/')

        if not os.path.exists(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/'):
            os.mkdir(f'../shot_allocation/l_eq_{l}/res_ours_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/')

    comm.Barrier()

    os.system(f'python3 allocateShots.py -n {Nl} -p {p} -t {tf} -d {dt} -D {de} -f {frac} -l {l} -s {avg} -S {rank} -r {runs[rank]}')

    comm.Barrier()

