#!/usr/bin/env python
# coding: utf-8

from qutip import *
import cirq
import numpy as np
import random as rnd
from random import randint
from timeit import default_timer as timer
import scipy as sp
from scipy import sparse
from scipy.optimize import lsq_linear
import pandas as pd
import pickle
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("-n", "--n", type=int, default=4, help="System size.")
    parser.add_argument("-p", "--p", type=int, default=3, help="Number of layers.")
    parser.add_argument("-t", "--t", type=float, default=1, help="Final time")
    parser.add_argument("-d", "--d", type=float, default=0.001, help="McLachlan step size")
    parser.add_argument("-D", "--D", type=float, default=0.0001, help="Tikhonov regulator")
    parser.add_argument("-f", "--f", type=float, default=1, help="1 is uniform dist. 0 is maximally non-uniform dist.")
    parser.add_argument("-l", "--l", type=float, default=1, help="l-norm value.")
    parser.add_argument("-s", "--s", type=int, default=20000, help="Average number of shots per matrix element.")
    parser.add_argument("-S", "--S", type=int, default=0, help="Random seed initialization.")
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
    seed = args.S
    run = args.r
    # R = args.R

    # The functions:

    def bonds(N, # The total number of sites
             ):
        
        '''
        This function creates all the potential bonds on the lattice for a
        square lattice.
        '''
        
        ####################################################################
        # Make list of tuples that are the site pairings to make the bonds #
        ####################################################################
        
        # Instantiate the list
        tups = []
        
        m = 0
        
        # Populate the list
        for x in range(1,N):
            tups.append((x,x+1))
        
        tups.sort()
        
        return tups

    def spin_op_x(dim,site):
        
        '''
        Want to take in a number dim = N (dimension of the Hilbert space) and make a spin operator
        for $\sigma_x$ on spin site = j.
        '''
        
        # Instantiate the empty list to build the operator into
        
        op_list = []
        
        # Build the list op_list where all j != site are 2x2 identity matrices 
        # and j == site is \sigma_y
        for n in range(1,dim+1):
            if n == site:
                op_list.append(sigmax())
            else:
                op_list.append(qeye(2))

        # Return a tensor made from the list
        return tensor(op_list)

    def spin_op_y(dim,site):
        
        '''
        Want to take in a number dim = N (dimension of the Hilbert space) and make a spin operator
        for $\sigma_y$ on spin site = j.
        '''
        
        # Instantiate the empty list to build the operator into
        
        op_list = []
        
        # Build the list op_list where all j != site are 2x2 identity matrices
        # and j == site is \sigma_y
        for n in range(1,dim+1):
            if n == site:
                op_list.append(sigmay())
            else:
                op_list.append(qeye(2))

        # Return a tensor made from the list
        return tensor(op_list)

    def spin_op_z(dim,site):
        
        '''
        Want to take in a number dim = N (dimension of the Hilbert space) and make a spin operator
        for $\sigma_z$ on spin site = j.
        '''
        
        # Instantiate the empty list to build the operator into
        
        op_list = []
        
        # Build the list op_list where all j != site are 2x2 identity matrices
        # and j == site is \sigma_y
        for n in range(1,dim+1):
            if n == site:
                op_list.append(sigmaz())
            else:
                op_list.append(qeye(2))

        # Return a tensor made from the list
        return tensor(op_list)

    def op_list(N, # Total number of sites
               ):
        
        '''
        Outputs the list of operators for the chain. The last operator is
        always an identity matrix (global phase).
        '''
        
        ####################################################################
        # Make list of tuples that are the site pairings to make the bonds #
        ####################################################################
        
        # Instantiate the list
        tups = bonds(N)
        
        dim = [[2 for _ in range(N)],[2 for _ in range(N)]]
        
        identity = Qobj(np.diag([1 for i in range(2**N)]),
                       dims=dim,shape=[[2**N],[2**N]])
        
        opx = []
        opz = []
        opzz = []
        opyy = []
        opxx = []

        for a,b in tups:
            opzz.append(spin_op_z(N,a)*spin_op_z(N,b))
            opyy.append(spin_op_y(N,a)*spin_op_y(N,b))
            opxx.append(spin_op_x(N,a)*spin_op_x(N,b))
            
        holder = set({})
        for a,b in tups:
            holder.add(a)
            holder.add(b)
        
        for k in holder:
            opx.append(spin_op_x(N,k))
            opz.append(spin_op_z(N,k))
        
        total_ops = []
        
        for i in range(len(opzz)):
            total_ops.append(opzz[i])
            total_ops.append(opyy[i])
            total_ops.append(opxx[i])
            
        for i in range(len(opx)):
            total_ops.append(opx[i])

        for i in range(len(opz)):
            total_ops.append(opz[i])
                
        # This attaches the operator for the phase at the end of the list
        total_ops.append(identity)
                
        # Return the full list of operators
        return total_ops

    def theta_state(thetas, # array of thetas
                    ops, # The operators, create separately
                    ref_state, # the reference state
                   ):
        
        '''
        Used to generate the parameterized state. Give it the list of thetas,
        the list of operators, and the reference state. It outputs the 
        parameterized state.
        '''
        
        # Figure out how many layers it's been given
        p = int((len(thetas) - 1)/len(ops[0:len(ops)-1]))
        # Make an identity matrix of the appropriate dims and shape
        identity = Qobj(np.diag([1 for i in range(ops[0].shape[0])]),
                       dims=ops[0].dims, shape=ops[0].shape)
        # param_state will get updated
        param_state = ref_state
        # Extend the list of operators such that there is one for each theta
        # (and in the appropriate order given the layer number p)
        op_list = []
        for i in range(p):
            op_list.extend(ops[0:len(ops)-1])
        op_list.append(ops[-1])
        # Build the parameteried state for this ansatz
        for th,op in zip(thetas,op_list):
            param_state = (np.cos(th) * identity - 1j * np.sin(th) * op) * param_state
        #         param_state = (np.cos(th, dtype=ld) * identity \
        #                        - 1j * np.sin(th, dtype=ld) * op).dot(param_state)
        
        return param_state

    def state_vec(n, # Local Hilbert space of a single site
                  N, # The number of total sites
                  a=None, # Can give it a list directly to make a particular state
                 ):
        
        '''
        The dimension of the basis for a single spin is M. Want to take in args such that
        (M*len(list(args)) = dimension of Hilbert space) and a list of arguments *args = 
        0 or 1 that have length len(list(args) = N and represent the desired state. For
        example, we might have M = 2 and the list of args = [1,0,0], which represents the 
        object: tensor(basis(2,1),basis(2,0),basis(2,0)).
        
        Note: This uses QuTiP functionality
        '''
        
        # Initialize the list to use in tensor()
        if a == None:
            vec = [basis(n,randint(0,1)) for _ in range(N)]
        else: 
            vec = [basis(n,j) for j in a]
        
        return tensor(vec)

    def hamil(beg, # The first site to include
          end, # The last site to include
          N, # The total number of sites
          coup, # The couplings, zz, x, z
          h=False, # Include transverse field
          g=False, # Include longitudinal field
         ):
    
        '''
        The Hamiltonian. N is the total number of sites, and coup is the 
        couplings. Flag h True for transverse Ising, and flag g True for 
        longitudinal Ising.
        '''
        
        H = 0
        
        ######################################################
        # Make a list of all the spin operators at each site #
        ######################################################
        
        tups = bonds(N)
        
        # Instantiate the list of operators
        op_x = []
        op_z = []

        # Populate the lists
        for i in range(1,N+1):
            op_x.append(spin_op_x(N,i))
            op_z.append(spin_op_z(N,i))
        
        # Use the lists to make the Hamiltonian, where the list of couplings = coup
        for i,j in tups:
            if i >= beg and i < end:
                H += coup[0]*op_z[i-1]*op_z[j-1]
        
        if h or g:
            holder = set({})
            for a,b in tups:
                holder.add(a)
                holder.add(b)
        
        if h:
            for k in holder:
                if (k >= beg and k <= end):
                    H += coup[1]*op_x[k-1]
                
        if g:
            for k in holder:
                if (k >= beg and k <= end):
                    H += coup[2]*op_z[k-1]
        
        return H

    def Rxx(q0, q1, theta):
        yield [cirq.H(q0), cirq.H(q1)]
        yield cirq.CNOT(q0, q1)
        yield cirq.rz(theta).on(q1)
        yield cirq.CNOT(q0, q1)
        yield [cirq.HPowGate(exponent=-1).on(q0), cirq.HPowGate(exponent=-1).on(q1)]

    def Ryy(q0, q1, theta):
        yield [cirq.ZPowGate(exponent=0.5).on(q0),cirq.ZPowGate(exponent=0.5).on(q1)]
        yield [cirq.H(q0), cirq.H(q1)]
        yield cirq.CNOT(q0, q1)
        yield cirq.rz(theta).on(q1)
        yield cirq.CNOT(q0, q1)
        yield [cirq.HPowGate(exponent=-1).on(q0), cirq.HPowGate(exponent=-1).on(q1)]
        yield [cirq.ZPowGate(exponent=-0.5).on(q0),cirq.ZPowGate(exponent=-0.5).on(q1)]
        
    def Rzz(q0, q1, theta):
        yield cirq.CNOT(q0, q1)
        yield cirq.rz(theta).on(q1)
        yield cirq.CNOT(q0, q1)

    def cRzz(q0,q1,q2, theta):
        yield cirq.CCNOT(q0, q1, q2)
        yield cirq.rz(theta).on(q2).controlled_by(q0)
        yield cirq.CCNOT(q0, q1, q2)
        
    def crx(q0, q1, theta):
        yield cirq.rx(theta).on(q1).controlled_by(q0)

    def crz(q0, q1, theta):
        yield cirq.rz(theta).on(q1).controlled_by(q0)

    def cZZ(q0,q1,q2):
        yield cirq.CZ(q0,q1)
        yield cirq.CZ(q0,q2)

    def cXX(q0,q1,q2):
        yield cirq.CNOT(q0, q1)
        yield cirq.CNOT(q0, q2)
        
    def cYY(q0,q1,q2):
        yield [cirq.ZPowGate(exponent=0.5).on(q1), cirq.ZPowGate(exponent=0.5).on(q2)]
        yield cirq.CNOT(q0, q1)
        yield cirq.CNOT(q0, q2)
        yield [cirq.ZPowGate(exponent=-0.5).on(q1), cirq.ZPowGate(exponent=-0.5).on(q2)]
        
    def ZZ(q0, q1):
        yield cirq.Z(q0)*cirq.Z(q1)
        
    def measure(q0,q1):
        yield cirq.measure(q0, key='q0'), cirq.measure(q1, key='q1')

    def indices_FENG_FULL(mu, num_sites):
        if mu < (5*num_sites - 3):
            layer = 0
            i = mu
            if i < 3 * (num_sites - 1):
                site = i // 3
            else:
                site = (i - 3 * num_sites + 3) % num_sites
        elif mu == (5*num_sites - 3): # global phase term
            layer = 1
            i = 0
            site = 0
        else: 
            raise (ValueError(f'Index mu = {mu} outside bounds of ansatz.'))
        return (layer, i, site)

    def circuit_M_1(m, n, thetas, num_layers, num_sites, sv_bool = False):
        if m > n : # make sure m < n (the output is symmetric)
            mm = m
            nn = n
            m = nn
            n = mm
        
        l_m, i_m, site_m = indices_FENG_FULL(m, num_sites)
        l_n, i_n, site_n = indices_FENG_FULL(n, num_sites)
        
        # l_m is the layer a site is in
        # i_m is the 
        
        #print(f"l_m = {l_m}, i_m = {i_m}, site_m = {site_m}")
        #print(f"l_n = {l_n}, i_n = {i_n}, site_n = {site_n}")
            
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on(ancilla))
        circuit.append(cirq.X.on(ancilla))
        
        # if n == global phase, only go through available layers
        if l_n == num_layers: # n = global phase parameter
            max_layer = l_n - 1
        else: 
            max_layer = l_n
            
        # go through layers 0 to l_n, unless l_n = num_layers in which case n is 
        # global phase term, and we only go through layers 0 to num_layers - 1.
        for layer in range(max_layer + 1):
            #print(f"layer = {layer}")
            
            # if l_m = l_n
            if l_m == l_n: # could be either actual layer or global phase term
                # build full ansatz for layers < l_m == l_n
                if layer < l_m:
                    j = 0
                    for site in range(num_sites - 1):
                        circuit.append(Rzz(qb[site],qb[site + 1], thetas[j]))
                        j += 1
                        circuit.append(Ryy(qb[site],qb[site + 1], thetas[j]))
                        j += 1
                        circuit.append(Rxx(qb[site],qb[site + 1], thetas[j]))
                        j += 1
                    for site in range(num_sites):
                        circuit.append(cirq.rx(thetas[j]).on(qb[site]))
                        j += 1
                    for site in range(num_sites):
                        circuit.append(cirq.rz(thetas[j]).on(qb[site]))
                        j += 1
            
                # in layer == l_m: build ansatz up to i_m, then insert controlled gate, 
                # the build ansatz for i > i_m
                elif layer == l_m: # will only be called if m is not the global phase term
                    # go through variables i < i_m
                    for i in range(i_m):
                        if i < (3 * num_sites - 3):
                            site = i // 3
                            if i % 3 == 0:
                                circuit.append(Rzz(qb[site],qb[site + 1], thetas[i]))
                            elif i % 3 == 1:
                                circuit.append(Ryy(qb[site],qb[site + 1], thetas[i]))
                            else:
                                circuit.append(Rxx(qb[site],qb[site + 1], thetas[i]))
                        elif (3 * num_sites - 3) <= i < (4 * num_sites - 3):
                            site = i - (3 * num_sites - 3)
                            circuit.append(cirq.rx(thetas[i]).on(qb[site]))
                        else:
                            site = i - (4 * num_sites - 3)
                            circuit.append(cirq.rz(thetas[i]).on(qb[site]))

                    # now insert the controlled sigma_m gate (is cRzz, cRx, cRz 
                    # depending on type of site i_m)
                    if i_m < (3 * num_sites - 3):
                        site = i_m // 3
                        if i_m % 3 == 0:
                            # insert cZZ
                            #print(f"insert an Rzz gate at site {site}")
                            circuit.append(Rzz(qb[site],qb[site + 1], thetas[i_m]))
                            circuit.append(cZZ(ancilla, qb[site], qb[site + 1]))
                        elif i_m % 3 == 1:
                            # insert cYY
                            #print(f"insert an Ryy gate at site {site}")
                            circuit.append(Ryy(qb[site],qb[site + 1], thetas[i_m]))
                            circuit.append(cYY(ancilla, qb[site], qb[site + 1]))
                        else:
                            # insert cXX
                            #print(f"insert an Rxx gate at site {site}")
                            circuit.append(Rxx(qb[site],qb[site + 1], thetas[i_m]))
                            circuit.append(cXX(ancilla, qb[site], qb[site + 1]))
                    elif (3 * num_sites - 3) <= i_m < (4 * num_sites - 3):
                        site = i_m - (3 * num_sites - 3)
                        # insert cX
                        circuit.append(cirq.rx(thetas[i_m]).on(qb[site]))
                        circuit.append(cirq.CNOT(ancilla, qb[site]))
                    else:
                        site = i_m - (4 * num_sites - 3)
                        # insert cZ
                        circuit.append(cirq.rz(thetas[i_m]).on(qb[site]))
                        circuit.append(cirq.CZ(ancilla, qb[site]))
                        
                    circuit.append(cirq.X.on(ancilla))
                        
                    # go through variables i_n > i > i_m
                    for i in range(i_m+1, i_n):
                        if i < (3 * num_sites - 3):
                            site = i // 3
                            if i % 3 == 0:
                                circuit.append(Rzz(qb[site],qb[site + 1], thetas[i]))
                            elif i % 3 == 1:
                                circuit.append(Ryy(qb[site],qb[site + 1], thetas[i]))
                            else:
                                circuit.append(Rxx(qb[site],qb[site + 1], thetas[i]))
                        elif (3 * num_sites - 3) <= i < (4 * num_sites - 3):
                            site = i - (3 * num_sites - 3)
                            circuit.append(cirq.rx(thetas[i]).on(qb[site]))
                        else:
                            site = i - (4 * num_sites - 3)
                            circuit.append(cirq.rz(thetas[i]).on(qb[site]))
                            
                    # now insert the controlled sigma_n gate (is cRzz, cRx, cRz depending 
                    # on type of site i_n)
                    if i_n < (3 * num_sites - 3):
                        site = i_n // 3
                        if i_n % 3 == 0:
                            # insert cZZ
                            #print(f"insert an Rzz gate at site {site}")
                            circuit.append(cZZ(ancilla, qb[site], qb[site + 1]))
                        elif i_n % 3 == 1:
                            # insert cYY
                            #print(f"insert an Ryy gate at site {site}")
                            circuit.append(cYY(ancilla, qb[site], qb[site + 1]))
                        else:
                            # insert cXX
                            #print(f"insert an Rxx gate at site {site}")
                            circuit.append(cXX(ancilla, qb[site], qb[site + 1]))
                    elif (3 * num_sites - 3) <= i_n < (4 * num_sites - 3):
                        site = i_n - (3 * num_sites - 3)
                        # insert cX
                        circuit.append(cirq.CNOT(ancilla, qb[site]))
                    else:
                        site = i_n - (4 * num_sites - 3)
                        # insert cZ
                        circuit.append(cirq.CZ(ancilla, qb[site]))
                        
                if l_m == num_layers: # m is global phase term
                    circuit.append(cirq.X.on(ancilla))
        
            else: # l_n > l_m. Therefore, m cannot be the global phase term, but n might be.
            # build full ansatz for layers < l_m
                if layer < l_m:
                    j = 0
                    for site in range(num_sites - 1):
                        circuit.append(Rzz(qb[site],qb[site + 1], thetas[j]))
                        j += 1
                        circuit.append(Ryy(qb[site],qb[site + 1], thetas[j]))
                        j += 1
                        circuit.append(Rxx(qb[site],qb[site + 1], thetas[j]))
                        j += 1
                    for site in range(num_sites):
                        circuit.append(cirq.rx(thetas[j]).on(qb[site]))
                        j += 1
                    for site in range(num_sites):
                        circuit.append(cirq.rz(thetas[j]).on(qb[site]))
                        j += 1

                # in layer == l_m: build ansatz up to i_m, then insert controlled gate, 
                # the build ansatz for i > i_m
                elif layer == l_m:
                    # go through variables i < i_m
                    for i in range(i_m):
                        if i < (3 * num_sites - 3):
                            site = i // 3
                            if i % 3 == 0:
                                circuit.append(Rzz(qb[site],qb[site + 1], thetas[i]))
                            elif i % 3 == 1:
                                circuit.append(Ryy(qb[site],qb[site + 1], thetas[i]))
                            else:
                                circuit.append(Rxx(qb[site],qb[site + 1], thetas[i]))
                        elif (3 * num_sites - 3) <= i < (4 * num_sites - 3):
                            site = i - (3 * num_sites - 3)
                            circuit.append(cirq.rx(thetas[i]).on(qb[site]))
                        else:
                            site = i - (4 * num_sites - 3)
                            circuit.append(cirq.rz(thetas[i]).on(qb[site]))

                    # now insert the controlled sigma_m gate (is cRzz, cRx, cRz 
                    # depending on type of site i_m)
                    if i_m < (3 * num_sites - 3):
                        site = i_m // 3
                        if i_m % 3 == 0:
                            # insert cZZ
                            #print(f"insert an Rzz gate at site {site}")
                            circuit.append(Rzz(qb[site],qb[site + 1], thetas[i_m]))
                            circuit.append(cZZ(ancilla, qb[site], qb[site + 1]))
                        elif i_m % 3 == 1:
                            # insert cYY
                            #print(f"insert an Ryy gate at site {site}")
                            circuit.append(Ryy(qb[site],qb[site + 1], thetas[i_m]))
                            circuit.append(cYY(ancilla, qb[site], qb[site + 1]))
                        else:
                            # insert cXX
                            #print(f"insert an Rxx gate at site {site}")
                            circuit.append(Rxx(qb[site],qb[site + 1], thetas[i_m]))
                            circuit.append(cXX(ancilla, qb[site], qb[site + 1]))
                    elif (3 * num_sites - 3) <= i_m < (4 * num_sites - 3):
                        site = i_m - (3 * num_sites - 3)
                        # insert cX
                        circuit.append(cirq.rx(thetas[i_m]).on(qb[site]))
                        circuit.append(cirq.CNOT(ancilla, qb[site]))
                    else:
                        site = i_m - (4 * num_sites - 3)
                        # insert cZ
                        circuit.append(cirq.rz(thetas[i_m]).on(qb[site]))
                        circuit.append(cirq.CZ(ancilla, qb[site]))

                    circuit.append(cirq.X.on(ancilla)) 
                    
                    # go through variables i > i_m
                    for i in range(i_m+1, 5*num_sites - 3):
                        if i < (3 * num_sites - 3):
                            site = i // 3
                            if i % 3 == 0:
                                circuit.append(Rzz(qb[site],qb[site + 1], thetas[i]))
                            elif i % 3 == 1:
                                circuit.append(Ryy(qb[site],qb[site + 1], thetas[i]))
                            else:
                                circuit.append(Rxx(qb[site],qb[site + 1], thetas[i]))
                        elif (3 * num_sites - 3) <= i < (4 * num_sites - 3):
                            site = i - (3 * num_sites - 3)
                            circuit.append(cirq.rx(thetas[i]).on(qb[site]))
                        else:
                            site = i - (4 * num_sites - 3)
                            circuit.append(cirq.rz(thetas[i]).on(qb[site]))

                # build full ansatz for layers > l_m
                elif (layer > l_m) and (layer < l_n):
                    j = 0
                    for site in range(num_sites - 1):
                        circuit.append(Rzz(qb[site],qb[site + 1], thetas[j]))
                        j += 1
                        circuit.append(Ryy(qb[site],qb[site + 1], thetas[j]))
                        j += 1
                        circuit.append(Rxx(qb[site],qb[site + 1], thetas[j]))
                        j += 1
                    for site in range(num_sites):
                        circuit.append(cirq.rx(thetas[j]).on(qb[site]))
                        j += 1
                    for site in range(num_sites):
                        circuit.append(cirq.rz(thetas[j]).on(qb[site]))
                        j += 1

                # in layer == l_n: build ansatz up to i_n, then insert controlled gate, 
                # the build ansatz for i > i_n
                elif layer == l_n:
                    # go through variables i < i_n
                    for i in range(i_n):
                        if i < (3 * num_sites - 3):
                            site = i // 3
                            if i % 3 == 0:
                                circuit.append(Rzz(qb[site],qb[site + 1], thetas[i]))
                            elif i % 3 == 1:
                                circuit.append(Ryy(qb[site],qb[site + 1], thetas[i]))
                            else:
                                circuit.append(Rxx(qb[site],qb[site + 1], thetas[i]))
                        elif (3 * num_sites - 3) <= i < (4 * num_sites - 3):
                            site = i - (3 * num_sites - 3)
                            circuit.append(cirq.rx(thetas[i]).on(qb[site]))
                        else:
                            site = i - (4 * num_sites - 3)
                            circuit.append(cirq.rz(thetas[i]).on(qb[site]))

                    if i_n < (3 * num_sites - 3):
                        site = i_n // 3
                        if i_n % 3 == 0:
                            # insert cZZ
                            #print(f"insert an Rzz gate at site {site}")
                            circuit.append(Rzz(qb[site],qb[site + 1], thetas[i_n]))
                            circuit.append(cZZ(ancilla, qb[site], qb[site + 1]))
                        elif i_n % 3 == 1:
                            # insert cYY
                            #print(f"insert an Ryy gate at site {site}")
                            circuit.append(Ryy(qb[site],qb[site + 1], thetas[i_n]))
                            circuit.append(cYY(ancilla, qb[site], qb[site + 1]))
                        else:
                            # insert cXX
                            #print(f"insert an Rxx gate at site {site}")
                            circuit.append(Rxx(qb[site],qb[site + 1], thetas[i_n]))
                            circuit.append(cXX(ancilla, qb[site], qb[site + 1]))
                    elif (3 * num_sites - 3) <= i_n < (4 * num_sites - 3):
                        site = i_n - (3 * num_sites - 3)
                        # insert cX
                        circuit.append(cirq.rx(thetas[i_n]).on(qb[site]))
                        circuit.append(cirq.CNOT(ancilla, qb[site]))
                    else:
                        site = i_n - (4 * num_sites - 3)
                        # insert cZ
                        circuit.append(cirq.rz(thetas[i_n]).on(qb[site]))
                        circuit.append(cirq.CZ(ancilla, qb[site]))
                        
        circuit.append(cirq.H.on(ancilla))
        # add measurement gate on ancilla if not sv simulation
        if sv_bool == False: 
            circuit.append(cirq.measure(ancilla, key = 'ancilla'))
        
        return circuit 

    def circuit_V_1(ham_type, ham_site, m, thetas, num_layers, num_sites, sv_bool = False):

        l_m, i_m, site_m = indices_FENG_FULL(m, num_sites)
        
        #print(f"l_m = {l_m}, i_m = {i_m}, site_m = {site_m}")
        
        # if n == global phase, only go through available layers
        if l_m == num_layers:
            max_layer = l_m - 1
        else: 
            max_layer = l_m
        
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on(ancilla))
        circuit.append(cirq.X.on(ancilla))
        
        # go through all layers 
        for layer in range(num_layers):
            #print(f"layer = {layer}")

            # build full ansatz for layers < l_m
            if layer < l_m:
                j = 0
                for site in range(num_sites - 1):
                    circuit.append(Rzz(qb[site],qb[site + 1], thetas[j]))
                    j += 1
                    circuit.append(Ryy(qb[site],qb[site + 1], thetas[j]))
                    j += 1
                    circuit.append(Rxx(qb[site],qb[site + 1], thetas[j]))
                    j += 1
                for site in range(num_sites):
                    circuit.append(cirq.rx(thetas[j]).on(qb[site]))
                    j += 1
                for site in range(num_sites):
                    circuit.append(cirq.rz(thetas[j]).on(qb[site]))
                    j += 1

            # in layer == l_m: build ansatz up to i_m, then insert controlled gate,
            # then build ansatz for i > i_m
            elif layer == l_m:
                # go through variables i < i_m
                for i in range(i_m):
                    if i < (3 * num_sites - 3):
                        site = i // 3
                        if i % 3 == 0:
                            circuit.append(Rzz(qb[site],qb[site + 1], thetas[i]))
                        elif i % 3 == 1:
                            circuit.append(Ryy(qb[site],qb[site + 1], thetas[i]))
                        else:
                            circuit.append(Rxx(qb[site],qb[site + 1], thetas[i]))
                    elif (3 * num_sites - 3) <= i < (4 * num_sites - 3):
                        site = i - (3 * num_sites - 3)
                        circuit.append(cirq.rx(thetas[i]).on(qb[site]))
                    else:
                        site = i - (4 * num_sites - 3)
                        circuit.append(cirq.rz(thetas[i]).on(qb[site]))

                # now insert the controlled sigma_m gate (is cRzz, cRx, cRz [not cRz anymore]
                # depending on type of site i_m)
                if i_m < (3 * num_sites - 3):
                    site = i_m // 3
                    if i_m % 3 == 0:
                        # insert cZZ
                        #print(f"insert an Rzz gate at site {site}")
                        circuit.append(Rzz(qb[site],qb[site + 1], thetas[i_m]))
                        circuit.append(cZZ(ancilla, qb[site], qb[site + 1]))
                    elif i_m % 3 == 1:
                        # insert cYY
                        #print(f"insert an Ryy gate at site {site}")
                        circuit.append(Ryy(qb[site],qb[site + 1], thetas[i_m]))
                        circuit.append(cYY(ancilla, qb[site], qb[site + 1]))
                    else:
                        # insert cXX
                        #print(f"insert an Rxx gate at site {site}")
                        circuit.append(Rxx(qb[site],qb[site + 1], thetas[i_m]))
                        circuit.append(cXX(ancilla, qb[site], qb[site + 1]))
                elif (3 * num_sites - 3) <= i_m < (4 * num_sites - 3):
                    site = i_m - (3 * num_sites - 3)
                    # insert cX
                    circuit.append(cirq.rx(thetas[i_m]).on(qb[site]))
                    circuit.append(cirq.CNOT(ancilla, qb[site]))
                else:
                    site = i_m - (4 * num_sites - 3)
                    # insert cZ
                    circuit.append(cirq.rz(thetas[i_m]).on(qb[site]))
                    circuit.append(cirq.CZ(ancilla, qb[site]))

                circuit.append(cirq.X.on(ancilla))

                # go through variables i > i_m
                for i in range(i_m+1, 5*num_sites - 3):
                    if i < (3 * num_sites - 3):
                        site = i // 3
                        if i % 3 == 0:
                            circuit.append(Rzz(qb[site],qb[site + 1], thetas[i]))
                        elif i % 3 == 1:
                            circuit.append(Ryy(qb[site],qb[site + 1], thetas[i]))
                        else:
                            circuit.append(Rxx(qb[site],qb[site + 1], thetas[i]))
                    elif (3 * num_sites - 3) <= i < (4 * num_sites - 3):
                        site = i - (3 * num_sites - 3)
                        circuit.append(cirq.rx(thetas[i]).on(qb[site]))
                    else:
                        site = i - (4 * num_sites - 3)
                        circuit.append(cirq.rz(thetas[i]).on(qb[site]))

            # build full ansatz for all layers > l_m until the end
            elif layer > l_m:
                j = 0
                for site in range(num_sites - 1):
                    circuit.append(Rzz(qb[site],qb[site + 1], thetas[j]))
                    j += 1
                    circuit.append(Ryy(qb[site],qb[site + 1], thetas[j]))
                    j += 1
                    circuit.append(Rxx(qb[site],qb[site + 1], thetas[j]))
                    j += 1
                for site in range(num_sites):
                    circuit.append(cirq.rx(thetas[j]).on(qb[site]))
                    j += 1
                for site in range(num_sites):
                    circuit.append(cirq.rz(thetas[j]).on(qb[site]))
                    j += 1
        
        # now insert the controlled sigma_n gate that appears in the Hamiltonian
        # (is cZZ, cRx, cRz depending on ham_type)
        if ham_type == 'ZZ':
            site = ham_site # between 0 and num_sites - 1
            # insert cZZ. Note that I do not need to insert Rzz here as 
            # (in contrast to i_m above).
            circuit.append(cZZ(ancilla, qb[site], qb[site + 1]))
        elif ham_type == 'YY':
            site = ham_site # between 0 and num_sites - 1
            # insert cYY. Note that I do not need to insert Ryy here as 
            # (in contrast to i_m above).
            circuit.append(cYY(ancilla, qb[site], qb[site + 1]))
        elif ham_type == 'XX':
            site = ham_site # between 0 and num_sites - 1
            # insert cXX. Note that I do not need to insert Rxx here as 
            # (in contrast to i_m above).
            circuit.append(cXX(ancilla, qb[site], qb[site + 1]))
        elif ham_type == 'X':
            site = ham_site
            # insert cX. Note that I do not need to insert rx here as
            # (in contrast to i_m above).
            circuit.append(cirq.CNOT(ancilla, qb[site]))
        elif ham_type == 'Z':
            site = ham_site
            # insert cZ. Note that I do not need to insert rz here as
            # (in contrast to i_m above).
            circuit.append(cirq.CZ(ancilla, qb[site]))
        else:
            raise(ValueError('ham_type not supported. Supported choices are ZZ, YY, XX, X, Z.'))

        circuit.append(cirq.H.on(ancilla))
        # add measurement gate on ancilla if not sv simulation
        if sv_bool == False: 
            circuit.append(cirq.measure(ancilla, key = 'ancilla'))
        return circuit

    def M_1(m, n, thetas, num_layers, num_sites, num_circuits, sv_bool = False):
        simulator = cirq.Simulator()
        if sv_bool == True:         
            result = simulator.simulate_expectation_values(circuit_M_1(m, n, thetas,
                        num_layers, num_sites, sv_bool), observables = [cirq.Z(ancilla)])
            M_1_res = result[0].real
        else: 
            # need to append measurement gate if using run (shots)
            # circuit.append(cirq.measure(ancilla, key = 'ancilla'))
            result = simulator.run(circuit_M_1(m, n, thetas, num_layers,
                        num_sites), repetitions=num_circuits)
            res = result.data
            p1 = res['ancilla'].sum()/res['ancilla'].shape[0]
            M_1_res = 1 - 2.*p1
        return M_1_res

    def emm_sv():
        # M_1_matrix_sv = np.zeros((len(thetas), len(thetas)))
        M_1_matrix_sv = np.identity(len(thetas))
        for m in range(len(thetas)):
            for n in range(m, len(thetas)):
                M_1_matrix_sv[m, n] = M_1(m, n, thetas, p, Nl, shotsF[m, n], sv_bool = True)
                if n > m:
                    M_1_matrix_sv[n, m] = M_1_matrix_sv[m, n]
                
        return 2.*M_1_matrix_sv

    def emm(shotsF):
        # M_1_matrix = np.zeros((len(thetas), len(thetas)))
        M_1_matrix = np.identity(len(thetas))
        circuit = cirq.Circuit()
        for m in range(len(thetas)):
            for n in range(m, len(thetas)):
                shots = int(shotsF[m, n])
                if shots < 100:
                    shots = 100
                M_1_matrix[m, n] = M_1(m, n, thetas, p, Nl, shots, False)
                if n > m:
                    M_1_matrix[n, m] = M_1_matrix[m, n]
                
        return 2.*M_1_matrix

    def vee_sv():
        V_vector_sv = np.zeros((len(thetas)))

        simulator = cirq.Simulator()

        # result_H_sv = sum(simulator.simulate_expectation_values(circuit_H('ZZ', thetas, p,
        # Nl, sv_bool = True), observables = obs_ZZ + obs_X + obs_Z))
        # print(f"<H> = {result_H_sv}")

        for m in range(len(thetas)):

            # JzVal = 1.
            # hxVal = float(1/np.sqrt(3))
            # hxVal = 1.

            obs_ZZ = []
            # obs_Z = []
            obs_X = []

            # this builds all terms in the Hamiltonian. If simulating a subregion, 
            # need to build only using those terms. For this, replace range(len(qb)-1)
            # by a list of sites of the subregion

            for i in range(len(qb)-1):
                obs_ZZ.append(JzVal*cirq.PauliString(cirq.ops.Z.on(qb[i]),
                                                     cirq.ops.Z.on(qb[i+1])))

            for i in range(len(qb)):
                # obs_Z.append(hzVal*cirq.PauliString(cirq.ops.Z.on(qb[i])))
                obs_X.append(hxVal*cirq.PauliString(cirq.ops.X.on(qb[i])))

            result_V1_sv = 0.

            for ham_site in range(Nl-1):
                result_V1_sv += JzVal*simulator.simulate_expectation_values(circuit_V_1('ZZ',
                            ham_site, m, thetas, p, Nl, sv_bool = True), 
                            observables = [cirq.Z(ancilla)])[0]

            for ham_type in ['X']:
                for ham_site in range(Nl):
                    if ham_type == 'X':
                        result_V1_sv += hxVal*simulator.simulate_expectation_values(circuit_V_1(
                            ham_type, ham_site, m, thetas, p, Nl, sv_bool = True), 
                            observables = [cirq.Z(ancilla)])[0]
                        #print(f"hx*X[{ham_site}] = {hxVal*simulator.simulate_expectation_values(
                        # circuit_V_1(ham_type, ham_site, m, theta_params, num_layers, num_sites, 
                        # sv_bool = True), observables = [cirq.Z(ancilla)])[0]}")
                    # elif ham_type == 'Z':
                    #     result_V1_sv += hzVal*simulator.simulate_expectation_values(
                        # circuit_V_1(ham_type, ham_site, m, thetas, p, Nl, sv_bool = True), 
                        # observables = [cirq.Z(ancilla)])[0]
                        # print(f"hz*Z[{ham_site}] = {hzVal*simulator.simulate_expectation_values(

            result_V_sv = 2.*(result_V1_sv) # + np.imag(np.conj(M_2_sv)*result_H_sv))
            V_vector_sv[m] = np.real(result_V_sv)
        
        return V_vector_sv

    def vee(shotsG):
        V_vector = np.zeros((len(thetas)))

        simulator = cirq.Simulator()
        circuit = cirq.Circuit()

        for m in range(len(thetas)):

            result_V1 = 0.

            # this builds all terms in the Hamiltonian. If simulating a subregion, need to 
            # build only using those terms. For this, replace range(len(qb)-1) by a list 
            # of sites of the subregion
            for ham_site in range(Nl-1):
                shots = int(np.round(shotsG[m]))
                if shots < 100:
                    shots = 100
                result = simulator.run(circuit_V_1('ZZ', ham_site, m,
                                    thetas, p, Nl), repetitions=shots)
                res = result.data
                p1 = res['ancilla'].sum()/res['ancilla'].shape[0]
                Z_ancilla = 1 - 2*p1
                result_V1 += JzVal*Z_ancilla
                #print(f"Jz*ZZ[{ham_site}] = {JzVal*Z_ancilla}")

            for ham_type in ['X']:
                for ham_site in range(Nl):
                    shots = int(np.round(shotsG[m]))
                    if shots < 100:
                        shots = 100
                    result = simulator.run(circuit_V_1(ham_type, ham_site, m,
                                    thetas, p, Nl), repetitions=shots)
                    res = result.data
                    p1 = res['ancilla'].sum()/res['ancilla'].shape[0]
                    Pauli_ancilla = 1 - 2*p1
                    if ham_type == 'X':
                        result_V1 += hxVal*Pauli_ancilla
                        #print(f"hx*X[{ham_site}] = {hxVal*Pauli_ancilla}")
        #             elif ham_type == 'Z':
        #                 result_V1 += hzVal*Pauli_ancilla
        #                 #print(f"hz*Z[{ham_site}] = {hzVal*Pauli_ancilla}")

            result_V = 2.*(result_V1) # + np.imag(np.conj(result_M2)*result_H))
            V_vector[m] = np.real(result_V)
            
        return V_vector

    def distM(M, 
              V, 
              d=None # Defines the norm you want to use.
             ):
        ''' Use to find the shot distribution for M. '''
        
        if d == None:
            d = 1/2
        
        m_inv = np.linalg.inv(M + de * np.identity(len(thetas)))
        
        denom = (sum([sum([abs(sum([sum(
            [(sum([V[k2]*m_inv[k2,j2]*m_inv[j1,i2] for k2 in range(len(thetas))])
            * sum([V[k1]*m_inv[i1,j1]*m_inv[j2,k1] for k1 in range(len(thetas))])*M[i2,i1]/2)
            for i2 in range(len(thetas))]) for i1 in range(len(thetas))]))**(2*d)
            for j2 in range(j1 + 1, len(thetas))]) for j1 in range(len(thetas))]))**(1/d)
        
        mat = np.zeros((len(thetas),len(thetas)))
        
        for j1 in range(len(thetas)):
            for j2 in range(j1 + 1, len(thetas)):
                mat[j1,j2] = (sum([sum([
                    sum([V[k2]*m_inv[k2,j2]*m_inv[j1,i2] for k2 in range(len(thetas))])
                    * sum([V[k1]*m_inv[i1,j1]*m_inv[j2,k1] for k1 in range(len(thetas))])
                    * M[i2,i1]/2 for i2 in range(len(thetas))])
                    for i1 in range(len(thetas))]))**2/denom
        
        return mat

    def distV(M, 
          d=None # Defines the norm you want to use.
         ):
        
        ''' Use to find the shot distribution for V. '''
        
        if d == None:
            d = 1/2
        
        m_inv = np.linalg.inv(M + de * np.identity(len(thetas)))
        
        denom = (sum([abs(sum([sum([m_inv[i,j]*m_inv[k,i]*M[j,k]/2 for j in range(len(thetas))])
                    for k in range(len(thetas))]))**(2*d) for i in range(len(thetas))]))**(1/d)

        vec = np.zeros((len(thetas),))
        
        for i in range(len(thetas)):
            vec[i] = sum([sum([(m_inv[i,j] * m_inv[k,i] * M[j,k]/2) 
                        for j in range(len(thetas))]) for k in range(len(thetas))])**2/denom
        
        return vec

    def nM(f, # 1 means uniform dist. 0 means maximally non-uniform dist.
           M,
           V,
           d=None, # For the l-norm, default is l = 2 norm
          ):
        
        ''' Gives 'optimal' shot distribution for a value of f. '''
        
        if d == None:
            d = 1/2
        
        L = len(thetas)
        
        N = nTot/((L**2 - L)/2 + L)
        
        return (1 - f) * N * (L**2 - L)/2 * ((distM(M,V,d))**d) + f * N * np.triu(np.ones((L,L)) - np.identity(L))

    def nV(f, # 1 means uniform dist. 0 means maximally non-uniform dist.
           M,
           d=None, # For the l-norm, default is l = 2 norm
          ):
        
        ''' Gives 'optimal' shot distribution for a value of f. '''
        
        if d == None:
            d = 1/2
        
        L = len(thetas)
        
        N = nTot/((L**2 - L)/2 + L)
        
        return (1 - f) * N * L * ((distV(M,d))**d) + f * N * np.ones((len(thetas),))

    # The script:

    b = 2

    rnd.seed(seed)
    thetas = np.array([rnd.random() for _ in range(p*(5*Nl - 3) +1)])

    shotsF = (avg * np.triu(np.ones((len(thetas), len(thetas))) - np.identity(len(thetas)))).astype(int)
    shotsG = (avg * np.ones((len(thetas), ))).astype(int)

    sF = [shotsF]
    sG = [shotsG]

    JzVal = -1
    hxVal = -1

    nTot = int(avg * (len(thetas)**2 - len(thetas))/2 + avg * len(thetas))

    # Define ancilla qubit and line of qb to represent system
    ancilla = cirq.NamedQubit("ancilla")
    qb = [cirq.GridQubit(x, y) for x in range(0, Nl) for y in range(1)]
    circuit = cirq.Circuit()
    simulator = cirq.Simulator()

    ref_state = state_vec(2, Nl, [0 for _ in range(Nl)])
    ops = op_list(Nl)
    H = hamil(1, Nl, Nl, [JzVal, hxVal], True, False)

    if not os.path.exists(f'../shot_allocation/l_eq_{l}/'):
        os.mkdir(f'../shot_allocation/l_eq_{l}/')

    if not os.path.exists(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/'):
        os.mkdir(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/')

    if os.path.exists(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/'):
        if os.path.exists(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/th_opt_{avg}_{Nl}_{p}_{frac}.csv'):
            thets = pd.read_csv(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/th_opt_{avg}_{Nl}_{p}_{frac}.csv', index_col=0)
            thetas = np.array(thets.iloc[-1])
            infid = pd.read_csv(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/inf_opt_{avg}_{Nl}_{p}_{frac}.csv', index_col=0)
            ts = infid.index
            with open(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/sF_{avg}_{Nl}_{p}_{frac}.pkl', 'rb') as g:
                sF = pickle.load(g)
            with open(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/sG_{avg}_{Nl}_{p}_{frac}.pkl', 'rb') as g:
                sG = pickle.load(g)
            shotsF = sF[-1]
            shotsG = sG[-1]
        else:
            param_state = theta_state(thetas, ops, ref_state)
            thets = pd.DataFrame(np.array([thetas]), index=np.array([0]))
            thets.to_csv(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/th_opt_{avg}_{Nl}_{p}_{frac}.csv')
            infid = pd.DataFrame(np.array([1 - np.abs(param_state.overlap(param_state))**2]), index=np.array([0]))
            infid.to_csv(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/inf_opt_{avg}_{Nl}_{p}_{frac}.csv')
            ts = infid.index
            with open(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/sF_{avg}_{Nl}_{p}_{frac}.pkl', 'rb') as g:
                sF = pickle.load(g)
            with open(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/sG_{avg}_{Nl}_{p}_{frac}.pkl', 'rb') as g:
                sG = pickle.load(g)
            shotsF = sF[-1]
            shotsG = sG[-1]
    else:
        os.mkdir(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/')
        param_state = theta_state(thetas, ops, ref_state)
        thets = pd.DataFrame(np.array([thetas]), index=np.array([0]))
        thets.to_csv(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/th_opt_{avg}_{Nl}_{p}_{frac}.csv')
        infid = pd.DataFrame(np.array([1 - np.abs(param_state.overlap(param_state))**2]), index=np.array([0]))
        infid.to_csv(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/inf_opt_{avg}_{Nl}_{p}_{frac}.csv')
        ts = infid.index

    while ts[-1] < tf:
        
        START = timer()
        
        # Do the statevector evolution
        psi_sv = (-1j * H * dt).expm() * theta_state(thetas/2, ops, ref_state)
        
        # Do the shots evolution with (allegedly) optimal shot allotment.
        M = emm(shotsF) # + de * np.identity(len(thetas))
        V = vee(shotsG)
        # dtheta = np.array(np.linalg.inv(M) @ V) * 2 * dt
        dtheta = (lsq_linear(M,V,(-b,b)).x) * 2 * dt
        thetas = thetas.copy() + dtheta
        psi = theta_state(thetas/2, ops, ref_state)
        
        # Redistribute shots
        shotsF = np.round( nM(frac, M + de * np.identity(len(thetas)), V, l) ).astype(int)
        shotsG = np.round( nV(frac, M + de * np.identity(len(thetas)), l) ).astype(int)
        
        sF.append(shotsF)
        sG.append(shotsG)
        
        chk = int(nTot - (sum([sum(x) for x in shotsF]) + sum(shotsG)))
        # print(chk)
        if chk > 0:
            shotsF[0, 1] += chk

        inf = np.array(1 - np.abs(psi.overlap(psi_sv))**2)
        
        if inf < 0.01: # and inf_ < 0.01:

            thets.loc[ts[-1] + dt] = np.array(thetas[:])
            infid.loc[ts[-1] + dt] = inf
            thets.to_csv(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/th_opt_{avg}_{Nl}_{p}_{frac}.csv')
            infid.to_csv(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/inf_opt_{avg}_{Nl}_{p}_{frac}.csv')
            with open(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/sF_{avg}_{Nl}_{p}_{frac}.pkl', 'wb') as g:
                pickle.dump(sF, g)
            with open(f'../shot_allocation/l_eq_{l}/res_EXPANZ_{avg}_{Nl}_{p}_{frac}_{dt}_{de}/run_{run}/sG_{avg}_{Nl}_{p}_{frac}.pkl', 'wb') as g:
                pickle.dump(sG, g)
            
            ts = infid.index
                
        # print(ts[-1], ': ', inf, 'in ', timer() - START)

    run += 1

    if run < 101:
        os.system(f'python3 allocateShotsFF.py -n {Nl} -p {p} -t {tf} -d {dt} -D {de} -f {frac} -l {l} -s {avg} -S {run} -r {run}')
    
