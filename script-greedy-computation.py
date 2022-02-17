#!/usr/bin/env python
# coding: utf-8

import pymoo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.integrate as integrate
from pymoo.factory import get_performance_indicator
import argparse
import os
import json
import random

def f(x, name, r2):
    if x < 0 or x > 1:
        return r2
    # Convex functions
    if name == 'convex-biL':
        e = np.exp(1)
        return e/(e-1) * np.exp(-x) + 1 - e/(e-1)
    elif name == 'convex-doublesphere': 
        return 1 + x - 2*np.sqrt(x)
    elif name == 'convex-zdt1':
        return 1 - np.sqrt(x)
    # Concave functions
    elif name == 'concave-biL': # bi-Lipschitz
        return 1 - x/2 - x**2/2
    elif name == 'concave-dtlz2':
        if 1 - x**2 >= 0:
            return np.sqrt(1-x**2)
        else:
            return r2
    elif name == 'concave-zdt2':
        return 1 - x**2
    # Linear functions
    elif name == 'linear':
        return 1 - x

# Function to compute the greedy sequence
def greedy_next(pop, name, r1, r2, nb_runs):
    r = [r1, r2]
    pop_ordered = sorted(list(pop))
    hvc = 0
    new_x = 0
    # explore the middle gap regions
    for i in range(len(pop)-1):
        r = [pop_ordered[i+1], f(pop_ordered[i], name, r2)]
        hvr = get_performance_indicator("hv", ref_point=np.array(r))
        hvr_pop = lambda pop: hvr.calc(np.array([[x,f(x,name,r2)] for x in pop]))
        # compute the best vector of this gap region
        store = []
        for j in range(nb_runs):
            x0 = [pop_ordered[i] + (pop_ordered[i+1] - pop_ordered[i]) * random.random()]
            res = minimize(lambda x: -hvr_pop(x), x0, method='SLSQP', bounds=[[pop_ordered[i],pop_ordered[i+1]]],
                           options={'disp': False, 'ftol':1e-13,'maxiter':1000})
            store.append(res.fun)
        if any([store[i+1] - store[i] > 1e-12 for i in range(nb_runs -1)]):
            raise ValueError('The optimizer hvc values differ of more than 1e-12 from one run to the other, raising some doubt on whether they find the global optimum.')
        if -res.fun > hvc:
            hvc = -res.fun
            new_x = res.x
    # explore the left extreme gap region
    r = [pop_ordered[0], r2]
    hvr = get_performance_indicator("hv", ref_point=np.array(r))
    hvr_pop = lambda pop: hvr.calc(np.array([[x,f(x,name,r2)] for x in pop]))
    store = []
    for j in range(nb_runs):
        x0 = [pop_ordered[0] * random.random()]
        res = minimize(lambda x: -hvr_pop(x), x0, 
                    method='SLSQP', bounds=[[0,pop_ordered[0]]],
                    options={'disp': False, 'ftol':1e-13,'maxiter':1000})
        store.append(res.fun)
    if any([store[i+1] - store[i] > 1e-12 for i in range(nb_runs -1)]):
        raise ValueError('The optimizer hvc values differ of more than 1e-12 from one run to the other, raising some doubt on whether they find the global optimum.')
    if -res.fun > hvc:
        hvc = -res.fun
        new_x = res.x
    # explore the right extreme gap region
    r = [r1, f(pop_ordered[-1], name, r2)]
    hvr = get_performance_indicator("hv", ref_point=np.array(r))
    hvr_pop = lambda pop: hvr.calc(np.array([[x,f(x,name,r2)] for x in pop]))
    store = []
    for j in range(nb_runs):
        x0 = [1-(1-pop_ordered[-1])*random.random()]
        res = minimize(lambda x: -hvr_pop(x), x0, 
                    method='SLSQP', bounds=[[pop_ordered[-1],1]],
                    options={'disp': False, 'ftol':1e-13,'maxiter':1000})
        store.append(res.fun)
    if any([store[i+1] - store[i] > 1e-12 for i in range(nb_runs -1)]):
        raise ValueError('The optimizer hvc values differ of more than 1e-12 from one run to the other, raising some doubt on whether they find the global optimum.')
    if -res.fun > hvc:
        hvc = -res.fun
        new_x = res.x
    return new_x[0]

def main():
    """Main function, to get the arguments and computing the greedy sets."""
    # Get parameters
    parser = argparse.ArgumentParser(description='Obtain the parameters.\n')
    parser.add_argument('--p', type=int, default=100,help="The greedy set computed is the p-th one.")
    parser.add_argument('--fun', type=str, default='convex-biL', help="The function f describing the Pareto front." 
                        +"Possibilities are : 'convex-biL', 'convex-doublesphere', 'convex-zdt1', 'concave-biL',"
                        +"'concave-dtlz2', 'concave-zdt2' and 'linear'")
    parser.add_argument('--r1', type=float, default=1,help="First coordinate of the reference point.")
    parser.add_argument('--r2', type=float, default=1,help="Second coordinate of the reference point.")
    parser.add_argument('--nb_runs', type=int, default=1,help="Number of times the new greedy vector is computed - to check for incoherence. ")
    args = parser.parse_args()

    # Create the required directories
    new_dir = "/Data-greedy/"
    path = os.getcwd() + new_dir
    if not os.path.exists(path):
        os.mkdir(path)

    # Define the hypervolume indicator
    hv = get_performance_indicator("hv", ref_point=np.array([args.r1, args.r2]))
    hv_pop = lambda pop: hv.calc(np.array([[x,f(x,args.fun,args.r2)] for x in pop]))

    # Compute the p-th greedy set sequence
    # the first
    store = []
    for j in range(args.nb_runs):
        x0 = [random.random()]
        res = minimize(lambda x: -hv_pop(x), x0, 
                    method='SLSQP', bounds=[[0,1]],
                    options={'disp': False, 'ftol':1e-13,'maxiter':1000})
        store.append(res.fun)
    if any([store[i+1] - store[i] > 1e-12 for i in range(args.nb_runs -1)]):
        raise ValueError('The optimizer hvc values differ of more than 1e-12 from one run to the other, raising some doubt on whether they find the global optimum.')
    greedy_set=[res.x[0]]
    # all but the first
    for i in range(1, args.p):
        greedy_set.append(greedy_next(greedy_set, args.fun, args.r1, args.r2, args.nb_runs))
        if i%10 == 0:
            print("p = ", i)
    
    # Write the p-th greedy set sequence
    name_file = ""
    for arg in vars(args):
        name_file += arg + "=" + str(getattr(args, arg)) + "_"
    with open(path + name_file[:-1] + '.txt', 'w') as file:
        file.write(json.dumps(greedy_set))


if __name__ == '__main__':
    main()
