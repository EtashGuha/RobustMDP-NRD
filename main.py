import numpy as np
from robust import oftpl_exp
from argparser import get_args
import pickle
import os
from plotter import plot

if __name__ == '__main__':
    args = get_args()
    # Create a stochastic transition matrix for the gridworld
    np.random.seed(args.seed)

    if not os.path.exists("saved_results/{}.pkl".format(args.name)):
        obj_vals = oftpl_exp(args)
        
        with open("saved_results/{}.pkl".format(args.name), "wb") as f:
            pickle.dump(obj_vals, f)
    else:
        with open("saved_results/{}.pkl".format(args.name), "rb") as f:
            obj_vals = pickle.load(f)
        plot(obj_vals, args.name)