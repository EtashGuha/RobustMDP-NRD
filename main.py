import numpy as np
from robust import oftpl_exp
from argparser import get_args
import pickle

if __name__ == '__main__':
    args = get_args()
    # Create a stochastic transition matrix for the gridworld
    np.random.seed(args.seed)

    obj_vals = oftpl_exp(args)
    
    with open("saved_results/{}.pkl".format(args.name)) as f:
        pickle.dump(obj_vals, f)