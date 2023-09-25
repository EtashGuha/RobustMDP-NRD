import numpy as np
from scipy.optimize import minimize
from mdp import generate_random_policy, generate_valid_transition_matrix

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Subtracting the max for numerical stability
    return e_x / e_x.sum(axis=-1, keepdims=True)

def policy_objective_function(policy, env):
    gah = softmax(policy)
    if (gah < 0).any():
        breakpoint()
    return -env.calculate_initial_state_value(gah)

def transition_objective_function(transition, policy, env):

    env.transition_matrix = softmax(transition)
    return env.calculate_initial_state_value(policy)

def prox_transition_constraints_radius(transition, initial_transition, radius, shape):
    return radius - np.linalg.norm(softmax(transition).flatten() - initial_transition.flatten(), ord=shape) 
    
def find_simple_policy(args, env, transition_matrix, initial_policy=None):
    num_states, num_actions, grid_size = get_constants(env)
    env.transition = transition_matrix
    real_object_function = lambda pol: policy_objective_function(pol.reshape(num_states, num_actions), env)
    return find_minimizing_policy(env, real_object_function, initial_policy = initial_policy)

def find_oftpl_policy(args, env, transitions, initial_policy=None):
    num_states, num_actions, grid_size = get_constants(env)

    sample = np.random.exponential(scale=.01)
    def oftpl_transition_function(pol):
        val = 0
        

        # Repeat the sampled value 10 times to create a vector
        vector = np.repeat(sample, len(pol))
        val += np.dot(pol, vector)
        for transition in transitions:
            env.transition = transition
            val += policy_objective_function(pol.reshape(num_states, num_actions), env)
        return val
    return find_minimizing_policy(env, oftpl_transition_function, initial_policy = initial_policy)

def find_minimizing_policy(env, real_object_function, initial_policy=None):   
    num_states, num_actions, grid_size = get_constants(env)
    # Initial random policy (each action is equally probable)

    if initial_policy is None:
        initial_policy = generate_random_policy(num_states, num_actions).flatten()

    options = {"disp": True, "xtol": 1e-12,"maxiter":5000}
    slow_result = minimize(real_object_function, initial_policy.flatten(), method='Nelder-Mead', tol=1e-12, options=options)
    
    # Extract the optimized policy
    optimized_policy = softmax(slow_result.x.reshape((num_states, num_actions)))
    objective_value = -1 * slow_result.fun
    return optimized_policy, objective_value 

def get_constants(env):
    grid_size = int(np.sqrt(len(env.transition_matrix)))
    num_actions = env.transition_matrix.shape[1]


    num_states = grid_size**2
    return num_states, num_actions, grid_size

 
def find_best_response_transition(args, env, policy, initial_transition=None):
    num_states, num_actions, grid_size = get_constants(env)
    real_object_function = lambda trans: transition_objective_function(trans.reshape(num_states, num_actions, num_states), policy, env)
    return find_maximizing_transition(args, env, policy, real_object_function, initial_transition=initial_transition)


def find_maximizing_transition(args, env, policy, real_object_function, initial_transition=None):
    num_states, num_actions, grid_size = get_constants(env)
    if initial_transition is None:
        initial_transition = generate_valid_transition_matrix(grid_size, num_actions).flatten()

    transition_constraints_radius = lambda transition: prox_transition_constraints_radius(transition, initial_transition, args.radius, args.shape)
    transition_constraints_args = [{'type': 'ineq', 'fun': transition_constraints_radius}]

    options = {"disp": True,"xtol": 1e-6, "maxiter":1000}

    result = minimize(real_object_function, initial_transition.flatten(), method='trust-constr', tol=1e-12, constraints=transition_constraints_args, options=options)

    

    optimized_transition = softmax(result.x.reshape((num_states, num_actions, num_states)))
    objective_value = result.fun


    optimized_transition = optimized_transition/ optimized_transition.sum(axis=2, keepdims=True)
    return optimized_transition, objective_value 