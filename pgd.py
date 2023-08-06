import numpy as np
from scipy.optimize import minimize
from mdp import generate_random_policy, generate_valid_transition_matrix

def policy_objective_function(policy, env):
    return -env.calculate_initial_state_value(policy)

def transition_objective_function(transition, policy, env):
    env.transition_matrix = transition
    return env.calculate_initial_state_value(policy)

def prox_policy_constraints_sum(policy, num_states, num_actions):
    return np.ones(num_states) - np.sum(policy.reshape(num_states, num_actions), axis=1)

def policy_constraints_nonnegative(policy):
    return policy

def transition_constraints_nonnegative(transition):
    return transition

def prox_transition_constraints_sum(transition, num_states, num_actions):
    return (np.ones((num_states, num_actions)) - np.sum(transition.reshape(num_states, num_actions, num_states), axis=2)).flatten()


def find_simple_policy(env, transition_matrix, real_object_function, initial_policy=None):
    num_states, num_actions, grid_size = get_constants(env)
    real_object_function = lambda pol: policy_objective_function(pol.reshape(num_states, num_actions), env)
    return find_minimizing_policy(env, transition_matrix, real_object_function, initial_policy = initial_policy)

def find_minimizing_policy(env, transition_matrix, real_object_function, initial_policy=None):
    env.transition_matrix = transition_matrix
    
    num_states, num_actions, grid_size = get_constants(env)
    # Initial random policy (each action is equally probable)

    if initial_policy is None:
        initial_policy = generate_random_policy(num_states, num_actions).flatten()

    # Define constraints for the policy optimization
    policy_constraints_sum = lambda pol: prox_policy_constraints_sum(pol, num_states, num_actions)
    policy_constraints_args = [{'type': 'eq', 'fun': policy_constraints_sum} ,{'type': 'ineq', 'fun': policy_constraints_nonnegative}]
    
    # Optimize the policy using BFGS

    options = {"verbose": 1, "gtol":1e-5}
    result = minimize(real_object_function, initial_policy, method='trust-constr', constraints=policy_constraints_args, options=options)

    # Extract the optimized policy
    optimized_policy = result.x.reshape((num_states, num_actions))
    objective_value = -1 * result.fun
    return optimized_policy, objective_value 

def get_constants(env):
    grid_size = int(np.sqrt(env.observation_space.n))
    num_actions = env.action_space.n

    num_states, num_actions = grid_size**2, 4
    return num_states, num_actions, grid_size


def find_best_response_transition(env, policy, initial_transition=None):
    num_states, num_actions, grid_size = get_constants(env)
    real_object_function = lambda trans: transition_objective_function(trans.reshape(num_states, num_actions, num_states), policy, env)
    return find_maximizing_transition(env, policy, real_object_function, initial_transition=initial_transition)


def find_maximizing_transition(env, policy, real_object_function, initial_transition=None):
    num_states, num_actions, grid_size = get_constants(env)
    if initial_transition is None:
        initial_transition = generate_valid_transition_matrix(grid_size, num_actions).flatten()
    transition_constraints_sum = lambda transition: prox_transition_constraints_sum(transition, num_states, num_actions)
    transition_constraints_args = [{'type': 'eq', 'fun': transition_constraints_sum} ,{'type': 'ineq', 'fun': transition_constraints_nonnegative}]

    options = {"verbose": 1, "gtol":1e-5}
    result = minimize(real_object_function, initial_transition, method='trust-constr', constraints=transition_constraints_args, options=options)

    # Extract the optimized policy
    optimized_transition = result.x.reshape((num_states, num_actions, num_states))
    objective_value = result.fun

    return optimized_transition, objective_value 