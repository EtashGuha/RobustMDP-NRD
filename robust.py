 
from pgd import find_minimizing_policy, find_maximizing_transition, find_oftpl_policy, find_best_response_transition
from mdp import SmallGridworld, generate_valid_transition_matrix, generate_random_policy
from tqdm import tqdm


def oftpl_exp(args):
    grid_size = 3
    num_actions = 4

    num_states = grid_size ** 2
    transition_matrix = generate_valid_transition_matrix(grid_size, num_actions)
    env = SmallGridworld(transition_matrix)

    policy = generate_random_policy(num_states, num_actions)
    transition_matrix, obj_val = find_best_response_transition(args, env, policy, initial_transition=transition_matrix)
    T = 60
    transitions = []
    transitions.append(transition_matrix)
    vals = []
    vals.append(obj_val)
    policy_obj_vals = []
    for i in tqdm(range(T)):
        policy, obj_val = find_oftpl_policy(args, env, transitions, initial_policy=policy)
        policy_obj_vals.append(obj_val)
        transition_matrix, obj_val = find_best_response_transition(args, env, policy, initial_transition=transition_matrix)
        vals.append(obj_val)
    return vals
