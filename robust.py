 
from pgd import find_minimizing_policy, find_maximizing_transition, find_oftpl_policy, find_best_response_transition
from mdp import SmallGridworld, generate_valid_transition_matrix, generate_random_policy
from tqdm import tqdm

def simple_moving_average(data, window_size):
    """
    Calculate the simple moving average of a list of data points.
    
    Args:
    data (list): The input data to be smoothed.
    window_size (int): The size of the moving average window.
    
    Returns:
    list: A list of smoothed values.
    """
    smoothed_data = []
    for i in range(len(data)):
        if i < window_size - 1:
            smoothed_data.append(data[i])
        else:
            window = data[i - window_size + 1 : i + 1]
            smoothed_value = sum(window) / window_size
            smoothed_data.append(smoothed_value)
    return smoothed_data

    
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
    breakpoint()
    return vals
