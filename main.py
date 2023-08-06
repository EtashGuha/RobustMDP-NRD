from mdp import SmallGridworld, generate_valid_transition_matrix, generate_random_policy
import numpy as np
from pgd import find_minimizing_policy, find_maximizing_transition

if __name__ == '__main__':
    # Create a stochastic transition matrix for the gridworld
    grid_size = 3
    num_states = grid_size * grid_size
    num_actions = 4

    # Generate a random transition matrix with stochastic transitions
    np.random.seed(42)
    transition_matrix = generate_valid_transition_matrix(grid_size, num_actions)
    policy = generate_random_policy(num_states, num_actions)


    env = SmallGridworld(transition_matrix)

    policy, obj_val = find_minimizing_policy(env, transition_matrix)
    transition_matrix, obj_val = find_maximizing_transition(env, policy, initial_transition=transition_matrix)
    breakpoint()