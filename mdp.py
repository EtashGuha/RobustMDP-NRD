import gym
from gym import spaces
import numpy as np

class SmallGridworld(gym.Env):
    def __init__(self, transition_matrix):
        # Define the gridworld dimensions
        self.grid_size = 3

        # Set up the action space and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.grid_size ** 2)

        # Define the possible actions
        self.actions = {
            0: 'up',
            1: 'right',
            2: 'down',
            3: 'left'
        }

        # Store the transition matrix
        self.transition_matrix = transition_matrix

        # Define the gridworld layout and rewards
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.goal_pos = (2, 2)
        self.goal_reward = 10
        self.step_reward = -1

        # Set the initial state to the top-left corner
        self.current_state = (0, 0)

    def reset(self):
        # Reset the environment to the initial state
        self.current_state = (0, 0)
        return self._state_to_observation(self.current_state)

    def step(self, action):
        # Perform the specified action and return the next state, reward, done, and additional info
        next_state_probs = self.transition_matrix[self._state_to_observation(self.current_state), action]
        next_state = np.random.choice(np.arange(len(next_state_probs)), p=next_state_probs)

        reward = self._get_reward(next_state)
        done = self._is_terminal(next_state)

        self.current_state = self._observation_to_state(next_state)
        return self._state_to_observation(self.current_state), reward, done, {}

    def _get_reward(self, state):
        # Return the reward for the given state
        if state == self._state_to_observation(self.goal_pos):
            return self.goal_reward
        return self.step_reward

    def _is_terminal(self, state):
        # Check if the given state is the terminal state
        return state == self._state_to_observation(self.goal_pos)

    def _state_to_observation(self, state):
        # Convert the state to a scalar observation (state number)
        x, y = state
        return x + y * self.grid_size

    def _observation_to_state(self, observation):
        # Convert the observation (state number) to a state tuple
        x = observation % self.grid_size
        y = observation // self.grid_size
        return x, y

    def render(self, mode='human'):
        # Print the current state of the gridworld
        grid = np.copy(self.grid)
        x, y = self.current_state
        grid[y, x] = 1
        print(grid)

    def calculate_initial_state_value(self, policy, gamma=0.9, tol=1e-6, max_iter=1000):
        num_states, num_actions = self.transition_matrix.shape[0], self.transition_matrix.shape[1]
        value_function = np.zeros(num_states)

        initial_state_observation = self._state_to_observation(self.current_state)
        
        for i in range(max_iter):
            prev_value_function = np.copy(value_function)

            for state in range(num_states):
                next_state_values = np.zeros(num_actions)
                for action in range(num_actions):
                    for next_state in range(num_states):
                        prob = self.transition_matrix[state, action, next_state] * policy[state, action]
                        reward = self._get_reward(next_state)
                        next_state_values[action] += prob * (reward + gamma * value_function[next_state])

                value_function[initial_state_observation] = self._get_reward(state) + gamma * np.dot(policy[state], next_state_values)

            if np.abs(value_function[initial_state_observation] - prev_value_function[initial_state_observation]) < tol:
                break
            
        return value_function[initial_state_observation]

        
def generate_random_policy(num_states, num_actions):
    policy = np.random.rand(num_states, num_actions)
    policy /= policy.sum(axis=1, keepdims=True)  # Normalize the probabilities to sum up to 1 for each state
    return policy

def generate_valid_transition_matrix(grid_size, num_actions):
    num_states = grid_size * grid_size

    # Generate a random transition matrix with probabilities between 0 and 1
    transition_matrix = np.random.rand(num_states, num_actions, num_states)
    transition_matrix = transition_matrix/ transition_matrix.sum(axis=2, keepdims=True)

    return transition_matrix


if __name__ == '__main__':
    # Create a stochastic transition matrix for the gridworld
    grid_size = 3
    num_states = grid_size * grid_size
    num_actions = 4

    # Generate a random transition matrix with stochastic transitions
    np.random.seed(42)
    transition_matrix = generate_valid_transition_matrix(num_states, num_actions, num_states)

    env = SmallGridworld(transition_matrix)

    # Perform a random walk in the environment
    env.reset()
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        env.render()

    print("Final state:", env.current_state)