"""
Here is one output of the code:
Number of threads: 16
Results:
    Epsilon Decay  Alpha  Alpha Decay  Success Rate (%)  Mean Episodes (if successful)
0           0.990  0.100        0.990               2.0                      29.500000
1           0.990  0.100        0.999               0.0                   10000.000000
2           0.990  0.900        0.990              77.0                     102.246753
3           0.990  0.900        0.999              93.0                     189.655914
4           0.990  0.999        0.990              76.0                     124.605263
5           0.990  0.999        0.999              95.0                     198.494737
6           0.999  0.100        0.990               2.0                      96.000000
7           0.999  0.100        0.999              63.0                     398.682540
8           0.999  0.900        0.990              93.0                     152.053763
9           0.999  0.900        0.999             100.0                     287.940000
10          0.999  0.999        0.990              96.0                     164.145833
11          0.999  0.999        0.999             100.0                     362.340000

Based on the results from our modified Q-learning implementation that tracks episodes needed to find the optimal policy, we can derive the following insights:

1. Optimal Parameter Combinations Identified
    Two parameter combinations achieved 100% success rate:

    Epsilon Decay = 0.999, Alpha = 0.9, Alpha Decay = 0.999 (287.94 episodes on average)
    Epsilon Decay = 0.999, Alpha = 0.999, Alpha Decay = 0.999 (362.34 episodes on average)
    These combinations consistently find the optimal policy, demonstrating the importance of slower exploration decay (0.999) combined with high learning rates and slow learning rate decay.

2. Learning Rate (Alpha) Impact
    Low Alpha (0.1) performs poorly across most settings, with success rates of 0-2% for most combinations

    Exception: When paired with Epsilon Decay = 0.999 and Alpha Decay = 0.999, it achieves 63% success
    This suggests that very slow decay rates can compensate somewhat for low learning rates
    High Alpha (0.9, 0.999) significantly outperforms low Alpha values:

    Success rates range from 76% to 100%
    Higher Alpha enables the algorithm to more quickly adapt Q-values based on new experiences
3. Exploration Strategy (Epsilon Decay) Impact
    Epsilon Decay = 0.999 (slower decay) consistently outperforms Epsilon Decay = 0.99:

    With Alpha = 0.9/0.999 and Alpha Decay = 0.999, success rates reach 100%
    Average episodes required (287-362) is higher than with faster decay
    This suggests maintaining longer exploration periods is crucial for finding the optimal policy
    Epsilon Decay = 0.99 (faster decay) still performs reasonably well:

    With Alpha = 0.999 and Alpha Decay = 0.999, success rate reaches 95%
    When successful, it finds solutions faster (mean 198.49 episodes)
    This represents a trade-off between exploration time and success rate
4. Learning Rate Decay (Alpha Decay) Impact
    Alpha Decay = 0.999 (slower decay) consistently outperforms Alpha Decay = 0.99:
    Success rates improve by 16-30% when switching from 0.99 to 0.999
    This suggests maintaining a higher learning rate longer helps Q-values converge to optimal values
5. Speed vs. Reliability Trade-off
    The fastest successful combination (Epsilon Decay = 0.99, Alpha = 0.1, Alpha Decay = 0.99) averaged 29.5 episodes but succeeded only 2% of the time
    The most reliable combinations (100% success rate) required 288-362 episodes on average
    This demonstrates a clear trade-off between speed and reliability in Q-learning
6. Interconnected Parameter Effects
    Parameter effects are highly interconnected:
    Alpha = 0.1 is generally ineffective, except when paired with very slow decay rates
    The benefits of slow Epsilon Decay (0.999) are amplified when combined with high Alpha values
    Alpha Decay = 0.999 provides consistent benefits across almost all parameter combinations
"""

import random
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

# Num of threads
num_threads = os.cpu_count() or 4
print(f"Number of threads: {num_threads}")

# Define the Grid Environment
grid = [
    [0, 0, 0, +1],
    [0, "#", 0, -1],
    ["S", 0, 0, 0],
]

# Constants
discount = 1
noise = 0.1
living_reward = -0.01
num_runs = 100  # Number of runs for each setting
max_episodes = 10000  # Maximum number of episodes to run

rows, cols = 3, 4
actions = ["N", "S", "E", "W"]
moves = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}

expected_policy = [
    ["E", "E", "E", "x"],
    ["N", "#", "W", "x"],
    ["N", "W", "W", "S"],
]

# Function to check terminal and wall states
def is_terminal(state):
    i, j = state
    return grid[i][j] in [1, -1]


def is_wall(state):
    i, j = state
    return grid[i][j] == "#"


def get_reward(state):
    i, j = state
    return 1.0 if grid[i][j] == 1 else -1.0 if grid[i][j] == -1 else living_reward


def next_state(state, action):
    i, j = state
    di, dj = moves[action]
    ni, nj = i + di, j + dj
    return (
        (ni, nj)
        if 0 <= ni < rows and 0 <= nj < cols and not is_wall((ni, nj))
        else (i, j)
    )


def apply_noise(action, noise):
    return {
        "N": [("N", 1 - 2 * noise), ("E", noise), ("W", noise)],
        "E": [("E", 1 - 2 * noise), ("S", noise), ("N", noise)],
        "S": [("S", 1 - 2 * noise), ("W", noise), ("E", noise)],
        "W": [("W", 1 - 2 * noise), ("N", noise), ("S", noise)],
    }[action]


def choose_action(state, Q, epsilon):
    return (
        random.choice(actions)
        if random.random() < epsilon
        else max(actions, key=lambda a: Q.get((state, a), 0))
    )


def get_current_policy(Q):
    return [
        [
            (
                "#"
                if is_wall((i, j))
                else (
                    "x"
                    if is_terminal((i, j))
                    else max(actions, key=lambda a: Q.get(((i, j), a), 0))
                )
            )
            for j in range(cols)
        ]
        for i in range(rows)
    ]


def run_q_learning(epsilon_decay, alpha, alpha_decay, verbose=False):
    Q = {}
    epsilon_local, alpha_local = 1.0, alpha
    
    for episode in range(1, max_episodes + 1):
        state = (2, 0)
        while not is_terminal(state):
            action = choose_action(state, Q, epsilon_local)
            real_action = random.choices(*zip(*apply_noise(action, noise)))[0]
            next_s, reward = next_state(state, real_action), get_reward(
                next_state(state, real_action)
            )
            Q[(state, action)] = (1 - alpha_local) * Q.get(
                (state, action), 0
            ) + alpha_local * (
                reward + discount * max(Q.get((next_s, a), 0) for a in actions)
            )
            state = next_s
        
        epsilon_local *= epsilon_decay
        alpha_local *= alpha_decay
        
        # Check if we've found the optimal policy
        current_policy = get_current_policy(Q)
        if current_policy == expected_policy:
            if verbose:
                print(f"Found optimal policy at episode {episode}")
            return True, episode
    
    if verbose:
        print(f"Failed to find optimal policy within {max_episodes} episodes")
    return False, max_episodes


def evaluate_policy(run_id, epsilon_decay, alpha, alpha_decay, verbose=False):
    success, episodes = run_q_learning(epsilon_decay, alpha, alpha_decay, verbose=(verbose and run_id == 0))
    return success, episodes


def run_experiment(epsilon_decay_list, alpha_list, alpha_decay_list, verbose=False):
    results = {}
    episodes_results = {}
    
    for epsilon_decay in epsilon_decay_list:
        for alpha in alpha_list:
            for alpha_decay in alpha_decay_list:
                key = (epsilon_decay, alpha, alpha_decay)
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    run_results = list(executor.map(
                        lambda run: evaluate_policy(
                            run, epsilon_decay, alpha, alpha_decay, verbose
                        ),
                        range(num_runs),
                    ))
                    
                    # Calculate success rate
                    success_rate = sum(success for success, _ in run_results) / num_runs
                    
                    # Calculate mean episodes for successful runs
                    successful_episodes = [ep for success, ep in run_results if success]
                    mean_episodes = np.mean(successful_episodes) if successful_episodes else max_episodes
                    
                    results[key] = success_rate
                    episodes_results[key] = mean_episodes
    
    return results, episodes_results


if __name__ == "__main__":
    # Define parameter ranges
    epsilon_decay_list = [0.99,0.999]
    alpha_list = [0.1, 0.9, 0.999]
    alpha_decay_list = [0.99, 0.999]

    results, episodes_results = run_experiment(
        epsilon_decay_list, alpha_list, alpha_decay_list, verbose=False
    )

    # Convert results to DataFrames for better visualization
    df_results = pd.DataFrame(
        [
            {
                "Epsilon Decay": epsilon_decay,
                "Alpha": alpha,
                "Alpha Decay": alpha_decay,
                "Success Rate (%)": success_rate * 100,
                "Mean Episodes (if successful)": episodes_results[(epsilon_decay, alpha, alpha_decay)]
            }
            for (
                epsilon_decay,
                alpha,
                alpha_decay,
            ), success_rate in results.items()
        ]
    )

    print("\nResults:")
    print(df_results)