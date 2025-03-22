"""
Here is one output of the code:
Number of threads: 16
    Episodes  Epsilon Decay  Alpha  Alpha Decay  Success Rate (%)
0        100           0.99   0.10         0.99               1.0
1        100           0.99   0.90         0.99              13.0
2        100           0.99   0.95         0.99              17.0
3        500           0.99   0.10         0.99               0.0
4        500           0.99   0.90         0.99              50.0
5        500           0.99   0.95         0.99              61.0
6       1000           0.99   0.10         0.99               0.0
7       1000           0.99   0.90         0.99              59.0
8       1000           0.99   0.95         0.99              63.0
9       2000           0.99   0.10         0.99               0.0
10      2000           0.99   0.90         0.99              60.0
11      2000           0.99   0.95         0.99              63.0

# Findings and Analysis:

1. Low Alpha (Learning Rate) Leads to Poor Convergence
When Alpha=0.1, the success rate is very low (0% in most cases), regardless of the number of episodes.

This suggests that a low learning rate causes Q-value updates to be too slow, preventing the algorithm from learning the optimal policy effectively.

2. Higher Alpha Improves Learning but Still Faces Limitations
When Alpha=0.9 or Alpha=0.95, the success rate significantly increases. For example:

Episodes=1000, Alpha=0.95 → Success rate = 63%

Episodes=500, Alpha=0.95 → Success rate = 61%, which is better than Alpha=0.9 at 50%.

However, the success rate never reaches 100%, indicating potential issues such as:

Epsilon Decay (0.99) is too slow, leading to excessive exploration that hinders convergence.

Learning rate (Alpha) is too high, causing instability in Q-value updates.

3. Increasing Training Episodes Does Not Always Improve Success Rate
As Episodes increases from 500 → 1000 → 2000, the success rate only improves slightly (50% → 59% → 60%).

This suggests a diminishing return from increasing training episodes, possibly due to:

Slow decay of exploration (Epsilon Decay = 0.99), causing suboptimal actions to persist longer.

Large Alpha values causing Q-values to oscillate, making it harder to stabilize the policy.he results indicate that increasing the number of episodes improves the success rate, but only when combined with appropriate learning rates and decay factors.


# How to Run the Code:
python p4.py
"""

import random
import os # for CPU count
from concurrent.futures import ThreadPoolExecutor # use multiprocessing to speed up the process
import pandas as pd 

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


def run_q_learning(episodes, epsilon_decay, alpha, alpha_decay):
    Q = {}
    epsilon_local, alpha_local = 1.0, alpha
    for _ in range(episodes):
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


def evaluate_policy(run_id, episodes, epsilon_decay, alpha, alpha_decay):
    return (
        run_q_learning(episodes, epsilon_decay, alpha, alpha_decay) == expected_policy
    )


def run_experiment(episodes_list, epsilon_decay_list, alpha_list, alpha_decay_list):
    results = {}
    for episodes in episodes_list:
        for epsilon_decay in epsilon_decay_list:
            for alpha in alpha_list:
                for alpha_decay in alpha_decay_list:
                    key = (episodes, epsilon_decay, alpha, alpha_decay)
                    with ThreadPoolExecutor(max_workers=num_threads) as executor:
                        results[key] = (
                            sum(
                                executor.map(
                                    lambda run: evaluate_policy(
                                        run, episodes, epsilon_decay, alpha, alpha_decay
                                    ),
                                    range(num_runs),
                                )
                            )
                            / num_runs
                        )
    return results


if __name__ == "__main__":
    # Define parameter ranges
    episodes_list = [100, 500, 1000, 2000]
    epsilon_decay_list = [0.99]
    alpha_list = [0.1, 0.9, 0.95]
    alpha_decay_list = [0.99]

    results = run_experiment(
        episodes_list, epsilon_decay_list, alpha_list, alpha_decay_list
    )

    # Convert results to a DataFrame for better visualization
    df_results = pd.DataFrame(
        [
            {
                "Episodes": episodes,
                "Epsilon Decay": epsilon_decay,
                "Alpha": alpha,
                "Alpha Decay": alpha_decay,
                "Success Rate (%)": success_rate * 100,
            }
            for (
                episodes,
                epsilon_decay,
                alpha,
                alpha_decay,
            ), success_rate in results.items()
        ]
    )

    print(df_results)
