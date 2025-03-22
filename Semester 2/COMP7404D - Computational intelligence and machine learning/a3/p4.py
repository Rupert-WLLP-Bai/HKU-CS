"""

"""

import random
from concurrent.futures import ThreadPoolExecutor
import os

# Num of threads
num_threads = os.cpu_count() or 4
print(f"Number of threads: {num_threads}")

# Define the Grid Environment
grid = [
    [0, 0, 0, +1],
    [0, "#", 0, -1],
    ["S", 0, 0, 0],
]

# constants defined in the original test case
discount = 1
noise = 0.1
living_reward = -0.01

episodes = 1000  # Number of training episodes
epsilon = 1.0  # Initial exploration probability
epsilon_decay = 0.99  # Decay factor for epsilon
alpha = 0.5  # Learning rate
alpha_decay = 0.99  # Decay factor for alpha
num_runs = 100  # Number of times to run the learning algorithm

rows, cols = 3, 4

actions = ["N", "S", "E", "W"]
moves = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}

expected_policy = [
    ["E", "E", "E", "x"],
    ["N", "#", "W", "x"],
    ["N", "W", "W", "S"],
]


def is_terminal(state):
    i, j = state
    return grid[i][j] in [1, -1]


def is_wall(state):
    i, j = state
    return grid[i][j] == "#"


def get_reward(state):
    i, j = state
    if grid[i][j] == 1:
        return 1.0
    elif grid[i][j] == -1:
        return -1.0
    else:
        return living_reward


def next_state(state, action):
    i, j = state
    di, dj = moves[action]
    ni, nj = i + di, j + dj

    if 0 <= ni < rows and 0 <= nj < cols and not is_wall((ni, nj)):
        return (ni, nj)
    return (i, j)


def apply_noise(action, noise):
    transition = {
        "N": [("N", 1 - 2 * noise), ("E", noise), ("W", noise)],
        "E": [("E", 1 - 2 * noise), ("S", noise), ("N", noise)],
        "S": [("S", 1 - 2 * noise), ("W", noise), ("E", noise)],
        "W": [("W", 1 - 2 * noise), ("N", noise), ("S", noise)],
    }
    return transition[action]


def choose_action(state, Q, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)
    return max(actions, key=lambda a: Q.get((state, a), 0))


def run_q_learning():
    Q = {}
    epsilon_local = epsilon
    alpha_local = alpha
    for ep in range(episodes):
        state = (2, 0)
        while not is_terminal(state):
            action = choose_action(state, Q, epsilon_local)
            real_action = random.choices(
                [a for a, _ in apply_noise(action, noise)],
                weights=[p for _, p in apply_noise(action, noise)],
            )[0]
            next_s = next_state(state, real_action)
            reward = get_reward(next_s)
            max_Q = max([Q.get((next_s, a), 0) for a in actions])
            sample = reward + discount * max_Q
            Q[(state, action)] = (1 - alpha_local) * Q.get(
                (state, action), 0
            ) + alpha_local * sample
            state = next_s
        epsilon_local *= epsilon_decay
        alpha_local *= alpha_decay
    policy = [["" for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if is_wall((i, j)):
                policy[i][j] = "#"
            elif is_terminal((i, j)):
                policy[i][j] = "x"
            else:
                s = (i, j)
                best_a = max(actions, key=lambda a: Q.get((s, a), 0))
                policy[i][j] = best_a
    return policy


def evaluate_policy(run_id):
    learned_policy = run_q_learning()
    output = []
    output.append(f"Run {run_id + 1}: Learned Policy")
    for row in learned_policy:
        output.append(str(row))
    match = learned_policy == expected_policy
    output.append(
        "✔ Optimal policy matched!" if match else "✘ Optimal policy did not match."
    )
    return match, "\n".join(output)


if __name__ == "__main__":
    success_count = 0
    output_policy = False
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = list(executor.map(evaluate_policy, range(num_runs)))

    for match, output in futures:
        if output_policy:
            print(output)
        if match:
            success_count += 1

    print(f"Optimal policy found {success_count}/{num_runs} times")
