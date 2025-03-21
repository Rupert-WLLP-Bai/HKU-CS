import sys, grader, parse

import random


def find_start(grid):
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == "S":
                return (i, j)
    return None


def get_intended_action(policy, state):
    i, j = state
    return policy[i][j]


def apply_noise(action, noise):
    d = {
        "N": ["N", "E", "W"],
        "E": ["E", "S", "N"],
        "S": ["S", "W", "E"],
        "W": ["W", "N", "S"],
    }
    return random.choices(population=d[action], weights=[1 - noise * 2, noise, noise])[
        0
    ]


def get_next_state(grid, state, action):
    i, j = state
    new_i, new_j = i, j
    if action == "N":
        new_i = max(i - 1, 0)
    elif action == "S":
        new_i = min(i + 1, len(grid) - 1)
    elif action == "E":
        new_j = min(j + 1, len(grid[0]) - 1)
    elif action == "W":
        new_j = max(j - 1, 0)

    if grid[new_i][new_j] == "#":
        return state
    return (new_i, new_j)


def play_episode(problem):
    grid = problem["grid"]
    policy = problem["policy"]
    noise = problem["noise"]
    living_reward = problem["livingReward"]

    state = find_start(grid)
    cumulative_reward = 0.0
    experience = "Start state:\n"

    for row in grid:
        experience += "    " + "    ".join(row) + "\n"
    experience += f"Cumulative reward sum: {cumulative_reward}\n"
    experience += "-" * 44 + "\n"

    while True:
        intended_action = get_intended_action(policy, state)
        actual_action = apply_noise(intended_action, noise)
        next_state = get_next_state(grid, state, actual_action)

        reward = living_reward
        i, j = next_state
        if grid[i][j] == "1":
            reward = 1.0
        elif grid[i][j] == "-1":
            reward = -1.0

        cumulative_reward += reward

        experience += f"Taking action: {actual_action} (intended: {intended_action})\n"
        experience += f"Reward received: {reward}\n"
        experience += "New state:\n"

        new_grid = [row.copy() for row in grid]
        new_grid[i][j] = "P"
        for row in new_grid:
            experience += "    " + "    ".join(row) + "\n"

        experience += f"Cumulative reward sum: {cumulative_reward}\n"
        experience += "-" * 44 + "\n"

        if grid[i][j] in ["1", "-1"]:
            break

        state = next_state

    return experience


if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    # test_case_id = 1
    problem_id = 1
    grader.grade(problem_id, test_case_id, play_episode, parse.read_grid_mdp_problem_p1)
