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
    seed = problem["seed"]
    
    random.seed(seed)

    state = find_start(grid)
    
    # replace the 'S' with 'P'
    start_i, start_j = state
    grid[start_i][start_j] = "P"
    
    cumulative_reward = 0.0
    experience = "Start state:\n"

    for row in grid:
        experience += "".join(f"{cell:>5}" for cell in row) + "\n"
    experience += f"Cumulative reward sum: {cumulative_reward}\n"
    experience += "-" * 44 + " \n"

    while True:
        intended_action = get_intended_action(policy, state)
        if intended_action == "exit":
            actual_action = "exit"
            break
        
        actual_action = apply_noise(intended_action, noise)
        next_state = get_next_state(grid, state, actual_action)

        reward = living_reward
        cumulative_reward = round(cumulative_reward + reward, 2)

        experience += f"Taking action: {actual_action} (intended: {intended_action})\n"
        experience += f"Reward received: {reward}\n"
        experience += "New state:\n"

        # Reset the previous position of 'P' to '_'
        prev_i, prev_j = state
        grid[prev_i][prev_j] = "_"  # Clear the previous position

        new_grid = [row.copy() for row in grid]
        i, j = next_state
        new_grid[i][j] = "P"
        
        # print(f"i: {i}, j: {j}, start_i: {start_i}, start_j: {start_j}")
        
        # If i,j is not the start state, put 'S' back
        if i != start_i or j != start_j:
            new_grid[start_i][start_j] = "S"
        
        for row in new_grid:
            experience += "".join(f"{cell:>5}" for cell in row) + "\n"

        experience += f"Cumulative reward sum: {cumulative_reward}\n"
        experience += "-" * 44 + " \n"

        if grid[i][j] in ["1", "-1"]:
            # Handle exit logic
            exit_reward = 1.0 if grid[i][j] == "1" else -1.0
            cumulative_reward = round(cumulative_reward + exit_reward, 2)

            experience += f"Taking action: exit (intended: exit)\n"
            experience += f"Reward received: {exit_reward}\n"
            experience += "New state:\n"

            final_grid = [row.copy() for row in grid]
            final_grid[i][j] = grid[i][j]
            
            # Put 'S' back
            final_grid[start_i][start_j] = "S"
            
            for row in final_grid:
                experience += "".join(f"{cell:>5}" for cell in row) + "\n"

            experience += f"Cumulative reward sum: {cumulative_reward}"
            break

        state = next_state

    return experience


if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    # test_case_id = 1
    problem_id = 1
    grader.grade(problem_id, test_case_id, play_episode, parse.read_grid_mdp_problem_p1)