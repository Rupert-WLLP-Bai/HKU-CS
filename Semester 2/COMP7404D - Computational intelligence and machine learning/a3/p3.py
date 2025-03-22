import sys, grader, parse

import numpy as np

def apply_noise(action, noise):
    """Returns the possible stochastic actions due to noise."""
    transition = {
        "N": [("N", 1 - 2 * noise), ("E", noise), ("W", noise)],
        "E": [("E", 1 - 2 * noise), ("S", noise), ("N", noise)],
        "S": [("S", 1 - 2 * noise), ("W", noise), ("E", noise)],
        "W": [("W", 1 - 2 * noise), ("N", noise), ("S", noise)],
    }
    return transition[action]

def value_iteration(problem):
    """Performs value iteration for a given MDP problem."""
    grid = problem["grid"]
    discount = problem["discount"]
    noise = problem["noise"]
    living_reward = problem["livingReward"]
    iterations = problem["iterations"]
    
    rows, cols = len(grid), len(grid[0])
    V = np.zeros((rows, cols))
    policy = [["" for _ in range(cols)] for _ in range(rows)]
    return_value = ""

    def format_grid(V, grid):
        result = ""
        for i in range(len(V)):
            for j in range(len(V[0])):
                if grid[i][j] == "#":
                    result += "| ##### |"
                else:
                    result += "|{:7.2f}|".format(V[i][j])
            result += "\n"
        return result
    
    def format_policy(policy, grid):
        result = ""
        for i in range(len(policy)):
            for j in range(len(policy[0])):
                if grid[i][j] == "#":
                    result += "| # |"
                elif policy[i][j] == "exit":
                    result += "| x |"
                else:
                    result += f"| {policy[i][j]} |"
            result += "\n"
        return result
    
    moves = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
    actions = ["N", "E", "S", "W"]
    
    return_value += "V_k=0\n" + format_grid(V, grid)
    
    for k in range(1, iterations):
        new_V = np.copy(V)
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == "#":
                    continue  # Skip walls
                if isinstance(grid[i][j], (int, float)) or str(grid[i][j]).lstrip('-').isdigit():
                    new_V[i][j] = float(grid[i][j])  # Terminal state
                    policy[i][j] = "exit"
                    continue
                
                best_value = float('-inf')
                best_action = "N"
                
                for action in actions:
                    expected_value = 0.0
                    for next_action, prob in apply_noise(action, noise):
                        di, dj = moves[next_action]
                        ni, nj = max(0, min(i + di, rows - 1)), max(0, min(j + dj, cols - 1))
                        if grid[ni][nj] == "#":
                            ni, nj = i, j  # Stay in place if hitting a wall
                        expected_value += prob * V[ni][nj]

                    q_value = living_reward + discount * expected_value
                    # print(f"State ({i},{j}) - Action {action}: Q-Value = {q_value}")

                    if q_value > best_value:
                        best_value = q_value
                        best_action = action

                
                new_V[i][j] = best_value
                policy[i][j] = best_action
        
        V = new_V
        return_value += f"V_k={k}\n" + format_grid(V, grid)
        return_value += f"pi_k={k}\n" + format_policy(policy, grid)
    
    return return_value.strip()


if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    # test_case_id = -4
    problem_id = 3
    grader.grade(
        problem_id, test_case_id, value_iteration, parse.read_grid_mdp_problem_p3
    )
