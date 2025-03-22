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


def policy_evaluation(problem):
    """Performs policy evaluation for a given MDP problem."""
    grid = problem["grid"]
    policy = problem["policy"]
    discount = problem["discount"]
    noise = problem["noise"]
    living_reward = problem["livingReward"]
    iterations = problem["iterations"]

    rows, cols = len(grid), len(grid[0])
    V = np.zeros((rows, cols))
    return_value = ""

    # Function to format and print values correctly
    def format_grid(V, grid):
        return_value = ""
        for i in range(len(V)):
            for j in range(len(V[0])):
                if grid[i][j] == "#":
                    return_value += "| ##### |"
                else:
                    return_value += "|{:7.2f}|".format(V[i][j])
            return_value += "\n"
        return return_value

    def get_expected_value(i, j):
        """Compute the expected value from state (i, j)."""
        if policy[i][j] == "exit":
            return float(grid[i][j])  # Terminal state

        action = policy[i][j]
        moves = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
        expected_value = 0.0

        for next_action, prob in apply_noise(action, noise):
            di, dj = moves[next_action]
            ni, nj = max(0, min(i + di, rows - 1)), max(0, min(j + dj, cols - 1))

            if grid[ni][nj] == "#":  # Hitting a wall, stay in place
                ni, nj = i, j

            expected_value += prob * V[ni][nj]

        return expected_value

    # Step 1: Initialize V to all zeros (first output)
    V = np.zeros((rows, cols))
    return_value += "V^pi_k=0\n"
    return_value += format_grid(V, grid)

    # Step 2: Initialize terminal states and non-terminal states
    for i in range(rows):
        for j in range(cols):
            if policy[i][j] == "exit":
                V[i][j] = float(grid[i][j])  # Set terminal values
            elif grid[i][j] != "#":
                V[i][j] = living_reward  # Non-terminal states get living reward

    return_value += "V^pi_k=1\n"
    return_value += format_grid(V, grid)

    # Step 3: Perform actual policy evaluation (from k=2 onward)
    for k in range(2, iterations):
        new_V = np.copy(V)
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == "#" or policy[i][j] == "exit":
                    continue  # Skip walls and terminal states

                new_V[i][j] = living_reward + discount * get_expected_value(i, j)

        V = new_V

        return_value += f"V^pi_k={k}\n"
        return_value += format_grid(V, grid)

    # fix: remove the last newline character
    return_value = return_value[:-1]

    return return_value


if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 2
    grader.grade(
        problem_id, test_case_id, policy_evaluation, parse.read_grid_mdp_problem_p2
    )
