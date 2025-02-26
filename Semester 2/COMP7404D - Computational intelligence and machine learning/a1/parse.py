import os, sys
def read_graph_search_problem(file_path):
    """Reads a graph search problem from a file, including state values."""
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Parse the file
    start_state = lines[0].split()[1]  # Start state
    goal_states = lines[1].split()[1]  # Goal state(s)

    # Initialize graph dictionary and state values
    graph = {}
    state_values = {}  # Dictionary to store state values (e.g., S: 4)
    
    # Read the states and edges
    for line in lines[2:]:
        parts = line.split()
        
        if len(parts) == 2:  # State definition (e.g., S 4)
            state = parts[0]
            value = float(parts[1])  # Store the associated value
            state_values[state] = value  # Save state and its value
            graph[state] = {}  # Initialize an empty dictionary for neighbors of this state
            
        elif len(parts) == 3:  # Edge definition (e.g., S B 1.5)
            state1, state2, cost = parts
            cost = float(cost)
            if state1 not in graph:
                graph[state1] = {}
            graph[state1][state2] = cost

    # Return the problem as a dictionary
    problem = {
        'start_state': start_state,
        'goal_states': goal_states,
        'graph': graph,
        'state_values': state_values  # Return the state values as part of the problem
    }
    return problem


def read_8queens_search_problem(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    queens_positions = []

    # Parse the grid and extract the queen positions (where 'q' is located)
    for row_index, line in enumerate(lines):
        for col_index, cell in enumerate(line.strip()):
            if cell == 'q':
                queens_positions.append((row_index, col_index))
    
    return queens_positions  # Return queen positions and grid size (n)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        problem_id, test_case_id = sys.argv[1], sys.argv[2]
        if int(problem_id) <= 5:
            problem = read_graph_search_problem(os.path.join('test_cases','p'+problem_id, test_case_id+'.prob'))
        else:
            problem = read_8queens_search_problem(os.path.join('test_cases','p'+problem_id, test_case_id+'.prob'))
        print(problem)
    else:
        print('Error: I need exactly 2 arguments!')