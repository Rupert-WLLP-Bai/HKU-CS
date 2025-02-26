import sys, parse, grader

import heapq

def greedy_search(problem):
    start_state = problem['start_state']
    goal_states = problem['goal_states']
    graph = problem['graph']
    heuristic = problem['state_values']  # Heuristic for each state
    
    # Initialize open list (priority queue) and closed list
    open_list = []
    heapq.heappush(open_list, (heuristic[start_state], start_state))  # Priority queue with f(x) = h(x)
    came_from = {}  # To reconstruct the path
    explored = []  # List to store explored nodes
    path = []  # List to store the path
    goal_state = None

    while open_list:
        # Get the state with the lowest heuristic value (Greedy approach)
        _, current_state = heapq.heappop(open_list)
        
        if current_state in explored:
            continue
        
        explored.append(current_state)
        
        # Check if we have found a goal state
        if current_state in goal_states:
            goal_state = current_state
            break
        
        # Explore neighbors of the current state
        for neighbor in graph[current_state]:
            if neighbor not in explored:
                came_from[neighbor] = current_state
                heapq.heappush(open_list, (heuristic[neighbor], neighbor))
    
    # Reconstruct the path if a goal state is found
    if goal_state:
        while goal_state in came_from:
            path.append(goal_state)
            goal_state = came_from[goal_state]
        path.append(start_state)
        path.reverse()
        # Remove the goal state from the explored list
        explored.remove(path[-1])
    
    # Return the explored states and the path
    return f'{" ".join(explored)}\n{" ".join(path)}' if path else f'{" ".join(explored)}\n'


if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 4
    grader.grade(problem_id, test_case_id, greedy_search, parse.read_graph_search_problem)