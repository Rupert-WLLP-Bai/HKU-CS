import sys, parse, grader

import heapq

def ucs_search(problem):
    start_state = problem['start_state']
    goal_states = problem['goal_states']
    graph = problem['graph']
    
    # Initialize open list (priority queue), closed list, and other helpers
    open_list = []
    heapq.heappush(open_list, (0, start_state))  # Priority queue with (cost, state)
    came_from = {}  # To reconstruct the path
    g_cost = {start_state: 0}  # g(x): Cost to reach the current state
    explored = []  # List to store explored nodes
    path = []  # List to store the path
    goal_state = None

    while open_list:
        # Get the state with the lowest cost from the priority queue
        current_cost, current_state = heapq.heappop(open_list)
        
        if current_state in explored:
            continue
        
        explored.append(current_state)
        
        # Check if we have found a goal state
        if current_state in goal_states:
            goal_state = current_state
            break
        
        # Explore neighbors of the current state
        for neighbor, cost in graph[current_state].items():
            tentative_cost = current_cost + cost
            
            # If the neighbor has not been explored or found a cheaper path
            if neighbor not in g_cost or tentative_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_cost
                came_from[neighbor] = current_state
                heapq.heappush(open_list, (tentative_cost, neighbor))
    
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
    problem_id = 3
    grader.grade(problem_id, test_case_id, ucs_search, parse.read_graph_search_problem)