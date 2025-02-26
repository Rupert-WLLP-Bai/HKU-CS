import sys, parse, grader

import heapq

def astar_search(problem):
    start_state = problem['start_state']
    goal_states = problem['goal_states']
    graph = problem['graph']
    heuristic = problem['state_values']  # A dictionary of heuristic values for each state
    
    # Initialize open list (priority queue), closed list, and parent mapping
    open_list = []
    heapq.heappush(open_list, (0 + heuristic[start_state], start_state))  # Priority queue with f = g + h
    came_from = {}  # To reconstruct the path
    g_cost = {start_state: 0}  # g(x): Cost to reach the current state
    f_cost = {start_state: heuristic[start_state]}  # f(x) = g(x) + h(x)
    explored = []
    path = []
    goal_state = None

    while open_list:
        # Get the state with the lowest f(x) value from the priority queue
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
            tentative_g_cost = g_cost[current_state] + graph[current_state][neighbor]
            
            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                # Update the g cost and f cost for the neighbor
                g_cost[neighbor] = tentative_g_cost
                f_cost[neighbor] = tentative_g_cost + heuristic[neighbor]
                came_from[neighbor] = current_state
                
                # Add the neighbor to the priority queue with its new f cost
                heapq.heappush(open_list, (f_cost[neighbor], neighbor))
    
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
    problem_id = 5
    grader.grade(problem_id, test_case_id, astar_search, parse.read_graph_search_problem)