import sys, grader, parse

from collections import deque

def bfs_search(problem):
    start_state = problem['start_state']
    goal_states = problem['goal_states']
    graph = problem['graph']
    
    # BFS-GSA
    # Returns: f'{" ".join(explored)}\n{" ".join(path)}'
    
    # Initialize explored list, frontier queue, and path list
    explored = []
    frontier = deque([start_state])  # Use deque as queue (FIFO)
    path = []
    parent = {start_state: None}
    goal_state = None

    while frontier:
        state = frontier.popleft()  # Pop from the queue (FIFO)
        
        if state not in explored:
            explored.append(state)
            
            # Check if we have found a goal state
            if state in goal_states:
                goal_state = state
                break

            # Add neighbors to the frontier (queue) for exploration
            for neighbor in graph[state]:
                if neighbor not in explored and neighbor not in frontier:
                    frontier.append(neighbor)
                    parent[neighbor] = state
    
    # If a goal state is found, reconstruct the path
    if goal_state:
        path.append(goal_state)
        while parent[goal_state] is not None:
            goal_state = parent[goal_state]
            path.append(goal_state)
        
        path.reverse()  # Reverse path to get the correct order from start to goal
        
        # remove the goal state from explored list
        explored.remove(path[-1])
    else:
        path = []  # No path if no goal state is found
    
    return f'{" ".join(explored)}\n{" ".join(path)}'


if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 2
    grader.grade(problem_id, test_case_id, bfs_search, parse.read_graph_search_problem)