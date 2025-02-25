import os, sys
def read_graph_search_problem(file_path):
    #Your p1 code here
    """ a sample problem file
    start_state: A
    goal_states: G
    A 0
    B 0
    C 0
    G 0
    A B 1.0
    B A 2.0
    B C 4.0
    C A 8.0
    C G 16.0
    C B 32.0
    """
    # read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # parse the file
    start_state = lines[0].split()[1]
    goal_states = lines[1].split()[1]
    graph = {}
    for line in lines[2:]:
        if len(line.split()) == 2:
            state = line.split()[0]
            graph[state] = {}
        else:
            state1, state2, cost = line.split()
            graph[state1][state2] = float(cost)
    # return the problem
    problem = {'start_state': start_state, 'goal_states': goal_states, 'graph': graph}
    return problem

def read_8queens_search_problem(file_path):
    #Your p6 code here
    problem = ''
    return problem

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