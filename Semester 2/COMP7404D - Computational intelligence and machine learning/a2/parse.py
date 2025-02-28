import os, sys
def read_layout_problem(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract the seed from the first line
    seed = int(lines[0].strip().split(':')[1].strip())

    # Extract the grid layout from the remaining lines
    layout = []
    for line in lines[1:]:
        layout.append(line.strip())

    # Return the parsed problem as a tuple or a dictionary
    problem = {
        'seed': seed,
        'layout': layout
    }

    return problem


if __name__ == "__main__":
    if len(sys.argv) == 3:
        problem_id, test_case_id = sys.argv[1], sys.argv[2]
        problem = read_layout_problem(os.path.join('test_cases','p'+problem_id, test_case_id+'.prob'))
        print(problem)
    else:
        print('Error: I need exactly 2 arguments!')