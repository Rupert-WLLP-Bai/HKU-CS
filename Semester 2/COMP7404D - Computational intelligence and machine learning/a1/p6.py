import sys, parse, grader

def number_of_attacks(problem):
    n = len(problem)
    # Create a board of size n x n initialized to 0
    attack_matrix = [[0] * n for _ in range(n)]
    
    # Iterate over every square on the board
    for row in range(n):
        for col in range(n):
            # Store the attack count for this square
            attack_matrix[row][col] = calculate_attack_cost(row, col, problem)
    # Return the matrix
    attack_matrix_str = '\n'.join([' '.join([f'{cell:2}' for cell in row]) for row in attack_matrix])
    return attack_matrix_str

def calculate_attack_cost(x, y, q_positions):
    attack_cost = 0
    n = len(q_positions)
    q_positions = q_positions.copy() # make a copy of the current queen positions
    
    # 0. move the queen from the same column to (x, y)
    q_positions[y] = x
    
    # 1. check the row and diagonal attacks
    for i in range(n):
        for j in range(i+1, n):
            if abs(i - j) == abs(q_positions[i] - q_positions[j]):
                attack_cost += 1
            elif q_positions[i] == q_positions[j]:
                attack_cost += 1

    return attack_cost
    

if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 6
    grader.grade(problem_id, test_case_id, number_of_attacks, parse.read_8queens_search_problem)