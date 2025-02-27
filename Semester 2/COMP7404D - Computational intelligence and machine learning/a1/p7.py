import sys, parse, grader

def better_board(problem):
    # 1. get the attack matrix
    attack_matrix = number_of_attacks(problem)
    
    # 2. find the lowest attack cost, if there are multiple, choose the one with the lowest row index
    min_attack_cost = float('inf')
    best_col = None
    best_row = None
    for col in range(len(attack_matrix[0])):
        for row in range(len(attack_matrix)):
            if attack_matrix[row][col] < min_attack_cost:
                min_attack_cost = attack_matrix[row][col]
                best_col = col
                best_row = row
    
    # 3. move the queen to the best position
    problem[best_col] = best_row
    
    # 4. reconstruct the board
    board = [['.' for _ in range(len(problem))] for _ in range(len(problem))]
    for col, row in enumerate(problem):
        board[row][col] = 'q'
        
    board_str = '\n'.join([' '.join(row) for row in board])
    return board_str
                
# functions from p6.py and remove the string formatting
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
    return attack_matrix

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
    problem_id = 7
    grader.grade(problem_id, test_case_id, better_board, parse.read_8queens_search_problem)