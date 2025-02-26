import sys, parse, grader

def number_of_attacks(problem):
    n = len(problem)
    # Create a board of size n x n initialized to 0
    attack_matrix = [[0] * n for _ in range(n)]
    
    # Iterate over every square on the board
    for row in range(n):
        for col in range(n):
            # Store the attack count for this square
            attack_matrix[row][col] = calculate_cost(problem, row, col)
    
    # Return the matrix
    return attack_matrix

def calculate_cost(problem, x, y):
    # move the queen from the same column to the (x,y) position
    # then calculate the number of attacks with the other queens
    pass
    

if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 6
    grader.grade(problem_id, test_case_id, number_of_attacks, parse.read_8queens_search_problem)