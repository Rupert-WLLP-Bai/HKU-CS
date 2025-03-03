import sys, parse
import time, os, copy

import sys, random, grader, parse
from parse import DIRECTIONS, EAT_FOOD_SCORE, PACMAN_EATEN_SCORE, PACMAN_WIN_SCORE, PACMAN_MOVING_SCORE

def better_play_single_ghosts(problem):
    seed = problem['seed']
    if seed != -1:
        random.seed(seed, version=1)
    
    layout = problem['layout']
    pacman_position, ghost_position, food_positions, wall_positions, board_size = extract_board_state(layout)
    score = 0
    step = 0
    result_str = f'seed: {seed}\n0\n'
    result_str += reconstruct_board_state(pacman_position, ghost_position, food_positions, wall_positions, board_size)

    while True:
        step += 1
        # Pacman chooses best move based on evaluation function
        direction = choose_best_direction(pacman_position, ghost_position, food_positions, wall_positions)
        pacman_position = move(pacman_position, direction)
        score += PACMAN_MOVING_SCORE
        p_moving_str = f'{step}: P moving {direction}'

        # Pacman eats food
        if pacman_position in food_positions:
            food_positions.remove(pacman_position)
            score += EAT_FOOD_SCORE
            if not food_positions:
                score += PACMAN_WIN_SCORE
                result_str += p_moving_str + '\n' + reconstruct_board_state(pacman_position, ghost_position, food_positions, wall_positions, board_size)
                result_str += f'score: {score}\nWIN: Pacman'
                return result_str, 'Pacman'

        result_str += p_moving_str + '\n' + reconstruct_board_state(pacman_position, ghost_position, food_positions, wall_positions, board_size)

        # Pacman is eaten by the ghost
        if pacman_position == ghost_position:
            score += PACMAN_EATEN_SCORE
            result_str += f'score: {score}\nWIN: Ghost'
            return result_str, 'Ghost'
        else:
            result_str += f'score: {score}\n'

        # Ghost moves randomly
        step += 1
        direction = choose_random_direction(layout, ghost_position)
        ghost_position = move(ghost_position, direction)
        w_moving_str = f'{step}: W moving {direction}'

        result_str += w_moving_str + '\n' + reconstruct_board_state(pacman_position, ghost_position, food_positions, wall_positions, board_size)

        # Check if Pacman is caught
        if pacman_position == ghost_position:
            score += PACMAN_EATEN_SCORE
            result_str += f'score: {score}\nWIN: Ghost'
            return result_str, 'Ghost'
        else:
            result_str += f'score: {score}\n'

def choose_best_direction(pacman_position, ghost_position, food_positions, wall_positions):
    '''
    Choose the best direction for Pacman based on the evaluation function
    
    Use manhattan distance to calculate the distance between Pacman and food/ghost
    '''
    best_direction = None
    best_score = float('-inf')

    for direction in DIRECTIONS:
        new_position = move(pacman_position, direction)
        if new_position in wall_positions:
            continue
        
        # Compute score: prioritize food, avoid ghost
        food_distance = min(manhattan_distance(new_position, food) for food in food_positions) if food_positions else 0
        ghost_distance = manhattan_distance(new_position, ghost_position)
        
        score = -food_distance + ghost_distance
        if score > best_score:
            best_score = score
            best_direction = direction

    return best_direction if best_direction else random.choice(DIRECTIONS)

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Code from p1.py
def choose_random_direction(layout, pacman_position):
    available_directions = []
    for direction in DIRECTIONS:
        i, j = move(pacman_position, direction)
        if layout[i][j] != '%':
            available_directions.append(direction)
    direction = random.choice(available_directions)
    return direction

def move(position, direction):
    i, j = position
    if direction == 'E':
        j += 1
    elif direction == 'W':
        j -= 1
    elif direction == 'N':
        i -= 1
    elif direction == 'S':
        i += 1
    return i, j

def extract_board_state(layout):
    '''
    returns:
    1. pacman position
    2. ghost position
    3. food positions
    4. wall positions
    5. board size
    '''
    pacman_position = None
    ghost_position = None
    food_positions = []
    wall_positions = []
    board_size = (len(layout), len(layout[0]))
    for i in range(len(layout)):
        for j in range(len(layout[i])):
            if layout[i][j] == 'P':
                pacman_position = (i, j)
            elif layout[i][j] == 'W':
                ghost_position = (i, j)
            elif layout[i][j] == '.':
                food_positions.append((i, j))
            elif layout[i][j] == '%':
                wall_positions.append((i, j))
    return pacman_position, ghost_position, food_positions, wall_positions, board_size

def reconstruct_board_state(pacman_position, ghost_position, food_positions, wall_positions, board_size):
    """
    '%': Wall
    'W': Ghost
    'P': Pacman
    '.': Food
    ' ': empty Square
    """
    # initialize the board
    layout = [[' ' for _ in range(board_size[1])] for _ in range(board_size[0])]
    # wall
    for i, j in wall_positions:
        layout[i][j] = '%'
    # pacman
    i, j = pacman_position
    layout[i][j] = 'P'
    # ghost
    i, j = ghost_position
    layout[i][j] = 'W'
    # food
    for i, j in food_positions:
        layout[i][j] = '.'
    # if ghost is on the food, change the food to ghost
    if ghost_position in food_positions:
        i, j = ghost_position
        layout[i][j] = 'W'
    # return the layout to string format
    return '\n'.join([''.join(row) for row in layout]) + '\n'


if __name__ == "__main__":
    test_case_id = int(sys.argv[1])    
    problem_id = 2
    file_name_problem = str(test_case_id)+'.prob' 
    file_name_sol = str(test_case_id)+'.sol'
    path = os.path.join('test_cases','p'+str(problem_id)) 
    problem = parse.read_layout_problem(os.path.join(path,file_name_problem))
    num_trials = int(sys.argv[2])
    verbose = bool(int(sys.argv[3]))
    print('test_case_id:',test_case_id)
    print('num_trials:',num_trials)
    print('verbose:',verbose)
    start = time.time()
    win_count = 0
    for i in range(num_trials):
        solution, winner = better_play_single_ghosts(copy.deepcopy(problem))
        if winner == 'Pacman':
            win_count += 1
        if verbose:
            print(solution)
    win_p = win_count/num_trials * 100
    end = time.time()
    print('time: ',end - start)
    print('win %',win_p)