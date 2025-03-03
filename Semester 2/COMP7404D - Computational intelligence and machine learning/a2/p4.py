import sys, parse
import time, os, copy

import sys, random, grader, parse
from parse import DIRECTIONS, EAT_FOOD_SCORE, PACMAN_EATEN_SCORE, PACMAN_WIN_SCORE, PACMAN_MOVING_SCORE

def better_play_multiple_ghosts(problem):
    """ 
    Improved Pacman agent playing against multiple random ghosts.
    Pacman moves intelligently, while ghosts move randomly.
    """
    seed = problem['seed']
    if seed != -1:
        random.seed(seed, version=1)
    
    layout = problem['layout']
    pacman_position, ghost_positions, food_positions, wall_positions, board_size = extract_board_state(layout)
    score = 0
    step = 0
    result_str = f'seed: {seed}\n0\n'
    result_str += reconstruct_board_state(pacman_position, ghost_positions, food_positions, wall_positions, board_size)

    while True:
        step += 1
        # Pacman chooses the best move
        direction = choose_best_direction(pacman_position, ghost_positions, food_positions, wall_positions)
        pacman_position = move(pacman_position, direction)
        score += PACMAN_MOVING_SCORE
        p_moving_str = f'{step}: P moving {direction}'

        # Pacman eats food
        if pacman_position in food_positions:
            food_positions.remove(pacman_position)
            score += EAT_FOOD_SCORE
            if not food_positions:
                score += PACMAN_WIN_SCORE
                result_str += p_moving_str + '\n' + reconstruct_board_state(pacman_position, ghost_positions, food_positions, wall_positions, board_size)
                result_str += f'score: {score}\nWIN: Pacman'
                return result_str, 'Pacman'

        result_str += p_moving_str + '\n' + reconstruct_board_state(pacman_position, ghost_positions, food_positions, wall_positions, board_size)

        # Check if Pacman is eaten by any ghost
        if pacman_position in ghost_positions.values():
            score += PACMAN_EATEN_SCORE
            result_str += f'score: {score}\nWIN: Ghost'
            return result_str, 'Ghost'
        else:
            result_str += f'score: {score}\n'

        # Each ghost moves
        for ghost_name in sorted(ghost_positions.keys()):
            step += 1
            old_position = ghost_positions[ghost_name]
            direction = choose_random_direction(layout, old_position, ghost_positions, wall_positions)
            
            # Only move if there's a valid direction
            if direction:
                new_position = move(old_position, direction)
                if new_position not in ghost_positions.values():  # Ensure no overlapping ghosts
                    ghost_positions[ghost_name] = new_position

            w_moving_str = f'{step}: {ghost_name} moving {direction}' if direction else f'{step}: {ghost_name} Moving'
            result_str += w_moving_str + '\n' + reconstruct_board_state(pacman_position, ghost_positions, food_positions, wall_positions, board_size)

            # Check if Pacman is caught after each ghost move
            if pacman_position in ghost_positions.values():
                score += PACMAN_EATEN_SCORE
                result_str += f'score: {score}\nWIN: Ghost'
                return result_str, 'Ghost'
            else:
                result_str += f'score: {score}\n'


def choose_best_direction(pacman_position, ghost_positions, food_positions, wall_positions):
    """ 
    Selects the best direction for Pacman using an evaluation function.
    Pacman moves towards food while avoiding ghosts.
    """
    best_direction = None
    best_score = float('-inf')

    for direction in DIRECTIONS:
        new_position = move(pacman_position, direction)
        if new_position in wall_positions:
            continue
        
        # Compute distance to closest food
        food_distance = min(manhattan_distance(new_position, food) for food in food_positions) if food_positions else 0
        # Compute distance to closest ghost
        ghost_distance = min(manhattan_distance(new_position, ghost) for ghost in ghost_positions.values())

        # Evaluation function: prioritize food, avoid ghosts
        score = -food_distance + ghost_distance
        if score > best_score:
            best_score = score
            best_direction = direction

    return best_direction if best_direction else random.choice(DIRECTIONS)


def choose_random_direction(layout, position, ghost_positions, wall_positions):
    """ 
    Chooses a random direction for ghosts while avoiding walls and other ghosts.
    Returns an empty string if no valid move is possible.
    """
    available_directions = []
    for direction in DIRECTIONS:
        new_position = move(position, direction)
        if new_position not in wall_positions and new_position not in ghost_positions.values():
            available_directions.append(direction)
    
    return random.choice(available_directions) if available_directions else ""  # Return empty string if no move possible


def move(position, direction):
    """ Moves a character in the given direction. """
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


def manhattan_distance(pos1, pos2):
    """ Computes the Manhattan distance between two positions. """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def extract_board_state(layout):
    """
    Extracts the initial board state and returns:
    1. Pacman position
    2. Ghost positions (dictionary with names as keys)
    3. Food positions
    4. Wall positions
    5. Board size
    """
    pacman_position = None
    ghost_positions = {}
    food_positions = []
    wall_positions = []
    board_size = (len(layout), len(layout[0]))
    ghost_names = ['W', 'X', 'Y', 'Z']
    ghost_count = 0

    for i in range(len(layout)):
        for j in range(len(layout[i])):
            if layout[i][j] == 'P':
                pacman_position = (i, j)
            elif layout[i][j] in ghost_names:
                ghost_positions[layout[i][j]] = (i, j)
                ghost_count += 1
            elif layout[i][j] == '.':
                food_positions.append((i, j))
            elif layout[i][j] == '%':
                wall_positions.append((i, j))

    return pacman_position, ghost_positions, food_positions, wall_positions, board_size


def reconstruct_board_state(pacman_position, ghost_positions, food_positions, wall_positions, board_size):
    """
    Reconstructs the board as a string after each move.
    """
    layout = [[' ' for _ in range(board_size[1])] for _ in range(board_size[0])]
    
    # Add walls
    for i, j in wall_positions:
        layout[i][j] = '%'

    # Add Pacman
    i, j = pacman_position
    layout[i][j] = 'P'

    # Add ghosts
    for ghost_name, (i, j) in ghost_positions.items():
        layout[i][j] = ghost_name

    # Add food
    for i, j in food_positions:
        if layout[i][j] == ' ':
            layout[i][j] = '.'

    return '\n'.join([''.join(row) for row in layout]) + '\n'

if __name__ == "__main__":
    test_case_id = int(sys.argv[1])    
    problem_id = 4
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
        solution, winner = better_play_multiple_ghosts(copy.deepcopy(problem))
        if winner == 'Pacman':
            win_count += 1
        if verbose:
            print(solution)
    win_p = win_count/num_trials * 100
    end = time.time()
    print('time: ',end - start)
    print('win %',win_p)