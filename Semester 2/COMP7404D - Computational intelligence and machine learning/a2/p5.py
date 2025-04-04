import sys, parse
import time, os, copy

import sys, copy, random, os, time
from parse import DIRECTIONS, EAT_FOOD_SCORE, PACMAN_EATEN_SCORE, PACMAN_WIN_SCORE, PACMAN_MOVING_SCORE

def min_max_multiple_ghosts(problem, k):
    """
    Pacman agent using Minimax search against Minimax ghosts.
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
        # Pacman chooses the best move using Minimax
        direction = minimax_decision(pacman_position, ghost_positions, food_positions, wall_positions, depth=k)
        pacman_position = move(pacman_position, direction)
        score += PACMAN_MOVING_SCORE
        p_moving_str = f'{step}: P moving {direction}'
        
        # print('step:{}, P moving {}'.format(step,direction))
        
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
        
        # Each ghost moves using Minimax
        for ghost_name in sorted(ghost_positions.keys()):
            step += 1
            old_position = ghost_positions[ghost_name]
            direction = minimax_decision(ghost_positions[ghost_name], ghost_positions, food_positions, wall_positions, k)
            new_position = move(old_position, direction)
            
            # print('step:{}, {} moving {}'.format(step,ghost_name,direction))
            
            if new_position not in ghost_positions.values():  # Avoid overlapping ghosts
                ghost_positions[ghost_name] = new_position
            
            w_moving_str = f'{step}: {ghost_name} moving {direction}'
            result_str += w_moving_str + '\n' + reconstruct_board_state(pacman_position, ghost_positions, food_positions, wall_positions, board_size)
            
            # Check if Pacman is caught after each ghost move
            if pacman_position in ghost_positions.values():
                score += PACMAN_EATEN_SCORE
                result_str += f'score: {score}\nWIN: Ghost'
                return result_str, 'Ghost'
            else:
                result_str += f'score: {score}\n'

def minimax_decision(pacman_position, ghost_positions, food_positions, wall_positions, depth):
    """ Minimax search with random selection among best moves for Pacman and Ghosts. """
    def minimax(state, depth, agent_index):
        if depth == 0 or is_terminal(state):
            return evaluate(state)

        if agent_index == 0:  # Pacman (Max node)
            best_value = float('-inf')
            best_actions = []  # Record all best directions

            for action in DIRECTIONS:
                new_position = move(state['pacman'], action)
                if new_position in wall_positions:
                    continue

                value = minimax({'pacman': new_position, 'ghosts': state['ghosts'], 'food': state['food']}, depth - 1, 1)

                if value > best_value:
                    best_value = value
                    best_actions = [action]  # Update best directions
                elif value == best_value:
                    best_actions.append(action)  # Record multiple best directions

            return random.choice(best_actions) if depth == k else best_value  # Randomly choose one of the best directions

        else:  # Ghosts (Min nodes)
            ghost_keys = list(state['ghosts'].keys())
            ghost_name = ghost_keys[agent_index - 1]
            ghost_position = state['ghosts'][ghost_name]

            worst_value = float('inf')
            best_actions = []  # Record all worst directions

            valid_actions = [d for d in DIRECTIONS if move(ghost_position, d) not in wall_positions]

            for action in valid_actions:
                new_position = move(ghost_position, action)
                new_ghosts = state['ghosts'].copy()
                new_ghosts[ghost_name] = new_position

                next_agent = agent_index + 1 if agent_index < len(ghost_keys) else 0
                next_depth = depth - 1 if next_agent == 0 else depth
                value = minimax({'pacman': state['pacman'], 'ghosts': new_ghosts, 'food': state['food']}, next_depth, next_agent)

                if value < worst_value:  # Ghosts choose the minimum value
                    worst_value = value
                    best_actions = [action]  # Update worst directions
                elif value == worst_value:
                    best_actions.append(action)  # Record multiple worst directions

            return random.choice(best_actions) if depth == k else worst_value  # Randomly choose one of the worst directions

    state = {'pacman': pacman_position, 'ghosts': ghost_positions, 'food': food_positions}
    return minimax(state, depth, 0)  # Pacman chooses direction



def is_terminal(state):
    """ Check if the state is terminal. """
    return not state['food'] or state['pacman'] in state['ghosts'].values()

def evaluate(state):
    """ Heuristic evaluation function for Pacman. """
    pacman_position = state['pacman']
    food_positions = state['food']
    ghost_positions = state['ghosts'].values()
    
    if pacman_position in ghost_positions:
        return PACMAN_EATEN_SCORE
    if not food_positions:
        return PACMAN_WIN_SCORE
    
    food_distance = min(manhattan_distance(pacman_position, food) for food in food_positions) if food_positions else 0
    ghost_distance = min(manhattan_distance(pacman_position, ghost) for ghost in ghost_positions)
    
    return -food_distance + ghost_distance

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
    problem_id = 5
    file_name_problem = str(test_case_id)+'.prob' 
    file_name_sol = str(test_case_id)+'.sol'
    path = os.path.join('test_cases','p'+str(problem_id)) 
    problem = parse.read_layout_problem(os.path.join(path,file_name_problem))
    k = int(sys.argv[2])
    num_trials = int(sys.argv[3])
    verbose = bool(int(sys.argv[4]))
    print('test_case_id:',test_case_id)
    print('k:',k)
    print('num_trials:',num_trials)
    print('verbose:',verbose)
    start = time.time()
    win_count = 0
    for i in range(num_trials):
        solution, winner = min_max_multiple_ghosts(copy.deepcopy(problem), k)
        if winner == 'Pacman':
            win_count += 1
        if verbose:
            print(solution)
    win_p = win_count/num_trials * 100
    end = time.time()
    print('time: ',end - start)
    print('win %',win_p)