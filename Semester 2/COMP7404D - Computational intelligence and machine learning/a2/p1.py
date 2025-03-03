import sys, random, grader, parse

from parse import DIRECTIONS, EAT_FOOD_SCORE, PACMAN_EATEN_SCORE, PACMAN_WIN_SCORE, PACMAN_MOVING_SCORE

def random_play_single_ghost(problem):
    """
    '%': Wall
    'W': Ghost
    'P': Pacman
    '.': Food
    ' ': empty Square
    """
    seed = problem['seed']
    random.seed(seed, version=1)
    layout = problem['layout']
    pacman_position, ghost_position, food_positions, wall_positions, board_size = extract_board_state(layout)
    score = 0
    step = 0
    result_str = ''
    result_str += 'seed: {}\n'.format(seed)
    result_str += '0\n'
    result_str += reconstruct_board_state(pacman_position, ghost_position, food_positions, wall_positions, board_size)
    
    # 1. pacman eats all the food and wins
    # 2. pacman is eaten by the ghost
    while True:
        step += 1
        # pacman moves
        # choose a random direction which is not blocked by a wall
        direction = choose_random_direction(layout, pacman_position)
        # print('step:{}, available_directions:{}, direction:{}'.format(step, available_directions, direction))
        p_moving_str = f'{step}: P moving {direction}'
        pacman_position = move(pacman_position, direction)
        # print('pacman_move:', direction)
        score += PACMAN_MOVING_SCORE
        # pacman eats food
        if pacman_position in food_positions:
            food_positions.remove(pacman_position)
            score += EAT_FOOD_SCORE
            if not food_positions:
                score += PACMAN_WIN_SCORE
                board_str = reconstruct_board_state(pacman_position, ghost_position, food_positions, wall_positions, board_size)
                result_str += p_moving_str + '\n' + board_str
                result_str += f'score: {score}\nWIN: Pacman'
                break
        board_str = reconstruct_board_state(pacman_position, ghost_position, food_positions, wall_positions, board_size)
        result_str += p_moving_str + '\n' + board_str
        # pacman is eaten by the ghost
        if pacman_position == ghost_position:
            score += PACMAN_EATEN_SCORE
            result_str += f'score: {score}\nWIN: Ghost'
            break
        else:
            # pacman is not eaten by the ghost
            result_str += f'score: {score}\n'
        # ghost moves
        step += 1
        # choose a random direction which is not blocked by a wall
        direction = choose_random_direction(layout, ghost_position)
        # print('step:{}, available_directions:{}, direction:{}'.format(step, available_directions, direction))
        w_moving_str = f'{step}: W moving {direction}'
        ghost_position = move(ghost_position, direction)
        board_str = reconstruct_board_state(pacman_position, ghost_position, food_positions, wall_positions, board_size)
        result_str += w_moving_str + '\n' + board_str
        # pacman is eaten by the ghost
        if pacman_position == ghost_position:
            score += PACMAN_EATEN_SCORE
            result_str += f'score: {score}\nWIN: Ghost'
            break
        else:
            # pacman is not eaten by the ghost
            result_str += f'score: {score}\n'
        # print('ghost_move:', direction)
        
    return result_str

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
    problem_id = 1
    grader.grade(problem_id, test_case_id, random_play_single_ghost, parse.read_layout_problem)