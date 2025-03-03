import sys, random, grader, parse
from parse import DIRECTIONS, EAT_FOOD_SCORE, PACMAN_EATEN_SCORE, PACMAN_WIN_SCORE, PACMAN_MOVING_SCORE

def random_play_multiple_ghosts(problem):
    seed = problem['seed']
    random.seed(seed, version=1)

    layout = problem['layout']
    pacman_position, ghost_positions, food_positions, wall_positions, board_size = extract_board_state(layout)
    score = 0
    step = 0
    result_str = f'seed: {seed}\n0\n'
    result_str += reconstruct_board_state(pacman_position, ghost_positions, food_positions, wall_positions, board_size)

    while True:
        step += 1
        # Pacman moves randomly
        direction = choose_random_direction(layout, pacman_position)
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
                return result_str

        result_str += p_moving_str + '\n' + reconstruct_board_state(pacman_position, ghost_positions, food_positions, wall_positions, board_size)

        # Pacman is eaten by any ghost
        if pacman_position in ghost_positions.values():
            score += PACMAN_EATEN_SCORE
            result_str += f'score: {score}\nWIN: Ghost'
            return result_str

        result_str += f'score: {score}\n'

        # Ghosts move one by one in order (W -> X -> Y -> Z)
        for ghost_name in sorted(ghost_positions.keys()):
            step += 1
            old_position = ghost_positions[ghost_name]
            direction = choose_random_direction(layout, old_position, ghost_positions, wall_positions)
            new_position = move(old_position, direction)

            # Ghosts cannot move onto another ghost
            if new_position not in ghost_positions.values():
                ghost_positions[ghost_name] = new_position

            # if ghost cannot move, it stays in the same position, direction = ''
            if direction == None:
                direction = ''
            w_moving_str = f'{step}: {ghost_name} moving {direction}'
            result_str += w_moving_str + '\n' + reconstruct_board_state(pacman_position, ghost_positions, food_positions, wall_positions, board_size)

            # Check if Pacman is eaten
            if pacman_position == ghost_positions[ghost_name]:
                score += PACMAN_EATEN_SCORE
                result_str += f'score: {score}\nWIN: Ghost'
                return result_str

            result_str += f'score: {score}\n'

def choose_random_direction(layout, position, ghost_positions=None, wall_positions=None):
    available_directions = []
    for direction in DIRECTIONS:
        i, j = move(position, direction)
        if layout[i][j] != '%' and (ghost_positions is None or (i, j) not in ghost_positions.values()):
            available_directions.append(direction)
    return random.choice(available_directions) if available_directions else None

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
    pacman_position = None
    ghost_positions = {}
    food_positions = []
    wall_positions = []
    board_size = (len(layout), len(layout[0]))

    for i in range(len(layout)):
        for j in range(len(layout[i])):
            if layout[i][j] == 'P':
                pacman_position = (i, j)
            elif layout[i][j] in 'WXYZ':
                ghost_positions[layout[i][j]] = (i, j)
            elif layout[i][j] == '.':
                food_positions.append((i, j))
            elif layout[i][j] == '%':
                wall_positions.append((i, j))

    return pacman_position, ghost_positions, food_positions, wall_positions, board_size

def reconstruct_board_state(pacman_position, ghost_positions, food_positions, wall_positions, board_size):
    layout = [[' ' for _ in range(board_size[1])] for _ in range(board_size[0])]

    for i, j in wall_positions:
        layout[i][j] = '%'

    i, j = pacman_position
    layout[i][j] = 'P'

    for ghost, pos in ghost_positions.items():
        i, j = pos
        layout[i][j] = ghost

    for i, j in food_positions:
        if (i, j) not in ghost_positions.values():
            layout[i][j] = '.'

    return '\n'.join([''.join(row) for row in layout]) + '\n'

if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 3
    grader.grade(problem_id, test_case_id, random_play_multiple_ghosts, parse.read_layout_problem)