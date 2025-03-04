import sys, parse
import time, os, copy

import sys, copy, random, os, time
from parse import DIRECTIONS, EAT_FOOD_SCORE, PACMAN_EATEN_SCORE, PACMAN_WIN_SCORE, PACMAN_MOVING_SCORE

def min_max_multiple_ghosts(problem, k):
    seed = problem['seed']
    if seed != -1:
        random.seed(seed, version=1)

    layout = problem['layout']
    pacman_position, ghost_positions, food_positions, wall_positions, board_size = extract_board_state(layout)
    score = 0
    step = 0
    result_str = f'seed: {seed}\n0\n'
    result_str += reconstruct_board_state(pacman_position, ghost_positions, food_positions, wall_positions, board_size)

    visited_positions = []  # 记录最近 10 步的位置
    MAX_HISTORY = 10  
    MAX_STEPS = 1000  # 设定最大步数，防止无限循环

    while step < MAX_STEPS:
        step += 1

        # 记录当前位置
        visited_positions.append(pacman_position)
        if len(visited_positions) > MAX_HISTORY:
            visited_positions.pop(0)

        # 检查是否可能出现死循环
        loop_count = visited_positions.count(pacman_position)
        if loop_count > 3:  
            print(f"Warning: Potential loop detected at step {step}, position {pacman_position}!")

        # Pacman moves using minimax
        score_eval, direction = minimax(pacman_position, ghost_positions, food_positions, wall_positions, k, True)

        if direction:
            pacman_position = move(pacman_position, direction)
            p_moving_str = f'{step}: P moving {direction}'
        else:
            p_moving_str = f'{step}: P Moving'  

        score += PACMAN_MOVING_SCORE
        print(f"Step {step}, Pacman Score Eval: {score_eval}, Move: {direction}, Score: {score}")

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

        # Pacman is eaten by any ghost
        if pacman_position in ghost_positions:
            score += PACMAN_EATEN_SCORE
            result_str += f'score: {score}\nWIN: Ghost'
            return result_str, 'Ghost'

        # Ghosts move
        new_ghost_positions = []
        for i, ghost_position in enumerate(ghost_positions):
            step += 1
            score_eval, direction = minimax(pacman_position, ghost_positions, food_positions, wall_positions, k, False, i)
            # print(f"Step {step}, Ghost {chr(87+i)} Score Eval: {score_eval}, Move: {direction}")  # 调试信息

            if direction:
                new_ghost_position = move(ghost_position, direction)
                w_moving_str = f'{step}: {chr(87+i)} moving {direction}'
            else:
                new_ghost_position = ghost_position
                w_moving_str = f'{step}: {chr(87+i)} Moving'

            new_ghost_positions.append(new_ghost_position)
            result_str += w_moving_str + '\n' + reconstruct_board_state(pacman_position, new_ghost_positions, food_positions, wall_positions, board_size)

            if pacman_position in new_ghost_positions:
                score += PACMAN_EATEN_SCORE
                result_str += f'score: {score}\nWIN: Ghost'
                return result_str, 'Ghost'

        ghost_positions = new_ghost_positions
        result_str += f'score: {score}\n'

    # 终止游戏，防止无限循环
    print("Game terminated due to possible infinite loop!")
    result_str += f'score: {score}\nWIN: Unknown (Loop Detected)'
    return result_str, 'Unknown'



import random

def minimax(pacman_position, ghost_positions, food_positions, wall_positions, depth, is_pacman, ghost_index=0):
    """
    Minimax search function with depth-limited evaluation.
    Pacman (maximizing) moves first, followed by ghosts (minimizing).
    If multiple moves have the same best score, Pacman selects randomly.
    """
    if depth == 0:
        return evaluate_state(pacman_position, ghost_positions, food_positions, wall_positions, []), None

    if is_pacman:
        best_score = float('-inf')
        best_directions = []  # Store all equally best directions

        for direction in DIRECTIONS:
            new_position = move(pacman_position, direction)
            if new_position in wall_positions:
                continue  # Ignore moves into walls

            score, _ = minimax(new_position, ghost_positions, food_positions, wall_positions, depth - 1, False, 0)

            if score > best_score:
                best_score = score
                best_directions = [direction]  # Replace with new best move
            elif score == best_score:
                best_directions.append(direction)  # Add alternative best move

        # **Randomly choose from best options**
        if best_directions:
            return best_score, random.choice(best_directions)
        else:
            return best_score, None  # No valid move

    else:
        best_score = float('inf')
        best_direction = None
        ghost_position = ghost_positions[ghost_index]

        for direction in DIRECTIONS:
            new_position = move(ghost_position, direction)
            if new_position in wall_positions or new_position in ghost_positions:
                continue  # Ignore moves into walls or other ghosts

            new_ghost_positions = ghost_positions[:]
            new_ghost_positions[ghost_index] = new_position

            if ghost_index < len(ghost_positions) - 1:
                score, _ = minimax(pacman_position, new_ghost_positions, food_positions, wall_positions, depth - 1, False, ghost_index + 1)
            else:
                score, _ = minimax(pacman_position, new_ghost_positions, food_positions, wall_positions, depth - 1, True)

            if score < best_score:
                best_score = score
                best_direction = direction

        return best_score, best_direction



def evaluate_state(pacman_position, ghost_positions, food_positions, wall_positions, visited_positions):
    """
    Evaluates the game state to guide the minimax search.

    Score Components:
    - Encourages Pacman to eat food quickly.
    - Discourages revisiting the same position (to avoid loops).
    - Encourages Pacman to maintain distance from ghosts.
    - Favors states with more available moves to prevent deadlocks.
    """

    # If Pacman is caught, return the worst possible score
    if pacman_position in ghost_positions:
        return -10000  

    # If Pacman eats all food, return the best possible score
    if not food_positions:
        return 10000  

    # Distance to nearest food (lower is better)
    food_distances = [manhattan_distance(pacman_position, food) for food in food_positions]
    nearest_food_distance = min(food_distances) if food_distances else 1

    # Distance to nearest ghost (higher is better)
    ghost_distances = [manhattan_distance(pacman_position, ghost) for ghost in ghost_positions]
    nearest_ghost_distance = min(ghost_distances) if ghost_distances else 100  

    # Count available moves (to avoid being trapped)
    available_moves = sum(1 for d in DIRECTIONS if move(pacman_position, d) not in wall_positions)

    # **Avoid loops**: Track position frequency in the last 10 moves
    loop_penalty = 0
    if visited_positions.count(pacman_position) >= 3:  
        loop_penalty = -20  # If Pacman revisits 3+ times in last 10 steps, punish heavily
    elif visited_positions.count(pacman_position) == 2:
        loop_penalty = -10  # 2 times is still bad, but not as bad as 3 times

    # **Encourage Pacman to eat food aggressively**
    food_score = 30 / (nearest_food_distance + 1)  

    # **Encourage Pacman to stay away from ghosts**
    ghost_avoidance = nearest_ghost_distance * 3  

    # **Encourage mobility to avoid getting trapped**
    mobility_bonus = available_moves * 2  

    # Final score calculation
    score = food_score + ghost_avoidance + mobility_bonus + loop_penalty

    return score





def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


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
    ghost_positions = []
    food_positions = []
    wall_positions = []
    for i, row in enumerate(layout):
        for j, cell in enumerate(row):
            if cell == 'P':
                pacman_position = (i, j)
            elif cell in 'WXYZ':
                ghost_positions.append((i, j))
            elif cell == '.':
                food_positions.append((i, j))
            elif cell == '%':
                wall_positions.append((i, j))
    return pacman_position, ghost_positions, food_positions, wall_positions, (len(layout), len(layout[0]))


def reconstruct_board_state(pacman_position, ghost_positions, food_positions, wall_positions, board_size):
    board = [[' ' for _ in range(board_size[1])] for _ in range(board_size[0])]
    for i, j in wall_positions:
        board[i][j] = '%'
    for i, j in food_positions:
        board[i][j] = '.'
    for i, pos in enumerate(ghost_positions):
        board[pos[0]][pos[1]] = chr(87 + i)
    board[pacman_position[0]][pacman_position[1]] = 'P'
    return '\n'.join(''.join(row) for row in board) + '\n'


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