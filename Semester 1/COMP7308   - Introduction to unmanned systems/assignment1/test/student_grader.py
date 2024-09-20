# A custom autograder for this project

################################################################################
# A mini-framework for autograding
################################################################################

import argparse
import pickle
import random
import os, sys
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = PARENT_DIR if True else THIS_DIR 
sys.path.append(f'{BASE_DIR}/src/')

class WritableNull:
    def write(self, string):
        pass

    def flush(self):
        pass

class Tracker(object):
    def __init__(self, questions, maxes, prereqs, mute_output):
        self.questions = questions
        self.maxes = maxes
        self.prereqs = prereqs

        self.points = {q: 0 for q in self.questions}

        self.current_question = None

        self.current_test = None
        self.points_at_test_start = None
        self.possible_points_remaining = None

        self.mute_output = mute_output
        self.original_stdout = None
        self.muted = False

    def mute(self):
        if self.muted:
            return

        self.muted = True
        self.original_stdout = sys.stdout
        sys.stdout = WritableNull()

    def unmute(self):
        if not self.muted:
            return

        self.muted = False
        sys.stdout = self.original_stdout

    def begin_q(self, q):
        assert q in self.questions
        text = 'Question {}'.format(q)
        print('\n' + text)
        print('=' * len(text))

        for prereq in sorted(self.prereqs[q]):
            if self.points[prereq] < self.maxes[prereq]:
                print("""*** NOTE: Make sure to complete Question {} before working on Question {},
*** because Question {} builds upon your answer for Question {}.
""".format(prereq, q, q, prereq))
                return False

        self.current_question = q
        self.possible_points_remaining = self.maxes[q]
        return True

    def begin_test(self, test_name):
        self.current_test = test_name
        self.points_at_test_start = self.points[self.current_question]
        print("*** {}) {}".format(self.current_question, self.current_test))
        if self.mute_output:
            self.mute()

    def end_test(self, pts):
        if self.mute_output:
            self.unmute()
        self.possible_points_remaining -= pts
        if self.points[self.current_question] == self.points_at_test_start + pts:
            print("*** PASS: {}".format(self.current_test))
        elif self.points[self.current_question] == self.points_at_test_start:
            print("*** FAIL")

        self.current_test = None
        self.points_at_test_start = None

    def end_q(self):
        assert self.current_question is not None
        assert self.possible_points_remaining == 0
        print('\n### Question {}: {}/{} ###'.format(
            self.current_question,
            self.points[self.current_question],
            self.maxes[self.current_question]))

        self.current_question = None
        self.possible_points_remaining = None

    def finalize(self):
        import time
        print('\nFinished at %d:%02d:%02d' % time.localtime()[3:6])
        print("\nProvisional grades\n==================")

        for q in self.questions:
          print('Question %s: %d/%d' % (q, self.points[q], self.maxes[q]))
        print('------------------')
        print('Total: %d/%d' % (sum(self.points.values()),
            sum([self.maxes[q] for q in self.questions])))

        print("""
Your grades are NOT yet registered. To register your grades, make sure
to follow your instructor's guidelines to receive credit on your project.
""")

    def add_points(self, pts):
        self.points[self.current_question] += pts

TESTS = []
PREREQS = {}
def add_prereq(q, pre):
    if isinstance(pre, str):
        pre = [pre]

    if q not in PREREQS:
        PREREQS[q] = set()
    PREREQS[q] |= set(pre)

def test(q, points):
    def deco(fn):
        TESTS.append((q, points, fn))
        return fn
    return deco

def parse_options(argv):
    parser = argparse.ArgumentParser(description = 'Run public tests on student code')
    parser.set_defaults(
        edx_output=False,
        gs_output=False,
        graphics=False,
        mute_output=False,
        check_dependencies=False,
        answer_dir='/hint/',
        )
    parser.add_argument('--edx-output', 
                        dest = 'edx_output',
                        action = 'store_true',
                        help = 'Ignored, present for compatibility only')
    parser.add_argument('--gradescope-output',
                        dest = 'gs_output',
                        action = 'store_true',
                        help = 'Ignored, present for compatibility only')
    parser.add_argument('--mute',
                        dest = 'mute_output',
                        action = 'store_true',
                        help = 'Mute output from executing tests')
    parser.add_argument('--test',
                        dest = 'test',
                        action = 'store_true',
                        help = 'test implemented answer by TA')
    parser.add_argument('--question', '-q',
                        dest = 'grade_question',
                        default = None,
                        help = 'Grade only one question (e.g. `-q q1`)')
    parser.add_argument('--graphics',
                        dest = 'graphics',
                        action = 'store_true',
                        help = 'Display graphics ?(visualizing your implementation is highly recommended for debugging).')
    parser.add_argument('--check-dependencies',
                        dest = 'check_dependencies',
                        action = 'store_true',
                        help = 'check that numpy and matplotlib are installed')
    parser.add_argument('--answer_dir', 
                        dest = 'answer_dir',
                        help = 'where answers are saved')
    parser.add_argument('--iter-step',
                        dest = 'iter_step',
                        default = 9999999, type = int,
                        help = 'Ignored, present for compatibility only. The numbder of iteration steps, should be positive, default 9999999.')
    options, extras = parser.parse_known_args(argv)
    return options

def main():
    options = parse_options(sys.argv[1:])
    if options.check_dependencies:
        check_dependencies()
        return

    questions = set()
    maxes = {}
    for q, points, fn in TESTS:
        questions.add(q)
        maxes[q] = maxes.get(q, 0) + points
        if q not in PREREQS:
            PREREQS[q] = set()

    questions = list(sorted(questions))
    if options.grade_question:
        if options.grade_question not in questions:
            print("ERROR: question {} does not exist".format(options.grade_question))
            sys.exit(1)
        else:
            questions = [options.grade_question]
            PREREQS[options.grade_question] = set()

    tracker = Tracker(questions, maxes, PREREQS, options.mute_output)
    for q in questions:
        started = tracker.begin_q(q)
        if not started:
            continue

        for testq, points, fn in TESTS:
            if testq != q:
                continue
            tracker.begin_test(fn.__name__)
            try:
                fn(tracker, options)
            except KeyboardInterrupt:
                tracker.unmute()
                print("\n\nCaught KeyboardInterrupt: aborting autograder")
                tracker.finalize()
                print("\n[autograder was interrupted before finishing]")
                sys.exit(1)
            except:
                tracker.unmute()
                print(traceback.format_exc())
            tracker.end_test(points)
        tracker.end_q()
    tracker.finalize()

################################################################################
# Tests begin here
################################################################################

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ir_sim.env import EnvBase

def check_dependencies():
    pass
   
@test('q1', points=40)
def check_dynamics_without_noise(tracker, options):
    if options.test:
        from two_wheeled_robot_answer import RobotTwoWheel
    else:
        from two_wheeled_robot import RobotTwoWheel
    env = EnvBase(f'{BASE_DIR}/src/two_wheeled_robot_q1.yaml', 
        disable_all_plot = not options.graphics,
        display = options.graphics, custom_robot=RobotTwoWheel)
    trajectory_true = np.load(f'{BASE_DIR}/{options.answer_dir}/q1_trajectory.npy')
    instructions = [[0.6, 0.8], [-0.2, 0.2], [0.6, 0.4], [0.4, 0.6]]
    timepoints   = [2, 1, 2, 1]
    k = 0
    print(f"Instruction with omega_1 {instructions[k][0]} and omega_2 {instructions[k][1]}")
    next_time = timepoints[k]
    sim_time = np.sum(timepoints)
    for i in range(int(sim_time/env.step_time)):
        current_time = env.step_time * i
        if current_time >= next_time:
            k += 1
            print(f"Instruction with omega_1 {instructions[k][0]} and omega_2 {instructions[k][1]}")
            next_time += timepoints[k]
            trajectory = np.concatenate(env.robot.trajectory, axis=1)
            error = np.linalg.norm(trajectory-trajectory_true[:, :trajectory.shape[1]] )
            print(f'Trajectory error in part {k}: {error:3f}')
            if error < 1e-2:
                tracker.add_points(10)
        env.step(vel_list=[instructions[k]], )
        env.render(show_goal=False, show_traj=True,)
    trajectory = np.concatenate(env.robot.trajectory, axis=1)
    error = np.linalg.norm(trajectory-trajectory_true[:, :trajectory.shape[1]] )
    print(f'Trajectory error in part {k+1}: {error:3f}')
    if error < 1e-2:
        tracker.add_points(10)

    if options.graphics:
        env.end(show_traj=True, show_goal=False)
            
@test('q2', points=40)
def check_dynamics_with_noise(tracker, options):
    if options.test:
        from two_wheeled_robot_answer import RobotTwoWheel
    else:
        from two_wheeled_robot import RobotTwoWheel
    kwargs1 = {'robot_args': 
                {'noise_mode': 'linear', 
                 'noise_amplitude': np.c_[[0.005, 0.005, 0.05]]} }
    kwargs2 = {'robot_args': 
                {'noise_mode': 'nonlinear', 
                 'noise_amplitude': np.c_[[0.05, 0.05]]} }
    
    def generate_samples(env):
        instructions = [[0.2, 0.3]]
        timepoints   = [5]
        sim_time = 5
        samples_num = 100
        samples = []
        np.random.seed(16)
        for j in range(samples_num):
            k = 0
            next_time = timepoints[k]
            for i in range(int(sim_time/env.step_time)):
                current_time = env.step_time * i
                if current_time >= next_time:
                    k += 1
                    next_time += timepoints[k]
                env.step(vel_list = [instructions[k]], )
                env.render(1)
            samples.append(env.robot.state)
            env.reset()

        samples = np.concatenate(samples, axis =1)
        return samples

    def plot(ax, samples, robot, robot_color = 'g', goal_color='r', fontsize=10, pointsize=10,
         arrow_width=0.1, arrow_length=0.2):
    
        ax.set_aspect('equal')
        ax.set_xlim([-0.2, 1.5])
        ax.set_ylim([-0.5, 1])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        x = robot.init_state[0, 0]
        y = robot.init_state[1, 0]

        robot_circle = mpl.patches.Circle(xy=(x, y), radius = robot.radius/3, color = robot_color)
        robot_circle.set_zorder(1)
        ax.add_patch(robot_circle)
        ax.text(x + 0.15, y, 'start', fontsize = fontsize, color = robot_color)

        theta = robot.init_state[2, 0]
        arrow = mpl.patches.Arrow(x, y, arrow_length*np.cos(theta), arrow_length*np.sin(theta), color = robot_color, width = arrow_width)
        arrow.set_zorder(1)
        ax.add_patch(arrow)

        goal_x = robot.goal[0, 0]
        goal_y = robot.goal[1, 0]

        goal_circle = mpl.patches.Circle(xy=(goal_x, goal_y), radius = robot.radius/3, color=goal_color, alpha=0.5)
        goal_circle.set_zorder(1)
        ax.add_patch(goal_circle)
        ax.text(goal_x + 0.15, goal_y, 'goal', fontsize = fontsize, color = goal_color)

        goal_theta = robot.goal[2, 0]
        goal_arrow = mpl.patches.Arrow(goal_x, goal_y, arrow_length*np.cos(goal_theta), arrow_length*np.sin(goal_theta), color = goal_color, width = arrow_width)
        goal_arrow.set_zorder(1)
        ax.add_patch(goal_arrow)

        points = ax.scatter(samples[0], samples[1], s=pointsize, c='b', alpha = 0.5, )
        points.set_zorder(2)

    samples_true = np.load(f'{BASE_DIR}/{options.answer_dir}/q2_samples.npy', allow_pickle=True).item()
    env = EnvBase(f'{BASE_DIR}/src/two_wheeled_robot_q2.yaml', disable_all_plot = True, custom_robot=RobotTwoWheel, **kwargs1)
    samples1 = generate_samples(env)
    error = np.linalg.norm(samples_true['linear']-samples1)
    print(f'Sample error with linear noise: {error:3f}')
    if error < 1e-2:
        tracker.add_points(20)
    
    env = EnvBase(f'{BASE_DIR}/src/two_wheeled_robot_q2.yaml', disable_all_plot = True, custom_robot=RobotTwoWheel, **kwargs2)
    samples2 = generate_samples(env)
    error = np.linalg.norm(samples_true['nonlinear']-samples2)
    print(f'Sample error with nonlinear noise: {error:3f}')
    if error < 1e-2:
        tracker.add_points(20)
    
    if options.graphics:
        fig = plt.figure(figsize = (10, 5))
        ax1 = plt.subplot(1, 2, 1)
        plot(ax1, samples1, env.robot)
        ax1.set_title('Linear Noise')
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title('Nonlinear Noise')
        plot(ax2, samples2, env.robot)
        fig.tight_layout()
        plt.show(block=False)
        print(f'Figure will be closed within 3 seconds.')
        plt.pause(3)
        plt.close()

@test('q3', points=20)
def check_policy(tracker, options):
    print('Q3 is only for reference. We do not check the number of instructions, which will affect the whole final score.')
    if options.test:
        from two_wheeled_robot_answer import RobotTwoWheel
    else:
        from two_wheeled_robot import RobotTwoWheel
    env = EnvBase(f'{BASE_DIR}/src/two_wheeled_robot_q3.yaml', 
            disable_all_plot = not options.graphics, 
            display = options.graphics, custom_robot=RobotTwoWheel)

    sim_time = 10
    instructions, timepoints, answer = env.robot.policy()
    assert len(instructions) == len(timepoints)
    instruction_num = len(instructions)
    k = 0
    next_time = timepoints[0] if instruction_num > 0 else np.inf

    for i in range(int(sim_time/env.step_time)):
        current_time = env.step_time * i
        if current_time >= next_time:
            k += 1
            next_time += timepoints[k] if k < instruction_num else np.inf
        instruction = [np.clip(instructions[k], -5, 5).tolist()] if k < instruction_num else []
        env.step(vel_list = instruction, )
        env.render(show_traj=True, show_goal=True)
        if env.done():
            break

    if env.done():
        print(f'Robot reaches its goal ' + 
            f'with error {np.linalg.norm(env.robot.state[0:env.robot.goal_dim[0]] - env.robot.goal):.3f}. ' + 
            f'The threshold is {env.robot.goal_threshold:3f}.')
        trajectory = env.robot.trajectory
        trajectory.append(env.robot.state)
        length = 0

        for i in range(len(trajectory)-1):
            length += np.linalg.norm(trajectory[i+1][:2] - trajectory[i][:2])
        error = np.abs(answer-length)/answer
        print(f'Trajectory length: {length:.3f}, relative error to your answer: {error:.3f}.')
        if error < env.robot.goal_threshold:
            print(f'Error of the trajectory\'s length accepted with threshold {env.robot.goal_threshold:.3f}.')
            tracker.add_points(20)
        else:
            print(f'Error of the trajectory\'s length to the answer exceeds threshold {env.robot.goal_threshold:.3f}.')
    else:
        print(f'Robot does not reach its goal, ' + 
            f'with error {np.linalg.norm(env.robot.state[0:env.robot.goal_dim[0]] - env.robot.goal):.3f}. ' + 
            f'The threshold is {env.robot.goal_threshold:3f}.')
    
    if options.graphics:
        env.end(show_traj=True, show_goal=True)
    
if __name__ == '__main__':
    main()
