# Date: 2024/09/20

import numpy as np
import matplotlib as mpl
from math import sin, cos, atan2, inf, pi
from ir_sim.world import RobotBase
import copy

class RobotTwoWheel(RobotBase):

	student_id = 3036382909 # Bai Junhao

	robot_type = 'custom'
	appearance = 'circle'
	state_dim  = (3, 1) # the state dimension, x, y, theta(heading direction),
	vel_dim = (2, 1)    # the angular velocity of right and left wheel
	goal_dim = (3, 1)	# the goal dimension, x, y, theta 
	position_dim = (2, 1) # the position dimension, x, y

	def __init__(self, id, state, vel = np.zeros((2, 1)), goal=np.zeros((3, 1)), 
				 step_time = 0.01, **kwargs):
		self.shape  = kwargs.get('shape', [4.6, 1.6, 3, 1.6]) # Only for rectangle shape
		self.radius = kwargs.get('radius', 0.25)
		self.radius_wheel = kwargs.get('radius_wheel', 1)
		self.control_mode = kwargs.get('control_mode', 'auto')
		self.noise_mode   = kwargs.get('noise_mode', 'none')
		self.noise_amplitude = kwargs.get('noise_amplitude', np.c_[[0.005, 0.005, 0.001]])
		super(RobotTwoWheel, self).__init__(id, state, vel, goal, step_time, **kwargs)

	def dynamics(self, state, vel, **kwargs):
		r"""
		Choose dynamics function based on different noise mode.
		"""
		state_, vel_ = copy.deepcopy(state), copy.deepcopy(vel)
		if self.control_mode == 'keyboard':
			vel_ = self.keyboard_to_angular(vel_)

		if self.noise_mode == 'none':
			return self.dynamics_without_noise(state_, vel_, **kwargs)
		elif self.noise_mode == 'linear':
			return self.dynamics_with_linear_noise(state_, vel_, **kwargs)
		else:
			return self.dynamics_with_nonlinear_noise(state_, vel_, **kwargs)
	
	# function to calculate ICR
	def calculate_ICR(self, state, vel):
		omega1, omega2 = vel[0, 0], vel[1, 0]
		r_ICR = (omega2 + omega1) / (omega2 - omega1) * self.radius
		theta = state[2, 0]
		# calculate ICR_x, ICR_y
		ICR_x = state[0, 0] + r_ICR * sin(theta)
		ICR_y = state[1, 0] - r_ICR * cos(theta)
		return ICR_x, ICR_y
  
	def dynamics_without_noise(self, state, vel, **kwargs):
		r"""
		Question 1
		The dynamics of two-wheeled robot, be careful with the defined direction of the robot.
		
		Some parameters that you may use:
		@param dt: delta time
		@param r:  wheel radius
		@param l:  distancce from wheel to center
		@param vel:   2*1 matrix, the angular velocity of the right and left wheels, [omega1, omega2]
		@param state: 3*1 matrix, the state dimension, [x, y, theta (heading direction)]
		
		Return:
		@param next_state: 3*1 matrix, same as state
		"""
		dt = self.step_time
		r  = self.radius_wheel
		l  = self.radius
		
		"*** YOUR CODE STARTS HERE ***"
		# get x, y, theta
		x, y, theta = state[0, 0], state[1, 0], state[2, 0]
		# calculate omega0
		omega0 = (vel[0, 0] - vel[1, 0]) * r / (2 * l)
		# if omega0 is 0, then the robot moves straight forward 
		# then we don't need to calculate ICR
		if omega0 == 0:
			next_state = np.zeros((3, 1))
			next_state[0, 0] = x + vel[0, 0] * cos(theta) * dt
			next_state[1, 0] = y + vel[1, 0] * sin(theta) * dt
			next_state[2, 0] = theta
			return next_state
		else:
			# calculate ICR
			ICR = self.calculate_ICR(state, vel)
			# matrix1 for rotation
			matrix1 = np.array([[cos(omega0 * dt), -sin(omega0 * dt), 0],
								[sin(omega0 * dt), cos(omega0 * dt), 0],
								[0, 0, 1]])
			# matrix2
			matrix2 = np.array([x - ICR[0], y - ICR[1], theta])
			# matrix3
			matrix3 = np.array([ICR[0], ICR[1], omega0 * dt])
			# calculate next_state
			next_state = np.zeros((3, 1))
			result = np.dot(matrix1, matrix2) + matrix3
			next_state[0, 0] = result[0]
			next_state[1, 0] = result[1]
			next_state[2, 0] = result[2]
		"*** YOUR CODE ENDS HERE ***"
		return next_state


	def dynamics_with_linear_noise(self, state, vel, **kwargs):
		r"""
		Question 2(a)
		The dynamics of two-wheeled robot, be careful with the defined direction.
		
		Some parameters that you may use:
		@param dt: delta time
		@param r:  wheel radius
		@param l:  distancce from wheel to center
		@param vel:   2*1 matrix, the angular velocity of the right and left wheels, [omega1, omega2]
		@param state: 3*1 matrix, the state dimension, [x, y, theta (heading direction)]
		@param noise: 3*1 matrix, noises of the additive Gaussian disturbances 
						for the state, [epsilon_x, epsilon_y, epsilon_theta]
		
		Return:
		@param next_state: 3*1 matrix, same as state
		"""
		dt = self.step_time
		r  = self.radius_wheel
		l  = self.radius
		R  = self.noise_amplitude
		noise = np.random.normal(0, R)
		
		"*** YOUR CODE STARTS HERE ***"
		# Before the motion function, we add the Gaussian noise to omega and theta 2(b)
		# pass
	
  		# perform the motion function without noise
		state_without_noise = self.dynamics_without_noise(state, vel)

		# 2(a)
		# After the motion function, we add the Gaussian noise to x, y, and theta
		next_state = np.zeros((3, 1))
		# add noise to x
		next_state[0, 0] = state_without_noise[0, 0] + noise[0]
		# add noise to y
		next_state[1, 0] = state_without_noise[1, 0] + noise[1]
		# add noise to theta
		next_state[2, 0] = state_without_noise[2, 0] + noise[2]

		"*** YOUR CODE ENDS HERE ***"
		
		return next_state

	def dynamics_with_nonlinear_noise(self, state, vel, **kwargs):
		r"""
		Question 2(b)
		The dynamics of two-wheeled robot, be careful with the defined direction.
		
		Some parameters that you may use:
		@param dt: delta time
		@param r:  wheel radius
		@param l:  distancce from wheel to center
		@param vel:   2*1 matrix, the angular velocity of the right and left wheels, [omega1, omega2]
		@param state: 3*1 matrix, the state dimension, [x, y, theta (heading direction)]
		@param noise: 2*1 matrix, noises of the additive Gaussian disturbances 
						for the angular velocity of wheels and heading, [epsilon_omega, epsilon_theta]. 
						Assume that the noises for omega1 and omega2 are the same.
		
		Return:
		@param next_state: 3*1 matrix, same as state
		"""
		dt = self.step_time
		r  = self.radius_wheel
		l  = self.radius
		R  = self.noise_amplitude
		noise = np.random.normal(0, R)
		
		"*** YOUR CODE STARTS HERE ***"
		# Before the motion function, we add the Gaussian noise to omega and theta 2(b)
		# pass
		# get omega1, omega2
		omega1, omega2 = vel[0, 0], vel[1, 0]
		# add noise to omega1 and omega2
		omega1 += noise[0]
		omega2 += noise[0]

		# get theta
		theta = state[2, 0]
		# add noise to theta
		theta += noise[1]
		# change state
		state[2, 0] = theta

		# change vel to omega1 and omega2
		self.vel = np.c_[[omega1, omega2]]

		# perform the motion function without noise
		state_without_noise = self.dynamics_without_noise(state, self.vel)
  
		# form up next_state
		next_state = np.zeros((3, 1))
		next_state[0, 0] = state_without_noise[0, 0]
		next_state[1, 0] = state_without_noise[1, 0]
		next_state[2, 0] = state_without_noise[2, 0]
		
		"*** YOUR CODE ENDS HERE ***"
		
		return next_state


	def policy(self):
		r"""
		Question 3
		A simple policy for steering.

		Some parameters that you may use:
		@param dt: delta time
		@param r:  wheel radius
		@param l:  distancce from wheel to center

		Return:
		@instructions: A list containing instructions of wheels' angular velocities. 
					   Form: [[omega1_t1, omega2_t1], [omega1_t2, omega2_t2], ...]
		@timepoints: A list containing the duration time of each instruction.
					   Form: [t1, t2, ...], then the simulation time is \sum(t1+t2+...).
		@path_length: The shortest trajectory length after calculation by your hand.
		"""

		dt = self.step_time
		r  = self.radius_wheel
		l  = self.radius
		instructions = []
		timepoints = []
		path_length = 0
		
		"*** YOUR CODE STARTS HERE ***"

		# Initial and goal states
		x_start, y_start, theta_start = 0.5, 2.0, np.pi / 2
		x_goal, y_goal, theta_goal = 4.5, 2.0, np.pi / 2
		
		# Calculate the angle and distance
		theta_turn = np.pi / 2
		distance = x_goal - x_start

		# Step 1: Turn right for 90 degrees
		omega1 = -np.pi / 2
		omega2 = +np.pi / 2
		instructions.append([omega1, omega2])
		timepoints.append(theta_turn / abs(omega1-omega2)/2)
		# path_length += l * theta_turn

		# Step 2: Move forward
		omega1 = 2
		omega2 = 2
		instructions.append([omega1, omega2])
		timepoints.append(distance / (omega1)*r)
		path_length += distance
  
		# Step 3: Turn left for 90 degrees	
		omega1 = +np.pi / 2
		omega2 = -np.pi / 2
		instructions.append([omega1, omega2])
		timepoints.append(theta_turn / abs(omega1-omega2)/2)
		
		# TODO: why not add the length of rotation to path_length?
		# path_length += l * theta_turn

		"*** YOUR CODE ENDS HERE ***"
		
		return instructions, timepoints, path_length

	
	def plot_robot(self, ax, robot_color = 'g', goal_color='r', show_goal=True, show_text=False, show_traj=False, show_uncertainty=False, traj_type='-g', fontsize=10, **kwargs):
		x = self.state[0, 0]
		y = self.state[1, 0]

		goal_x = self.goal[0, 0]
		goal_y = self.goal[1, 0]

		robot_circle = mpl.patches.Circle(xy=(x, y), radius = self.radius, color = robot_color)
		robot_circle.set_zorder(3)

		ax.add_patch(robot_circle)
		if show_text: ax.text(x - 0.5, y, 'r'+ str(self.id), fontsize = fontsize, color = 'r')
		self.plot_patch_list.append(robot_circle)

		# arrow
		theta = self.state[2][0]
		arrow = mpl.patches.Arrow(x, y, 0.5*cos(theta), 0.5*sin(theta), width = 0.6)
		arrow.set_zorder(3)
		ax.add_patch(arrow)
		self.plot_patch_list.append(arrow)

		if show_goal:
			goal_circle = mpl.patches.Circle(xy=(goal_x, goal_y), radius = self.radius, color=goal_color, alpha=0.5)
			goal_circle.set_zorder(1)

			ax.add_patch(goal_circle)
			if show_text: ax.text(goal_x + 0.3, goal_y, 'g'+ str(self.id), fontsize = fontsize, color = 'k')
			self.plot_patch_list.append(goal_circle)

		if show_traj:
			x_list = [t[0, 0] for t in self.trajectory]
			y_list = [t[1, 0] for t in self.trajectory]
			self.plot_line_list.append(ax.plot(x_list, y_list, traj_type))


	def keyboard_to_angular(self, vel):
		r"""

		Change the velocity from [v, omega] to [omega1, omega2].
		
		Some parameters that you may use:
		@param r:  wheel radius
		@param l:  distancce from wheel to center
		@param vel: 2*1 matrix, the forward velocity and the rotation velocity [v, omega]

		Return:
		@param vel_new: 2*1 matrix,the angular velocity of right and left wheel, [omega1, omega2]
		"""
		l  = self.radius
		r  = self.radius_wheel
		
		vel_new = np.c_[[vel[0, 0]+vel[1, 0]*l, vel[0, 0]-vel[1, 0]*l]] / r
		return vel_new
	

