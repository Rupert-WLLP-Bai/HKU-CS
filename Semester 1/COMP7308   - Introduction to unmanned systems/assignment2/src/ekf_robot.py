import numpy as np
import matplotlib as mpl
from math import sin, cos, atan2, inf, pi
from ir_sim.world import RobotBase
from ir_sim.global_param import env_param
from ir_sim.util.util import get_transform, WrapToPi

class RobotEKF(RobotBase):

	student_id = None

	
	robot_type = 'custom'
	appearance = 'circle'
	state_dim  = (3, 1) # the state dimension, x, y, theta(heading direction),
	vel_dim = (2, 1)    # the angular velocity of right and left wheel
	goal_dim = (3, 1)	# the goal dimension, x, y, theta 
	position_dim = (2, 1) # the position dimension, x, y

	def __init__(self, id, state, vel = np.zeros((2, 1)), goal=np.zeros((3, 1)), 
				 step_time = 0.01, **kwargs):
		r""" FOR SETTING STARTS """
		self.shape  = kwargs.get('shape', [4.6, 1.6, 3, 1.6]) # Only for rectangle shape
		self.radius = kwargs.get('radius', 0.25)
		super(RobotEKF, self).__init__(id, state, vel, goal, step_time, **kwargs)
		r""" FOR SETTING ENDS """

		r""" FOR SIMULATION STARTS """
		self.landmark_map = self.get_landmark_map()
		# self.control_mode = kwargs.get('control_mode', 'auto') # 'auto' or 'policy'. Control the robot by keyboard or defined policy.

		self.s_mode  = kwargs.get('s_mode', 'sim') # 'sim', 'pre'. Plot simulate position or predicted position
		# self.s_mode   = kwargs.get('s_mode', 'none') # 'none', 'linear', 'nonlinear'. Simulation motion model with different noise mode
		self.s_R = kwargs.get('s_R', np.c_[[0.02, 0.02, 0.01]]) # Noise amplitude of simulation motion model
		r""" FOR SIMULATION ENDS """

		r""" FOR EKF ESTIMATION STARTS """
		self.e_state = {'mean': self.state, 'std': np.diag([1, 1, 1])}

		self.e_trajectory = []
		self.e_mode  = kwargs.get('e_mode', 'no_measure') # 'no_measure', 'no_bearing', 'bearing'. Estimation mode
		self.e_R     = kwargs.get('e_R', np.diag([0.02, 0.02, 0.01])) # Noise amplitude of ekf estimation motion model
		self.e_Q     = kwargs.get('e_Q', 0.2) # Noise amplitude of ekf estimation measurement model
		r""" FOR EKF ESTIMATION ENDS """

	def dynamics(self, state, vel, **kwargs):
		r"""
		Question 1
		The dynamics of two-wheeled robot for SIMULATION.

		NOTE that this function will be utilised in q3 and q4, 
		but we will not check the correction of sigma_bar. 
		So if you meet any problems afterwards, please check the
		calculation of sigma_bar here.

		Some parameters that you may use:
		@param dt:	  delta time
		@param vel  : 2*1 matrix, the forward velocity and the rotation velocity [v, omega]
		@param state: 3*1 matrix, the state dimension, [x, y, theta]
		@param noise: 3*1 matrix, noises of the additive Gaussian disturbances 
						for the state, [epsilon_x, epsilon_y, epsilon_theta]

		Return:
		@param next_state: 3*1 matrix, same as state
		"""
		dt     = self.step_time
		R_hat  = self.s_R
		noise  = np.random.normal(0, R_hat)

		"*** YOUR CODE STARTS HERE ***"
		


		"*** YOUR CODE ENDS HERE ***"
		return next_state

	
	def ekf_predict(self, vel, **kwargs):
		r"""
		Question 2
		Predict the state of the robot.

		Some parameters that you may use:
		@param dt: delta time
		@param vel   : 2*1 matrix, the forward velocity and the rotation velocity [v, omega]
		@param mu    : 3*1 matrix, the mean of the belief distribution, [x, y, theta]
		@param sigma : 3*3 matrix, the covariance matrix of belief distribution.
		@param R     : 3*3 matrix, the assumed noise amplitude for dynamics, usually diagnal.

		Goal:
		@param mu_bar    : 3*1 matrix, the mean at the next time, as in EKF algorithm
		@param sigma_bar : 3*3 matrix, the covariance matrix at the next time, as in EKF algorithm
		"""
		dt = self.step_time
		R  = self.e_R
		mu = self.e_state['mean']
		sigma = self.e_state['std']
		
		"*** YOUR CODE STARTS HERE ***"
		# Compute the Jacobian of g called G with respect to the state
		


		# Compute the mean 
		


		# Compute the covariance matrix
		

		"*** YOUR CODE ENDS HERE ***"
		self.e_state['mean'] = mu_bar
		self.e_state['std'] = sigma_bar

	def ekf_correct_no_bearing(self, **kwargs):
		r"""
		Question 3
		Update the state of the robot using range measurement.
		
		NOTE that ekf_predict() will be utilised here in q3 and q4, 
		If you meet any problems, you may need to check 
		the calculation of sigma_bar in q2.

		Some parameters that you may use:
		@param dt: delta time
		@param mu_bar    : 3*1 matrix, the mean of the belief distribution, [x, y, theta]
		@param sigma_bar : 3*3 matrix, the covariance matrix of belief distribution.
		@param Q         : 1*1 matrix, the assumed noise amplitude for measurement, usually diagnal.
		@param lm_map    : a dict containing positions [x, y] of all the landmarks.
							Access by lm_map[landmark_id]
		@param lm_measurements : a list containing all the landmarks detected by sensor. 
							Each element is a dict, keys are 'id', 'range', 'angle'.
							Access by lm_measurements[0]['id'] (example).

		Goal:
		@param mu    : 3*1 matrix, the updated mean, as in EKF algorithm
		@param sigma : 3*3 matrix, the updated covariance matrix, as in EKF algorithm
		"""
		dt = self.step_time
		mu_bar = self.e_state['mean']
		sigma_bar = self.e_state['std']

		lm_map   = self.landmark_map
		lm_measurements = self.get_landmarks()
		Q = np.array([[self.e_Q]])

		for lm in lm_measurements:
			# Update mu_bar and sigma_bar with each measurement individually,
			"*** YOUR CODE STARTS HERE ***"
			
			# Calculate the expected measurement vector
			


			# Compute H
			


			# Gain of Kalman
			


			# Kalman correction for mean_bar and covariance_bar
			


			"*** YOUR CODE ENDS HERE ***"
			pass
		mu    = mu_bar
		sigma = sigma_bar
		self.e_state['mean'] = mu
		self.e_state['std'] = sigma

	def ekf_correct_with_bearing(self, **kwargs):
		r"""
		Question 4
		Update the state of the robot using range and bearing measurement.
		
		NOTE that ekf_predict() will be utilised here in q3 and q4, 
		If you meet any problems, you may need to check 
		the calculation of sigma_bar in q2.

		Some parameters that you may use:
		@param dt: delta time
		@param mu_bar    : 3*1 matrix, the mean of the belief distribution, [x, y, theta]
		@param sigma_bar : 3*3 matrix, the covariance matrix of belief distribution.
		@param Q         : 2*2 matrix, the assumed noise amplitude for measurement, usually diagnal.
		@param lm_map    : a dict containing positions [x, y] of all the landmarks.
							Access by lm_map[landmark_id]
		@param lm_measurements : a list containing all the landmarks detected by sensor. 
							Each element is a dict, keys are 'id', 'range', 'angle'.
							Access by lm_measurements[0]['id'] (example).
		
		Goal:
		@param mu    : 3*1 matrix, the updated mean, as in EKF algorithm
		@param sigma : 3*3 matrix, the updated covariance matrix, as in EKF algorithm
		"""
		dt = self.step_time
		mu_bar = self.e_state['mean']
		sigma_bar = self.e_state['std']

		lm_map    = self.landmark_map
		lm_measurements = self.get_landmarks()
		Q = np.diag([self.e_Q, self.e_Q])
		
		for lm in lm_measurements:
			# Update mu_bar and sigma_bar with each measurement individually,
			"*** YOUR CODE STARTS HERE ***"

			# Calculate the expected measurement vector
			


			# Compute H
			


			# Gain of Kalman
			


			# Kalman correction for mean_bar and covariance_bar
			


			"*** YOUR CODE ENDS HERE ***"
			pass
		mu = mu_bar
		sigma = sigma_bar
		self.e_state['mean'] = mu
		self.e_state['std'] = sigma
	
	
	def get_landmark_map(self, ):
		env_map = env_param.obstacle_list.copy()
		landmark_map = dict()
		for obstacle in env_map:
			if obstacle.landmark:
				landmark_map[obstacle.id] = obstacle.center[0:2]
		return landmark_map

	def post_process(self):
		self.ekf(self.vel)

	def ekf(self, vel):
		if self.s_mode == 'pre':
			if self.e_mode == 'no_measure':
				self.ekf_predict(vel)
				self.e_trajectory.append(self.e_state['mean'])
			elif self.e_mode == 'no_bearing':
				self.ekf_predict(vel)
				self.ekf_correct_no_bearing()
				self.e_trajectory.append(self.e_state['mean'])
			elif self.e_mode == 'bearing':
				self.ekf_predict(vel)
				self.ekf_correct_with_bearing()
				self.e_trajectory.append(self.e_state['mean'])
			else:
				raise ValueError('Not supported e_mode. Try \'no_measure\', \'no_bearing\', \'bearing\' for estimation mode.')
		elif self.s_mode == 'sim':
			pass
		else:
			raise ValueError('Not supported s_mode. Try \'sim\', \'pre\' for simulation mode.')

	def plot_robot(self, ax, robot_color = 'g', goal_color='r', 
					show_goal=True, show_text=False, show_uncertainty=False, 
					show_traj=False, traj_type='-g', fontsize=10, **kwargs):
		x = self.state[0, 0]
		y = self.state[1, 0]
		theta = self.state[2, 0]

		robot_circle = mpl.patches.Circle(xy=(x, y), radius = self.radius, color = robot_color, alpha = 0.5)
		robot_circle.set_zorder(3)
		ax.add_patch(robot_circle)
		if show_text: ax.text(x - 0.5, y, 'r'+ str(self.id), fontsize = fontsize, color = 'r')
		self.plot_patch_list.append(robot_circle)

		# arrow
		arrow = mpl.patches.Arrow(x, y, 0.5*cos(theta), 0.5*sin(theta), width = 0.6)
		arrow.set_zorder(3)
		ax.add_patch(arrow)
		self.plot_patch_list.append(arrow)

		if self.s_mode == 'pre':
			x = self.e_state['mean'][0, 0]
			y = self.e_state['mean'][1, 0]
			theta = self.e_state['mean'][2, 0]

			e_robot_circle = mpl.patches.Circle(xy=(x, y), radius = self.radius, color = 'y', alpha = 0.7)
			e_robot_circle.set_zorder(3)
			ax.add_patch(e_robot_circle)
			self.plot_patch_list.append(e_robot_circle)

			# calculate and plot covariance ellipse
			covariance = self.e_state['std'][:2, :2]
			eigenvals, eigenvecs = np.linalg.eig(covariance)

			# get largest eigenvalue and eigenvector
			max_ind = np.argmax(eigenvals)
			max_eigvec = eigenvecs[:,max_ind]
			max_eigval = eigenvals[max_ind]

			# get smallest eigenvalue and eigenvector
			min_ind = 0
			if max_ind == 0:
			    min_ind = 1

			min_eigvec = eigenvecs[:,min_ind]
			min_eigval = eigenvals[min_ind]

			# chi-square value for sigma confidence interval
			chisquare_scale = 2.2789  

			scale = 2
			# calculate width and height of confidence ellipse
			width = 2 * np.sqrt(chisquare_scale*max_eigval) * scale
			height = 2 * np.sqrt(chisquare_scale*min_eigval) * scale
			angle = np.arctan2(max_eigvec[1],max_eigvec[0])

			# generate covariance ellipse
			ellipse = mpl.patches.Ellipse(xy=[x, y], 
				width=width, height=height, 
				angle=angle/np.pi*180, alpha = 0.25)

			ellipse.set_zorder(1)
			ax.add_patch(ellipse)
			self.plot_patch_list.append(ellipse)

		if show_goal:
			goal_x = self.goal[0, 0]
			goal_y = self.goal[1, 0]

			goal_circle = mpl.patches.Circle(xy=(goal_x, goal_y), radius = self.radius, color=goal_color, alpha=0.5)
			goal_circle.set_zorder(1)

			ax.add_patch(goal_circle)
			if show_text: ax.text(goal_x + 0.3, goal_y, 'g'+ str(self.id), fontsize = fontsize, color = 'k')
			self.plot_patch_list.append(goal_circle)

		if show_traj:
			x_list = [t[0, 0] for t in self.trajectory]
			y_list = [t[1, 0] for t in self.trajectory]
			self.plot_line_list.append(ax.plot(x_list, y_list, traj_type))
			
			if self.s_mode == 'pre':
				x_list = [t[0, 0] for t in self.e_trajectory]
				y_list = [t[1, 0] for t in self.e_trajectory]
				self.plot_line_list.append(ax.plot(x_list, y_list, '-y'))

