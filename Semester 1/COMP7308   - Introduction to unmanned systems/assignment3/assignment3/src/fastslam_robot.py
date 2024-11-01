import numpy as np
import matplotlib as mpl
import math
from math import sin, cos, atan2, inf, pi
from ir_sim.world import RobotBase
from ir_sim.global_param import env_param
from ir_sim.util.util import get_transform, WrapToPi
import copy

class Particle:
    def __init__(self, state, n_landmark, n_particle):
        # particle weight
        self.weight = 1.0 / n_particle
        # particle state
        self.state = state

        landmarks = []
        for i in range(n_landmark):
            landmark = dict()

            #initialize the landmark mean and covariance 
            landmark['mean'] = np.zeros((2, 1))
            landmark['std'] = np.zeros((2, 2))
            landmark['observed'] = False

            landmarks.append(landmark)
        # landmarks
        self.landmarks = landmarks

class RobotFastSLAM(RobotBase):

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
		super(RobotFastSLAM, self).__init__(id, state, vel, goal, step_time, **kwargs)
		r""" FOR SETTING ENDS """

		r""" FOR SIMULATION STARTS """
		self.landmark_map = self.get_landmark_map()
		# self.control_mode = kwargs.get('control_mode', 'auto') # 'auto' or 'policy'. Control the robot by keyboard or defined policy.

		self.s_mode  = kwargs.get('s_mode', 'pre') # 'sim', 'pre'. Plot simulate position or predicted position
		self.s_R = kwargs.get('s_R', np.array([0.02, 0, 0.002, 0.005])) # Noise amplitude of simulation motion model
		self.s_rng = np.random.default_rng(16)
		r""" FOR SIMULATION ENDS """

		r""" FOR FASTSLAM SETTING STARTS """
		self.e_state = {'mean': self.state, 'std': np.diag([1, 1])}

		self.e_trajectory = [self.state]
		self.e_mode  = kwargs.get('e_mode', 'all') # Estimation mode, all (default), no_resample, no_measure_resample
		self.e_R     = kwargs.get('e_R', np.array([0.04, 0.01, 0.01, 0.04])) # Noise amplitude of ekf estimation motion model
		self.e_Q     = kwargs.get('e_Q', np.diag([0.4, 0.2])) # Noise amplitude of ekf estimation measurement model
		self.e_rng   = np.random.default_rng(16)
		
		self.e_map = [{'mean': np.zeros((2, 1)), 'observed': False} for _ in range(len(self.landmark_map))]
		self.num_particle = kwargs.get('num_particle', 100)
		self.particles = [Particle(self.state, len(self.landmark_map), self.num_particle) for _ in range(self.num_particle)]
		self.resample_rng = np.random.default_rng(16)
		r""" FOR FASTSLAM SETTING ENDS """

	@staticmethod
	def motion_model(current_state, vel, step_time, noise_alpha = np.array([0.03, 0, 0, 0.03]), seed = np.random.default_rng()):
		std_linear = np.sqrt(noise_alpha[0] * (vel[0, 0] ** 2) + noise_alpha[1] * (vel[1, 0] ** 2))
		std_angular = np.sqrt(noise_alpha[2] * (vel[0, 0] ** 2) + noise_alpha[3] * (vel[1, 0] ** 2))
		real_vel = vel + seed.normal([[0], [0]], scale = [[std_linear], [std_angular]])

		coefficient_vel = np.zeros((3, 2))
		coefficient_vel[0, 0] = cos(current_state[2, 0])
		coefficient_vel[1, 0] = sin(current_state[2, 0])
		coefficient_vel[2, 1] = 1

		next_state = current_state + coefficient_vel @ real_vel * step_time
		next_state[2, 0] = WrapToPi(next_state[2, 0])
		return next_state

	def dynamics(self, state, vel, **kwargs):
		next_state = self.motion_model(state, vel, self.step_time, self.s_R, self.s_rng)
		return next_state

	def fastslam_particles_motion_predict(self, vel, **kwargs):
		r"""
		Question 1
		Predict the state of particles.

		Parameters that you may use:
		@param dt    : delta time
		@param vel   : 2*1 matrix, the forward velocity and the rotation velocity [v, omega]
		@param R_t   : 4*1 array,  the assumed noise amplitude for dynamics.
		@param seed  : a numpy Random Number Generator (RNG) for prediction.
		@param self.particles: a list containing all the particles, see below.
		@param particle: a class with attributes: weight, state, landmarks.
						 Access by (example):
						 particle.weight: float.
						 particle.state:  3*1 matrix, [x, y, theta].
						 particle.landmarks: list.
						 For the attibute 'landmarks', it is a list saving position information of all 
						 the landmarks. Each element is a dict, keys are 'mean', 'std', 'observed'.
						 Access by (example):
						 particle.landmarks[0]['mean']: 2*1 matrix, [x, y].
						 particle.landmarks[0]['std']: 2*2 matrix for covariance.
						 particle.landmarks[0]['observed']: boolen.
		
		Goal:
		Follow what we did in self.dynamics(), simply use self.motion_model() 
		(see above) to update the state of every particles.
		"""
		dt = self.step_time
		R_t  = self.e_R
		seed = self.e_rng
		for particle in self.particles:
			"*** YOUR CODE STARTS HERE ***"

			

			"*** YOUR CODE ENDS HERE ***"
			pass
		return

	def fastslam_particles_measurement_update(self, **kwargs):
		r"""
		Question 2
		Update the state of the robot using bearing and range measurements.
		
		Some parameters that you may use:
		@param dt:  delta time
		@param Q_t: 2*2 matrix, the assumed noise amplitude for measurement, usually diagonal.
		@param self.particles: a list containing all the particles, see below.
		@param particle: a class with attributes: weight, state, landmarks.
						 Access by (example):
						 particle.weight: float.
						 particle.state:  3*1 matrix, [x, y, theta].
						 particle.landmarks: list.
						 For the attibute 'landmarks', it is a list saving position information of all 
						 the landmarks. Each element is a dict, keys are 'mean', 'std', 'observed'.
						 Access by (example):
						 particle.landmarks[0]['mean']: 2*1 matrix, [x, y].
						 particle.landmarks[0]['std']: 2*2 matrix for covariance.
						 particle.landmarks[0]['observed']: boolen.
		@param lm_measurements : a list containing all the landmarks detected by sensor. 
							Each element is a dict, keys are 'id', 'range', 'angle'.
							Access by lm_measurements[0]['id'] (example).

		Goal:
		Complete measurement part of fastslam algorithm.
		"""
		dt = self.step_time
		lm_measurements = self.get_landmarks()
		Q_t = self.e_Q
		
		# Update landmarks and calculate weight for each particle
		for particle in self.particles:
			for lm in lm_measurements:
				lm_id = lm['id'] 

				# New landmark
				if not particle.landmarks[lm_id]['observed']:
					# Initialise mean and covariance of landmark lm_id
					"*** YOUR CODE STARTS HERE ***"
					# Initialise mean, at line 7
					


					# Calculate Jacobian related to landmark, at line 8
					


					# Initialise corvariance, at line 9
					


					# Remain default importance weight (skip), at line 10
					"*** YOUR CODE ENDS HERE ***"
					particle.landmarks[lm_id]['observed'] = True
					pass

				# Known landmark
				else:
					# Update mean and covariance of landmark lm_id
					# and update importance weight
					"*** YOUR CODE STARTS HERE ***"
					# Measurement prediction, at line 12
					


					# Calculate Jacobian, at line 13
					

					# Measurement covariance, at line 14


					# Calculate Kalman gain, at line 15


					# Update mean, at line 16


					# Update covariance, at line 17
					

					# Calculate importance factor w, at line 18
					w = 1 # Rewrite this line or update w after
					

					"*** YOUR CODE ENDS HERE ***"
					particle.weight *= w
					pass
		
		return

	def fastslam_particles_normalise_weight(self):
		# Normalise weight
		sum_of_weight = sum([p.weight for p in self.particles])

		try:
			for i in range(self.num_particle):
				self.particles[i].weight /= sum_of_weight
		except ZeroDivisionError:
			for i in range(self.num_particle):
				self.particles[i].weight = 1.0 / self.num_particle
		return

	def fastslam_particles_resample(self):
		r"""
		Question 3
		Resample particles.
		
		Some parameters that you may use:
		@param self.num_particle: the number of particles
		@param self.particles: a list containing all the particles. For each particle, 
						 it is a class with attributes: weight, state, landmarks.
						 Access by (example):
						 particle.weight: float.
						 particle.state:  3*1 matrix, [x, y, theta].
						 particle.landmarks: list.
						 For the attibute 'landmarks', it is a list saving position information of all 
						 the landmarks. Each element is a dict, keys are 'mean', 'std', 'observed'.
						 Access by (example):
						 self.particles[0].landmarks[0]['mean']: 2*1 matrix, [x, y].
						 self.particles[0].landmarks[0]['std']: 2*2 matrix.
						 self.particles[0].landmarks[0]['observed']: boolen.
		
		Goal:
		Complete resampling part using the low variance sampler.
		"""
		
		requirement = False
		"*** YOUR CODE STARTS HERE ***"
		# Here you should:
		# Check whether the resampling requirement is met,
		# Assign the result to 'requirement'.
		

		"*** YOUR CODE ENDS HERE ***"

		if requirement:
			# Here we make preparations for sampleing.
			new_particles = []					 # new particles set, at line 2
			step = 1.0 / self.num_particle		 # step between two samples
			r = self.resample_rng.uniform(0, step) # random start of first target position, at line 3
			c = self.particles[0].weight 		 # current accumulated weight position, at line 4
			i = 0 								 # current particle's id, at line 5
			
			for j in range(self.num_particle):
				"*** YOUR CODE STARTS HERE ***"
				# Here you should:
				# generate new particles using the low variance sampler
				# and add it to 'new_particles'.

				# Find the id of the new particle, at line 7-10
				


				# At line 12
				# Copy the old particle and add it to 'new_particles'
				# using copy.deepcopy(old_particle) (example).
				# You should change the weight to the default weight.
				

				
				"*** YOUR CODE ENDS HERE ***"
				pass
			
			if len(new_particles) > 0:
				self.particles = new_particles
		return

	def fastslam_post_process(self):
		self.e_state['mean'] = np.sum([p.weight * p.state for p in self.particles], axis=0)
		self.e_trajectory.append(self.e_state['mean'])
		for i in range(len(self.landmark_map)):
			if self.e_map[i]['observed']:
				self.e_map[i]['mean'] = np.sum([p.weight * p.landmarks[i]['mean'] for p in self.particles], axis=0)
			elif self.particles[0].landmarks[i]['observed']:
				self.e_map[i]['mean'] = np.sum([p.weight * p.landmarks[i]['mean'] for p in self.particles], axis=0)
				self.e_map[i]['observed'] = True
		return

	def post_process(self):
		self.fastslam(self.vel)
		return

	def fastslam(self, vel):
		if self.s_mode == 'pre':
			if self.e_mode == 'all':
				self.fastslam_particles_motion_predict(vel)
				self.fastslam_particles_measurement_update()
				self.fastslam_particles_normalise_weight()
				self.fastslam_particles_resample()
				self.fastslam_post_process()
			elif self.e_mode == 'no_resample':
				self.fastslam_particles_motion_predict(vel)
				self.fastslam_particles_measurement_update()
				self.fastslam_particles_normalise_weight()
				self.fastslam_post_process()
			elif self.e_mode == 'no_measure_resample':
				self.fastslam_particles_motion_predict(vel)
				self.fastslam_post_process()
			else:
				raise ValueError('Not supported e_mode. Try \'all (default)\', \'no_resample\', \'no_measure_resample\' for estimation mode.')
		elif self.s_mode == 'sim':
			pass
		else:
			raise ValueError('Not supported s_mode. Try \'sim\', \'pre\' for simulation mode.')

	def get_landmark_map(self, ):
		env_map = env_param.obstacle_list.copy()
		landmark_map = dict()
		for obstacle in env_map:
			if obstacle.landmark:
				landmark_map[obstacle.id] = obstacle.center[0:2]
		return landmark_map

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

			# Plot estimated robot position
			e_robot_circle = mpl.patches.Circle(xy=(x, y), radius = self.radius, color = 'y', alpha = 0.7)
			e_robot_circle.set_zorder(3)
			ax.add_patch(e_robot_circle)
			self.plot_patch_list.append(e_robot_circle)

			# Plot particles
			xP, yP = [], []
			for i in range(self.num_particle):
				xP.append(self.particles[i].state[0, 0])
				yP.append(self.particles[i].state[1, 0])
			self.plot_line_list.append(ax.plot(xP, yP, ".b"))

			# Plot estimated landmarks position
			xEst, yEst = [], []
			for i in range(len(self.e_map)):
				if self.e_map[i]['observed']:
					xEst.append(self.e_map[i]['mean'][0, 0])
					yEst.append(self.e_map[i]['mean'][1, 0])
			if len(xEst) > 0:
				self.plot_line_list.append(ax.plot(xEst, yEst, "xk"))

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

