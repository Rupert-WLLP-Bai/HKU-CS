import numpy as np
import matplotlib as mpl
import math
from math import sin, cos, atan2, inf, pi
from collections import namedtuple
from ir_sim.world import RobotBase
from ir_sim.global_param import env_param
from ir_sim.util.util import get_transform, WrapToPi
from ir_sim.util import collision_dectection_geo as cdg 

point_geometry = namedtuple('point', ['x', 'y'])
circle_geometry = namedtuple('circle', ['x', 'y', 'r'])

class RobotRRT(RobotBase):

	robot_type = 'custom'
	appearance = 'circle'
	state_dim  = (2, 1) # the state dimension, x, y, theta(heading direction),
	vel_dim = (2, 1)    # the angular velocity of right and left wheel
	goal_dim = (2, 1)	# the goal dimension, x, y, theta 
	position_dim = (2, 1) # the position dimension, x, y

	def __init__(self, id, state, vel = np.zeros((2, 1)), goal=np.zeros((2, 1)), 
				 vel_min=[-2, -2], vel_max=[2, 2], step_time = 0.01, acce=[inf, inf], 
				 **kwargs):
		r""" FOR SETTING STARTS """
		# self.shape  = kwargs.get('shape', [4.6, 1.6, 3, 1.6]) # Only for rectangle shape
		self.radius = kwargs.get('radius', 0.2)
		super(RobotRRT, self).__init__(id, state, vel, goal, step_time, 
			vel_min=vel_min, vel_max=vel_max, acce=acce, **kwargs)
		r""" FOR SETTING ENDS """

		r""" FOR SIMULATION STARTS """
		self.world_range = kwargs.get('world_range', (np.array([[0], [0]]), np.array([[10], [10]])))
		self.map = self.get_map()
		# self.landmark_map = self.get_landmark_map()
		# self.control_mode = kwargs.get('control_mode', 'auto') # 'auto' or 'policy'. Control the robot by keyboard or defined policy.

		self.s_mode  = kwargs.get('s_mode', 'pre') # 'sim', 'pre'. Plot simulate position or predicted position
		self.s_R = kwargs.get('s_R', np.c_[[0.01, 0.01]]) # Noise amplitude of simulation motion model
		self.s_rng = np.random.default_rng(16)
		r""" FOR SIMULATION ENDS """

		r""" FOR POLICY SETTING STARTS """
		self.p_mode = kwargs.get('p_mode', 'rrt')# 'rrt' (default) or 'rrt_star'. Policy mode
		self.p_max_iter = kwargs.get('p_max_iter', 1000)
		self.p_step_per_iter = kwargs.get('p_step_per_iter', 10)
		self.p_current_iter = 0
		assert(isinstance(self.p_max_iter, int) and self.p_max_iter > 0)
		assert(isinstance(self.p_step_per_iter, int) and self.p_step_per_iter > 0)
		
		self.p_status = 'idle' # 'idle', 'running', 'failed', 'successful'
		self.p_results = None
		self.p_tolerance = kwargs.get('p_tolerance', 2e-1)
		assert(self.p_tolerance > 0)
		
		self.p_expand_dis = kwargs.get('p_expand_dis', 0.4)
		self.p_discrete_dis = kwargs.get('p_discrete_dis', 0.1)
		self.p_safe_dis = kwargs.get('p_safe_dis', 0.01)
		assert(self.p_expand_dis > 0 and self.p_discrete_dis > 0 and self.p_safe_dis >= 0)
		if self.p_mode == 'rrt':
			self.nodes = [[self.init_state, None],]
		elif self.p_mode == 'rrt_star':
			self.nodes = [[self.init_state, None, 0],]
		else:
			raise ValueError('Not supported policy mode, try set p_mode as \'rrt\' or \'rrt_star\'.')
		self.p_rng = np.random.default_rng(16)
		r""" FOR POLICY SETTING ENDS """

	@staticmethod
	def motion_model(current_state, vel, step_time, noise = False, noise_alpha = np.c_[[0.01, 0.01]], seed = np.random.default_rng()):
		if noise:
			vel = vel + seed.normal(0, noise_alpha)

		next_state = current_state + vel * step_time
		return next_state

	def dynamics(self, state, vel, **kwargs):
		next_state = self.motion_model(state, vel, self.step_time, self.s_R, self.s_rng)
		return next_state

	def rrt_sample_free(self):
		# Only consider stable obstacles and no other robot here.
		# For complex condition, please refer to collision_check()
		x_rand = None
		while x_rand is None:
			x_rand = self.p_rng.uniform(self.world_range[0], self.world_range[1])
			for obj in self.map:
				collision_flag = self.collision_check_simple(x_rand, obj, robot_radius = self.radius)
				if collision_flag:
					x_rand = None
					break
		return x_rand

	def rrt_nearest(self, x_target):
		r"""
		Question 1(a)
		Find the nearest nodes to the target node.

		Parameters that you may use:
		@param self.nodes: a rrt/rrt_star tree, a list containing all the nodes
					 Access by (example):
					 node = self.nodes[2]
					 node is a rrt/rrt_star node, which is a list containing:
					 for 'rrt': [state, parent_node_index]
					 for 'rrt_star': [state, parent_node_index, path_cost]
					 Access by (example):
					 node[0]: 2*1 matrix, state
					 node[1]: int, parent_node_index, usage as self.nodes[parent_node_index]
					 node[2]: float, path_cost, the path length from root node to this node
		@param x_target: the state of the target node
		@param min_dist: a variable for you to save the minimum distance for comparision

		Target:
		@param x_nearest: the node state which is closest to x_target.
		@param index_nearest: the node index which is closest to x_target.
		"""
		x_nearest, index_nearest = None, None
		min_dist = float('inf')
		for i, node in enumerate(self.nodes):
			"*** YOUR CODE STARTS HERE ***"
			# Calculate the distance between node and x_target
			state = node[0]
			dist = np.linalg.norm(state-x_target)
			if dist < min_dist:
				# Update min_dist, x_nearest, index_nearest
				min_dist = dist
				x_nearest = state
				index_nearest = i
			"*** YOUR CODE ENDS HERE ***"
			pass
		return x_nearest, index_nearest

	def rrt_stear(self, x_nearest, x_rand):
		r"""
		Question 1(b)
		Calculate the x_new following the direction \arrow{x_nearest, x_rand}.

		Parameters that you may use:
		@param x_nearest: state of a node within the rrt/rrt_star tree
		@param x_rand: state of a node generated by sampling
		@param expand_dis: the expand distance from the nearest node
		
		Target:
		@param x_new: a node state following the direction \arrow{x_nearest, x_rand}
				with length(x_new, x_nearest) = expand distance.
		"""
		expand_dis = self.p_expand_dis
		x_new = None
		"*** YOUR CODE STARTS HERE ***"
		direction = x_rand - x_nearest
		distance = np.linalg.norm(direction)
		direction = direction / distance
		x_new = x_nearest + direction * self.p_expand_dis

		"*** YOUR CODE ENDS HERE ***"
		return x_new
	
	def rrt_collision_free(self, x1, x2):
		r"""
		Return:
		True if the segment between the two states of two
    	given points (x1, x2) is colliding the obstacles
		"""
		length = np.linalg.norm((x1-x2))
		vectx = (x1[0, 0] - x2[0, 0])/length
		vecty = (x1[1, 0] - x2[1, 0])/length

		dx_discrete = np.linspace(0.0, length, int(length/self.p_discrete_dis)+1)
		for dx in dx_discrete:
			x_alpha = x2 + np.array([[vectx*dx],[vecty*dx]])
			for obj in self.map:
				collision_flag = self.collision_check_simple(x_alpha, obj, robot_radius = self.radius+self.p_safe_dis)
				if collision_flag:
					return False
		return True

	def rrt_reach_goal(self, x):
		return np.linalg.norm(self.goal-x) < self.p_tolerance

	def rrt_generate_results(self):
		node = self.nodes[-1]
		self.p_results = [node[0]]
		while node[1] is not None:
			node =  self.nodes[node[1]]
			self.p_results.append(node[0])
		self.p_results.reverse()
		return

	def rrt_near(self, x_target):
		r"""
		Question 2(a)
		Find nodes within the rrt/rrt_star tree 
		whose distance to the target node is within radius r.

		Parameters that you may use:
		@param self.nodes: a rrt/rrt_star tree, a list containing all the nodes
					 Access by (example):
					 node = self.nodes[2]
					 node is a rrt/rrt_star node, which is a list containing:
					 for 'rrt': [state, parent_node_index]
					 for 'rrt_star': [state, parent_node_index, path_cost]
					 Access by (example):
					 node[0]: 2*1 matrix, state
					 node[1]: int, parent_node_index, usage as self.nodes[parent_node_index]
					 node[2]: float, path_cost, the path length from root node to this node
		@param x_target: the state of the target node
		@param eta: a constant described within the algorithm
		@param gamma: a constant described within the algorithm
		@param d: dimension of the world

		Target:
		@param node_near_list: a list containing nodes (NOT states of the nodes)
							   and their index  within the rrt/rrt_star tree. 
							   These nodes' distances to x_target are within radius r,
							   where r is described within the algorithm.
							   Form: [[node1, index1], [node2, index2]]
		"""
		gamma = 5
		eta = 0.6*gamma
		d = 2
		node_near_list = []
		"*** YOUR CODE STARTS HERE ***"
		# Calculate r 
		r = min(gamma * (np.log(len(self.nodes)) / len(self.nodes)) ** (1 / d), eta)

		# Loop within tree to find all satisfiable node
		for i, node in enumerate(self.nodes):
			state = node[0]
			dist = np.linalg.norm(state-x_target)
			if dist < r:
				node_near_list.append([node, i])

		"*** YOUR CODE ENDS HERE ***"

		return node_near_list

	def rrt_cost(self, x1, x2):
		r"""
		@param x1: the state of one node
		@param x2: the state of another node

		Return:
		The cost (distance here) between two nodes.
		"""
		return np.linalg.norm(x1-x2)

	def rrt(self):
		r"""
		Question 1(c)
		Finish the part of rrt algorithm.
		AND
		Question 2(b)
		Finish the part of rrt_star algorithm.

		Parameters that you may use:
		@param self.nodes: a rrt/rrt_star tree, a list containing all the nodes
					 Access by (example):
					 node = self.nodes[2]
					 node is a rrt/rrt_star node, which is a list containing:
					 for 'rrt': [state, parent_node_index]
					 for 'rrt_star': [state, parent_node_index, path_cost]
					 Access by (example):
					 node[0]: 2*1 matrix, state
					 node[1]: int, parent_node_index, usage as self.nodes[parent_node_index]
					 node[2]: float, path_cost, the path length from root node to this node
		@param x_nearest, index_nearest
		@param x_new

		Function:
		@func self.rrt_near
		@func self.rrt_collision_free
		@func self.rrt_cost

		Target:
		Finish the part of rrt algorithm.
		AND
		Finish the part of rrt_star algorithm.
		"""
		for i in range(self.p_step_per_iter):
			x_rand = self.rrt_sample_free()
			x_nearest, index_nearest = self.rrt_nearest(x_rand)
			x_new = self.rrt_stear(x_nearest, x_rand)
			if self.rrt_collision_free(x_nearest, x_new):
				if self.p_mode == 'rrt':
					"*** YOUR CODE STARTS HERE ***"
					# Here Question 1(c) starts
					
					# Line 7, save the new node to our list
					self.nodes.append([x_new, index_nearest])

					"*** YOUR CODE ENDS HERE ***"

				elif self.p_mode == 'rrt_star':
					"*** YOUR CODE STARTS HERE ***"
					# Here Question 2(b) starts

					# Line 7, find nodes near new node with state x_new,
					# get a list saving these nodes and their indexes
					node_near_list = self.rrt_near(x_new)


					# Line 9-12, loop from the list, find node with
					# the minimum path cost, get its index as well
					xmin = index_nearest
					cmin = self.rrt_cost(x_nearest, x_new) + self.nodes[index_nearest][2]  # Total cost for x_new
					
					for node, idx in node_near_list:
						if self.rrt_collision_free(node[0], x_new):
							cost = self.rrt_cost(node[0], x_new) + node[2]
							if cost < cmin:
								xmin = idx
								cmin = cost



					# Line 8 and 13, save the new node to our rrt_star tree
					# with form [state, parent_node_index, path_cost]
					self.nodes.append([x_new, xmin, cmin])

					# Line 14-16, loop from the list, check whether these nodes
					# can get a shorter path cost when they connect to our new node
					# If so, update their connection and the cost.
					for node, idx in node_near_list:
						if self.rrt_collision_free(x_new, node[0]):
							new_cost = cmin + self.rrt_cost(x_new, node[0])
							if new_cost < node[2]:
								node[1] = len(self.nodes) - 1  # Update parent to x_new
								node[2] = new_cost             # Update the cost

					"*** YOUR CODE ENDS HERE ***"

			if self.rrt_reach_goal(x_new):
				self.p_status = 'successful'
				print(f'{self.p_mode} succeeds in finding a policy.') 
				self.rrt_generate_results()
				return

		self.p_current_iter += 1
		if self.p_current_iter >= self.p_max_iter:
			self.p_status == 'failed'
			print(f'{self.p_mode} fails to find a policy.') 
		return

	def p_step(self):
		self.rrt()

	def p_start(self):
		self.p_status = 'running'

	def p_reset(self):
		self.p_status = 'idle'
		self.p_results = None
		self.p_current_iter = 0

	def collision_check_simple(self, state, obj, robot_radius):
		# Only consider stable obstacles, no other robot, and circle robot.
		# For complex condition, please refer to collision_check_state()

		collision_flag = False

		robot_circle = circle_geometry(state[0, 0], state[1, 0], robot_radius)  
		if obj.appearance == 'circle':
			obj_circle = circle_geometry(obj.center[0, 0], obj.center[1, 0], obj.radius)  

			collision_flag, _ =  cdg.collision_cir_cir(robot_circle, obj_circle)
			if collision_flag: return collision_flag

		if obj.appearance == 'polygon' or obj.appearance == 'rectangle':
			obj_poly = [ point_geometry(v[0], v[1]) for v in obj.vertex.T]

			collision_flag, _ = cdg.collision_cir_poly(robot_circle, obj_poly)
			if collision_flag: return collision_flag

		return collision_flag

	def get_map(self):
		return env_param.obstacle_list.copy()

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
		# theta = self.state[2, 0]

		robot_circle = mpl.patches.Circle(xy=(x, y), radius = self.radius, color = robot_color, alpha = 0.5)
		robot_circle.set_zorder(3) # the bigger, the upper
		ax.add_patch(robot_circle)
		if show_text: ax.text(x - 0.5, y, 'r'+ str(self.id), fontsize = fontsize, color = 'r')
		self.plot_patch_list.append(robot_circle)

		# # arrow
		# arrow = mpl.patches.Arrow(x, y, 0.5*cos(theta), 0.5*sin(theta), width = 0.6)
		# arrow.set_zorder(3)
		# ax.add_patch(arrow)
		# self.plot_patch_list.append(arrow)

		# Plot rrt/rrt_star tree
		xP, yP = [], []
		for node in reversed(self.nodes):
			if node[1] is not None:
				xP = [node[0][0, 0], self.nodes[node[1]][0][0, 0]]
				yP = [node[0][1, 0], self.nodes[node[1]][0][1, 0]]
				self.plot_line_list.append(ax.plot(xP, yP, color='y', marker='.', zorder=4))
		
		# Plot found trajectory if it exists
		if self.p_results is not None:
			xP, yP = [], []
			for state in self.p_results:
				xP.append(state[0, 0])
				yP.append(state[1, 0])
			self.plot_line_list.append(ax.plot(xP, yP, color='r', marker='.', zorder=5))

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

