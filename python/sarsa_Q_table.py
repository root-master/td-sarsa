import numpy as np
from puddleworld import PuddleWorld 
from random import random

env = PuddleWorld(nd = 2, 
                goal = np.array([1.0,1.0]),
                state_bound = [[0.0,1.0],[0.0,1.0]],
                nA = 4,
                action_list = [[0,1],[0,-1],[1,0],[-1,0]],
                ngrid = [10.0,10.0],
                maxStep = 80) 

def random_init_state(nd):
	# s = np.random.random(nd)
	s = np.zeros(nd)
	for i in range(nd):
		s[i] = np.random.choice(env.discrete_state_vec[i]) 
	return s

def find_index(s,nd):
	idx = np.zeros(nd, dtype=int)
	for i in range(nd):
		idx[i] = ( np.abs(env.discrete_state_vec[i] - s[i]) ).argmin() 
	return idx

def select_action(Q,epsilon):
	# greedy action
	nA = Q.size
	act_index = np.argmax(Q)
	action = np.zeros(nA)
	# explore
	if random() < epsilon:
		act_index = np.random.randint(nA)
	action[act_index] = 1
	return action

x_mesh_number = env.discrete_state_vec[0].size
y_mesh_number = env.discrete_state_vec[1].size
nd = env.nd
nA = env.nA

# initialize Q
Q_table = np.zeros( (x_mesh_number,y_mesh_number,nA) ) 

T = 80 # max steps per episodes
M = 10000 # max episodes


epsilon_min = 0.01
epsilon_max = 0.1
epsilon = epsilon_max

lr_min = 0.01
lr_max = 0.1
lr = lr_max

gamma = 0.99

# Temporal Difference -- SARSA Learning
for e in range(M):
	s = random_init_state(nd)
	s0 = s
	idx = find_index(s,nd)
	Q = np.copy( Q_table[idx[0],idx[1],:] )
	a = select_action(Q,epsilon)
	for t in range(T):
		# check if s0 is not a goal state
		if env.success(s):
			print('Episode: {} s0: {} steps: {}' .format(e, s0, t) )			
			break
		
		sp1, r, done = env.update_state_env_reward(s,a)
		idxp1 = find_index(sp1,nd)
		Qp1 = np.copy( Q_table[idxp1[0],idxp1[1],:] )
		ap1 = select_action(Qp1, epsilon)
		
		if ~done:
			delta = r + gamma * Qp1[np.argmax(ap1)] - Q[np.argmax(a)]
		else:
			delta = r - Q[np.argmax(a)]
		
		target = np.copy(Q)
		target[np.argmax(a)] += lr * delta
		
		Q_table[idx[0],idx[1],:] = np.copy( target )

		# if done:
		# 	epsilon = max(epsilon_min, 0.99 * epsilon)
		# 	lr = max(lr_min, 0.999 * lr )
		# else:
		# 	epsilon = min(epsilon_max, epsilon * 1.01)
		# 	lr = min(lr_min, 1.01 * lr )

		# break if sp1 is terminal
		if done:
			print('Episode: {} s0: {} steps: {}' .format(e, s0, t) )			
			break
		
		s = np.copy(sp1)
		a = np.copy(ap1)
		Q = np.copy(Qp1)
		idx = np.copy(idxp1)


	

