import math
from math import floor
import random

import numpy as np

FORWARD_ACCEL = 1
BACKWARD_ACCEL = 0


class QLearningAgent:
    def __init__(self, lr, gamma, track_length, epsilon=0, policy='greedy', x_bins=10, x_dot_bins=10, theta_bins=10, theta_dot_bins=10,
                 min_x_dot=-1.5, max_x_dot=1.5, min_theta_dot=-1.5, max_theta_dot=1.5):
        """
        A function for initializing your agent
        :param lr: learning rate
        :param gamma: discount factor
        :param track_length: how far the ends of the track are from the origin.
            e.g., while track_length is 2.4,
            the x-coordinate of the left end of the track is -2.4,
            the x-coordinate of the right end of the track is 2.4,
            and x-coordinate of the the cart is 0 initially.
        :param epsilon: epsilon for the mixed policy
        :param policy: can be 'greedy' or 'mixed'
        """
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.track_length = track_length
        self.policy = policy
        self.x_bins = x_bins
        self.x_dot_bins = x_dot_bins
        self.theta_bins = theta_bins
        self.theta_dot_bins = theta_dot_bins
        self.min_x_dot = min_x_dot
        self.max_x_dot = max_x_dot
        self.min_theta_dot = min_theta_dot
        self.max_theta_dot = max_theta_dot

        random.seed(11)
        np.random.seed(11)
        # you may add your code for initialization here, e.g., the Q-table
        self.q_table = {}
        pass

    def reset(self):
        """
        you may add code here to re-initialize your agent before each trial
        :return:
        """
        pass

    def get_action(self, x, x_dot, theta, theta_dot):
        """
        main.py calls this method to get an action from your agent
        :param x: the position of the cart
        :param x_dot: the velocity of the cart
        :param theta: the angle between the cart and the pole
        :param theta_dot: the angular velocity of the pole
        :return:
        """
        if self.policy == 'mixed' and random.random() < self.epsilon:
            action = random.sample([FORWARD_ACCEL, BACKWARD_ACCEL], 1)[0]
        else:
            # fill your code here to get an action from your agent
            state = self.discretize_state((x, x_dot, theta, theta_dot))
            forward = self.q_table.get((state, FORWARD_ACCEL), 0)
            backward = self.q_table.get((state, BACKWARD_ACCEL), 0)
            action = FORWARD_ACCEL if forward > backward else BACKWARD_ACCEL
            if forward == backward:
                action = random.sample([FORWARD_ACCEL, BACKWARD_ACCEL], 1)[0]
        return action

    def update_Q(self, prev_state, prev_action, cur_state, reward):
        """
        main.py calls this method so that you can update your Q-table
        :param prev_state: previous state, a tuple of (x, x_dot, theta, theta_dot)
        :param prev_action: previous action, FORWARD_ACCEL or BACKWARD_ACCEL
        :param cur_state: current state, a tuple of (x, x_dot, theta, theta_dot)
        :param reward: reward, 0.0 or -1.0
        e.g., if we have S_i ---(action a, reward)---> S_j, then
            prev_state is S_i,
            prev_action is a,
            cur_state is S_j,
            rewards is reward.
        :return:
        """
        prev_state = self.discretize_state(prev_state)
        cur_state = self.discretize_state(cur_state)
        curr_q_value = self.q_table.get((prev_state, prev_action), 0)
        next_q_value = max(self.q_table.get((cur_state, FORWARD_ACCEL), 0), self.q_table.get((cur_state, BACKWARD_ACCEL), 0))
        new_q_value = self.lr*(reward + self.gamma*next_q_value) + (1-self.lr)*curr_q_value
        self.q_table[(prev_state, prev_action)] = new_q_value

    # you may add more methods here for your needs. E.g., methods for discretizing the variables.

    def discretize_state(self, state):
        x, x_dot, theta, theta_dot = state
        
        # Compute the bin width for each variable
        x_bin_width = 2*self.track_length / self.x_bins  
        x_dot_bin_width = (self.max_x_dot - self.min_x_dot) / self.x_dot_bins
        theta_bin_width = 0.4188 / self.theta_bins  # theta ranges from -0.20944 to 0.20944
        theta_dot_bin_width = (self.max_theta_dot - self.min_theta_dot) / self.theta_dot_bins
        
        # Map each variable to its corresponding bin
        x_discrete = floor((x + 2.4) / x_bin_width)
        x_dot_discrete = floor((x_dot - self.min_x_dot) / x_dot_bin_width)
        theta_discrete = floor((theta + 0.20944) / theta_bin_width)
        theta_dot_discrete = floor((theta_dot - self.min_theta_dot) / theta_dot_bin_width)
        
        return x_discrete, x_dot_discrete, theta_discrete, theta_dot_discrete

if __name__ == '__main__':
    pass
