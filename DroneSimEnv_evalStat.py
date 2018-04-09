# -*- coding: utf-8 -*-
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from projection import *
from collections import deque

import dronesim

class DroneSimEnv(gym.Env):
    def __init__(self):
        '''
        general property
        '''
        # self.tmp_pos = np.array([None,None,None,None])
        self.episodes = 0
        self.fps = 100
        self.iteration, self.max_iteration = 0, 20 * self.fps

        self.width = 256
        self.height = 144

        self.gravity = 9.8
        
        self.max_absolute_thrust = 16 # 2 * self.gravity
        self.min_absolute_thrust = 4
        self.thrust_sensity = 6

        self.min_absolute_x, self.max_absolute_x = 0, self.width
        self.min_absolute_y, self.max_absolute_y = 0, self.height

        self.min_initial_distance, self.max_initial_distance = 5, 30
        self.min_detect_distance, self.max_detect_distance = 1, 30

        self.max_absolute_angle = 180
        self.max_roll_angle = 180
        self.max_pitch_angle = 180
        self.max_yaw_angle = 180

        #self.max_absolute_thrust = 2 * self.mass_hunter * self.gravity

        '''
        state related property
        '''
        # configuration of relative position in view
        self.queue_length = 4
        self.coordinate_queue = None
        self.distance_queue = None

        # threshold for orientation
        self.min_angle, self.max_angle = -1, 1 # -180 -> 180
        self.min_roll, self.max_roll = self.min_angle, self.max_angle
        self.min_pitch, self.max_pitch = self.min_angle, self.max_angle
        self.min_yaw, self.max_yaw = self.min_angle, self.max_angle

        # threshold for thrust
        self.min_thrust, self.max_thrust = -1, 1 # thrust = self.min_absolute_thrust + self.thrust_sensity * (x + 1)

        # threshold for relative position in view
        self.min_relative_x, self.max_relative_x = 0, 1 # 1 - 256
        self.min_relative_y, self.max_relative_y = 0, 1 # 1 - 144

        # threshold for distance within target and hunter
        self.min_distance, self.max_distance = 0, 1 # 0 -> 30

        # threshold for state
        self.low_state = np.array(
            [
                self.min_roll, self.min_pitch, self.min_yaw, self.min_thrust,
                self.min_relative_x, self.min_relative_y, self.min_relative_x, self.min_relative_y, self.min_relative_x, self.min_relative_y, self.min_relative_x, self.min_relative_y,
                self.min_relative_x, self.min_relative_y, self.min_relative_x, self.min_relative_y, self.min_relative_x, self.min_relative_y, self.min_relative_x, self.min_relative_y, 
                self.min_distance, self.min_distance, self.min_distance, self.min_distance
            ]
        )
        self.high_state = np.array(
            [
                self.max_roll, self.max_pitch, self.max_yaw, self.max_thrust,
                self.max_relative_x, self.max_relative_y, self.max_relative_x, self.max_relative_y, self.max_relative_x, self.max_relative_y, self.max_relative_x, self.max_relative_y,
                self.max_relative_x, self.max_relative_y, self.max_relative_x, self.max_relative_y, self.max_relative_x, self.max_relative_y, self.max_relative_x, self.max_relative_y, 
                self.max_distance, self.max_distance, self.max_distance, self.max_distance
            ]
        )

        '''
        action related property
        assume symmetric actions
        '''
        # threshold for orientation
        self.min_roll_action, self.max_roll_action = -1, 1 # -180 -> 180
        self.min_pitch_action, self.max_pitch_action = -1, 1 # -180 -> 180
        self.min_yaw_action, self.max_yaw_action = -1, 1 # -180 -> 180

        # threshold for thrust
        self.min_thrust_action, self.max_thrust_action = -1, 1 # -2 * self.mass_hunter * self.gravity -> 2 * self.mass_hunter * self.gravity

        # threshold for action
        self.low_action = np.array([self.min_roll_action, self.min_pitch_action, self.min_yaw_action, self.min_thrust_action])
        self.high_action = np.array([self.max_roll_action, self.max_pitch_action, self.max_yaw_action, self.max_thrust_action])

        '''
        define action space and observation space
        '''
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)

        self.seed()
        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        roll, pitch, yaw, thrust = action[0], action[1], action[2], action[3]

        # update hunter
        # dronesim.simcontrol([roll, pitch, yaw, thrust])
        dronesim.simrun(int(1e9 / self.fps), [roll, pitch, yaw, thrust])   #transform from second to nanoseconds
        
        # update state
        self.state = self.get_state()

        # calculate reward and check if done
        reward = (self.previous_distance - self.distance) * 500 # even if chasing at max speed, the reward will be 10 / 100 * 500 = 50
        self.previous_distance = self.distance

        done = False
        reason = None
        is_in_view = [ (self.min_relative_x < coordinate[0] < self.max_relative_x and self.min_relative_y < coordinate[1] < self.max_relative_y) for coordinate in self.coordinate_queue]
        '''
        if True not in is_in_view:
            done = True
            reward = 0
            self.episodes += 1
            self.iteration = 0
        '''
        if self.distance > self.max_detect_distance or self.distance < self.min_detect_distance or self.iteration > 100000:
            done = True
            reward = 0
            self.episodes += 1
            self.iteration = 0
            if self.distance < self.min_detect_distance: 
                reward = 200
                reason = 1
            if self.distance > self.max_detect_distance:
                reason = 2
            if self.iteration > 100000:
                reason = 3

        # control parameter
        self.iteration += 1

        return self.state, reward, done, {'distance': self.distance, 'reason': reason}



    def get_state(self):
        position_hunter, orientation_hunter, acc_hunter, position_target, orientation_target, acc_target, thrust_hunter = dronesim.siminfo()
        
        orientation_hunter = [math.degrees(degree) for degree in orientation_hunter]
        orientation_target = [math.degrees(degree) for degree in orientation_target]

        try:        
            (absolute_x, absolute_y), _ = projection(np.matrix(position_target), np.matrix(position_hunter), np.matrix(orientation_hunter), w=float(self.width), h=float(self.height)) 
        except ValueError:
            self.distance = 100
            return self.state

        relative_x, relative_y = absolute_x / self.width, absolute_y / self.height
        target_coordinate_in_view = np.array((relative_x, relative_y)).flatten()
        
        self.distance = np.linalg.norm(position_hunter - position_target)
        # get distance within hunter and target
        distance = np.array([self.distance / self.max_initial_distance])  

        # maintain a 8-length deque
        if self.coordinate_queue is None and self.distance_queue is None:
            self.coordinate_queue = deque([target_coordinate_in_view] * 8)
            self.distance_queue = deque([distance] * self.queue_length)
        else:
            self.coordinate_queue.append(target_coordinate_in_view)
            self.coordinate_queue.popleft()

            self.distance_queue.append(distance)
            self.distance_queue.popleft()

        coordinate_state = np.concatenate(list(self.coordinate_queue))
        distance_state = np.concatenate(list(self.distance_queue))

        # define state
        state = np.concatenate([np.array([orientation_hunter[0] / self.max_roll_angle, orientation_hunter[1] / self.max_pitch_angle, orientation_hunter[2] / self.max_yaw_angle]).flatten(),
                                np.array((thrust_hunter - self.min_absolute_thrust) / self.thrust_sensity + self.min_thrust).flatten(),
                                coordinate_state,
                                distance_state
                                ], 0)

        # if self.tmp_pos.any() != None :
        #     tmp_pos = self.tmp_pos
        # self.tmp_pos = position_hunter
        return state


    def reset(self):
        # state related property
        position_hunter = np.matrix([0.0, 0.0, 10.0]) # x, y, z
        orientation_hunter = np.matrix([0.0, 0.0, 0.0]) # roll, pitch, taw

        #the position of target is generated randomly and should not exceed the vision range of hunter
        position_target = np.matrix([10.0, 0.0, 10.0]) + np.random.normal(0, 5) # x, y, z
        orientation_target = np.matrix([0.0, 0.0, 0.0]) # roll, pitch, yaw

        (absolute_x, absolute_y), _ = projection(position_target, position_hunter, orientation_hunter, w=float(self.width), h=float(self.height)) 
        
        print('x,y: ', absolute_x, absolute_y)
        
        distance = np.linalg.norm(position_hunter - position_target)
        # invalid initialization
        while (absolute_x > self.max_absolute_x or absolute_x < self.min_absolute_x or absolute_y > self.max_absolute_y or absolute_y < self.min_absolute_y or distance > self.max_initial_distance or distance < self.min_initial_distance):
            position_target = np.matrix([10.0, 0.0, 10.0]) + np.random.normal(0, 5)
            (absolute_x, absolute_y), _ = projection(position_target, position_hunter, orientation_hunter, w=float(self.width), h=float(self.height)) 
            distance = np.linalg.norm(position_hunter - position_target)

        dronesim.siminit(np.squeeze(np.asarray(position_hunter)),np.squeeze(np.asarray(orientation_hunter)), \
                         np.squeeze(np.asarray(position_target)),np.squeeze(np.asarray(orientation_target)), 20, 10)
        
        self.previous_distance = distance
        self.state = self.get_state()

        self.iteration = 0
        self.episodes = 0

        return self.state

    def stop(self):
        dronesim.simstop()

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass

##################for test###############################
if __name__ == "__main__":
    env = DroneSimEnv()
    env.step([1,0,0,1])
