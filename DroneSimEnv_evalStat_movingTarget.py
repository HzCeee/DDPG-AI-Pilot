# -*- coding: utf-8 -*-
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

# from projection import *
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
        self.iteration, self.max_iteration = 0, 60 * self.fps

        self.width = 256
        self.height = 144

        self.gravity = 9.8
        
        self.max_absolute_thrust = 16 # 2 * self.gravity
        self.min_absolute_thrust = 4
        self.thrust_sensity = (self.max_absolute_thrust - self.min_absolute_thrust) / 2

        self.min_absolute_x, self.max_absolute_x = 0, self.width
        self.min_absolute_y, self.max_absolute_y = 0, self.height

        self.min_initial_distance, self.max_initial_distance = 5, 30
        self.min_detect_distance, self.max_detect_distance = 1, 30

        self.max_absolute_angle = 180
        self.max_roll_angle = 40
        self.max_pitch_angle = 40
        self.max_yaw_angle = 180

        #self.max_absolute_thrust = 2 * self.mass_hunter * self.gravity

        '''
        state related property
        '''
        # configuration of relative position in view
        self.queue_length = 8
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
                self.min_distance, self.min_distance, self.min_distance, self.min_distance, self.min_distance, self.min_distance, self.min_distance, self.min_distance
            ]
        )
        self.high_state = np.array(
            [
                self.max_roll, self.max_pitch, self.max_yaw, self.max_thrust,
                self.max_relative_x, self.max_relative_y, self.max_relative_x, self.max_relative_y, self.max_relative_x, self.max_relative_y, self.max_relative_x, self.max_relative_y,
                self.max_relative_x, self.max_relative_y, self.max_relative_x, self.max_relative_y, self.max_relative_x, self.max_relative_y, self.max_relative_x, self.max_relative_y, 
                self.max_distance, self.max_distance, self.max_distance, self.max_distance, self.max_distance, self.max_distance, self.max_distance, self.max_distance
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
        # print('norm: ', roll, pitch, yaw) 
        # _, ori_hunter, _, pos_target, ori_target, _, _ = dronesim.siminfo()
        
        # roll_target, pitch_target, yaw_target = ori_target[0], ori_target[1], ori_target[2]
        # print('target: ', roll_target, pitch_target, yaw_target)
        # print('hunter: ', ori_hunter[0], ori_hunter[1], ori_hunter[2])
        # print('diff: ', ori_hunter[0] - roll_target, ori_hunter[1] - pitch_target, ori_hunter[2] - yaw_target)
        if self.iteration % 50 == 0:
            self.roll_target = min(max(self.roll_target + 0.33 * np.random.randn(), -1), 1)
            self.pitch_target = min(max(self.pitch_target + 0.33 * np.random.randn(), -1), 1)
            self.yaw_target = min(max(self.yaw_target + 0.33 * np.random.randn(), -1), 1)
            self.thrust_target = min(max(self.thrust_target + 0.33 * np.random.randn(), -1), 1)
        # print('norm_target: ', self.roll_target, self.pitch_target, self.yaw_target)

        # update hunter
        # dronesim.simcontrol([roll, pitch, yaw, thrust], [self.roll_target, self.pitch_target, self.yaw_target, self.thrust_target])
        dronesim.simrun(int(1e9 / self.fps), [roll, pitch, yaw, thrust], [self.roll_target, self.pitch_target, self.yaw_target, self.thrust_target])   #transform from second to nanoseconds
        
        # update state
        self.state = self.get_state()

        # calculate reward and check if done
        reward = (self.previous_distance - self.distance) * 1000 # even if chasing at max speed, the reward will be 10 / 100 * 500 = 50
        self.previous_distance = self.distance

        # print('reward: ', reward)

        done = False
        reason = None
        is_in_view = [ (self.min_relative_x < coordinate[0] < self.max_relative_x and self.min_relative_y < coordinate[1] < self.max_relative_y) if coordinate[0] != -1 else False for coordinate in self.coordinate_queue]
        '''
        if True not in is_in_view:
            done = True
            reward = 0
            self.episodes += 1
            self.iteration = 0
        '''
        # print('reward of first term: ', reward)
        # print(self.coordinate_queue[-1])
        if self.coordinate_queue[-1][0] != -1:
            error_x, error_y = abs(self.coordinate_queue[-1][0] - 0.5), abs(self.coordinate_queue[-1][1] - 0.5)
            # reward = reward - 20*error_x - 10*error_y - 20*(error_x*error_y) # so the largest possible punishment is 20
            reward = reward - 200 * np.linalg.norm([error_x/self.distance, error_y/self.distance])
        else:
            reward = -100
        
        if self.distance > self.max_detect_distance*2 or self.distance < self.min_detect_distance or self.iteration > 100000:
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
        # print('reward: ', reward)
        return self.state, reward, done, {'distance': self.distance, 'reason': reason}



    def get_state(self):
        position_hunter, orientation_hunter, acc_hunter, position_target, orientation_target, acc_target, thrust_hunter = dronesim.siminfo()
        
        # print('ori: ', orientation_hunter)

        orientation_hunter = [math.degrees(degree) for degree in orientation_hunter]
        orientation_target = [math.degrees(degree) for degree in orientation_target]
        
        try:        
            absolute_x, absolute_y, target_in_front = dronesim.projection(position_hunter, orientation_hunter, position_target, float(self.width), float(self.height))
        except ValueError:
            self.distance = 100
            return self.state

        if target_in_front: relative_x, relative_y = absolute_x / self.width, absolute_y / self.height
        target_coordinate_in_view = np.array((relative_x, relative_y)).flatten() if target_in_front and self.min_absolute_x < absolute_x < self.max_absolute_x and self.min_absolute_y < absolute_y < self.max_absolute_y else np.array((-1, -1))
        
        # print('!!!!####')
        # print(position_hunter, orientation_hunter, position_target)
        # print(0 < absolute_x < 256, 0 < absolute_y < 144, target_in_front)
        # print(target_coordinate_in_view)

        self.distance = np.linalg.norm(np.array(position_hunter) - np.array(position_target))
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
        # camera
        dronesim.installcamera([0,-15,180], 110, 63, 0.01, 500)
        # state related property
        position_hunter = [0.0, 0.0, 10.0] # x, y, z
        orientation_hunter = [0.0, 0.0, 0.0] # roll, pitch, taw
        
        # position_hunter = np.matrix([44.41015975 -3.96066086  7.88967305])
        # orientation_hunter = np.matrix([-11.276222823141545, 8.627114153116235, 27.256829024182572])
        # position_target = np.matrix([17.49014858 -5.98676401 21.02755752])

        #the position of target is generated randomly and should not exceed the vision range of hunter
        position_target = (np.array([10.0, 0.0, 10.0]) + np.random.normal(0, 5)).tolist() # x, y, z
        orientation_target = [0.0, 0.0, 0.0] # roll, pitch, yaw
        self.roll_target, self.pitch_target, self.yaw_target, self.thrust_target = 0, 0, 0, 0
        # print('error c')
        # print('lala: ', position_hunter, orientation_hunter, position_target, self.width, self.height)
        absolute_x, absolute_y, target_in_front = dronesim.projection(position_hunter, orientation_hunter, position_target, float(self.width), float(self.height))
        # print('position: ', position_hunter)
        # print('error e')
        # print('pos_hunter: ', position_hunter)
        # print('pos_target: ', position_target)
        distance = np.linalg.norm(np.array(position_hunter) - np.array(position_target))
        # invalid initialization
        # print('error d')
        while (not target_in_front or absolute_x > self.max_absolute_x or absolute_x < self.min_absolute_x or absolute_y > self.max_absolute_y or absolute_y < self.min_absolute_y or distance > self.max_initial_distance or distance < self.min_initial_distance):
            position_target = (np.array([10.0, 0.0, 10.0]) + np.random.normal(0, 5)).tolist()
            absolute_x, absolute_y, target_in_front = dronesim.projection(position_hunter, orientation_hunter, position_target, float(self.width), float(self.height)) 
            distance = np.linalg.norm(np.array(position_hunter) - np.array(position_target))
        
        # print(position_hunter, orientation_hunter, position_target)
        
        # position_hunter = np.matrix([38.5, 26.8, 30.3])
        # orientation_hunter = np.matrix([20.5,12.9,84.9])
        # position_target = np.matrix([16.4,8.8,39.9])
        # print('error a')
        dronesim.siminit(np.squeeze(np.asarray(position_hunter)),np.squeeze(np.asarray(orientation_hunter)), \
                         np.squeeze(np.asarray(position_target)),np.squeeze(np.asarray(orientation_target)), 20, 5)
        # print('error b')
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
