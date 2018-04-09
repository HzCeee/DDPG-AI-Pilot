# -*- coding: utf-8 -*-
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from baselines import logger
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
        self.in_fov_queue = None

        # threshold for orientation
        self.min_angle, self.max_angle = -1, 1 # -180 -> 180
        self.min_roll, self.max_roll = self.min_angle, self.max_angle
        self.min_pitch, self.max_pitch = self.min_angle, self.max_angle
        self.min_yaw, self.max_yaw = self.min_angle, self.max_angle

        # threshold for thrust
        self.min_thrust, self.max_thrust = -1, 1 # thrust = self.min_absolute_thrust + self.thrust_sensity * (x + 1)

        # threshold for relative position in view
        self.min_relative_x, self.max_relative_x = -1, 1 # 1 - 256
        self.min_relative_y, self.max_relative_y = -1, 1 # 1 - 144

        # threshold for distance within target and hunter
        self.min_distance, self.max_distance = 0, 1 # 0 -> 30

        # threshold for state
        self.low_state = np.array(
            [self.min_roll, self.min_pitch, self.min_yaw, self.min_thrust] 
            + self.queue_length * [self.min_relative_x, self.min_relative_y]
            + self.queue_length * [self.min_distance]
            )
        self.high_state = np.array(
            [self.max_roll, self.max_pitch, self.max_yaw, self.max_thrust] 
            + self.queue_length * [self.max_relative_x, self.max_relative_y]
            + self.queue_length * [self.max_distance]
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
        # self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)

        self.observation_space = spaces.Tuple((
            spaces.Box(low=self.low_state, high=self.high_state),
            spaces.MultiBinary(self.queue_length) # if corrdinates in FOV, it is equal to if we can measure the distance
            ))

        self.seed()
        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.seed = seed
        return [seed]


    def step(self, action):
        roll, pitch, yaw, thrust = action[0], action[1], action[2], action[3]
        
        if self.iteration % 50 == 0:
            self.roll_target = min(max(self.roll_target + 0.33 * np.random.randn(), -1), 1)
            self.pitch_target = min(max(self.pitch_target + 0.33 * np.random.randn(), -1), 1)
            self.yaw_target = min(max(self.yaw_target + 0.33 * np.random.randn(), -1), 1)
            self.thrust_target = min(max(self.thrust_target + 0.33 * np.random.randn(), -1), 1)

        # update hunter
        # dronesim.simcontrol([roll, pitch, yaw, thrust], [self.roll_target, self.pitch_target, self.yaw_target, self.thrust_target])

        dronesim.simrun(int(1e9 / self.fps), [roll, pitch, yaw, thrust], [self.roll_target, self.pitch_target, self.yaw_target, self.thrust_target])   #transform from second to nanoseconds
        
        #####################

        self.test_roll_hunter = roll
        self.test_pitch_hunter = pitch
        self.test_yaw_hunter = yaw
        self.test_thrust_hunter = thrust

        #####################

        # update state
        self.state = self.get_state()

        # calculate reward and check if done
        reward = (self.previous_distance - self.distance) * 1000 # even if chasing at max speed, the reward will be 10 / 100 * 500 = 50
        self.previous_distance = self.distance

        done = False
        is_in_view = [True if is_in_fov == 1 else False for is_in_fov in self.in_fov_queue]
        if True not in is_in_view:
            done = True
            reward = 0
            self.episodes += 1
            self.iteration = 0

        if self.coordinate_queue[-1][0] != self.min_relative_x:
            error_x, error_y = abs(self.coordinate_queue[-1][0] - 0.5), abs(self.coordinate_queue[-1][1] - 0.5)
            # reward = reward - 20*error_x - 10*error_y - 20*(error_x*error_y) # so the largest possible punishment is 20
            reward = reward - 300 * np.linalg.norm([error_x/self.distance, error_y/self.distance])
            # reward = -100 * np.linalg.norm([error_x/self.distance, error_y/self.distance, (1/self.distance - 1)/5])
        else:
            reward = -100

        if self.distance > self.max_detect_distance or self.distance < self.min_detect_distance or self.iteration > self.max_iteration:
            done = True
            reward = 0
            self.episodes += 1
            self.iteration = 0
            if self.distance < self.min_detect_distance: reward = 200
            if self.iteration > self.max_iteration: reward = 0
        
        if self.flag:
            done = True
            reward = 0
            self.iteration = 0

        # control parameter
        self.iteration += 1

        return self.state, reward, done, {'distance': self.distance}



    def get_state(self):
        position_hunter, orientation_hunter, acc_hunter, position_target, orientation_target, acc_target, thrust_hunter = dronesim.siminfo()

        orientation_hunter = [math.degrees(degree) for degree in orientation_hunter]
        orientation_target = [math.degrees(degree) for degree in orientation_target]
        
        self.flag = False
        try:
            absolute_x, absolute_y, target_in_front = dronesim.projection(position_hunter, orientation_hunter, position_target, float(self.width), float(self.height))
        except ValueError:
            logger.info('###########################################')
            logger.info('episodes: {}, iteration: {}'.format(self.episodes, self.iteration))
            logger.info('prev pos hunter: {}'.format(self.prev_pos_hunter))
            logger.info('prev ori hunter: {}'.format(self.prev_ori_hunter))
            logger.info('prev pos target: {}'.format(self.prev_pos_target))
            logger.info('prev ori target: {}'.format(self.prev_ori_target))
            logger.info('prev acc hunter: {}'.format(self.prev_acc_hunter))
            logger.info('prev acc target: {}'.format(self.prev_acc_target))
            logger.info('prev thrust hunter: {}'.format(self.prev_thrust_hunter))
            logger.info('hunter position: {}'.format(position_hunter))
            logger.info('hunter ori: {}'.format(orientation_hunter))
            logger.info('target position: {}'.format(position_target))
            logger.info('target ori: {}'.format(orientation_target))
            logger.info('hunter acc: {}'.format(acc_hunter))
            logger.info('target acc: {}'.format(acc_target))
            logger.info('thrust hunter: {}'.format(thrust_hunter))
            logger.info('init pos target: {}'.format(self.init_pos_target))

            logger.info('target action: {},{},{},{}'.format(self.roll_target, self.pitch_target, self.yaw_target, self.thrust_target))
            logger.info('hunter action: {},{},{},{}'.format(self.test_roll_hunter, self.test_pitch_hunter, self.test_yaw_hunter, self.test_thrust_hunter))

            self.flag = True # force program to terminate
            # raise ValueError
            return self.state
        
        self.prev_pos_hunter = position_hunter
        self.prev_ori_hunter = orientation_hunter
        self.prev_pos_target = position_target
        self.prev_ori_target = orientation_target
        self.prev_acc_hunter = acc_hunter
        self.prev_acc_target = acc_target
        self.prev_thrust_hunter = thrust_hunter
        '''
        if (1 + self.iteration) % 200 == 0:
            logger.info('episodes: {},iteration: {}'.format(self.episodes, self.iteration))
            logger.info('pos hunter: {}'.format(position_hunter))
            logger.info('pos target: {}'.format(position_target))
            logger.info('ori hunter: {}'.format(orientation_hunter))
            logger.info('ori target: {}'.format(orientation_target))
            logger.info('acc hunter: {}'.format(acc_hunter))
            logger.info('acc target: {}'.format(acc_target))
            logger.info('thrust hunter: {}'.format(thrust_hunter))
            logger.info('target action: {},{},{},{}'.format(self.roll_target, self.pitch_target, self.yaw_target, self.thrust_target))
            logger.info('hunter action: {},{},{},{}'.format(self.test_roll_hunter, self.test_pitch_hunter, self.test_yaw_hunter, self.test_thrust_hunter))
        '''
        self.is_in_fov = 1 if target_in_front and self.min_absolute_x < absolute_x < self.max_absolute_x and self.min_absolute_y < absolute_y < self.max_absolute_y else 0

        if target_in_front: relative_x, relative_y = absolute_x / self.width, absolute_y / self.height
        target_coordinate_in_view = np.array((relative_x, relative_y)).flatten() if self.is_in_fov == 1 else np.array((self.min_relative_x, self.min_relative_y))
        
        self.distance = np.linalg.norm(np.array(position_hunter) - np.array(position_target))
        distance = np.array([self.distance / self.max_initial_distance]) if self.is_in_fov == 1 else np.array([self.min_distance]) 

        # maintain a 8-length deque
        if self.coordinate_queue is None and self.distance_queue is None and self.in_fov_queue is None:
            self.coordinate_queue = deque([target_coordinate_in_view] * self.queue_length)
            self.distance_queue = deque([distance] * self.queue_length)
            self.in_fov_queue = deque([self.is_in_fov] * self.queue_length)
        else:
            self.coordinate_queue.append(target_coordinate_in_view)
            self.coordinate_queue.popleft()

            self.distance_queue.append(distance)
            self.distance_queue.popleft()

            self.in_fov_queue.append(self.is_in_fov)
            self.in_fov_queue.popleft()

        coordinate_state = np.concatenate(list(self.coordinate_queue))
        distance_state = np.concatenate(list(self.distance_queue))
        in_fov_state = np.concatenate(list(self.in_fov_queue))

        # define state
        state = np.concatenate([np.array([orientation_hunter[0] / self.max_roll_angle, orientation_hunter[1] / self.max_pitch_angle, orientation_hunter[2] / self.max_yaw_angle]).flatten(),
                                np.array((thrust_hunter - self.min_absolute_thrust) / self.thrust_sensity + self.min_thrust).flatten(),
                                coordinate_state,
                                distance_state
                                ], 0)

        state = (state, in_fov_state)

        return state


    def reset(self):
        # camera
        dronesim.installcamera([0,-15,180], 63, 110, 0.01, 500)

        # state related property
        position_hunter = [0.0, 0.0, 10.0] # x, y, z
        orientation_hunter = [0.0, 0.0, 0.0] # roll, pitch, taw

        #the position of target is generated randomly and should not exceed the vision range of hunter
        position_target = (np.array([10.0, 0.0, 10.0]) + np.random.normal(0, 5)).tolist() # x, y, z
        orientation_target = [0.0, 0.0, 0.0] # roll, pitch, yaw
        self.roll_target, self.pitch_target, self.yaw_target, self.thrust_target = 0, 0, 0, 0

        absolute_x, absolute_y, target_in_front = dronesim.projection(position_hunter, orientation_hunter, position_target, float(self.width), float(self.height)) 
        distance = np.linalg.norm(np.array(position_hunter) - np.array(position_target))
        # invalid initialization
        while (not target_in_front or absolute_x > self.max_absolute_x or absolute_x < self.min_absolute_x or absolute_y > self.max_absolute_y or absolute_y < self.min_absolute_y or distance > self.max_initial_distance or distance < self.min_initial_distance):
            position_target = (np.array([10.0, 0.0, 10.0]) + np.random.normal(0, 5)).tolist()
            absolute_x, absolute_y, target_in_front = dronesim.projection(position_hunter, orientation_hunter, position_target, float(self.width), float(self.height)) 
            distance = np.linalg.norm(np.array(position_hunter) - np.array(position_target))

        dronesim.siminit(np.squeeze(np.asarray(position_hunter)),np.squeeze(np.asarray(orientation_hunter)), \
                         np.squeeze(np.asarray(position_target)),np.squeeze(np.asarray(orientation_target)), 20, 5)#, 720, 180)
        
        self.init_pos_target = position_target

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
