# -*- coding: utf-8 -*-
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from projection import *
from collections import deque

class DroneSimEnv(gym.Env):
    def __init__(self):
        '''
        general property
        '''
        self.episodes = 0
        self.fps = 100
        self.iteration, self.max_iteration = 0, 20 * self.fps

        self.mass_target = 1
        self.mass_hunter = 1
        self.gravity = 9.8

        self.width = 256
        self.height = 144

        self.roll_angular_acceleration = 60
        self.pitch_angular_acceleration = 60
        self.yaw_angular_acceleration = 60
        self.thrust_acceleration = 10
        self.max_velocity = 10

        self.min_absolute_x, self.max_absolute_x = 0, self.width
        self.min_absolute_y, self.max_absolute_y = 0, self.height

        self.min_initial_distance, self.max_initial_distance = 5, 30
        self.min_detect_distance, self.max_detect_distance = 1, 30

        self.max_absolute_angle = 180

        self.max_absolute_thrust = 2 * self.mass_hunter * self.gravity
        '''
        state related property
        '''
        # configuration of relative position in view
        self.queue_length = 4
        self.coordinate_queue = None
        self.distance_queue = None

        # threshold for position
        self.min_position_x, self.max_position_x = -np.inf, np.inf
        self.min_position_y, self.max_position_y = -np.inf, np.inf
        self.min_position_z, self.max_position_z = -np.inf, np.inf # assume in open space

        # threshold for orientation
        self.min_angle, self.max_angle = -1, 1 # 0 - 360
        self.min_roll, self.max_roll = self.min_angle, self.max_angle
        self.min_pitch, self.max_pitch = self.min_angle, self.max_angle
        self.min_yaw, self.max_yaw = self.min_angle, self.max_angle

        # threshold for thrust
        self.min_thrust, self.max_thrust = 0, 1 # 0 - 2 * self.mass_hunter * self.gravity

        # threshold for relative position in view
        self.min_relative_x, self.max_relative_x = 0, 1 # 1 - 256
        self.min_relative_y, self.max_relative_y = 0, 1 # 1 - 144

        # threshold for distance within target and hunter
        self.min_distance, self.max_distance = 0, 1 # 0 - 30

        # threshold for state
        self.low_state = np.array(
            [
                self.min_roll, self.min_pitch, self.min_yaw, self.min_thrust,
                self.min_relative_x, self.min_relative_y, self.min_relative_x, self.min_relative_y, self.min_relative_x, self.min_relative_y, self.min_relative_x, self.min_relative_y,  
                self.min_distance, self.min_distance, self.min_distance, self.min_distance 
            ]
        )
        self.high_state = np.array(
            [
                self.max_roll, self.max_pitch, self.max_yaw, self.max_thrust,
                self.max_relative_x, self.max_relative_y, self.max_relative_x, self.max_relative_y, self.max_relative_x, self.max_relative_y, self.max_relative_x, self.max_relative_y,
                self.max_distance, self.max_distance, self.max_distance, self.max_distance
            ]
        )

        '''
        action related property
        assume symmetric actions
        '''
        # threshold for orientation
        self.min_roll_action, self.max_roll_action = -1, 1 # 0 - 360
        self.min_pitch_action, self.max_pitch_action = -1, 1 # 0 - 360
        self.min_yaw_action, self.max_yaw_action = -1, 1 # 0 - 360

        # threshold for thrust
        self.min_thrust_action, self.max_thrust_action = -1, 1 # 0 - 2 * self.mass_hunter * self.gravity

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


    # Attention: remember to check the equation
    def get_accleration(self, orientation, thrust, mass):
        yaw_radians = radians(orientation.item(2))
        pitch_radians = radians(orientation.item(1))
        roll_radians = radians(orientation.item(0))
        f = np.matrix([0, 0, -float(thrust)/float(mass), 1])
        R = np.matrix([[cos(yaw_radians)*cos(pitch_radians), cos(yaw_radians)*sin(pitch_radians)*sin(roll_radians)-sin(yaw_radians)*cos(roll_radians), cos(yaw_radians)*sin(pitch_radians)*cos(roll_radians) + sin(yaw_radians)*sin(roll_radians), 0],
                       [sin(yaw_radians)*cos(pitch_radians), sin(yaw_radians)*sin(pitch_radians)*sin(roll_radians)+cos(yaw_radians)*cos(roll_radians), sin(yaw_radians)*sin(pitch_radians)*cos(roll_radians) - cos(yaw_radians)*sin(roll_radians), 0],
                       [-sin(pitch_radians), cos(pitch_radians)*sin(roll_radians), cos(pitch_radians)*cos(roll_radians), 0],
                       [0, 0, 0, 1]])
        a = R*np.transpose(f)
        a[2] *= -1
        a = np.transpose(a[:3])
        a += np.matrix([0.0, 0.0, -self.gravity])
        return a


    def step(self, action):
        roll, pitch, yaw, thrust = action[0], action[1], action[2], action[3]

        # update hunter
        self.orientation_hunter += np.matrix([self.roll_angular_acceleration * roll / self.fps, self.pitch_angular_acceleration * pitch / self.fps, self.yaw_angular_acceleration * yaw / self.fps])

        self.thrust_hunter += self.thrust_acceleration * thrust / self.fps
        self.thrust_hunter = np.clip(self.thrust_hunter, self.max_absolute_thrust * self.min_thrust, self.max_absolute_thrust * self.max_thrust)

        acceleration_hunter = self.get_accleration(self.orientation_hunter, self.thrust_hunter, self.mass_hunter)
        self.velocity_hunter += acceleration_hunter / self.fps
        if np.linalg.norm(self.velocity_hunter) > self.max_velocity: self.velocity_hunter *= self.max_velocity * np.linalg.norm(self.velocity_hunter)
        self.position_hunter += 10 * self.velocity_hunter / self.fps # add 10 to accelerate the approaching process

        # update target
        self.orientation_target += np.matrix([
            np.random.normal(0, self.roll_angular_acceleration * 0.5 / self.fps), 
            np.random.normal(0, self.pitch_angular_acceleration * 0.5 / self.fps), 
            np.random.normal(0, self.yaw_angular_acceleration * 0.5 / self.fps)
            ])
        self.thrust_target += np.random.normal(0, self.thrust_acceleration / self.fps)
        self.thrust_target = np.clip(self.thrust_target, self.max_absolute_thrust * self.min_thrust, self.max_absolute_thrust * self.max_thrust)

        acceleration_target = self.get_accleration(self.orientation_target, self.thrust_target, self.mass_target)
        self.velocity_target += acceleration_target / self.fps
        if np.linalg.norm(self.velocity_target) > self.max_velocity: self.velocity_target *= self.max_velocity * np.linalg.norm(self.velocity_target)
        # self.position_target += self.velocity_target / self.fps # stationary target

        # update state
        self.state = self.get_state()

        # calculate reward and check if done
        distance = np.linalg.norm(self.position_target - self.position_hunter)
        reward = (self.previous_distance - distance) * 500 # even if chasing at max speed, the reward will be 10 / 120 * 500 = 41.67
        self.previous_distance = distance

        done = False
        done_reason = None
        is_in_view = [ (self.min_relative_x < coordinate[0] < self.max_relative_x and self.min_relative_y < coordinate[1] < self.max_relative_y) for coordinate in self.coordinate_queue]
        if True not in is_in_view:
            done = True
            reward = 0
            self.episodes += 1
            self.iteration = 0
            done_reason = "not in view"
        
        if distance > self.max_detect_distance or distance <= self.min_detect_distance or self.iteration > self.max_iteration:
            done = True
            reward = 0
            self.episodes += 1
            self.iteration = 0
            if distance > self.max_detect_distance: done_reason = "too far to detect"
            if self.iteration > self.max_iteration: done_reason = "meet max iteration"
            if distance <= self.min_detect_distance: 
                reward = 200
                done_reason = "success!"

        # control parameter
        self.iteration += 1

        return self.state, reward, done, {'distance': distance, 'done_reason': done_reason}


    def get_state(self):
        # get target relative coordinate in view
        (absolute_x, absolute_y), _ = projection(self.position_target, self.position_hunter, self.orientation_hunter, w=float(self.width), h=float(self.height)) 
        relative_x, relative_y = absolute_x / self.width, absolute_y / self.height
        target_coordinate_in_view = np.array((relative_x, relative_y)).flatten()

        # get distance within hunter and target
        distance = np.array([np.linalg.norm(self.position_hunter - self.position_target) / self.max_initial_distance]) 

        # maintain a 8-length deque
        if self.coordinate_queue is None and self.distance_queue is None:
            self.coordinate_queue = deque([target_coordinate_in_view] * self.queue_length)
            self.distance_queue = deque([distance] * self.queue_length)
        else:
            self.coordinate_queue.append(target_coordinate_in_view)
            self.coordinate_queue.popleft()

            self.distance_queue.append(distance)
            self.distance_queue.popleft()

        coordinate_state = np.concatenate(list(self.coordinate_queue))
        distance_state = np.concatenate(list(self.distance_queue))

        # define state
        state = np.concatenate([np.array(self.orientation_hunter).flatten() / self.max_absolute_angle,
                                np.array(self.thrust_hunter).flatten() / self.max_absolute_thrust,
                                coordinate_state,
                                distance_state
                                ], 0)

        return state


    def reset(self):
        # state related property
        self.position_hunter = np.matrix([0.0, 0.0, 10.0]) # x, y, z
        self.orientation_hunter = np.matrix([0.0, 0.0, 0.0]) # roll, pitch, taw
        self.thrust_hunter = 9.8
        self.velocity_hunter = np.matrix([0.0, 0.0, 0.0])

        self.position_target = np.matrix([10.0, 0.0, 10.0]) + np.random.normal(0, 5) # x, y, z
        self.orientation_target = np.matrix([0.0, 0.0, 0.0]) # roll, pitch, yaw
        self.thrust_target = 9.8
        self.velocity_target = np.matrix([0.0, 0.0, 0.0])

        # generate absolute coordinate in view and distance
        (absolute_x, absolute_y), _ = projection(self.position_target, self.position_hunter, self.orientation_hunter, w=float(self.width), h=float(self.height)) 
        distance = np.linalg.norm(self.position_hunter - self.position_target)

        # invalid initialization
        while (absolute_x > self.max_absolute_x or absolute_x < self.min_absolute_x or absolute_y > self.max_absolute_y or absolute_y < self.min_absolute_y or distance > self.max_initial_distance or distance < self.min_initial_distance):
            self.position_target = np.matrix([10.0, 0.0, 10.0]) + np.random.normal(0, 5)
            (absolute_x, absolute_y), _ = projection(self.position_target, self.position_hunter, self.orientation_hunter, w=float(self.width), h=float(self.height)) 
            distance = np.linalg.norm(self.position_hunter - self.position_target)

        self.previous_distance = distance
        self.state = self.get_state()

        self.episodes = 0
        self.iteration = 0

        return self.state


    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass
