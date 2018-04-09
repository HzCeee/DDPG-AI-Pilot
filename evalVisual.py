from DroneSimEnv_evalStat import *
import random
import time
import tensorflow as tf
import baselines.common.tf_util as U
from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *
import numpy as np
# from matplotlib.pyplot as plt

def actiongenerator(obs):
    action, _ = agent.pi(obs, apply_noise=False, compute_Q=True)
    # print('action:')
    # print('---------------------------------------')
    # print(action[0], action[1],action[2],action[3])
    # print('----------------------------------------')
    return action

env = DroneSimEnv()
itetime = 1000

normalize_returns = False
normalize_observations = True
critic_l2_reg = 1e-2
batch_size = 64
actor_lr = 1e-4
critic_lr = 1e-3
popart = False
gamma = 0.99
tau = 0.01
reward_scale = 1.
clip_norm = None
layer_norm = True

action_noise = None
param_noise = None
noise_type = 'adaptive-param_0.2'  # choices are adaptive-param_xx, ou_xx, normal_xx, none
for current_noise_type in noise_type.split(','):
    current_noise_type = current_noise_type.strip()
    if current_noise_type == 'none':
        pass
    elif 'adaptive-param' in current_noise_type:
        _, stddev = current_noise_type.split('_')
        param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
    elif 'normal' in current_noise_type:
        _, stddev = current_noise_type.split('_')
        action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
    elif 'ou' in current_noise_type:
        _, stddev = current_noise_type.split('_')
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
    else:
        raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

nb_actions = env.action_space.shape[-1]
memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
critic = Critic(layer_norm=layer_norm)
actor = Actor(nb_actions, layer_norm=layer_norm)

tf.reset_default_graph()

agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
    gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
    batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
    actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
    reward_scale=reward_scale)

max_iteration = 1
step_number = []
success = []
reason = {1:0, 2:0, 3:0}

with U.single_threaded_session() as sess:
    agent.initialize(sess)
    # sess.graph.finalize()
    iteration = 0
    success_number = 0

    renderer = dronesim.visualdrone()

    while iteration < max_iteration:
        iteration += 1
        print(iteration)
        agent.reset()
        obs = env.reset()

        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), '/home/projectvenom/Documents/AIPilot/AIPilot-ProjectVenom-master/model_SMode/model_SMode_exp3/Exp3_SMode_best')

        done = False
        step = 0
        while not done:
            state, reward, done, dis = env.step(actiongenerator(obs))
            # print("reward: ",reward)
            # print("done: ",done)
            # print("distance: ",dis['distance'])
            
            pos_hunter, ori_hunter, acc_hunter, pos_target, ori_target, acc_target, thrust = dronesim.siminfo()

            if step % 1 == 0:
                renderer.render(pos_hunter, ori_hunter, pos_target, ori_target)

            obs = state

            # time.sleep(0.05)
            step += 1

            if done:
                # print('step: ', step)
                # print('done')

                reason[dis['reason']] += 1

                if dis['distance'] <= 1:
                    success.append(1)
                    success_number += 1
                    step_number.append(step)
                else:
                    success.append(0)

                # time.sleep(10)
                agent.reset()
                obs = env.reset()

        env.stop()

# print(success_number/max_iteration)
print('----------------------')
print('average step: ', sum(step_number)/len(step_number), len(step_number), np.mean(step_number), np.var(step_number))
print('----------------------')
print('success rate: ', sum(success)/len(success))
print('----------------------')
print('result: (1 = success, 2 = max distance, 3 = max time)\n', reason)
