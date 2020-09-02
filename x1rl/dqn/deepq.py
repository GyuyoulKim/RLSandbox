import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from x1rl.common.atari_wrappers import make_atari, wrap_deepmind
from x1rl import logger
from .model import create_q_model, create_duel_q_model
from .agent import DEEQAgent
from .replay_memory import ReplayMemory

import os
import time
import shutil

def arg_parser():
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def deepq_arg_parser():
    parser = arg_parser()
    parser.add_argument('--duel', help='use duel network',                      default=False, action='store_true')
    parser.add_argument('--mode', help='choose to use cpu or gpu',  type=str,   default='gpu')
    return parser

def make_env(env_id, seed, is_play=False):
    if is_play is False:
        env = make_atari(env_id)
    else:
        env = gym.make(env_id)
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env.seed(seed)
    return env

def learn(env_id, num_steps, render, args):
    arg_parser = deepq_arg_parser()
    args, _ = arg_parser.parse_known_args(args)

        # Configuration paramaters for the whole setup
    gamma = 0.99  # Discount factor for past rewards
    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    epsilon_interval = (
        epsilon_max - epsilon_min
    )  # Rate at which to reduce chance of random action being taken
    batch_size = 32  # Size of batch taken from replay buffer
    # Number of frames for exploration
    epsilon_greedy_frames = num_steps * 0.1
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 100000
    # Train the model after 4 actions
    train_frequency = 4
    update_after_actions = 4
    # How often to update the target network
    update_target_network = 1000
    # When agent starts learning
    learning_starts = 10000

    # Use the Baseline Atari environment because of Deepmind helper functions
    env = make_env(env_id, 42)

    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    if args.mode == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if args.duel is True:
        q_func = create_duel_q_model
    else:
        q_func = create_q_model

    agent = DEEQAgent(q_func, input_shape, num_actions, 1e-4, gamma)

    # Experience replay buffers
    replay_memory = ReplayMemory(max_memory_length)
    episode_rewards = [0.0]
    episode_count = 0

    state = env.reset()
    state = np.expand_dims(np.array(state), axis=0)

    start_time = time.time()

    for t in range(num_steps):
        if render is True:
            env.render('human')

        # Use epsilon-greedy for exploration
        if epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.int64(np.random.choice(num_actions))
        else:
            # Predict action Q-values
            # From environment state
            action, _, _, _ = agent.step(tf.constant(state))
            # Take best action
            action = action[0].numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step(action)
        state_next = np.expand_dims(np.array(state_next), axis=0) 

        # Save actions and states in replay buffer
        replay_memory.append(state[0], action, reward, state_next[0], float(done))
        state = state_next

        episode_rewards[-1] += reward
        
        if done:
            state = env.reset()
            state = np.expand_dims(np.array(state), axis=0)
            episode_rewards.append(0.0)
            episode_count += 1

        # Update every fourth frame and once batch size is over 32
        if t > learning_starts and t % train_frequency == 0:
            states, actions, rewards, next_states, dones = replay_memory.sample(batch_size)
            states, next_states = tf.constant(states), tf.constant(next_states)
            actions, rewards, dones = tf.constant(actions), tf.constant(rewards), tf.constant(dones)

            agent.train(states, actions, rewards, next_states, dones)


        if t > learning_starts and t % update_target_network == 0:
            # update the the target network with new weights
            agent.update_target()

        if done and episode_count % 100 == 0:
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 2)
            # Log details
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", episode_count)
            logger.record_tabular("mean 100 episodes reward", mean_100ep_reward)
            logger.dump_tabular()

    env.close()

    end_time = time.time()
    logger.record_tabular("running time", end_time - start_time)
    logger.dump_tabular()

    path_to = './data'
    os.makedirs(path_to, exist_ok=True)
    path_to_file = path_to + '/' + env_id + '-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '-dqn' + '-' + args.mode + '.h5'
    agent.save_model(path_to_file)

    return    

def play(env_id, model_file, episodes):
    import math
    # Use the Baseline Atari environment because of Deepmind helper functions
    env = make_env(env_id, math.trunc(time.time()))

    path_to = './data'
    path_to_file = path_to + '/' + model_file

    if os.path.isfile(path_to_file):
        q_func = keras.models.load_model(path_to_file)
    else:
        logger.error('filename:{} doesn\'t exist'.format(path_to_file))
        return

    for _ in range(episodes):
        is_done = False
        state = env.reset()
        state = np.expand_dims(np.array(state), axis=0)
        
        while not is_done:
            env.render('human')

            # Predict action Q-values
            # From environment state
            action_probs = q_func(tf.constant(state), training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
            
            state, _, is_done, _  = env.step(action)
            state = np.expand_dims(np.array(state), axis=0)

    env.close()
    return

if __name__ == '__main__':
    learn()