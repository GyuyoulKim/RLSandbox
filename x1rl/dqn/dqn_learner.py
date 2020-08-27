import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from x1rl.common.atari_wrappers import make_atari, wrap_deepmind
from x1rl import logger
from .model import create_q_model, create_duel_q_model
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

def make_env(env_id, seed):
    env = make_atari(env_id)
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env.seed(seed)
    return env

def learn(env_id, num_steps, render, args):
    arg_parser = deepq_arg_parser()
    args, _ = arg_parser.parse_known_args(args)

    # Use the Baseline Atari environment because of Deepmind helper functions
    env = make_env(env_id, 42)

    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    if args.mode == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
    if args.duel is True:
        model = create_duel_q_model(input_shape, num_actions)
        model_target = create_duel_q_model(input_shape, num_actions)
    else:
        model = create_q_model(input_shape, num_actions)
        model_target = create_q_model(input_shape, num_actions)

    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time
    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    loss_function = keras.losses.Huber()

    """
    ## Train
    """
    # Configuration paramaters for the whole setup
    gamma = 0.99  # Discount factor for past rewards
    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    epsilon_interval = (
        epsilon_max - epsilon_min
    )  # Rate at which to reduce chance of random action being taken
    batch_size = 32  # Size of batch taken from replay buffer
    # Number of frames to take random action and observe output
    epsilon_random_frames = 50000
    # Number of frames for exploration
    epsilon_greedy_frames = num_steps * 0.1
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 100000
    # Train the model after 4 actions
    update_after_actions = 4
    # How often to update the target network
    update_target_network = 10000

    # Experience replay buffers
    replay_memory = ReplayMemory(max_memory_length)
    episode_rewards = [0.0]
    episode_count = 0
    state = np.array(env.reset())

    start_time = time.time()

    for t in range(num_steps):
        if render is True:
            env.render('human')

        # Use epsilon-greedy for exploration
        if t < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)

        # Save actions and states in replay buffer
        replay_memory.append(state, action, reward, state_next, done)
        state = state_next

        episode_rewards[-1] += reward
        
        if done:
            state = env.reset()
            episode_rewards.append(0.0)
            episode_count += 1

        # Update every fourth frame and once batch size is over 32
        if t % update_after_actions == 0 and t > batch_size:

            state_sample, action_sample, rewards_sample, state_next_sample, done_sample = replay_memory.sample(batch_size)

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if t % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())

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
    model.save(filepath=path_to_file)

    return

def play(env_id, model_file, episodes):
    import math
    # Use the Baseline Atari environment because of Deepmind helper functions
    env = make_env(env_id, math.trunc(time.time()))

    path_to = './data'
    path_to_file = path_to + '/' + model_file

    if os.path.isfile(path_to_file):
        model = keras.models.load_model(path_to_file)
    else:
        logger.error('filename:{} doesn\'t exist'.format(path_to_file))
        return

    for _ in range(episodes):
        is_done = False
        state = np.array(env.reset())
        
        while not is_done:
            env.render('human')

            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
            
            state, _, is_done, _  = env.step(action)

    env.close()
    return

if __name__ == '__main__':
    learn()