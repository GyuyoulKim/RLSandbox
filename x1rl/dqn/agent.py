import tensorflow as tf
from tensorflow.python.keras.backend import dtype

class DEEQAgent(tf.Module):
    def __init__(self, q_func, observation_shape, num_actions, lr, gamma):
        self.num_actions = num_actions
        self.gamma = gamma
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.losses = tf.keras.losses.Huber()
        with tf.name_scope('q_network'):
            self.q_network = q_func(observation_shape, num_actions)
        with tf.name_scope('target_q_network'):
            self.target_q_network = q_func(observation_shape, num_actions)

    @tf.function
    def step(self, state):
        q_values = self.q_network(state)
        output_actions = tf.argmax(q_values, axis=1)

        return output_actions, None, None, None

    @tf.function
    def train(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            q_t = self.q_network(states)
            q_t_selected = tf.reduce_sum(q_t * tf.one_hot(actions, self.num_actions, dtype=tf.float32), axis=1)

            q_tp1 = self.target_q_network(next_states)

            q_tp1_using_online_net = self.q_network(next_states)
            q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, axis=1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, self.num_actions, dtype=tf.float32), axis=1)

            dones = tf.cast(dones, q_tp1_best.dtype)
            q_tp1_best_masked = (1.0 - dones) * q_tp1_best

            q_t_selected_target = rewards + self.gamma * q_tp1_best_masked
            
            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            error = self.losses(q_t_selected_target, q_t_selected)

        grads = tape.gradient(error, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        return td_error

    @tf.function(autograph=False)
    def update_target(self):
        q_vars = self.q_network.trainable_variables
        target_q_vars = self.target_q_network.trainable_variables
        for var, var_target in zip(q_vars, target_q_vars):
            var_target.assign(var)

    def save_model(self, path_to_file):
        self.q_network.save(filepath=path_to_file)

    def load_model(self, path_to_file):
        self.q_network.load_model(path_to_file)