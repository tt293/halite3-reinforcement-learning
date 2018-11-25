import sys
import math
import random
import numpy as np
from collections import namedtuple
from time import time

import tensorflow as tf
import keras
import keras.layers as L
tf.reset_default_graph()
sess = tf.InteractiveSession()
keras.backend.set_session(sess)

from tables import *

class DQNAgent():
    def __init__(self, name, state_shape, epsilon=0, weights_file=None, reuse=False):
        """A simple DQN agent"""
        with tf.variable_scope(name, reuse=reuse):
            self.network = keras.models.Sequential()
            self.network.add(L.InputLayer(input_shape=state_shape))
            self.network.add(L.Dense(1024, activation='relu'))
            self.network.add(L.Dense(256, activation='relu'))
            self.network.add(L.Dense(5))
            if weights_file:
                self.network.load_weights(weights_file)
            # prepare a graph for agent step
            self.state_t = tf.placeholder('float32', [None,] + list(state_shape))
            self.qvalues_t = self.get_symbolic_qvalues(self.state_t)

            self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            self.epsilon = epsilon

    def get_symbolic_qvalues(self, state_t):
        """takes agent's observation, returns qvalues. Both are tf Tensors"""
        return self.network(state_t)

    def get_qvalues(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        sess = tf.get_default_session()
        return sess.run(self.qvalues_t, {self.state_t: state_t})

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p = [1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


def sample_batch(table, batch_size):
    idxs = np.random.randint(0, table.nrows, size=(batch_size,))
    els = table[idxs]
    return {
        obs_ph: [x[4] for x in els], actions_ph: [x[0] for x in els], rewards_ph: [x[3] for x in els],
        next_obs_ph: [x[2] for x in els], is_done_ph: [x[1] for x in els]
    }


state_dim = (5122,)
n_actions = 5

obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_dim)
actions_ph = tf.placeholder(tf.int32, shape=[None])
rewards_ph = tf.placeholder(tf.float32, shape=[None])
next_obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_dim)
is_done_ph = tf.placeholder(tf.float32, shape=[None])
is_not_done = 1 - is_done_ph
gamma = 0.99


class TrainAgent():
    def __init__(self):
        self.agent = DQNAgent("dqn_agent", state_dim, epsilon=0.5, weights_file='./single_turtle_network.hdf5')
        self.target_network = DQNAgent("target_network", state_dim, weights_file='./single_turtle_target_network.hdf5')

        current_qvalues = self.agent.get_symbolic_qvalues(obs_ph)
        current_action_qvalues = tf.reduce_sum(tf.one_hot(actions_ph, n_actions) * current_qvalues, axis=1)

        # compute q-values for NEXT states with target network
        next_qvalues_target = self.target_network.get_symbolic_qvalues(next_obs_ph)

        # compute state values by taking max over next_qvalues_target for all actions
        next_state_values_target = tf.reduce_max(next_qvalues_target, axis=-1)

        # compute Q_reference(s,a) as per formula above.
        reference_qvalues = rewards_ph + gamma * next_state_values_target * is_not_done

        # Define loss function for sgd.
        td_loss = (current_action_qvalues - reference_qvalues) ** 2
        self.td_loss = tf.reduce_mean(td_loss)

        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(td_loss, var_list=self.agent.weights)

        sess.run(tf.global_variables_initializer())
        self.counter = 0


    def train(self):
        fileh = open_file('C:/temp/er_store.h5', mode='a')
        table = fileh.get_node('/experience_store/er_store')

        for i in range(10):
            _, loss_t = sess.run([self.train_step, self.td_loss], sample_batch(table, batch_size=64))
        #print("Trained network, ER entries: {}".format(table.nrows))

        if self.counter % 10 == 0:
            print(self.counter)
        if self.counter % 25 == 0:
            print("Updating target network, counter {}".format(self.counter))
            self.agent.network.save_weights('./single_turtle_network.hdf5')
            self.agent.network.save_weights('./single_turtle_target_network.hdf5')
            self.target_network = DQNAgent("target_network", state_dim, weights_file='./single_turtle_target_network.hdf5')

        if table.nrows > 100000:
            table.remove_rows(0, 25000)
            table.flush()

        self.counter += 1
        fileh.close()