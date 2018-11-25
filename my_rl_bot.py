#!/usr/bin/env python3
# Python 3.6

import hlt
from hlt import constants
from hlt.positionals import Direction

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import logging
import tensorflow as tf
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
import keras.layers as L
sys.stderr = stderr
import numpy as np
import pickle

tf.reset_default_graph()
sess = tf.InteractiveSession()
keras.backend.set_session(sess)

from tables import *

fileh = open_file('C:/temp/er_store.h5', mode='a')
table = fileh.get_node('/experience_store/er_store')

with open('epsilon.pickle', 'rb') as f:
    epsilon = pickle.load(f)

game = hlt.Game()
max_turns = constants.MAX_TURNS

class DQNAgent():
    def __init__(self, name, state_shape, epsilon=0, reuse=False):
        """A simple DQN agent"""
        with tf.variable_scope(name, reuse=reuse):
            self.network = keras.models.Sequential()
            self.network.add(L.InputLayer(input_shape=state_shape))
            self.network.add(L.Dense(1024, activation='relu'))
            self.network.add(L.Dense(256, activation='relu'))
            self.network.add(L.Dense(5))
            self.network.load_weights('./single_turtle_network.hdf5')
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


n_actions = 5
state_dim = (5122,)
actions = ['n', 's', 'e', 'w', 'o']

agent = DQNAgent("dqn_agent", state_dim, epsilon=epsilon)
target_network = DQNAgent("target_network", state_dim)
sess.run(tf.global_variables_initializer())

def get_nn_repr(game, ship, my_squares, enemy_squares):
    me = game.me
    game_map = game.game_map
    halites =  np.array([x.halite_amount for y in game_map._cells for x in y])
    ship_pos = [0] * (game_map.width * game_map.height)
    x, y = ship.position.x, ship.position.y
    ship_pos[32*y+x] = 1

    my_pos = [0] * (game_map.width * game_map.height)
    for pos in my_squares:
        my_pos[32*pos.y + pos.x] = 1

    enemy_pos = [0] * (game_map.width * game_map.height)
    for pos in enemy_squares:
        enemy_pos[32 * pos.y + pos.x] = 1

    dropoff_pos = [0] * (game_map.width * game_map.height)
    dropoff_pos[32 * me.shipyard.position.y + me.shipyard.position.x] = 1

    halite = ship.halite_amount
    return list(halites) + ship_pos + my_pos + enemy_pos + dropoff_pos + [game.turn_number / max_turns, halite / 1000.0]


game.ready("MyRLBot")

logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))
logging.info(max_turns)

last_halite = None
last_state = None
spawned = False

while True:
    game.update_frame()
    game_map = game.game_map
    me = game.me
    logging.info(game.turn_number)
    done = game.turn_number == max_turns

    occupied_squares = [ship.position for player in game.players.values() for ship in player.get_ships()]
    my_squares = [ship.position for ship in me.get_ships()]
    enemy_squares = [x for x in occupied_squares if x not in my_squares]

    spawn_reward = 1000 if spawned else 0
    spawned = False

    reward = 0
    if last_halite:
        reward = me.halite_amount - last_halite + spawn_reward
    last_halite = me.halite_amount
    logging.info("Reward: {}".format(reward))

    command_queue = []

    for ship in me.get_ships():
        state = get_nn_repr(game, ship, my_squares, enemy_squares)

        if last_state:
            state_rep = table.row
            state_rep['state'] = last_state
            state_rep['next_state'] = state
            state_rep['action'] = last_action
            state_rep['reward'] = reward
            state_rep['done'] = done

            state_rep.append()
            table.flush()

        qvalues = agent.get_qvalues([state])
        action = agent.sample_actions(qvalues)[0]
        if action == 4:
            command_queue.append(ship.stay_still())
        else:
            command_queue.append(ship.move(actions[action]))

        last_state = state
        last_action = action

    if game.turn_number == 1:
        command_queue.append(me.shipyard.spawn())
        spawned = True

    if done:
        with open('epsilon.pickle', 'wb') as f:
            pickle.dump(epsilon * 0.999, f)
        fileh.close()

    game.end_turn(command_queue)

