from collections import deque
from game import SnakeGameAI, Direction, Block, BLOCK_SIZE
from model import make_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

MAX_MEMORY = 100_000
BATCH_SIZE = 64
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 1
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.998

        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = make_model(
            input_shape=[11], hidden_size=128, output_size=3)
        self.loss_fn = keras.losses.mean_squared_error
        self.optimizer = keras.optimizers.Adam(learning_rate=LR)

    def _epsilon_greedy_policy(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        new_action = [0, 0, 0]

        if np.random.rand() < self.epsilon:
            action_choice = np.random.randint(0, 2)
            new_action[action_choice] = 1
        else:
            Q_values = self.model.predict(state[np.newaxis])
            action_choice = np.argmax(Q_values[0])
            new_action[action_choice] = 1

        return new_action

    def sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.memory), size=batch_size)
        batch = [self.memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)
        ]
        return states, actions, rewards, next_states, dones

    def play_one_step(self, env, state):
        action = self._epsilon_greedy_policy(state)
        next_state, reward, done, info = env.play_step(action)
        self.memory.append((state, action, reward, next_state, done))
        return next_state, action, reward, done, info

    def training_step(self, batch_size):
        experiences = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = experiences
        target_Q_values = (rewards + self.gamma *
                           np.max(self.model.predict(next_states)))

        next_Q_values = self.model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + self.gamma * max_next_Q_values)

        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(
                all_Q_values * actions, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))


if __name__ == "__main__":
    agent = Agent()
    env = SnakeGameAI()
    env.reset()

    state = env._get_state()

    while True:
        next_state, action, reward, done, info = agent.play_one_step(
            env, state)

        state = next_state

        if done:
            agent.n_games += 1
            env.reset()

        if len(agent.memory) > BATCH_SIZE:
            agent.training_step(BATCH_SIZE)
