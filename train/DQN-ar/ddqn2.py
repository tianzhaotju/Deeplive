# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Input
from keras.layers import Reshape
from keras.models import Model
import tensorflow as tf


EPISODES = 5000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200000)
        self.gamma = 0.95   # discount rate
        self.epsilon = 0.90  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.001
        self.learning_rate = 0.0001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        ip = Input(shape=(self.state_size,))
        y1 = Dense(64,activation='relu')(ip)
        y2 = Dense(64,activation='relu')(y1)
        y = Dense(self.action_size,activation='relu')(y2)
        model = Model(ip,y)
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.randrange(2),random.randrange(self.action_size-2)]
        act_values = self.model.predict(state)
        return [np.argmax(act_values[0][0:2]),np.argmax(act_values[0][2:])-2]  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action[0]] = reward
                target[0][action[1]] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t0 = self.target_model.predict(next_state)[0][0:2]
                t1 = self.target_model.predict(next_state)[0][2:]
                target[0][action[0]] = reward + self.gamma * np.amax(t0)
                target[0][action[1]] = reward + self.gamma * np.amax(t1)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
