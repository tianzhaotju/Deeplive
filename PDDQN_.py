import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import Reshape
from keras.layers import Flatten

BIT_RATE = [500.0,850.0,1200.0,1850.0]

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200000)
        self.gamma = 0.95   # discount rate
        self.epsilon = 0.0  # exploration rate
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.0
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
        model = Sequential()
        # model.add(Dense(128,input_dim=self.state_size, activation='relu'))
        model.add(Reshape((50,5),input_shape=(self.state_size,)))
        model.add(Conv1D(5,kernel_size=4, activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))
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
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                # target[0][action] = reward + self.gamma * np.amax(t)
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class Algorithm:
    def __init__(self):
        # fill your init vars
        self.state_size = 250
        self.action_size = 64
        self.history_len = 50
        self.BITRATE = [0, 1, 2, 3]
        self.TARGET_BUFFER = [0, 1, 2, 3]
        self.LATENCY_LIMIT = [1, 2, 3, 4]
        self.ACTION_SAPCE = []
        self.agent = DQNAgent(self.state_size, self.action_size)

    # Intial
    def Initial(self,model_name):
        # name = "save/16.h5"
        name = str(model_name+"100.h5")
        self.agent.load(name)

        for i in self.BITRATE:
            for j in self.TARGET_BUFFER:
                for k in self.LATENCY_LIMIT:
                    action_apace = []
                    action_apace.append(i)
                    action_apace.append(j)
                    action_apace.append(k)
                    self.ACTION_SAPCE.append(action_apace)

    #Define your al
    def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,
            S_end_delay, S_decision_flag, S_buffer_flag, S_cdn_flag, S_skip_time, end_of_video, cdn_newest_id,
            download_id, cdn_has_frame, IntialVars, start_avgbw):

        target_buffer = 1
        latency_limit = 4

        state = []
        length = len(S_time_interval)
        history_len = self.history_len
        for i in S_buffer_size[length-history_len:]:
            state.append(i*0.1)
        for i in S_send_data_size[length-history_len:]:
            state.append(i*0.00001)
        for i in S_time_interval[length-history_len:]:
            state.append(i*10)
        for i in S_end_delay[length-history_len:]:
            state.append(i*0.1)
        for i in S_rebuf[length-history_len:]:
            state.append(i)

        state = np.reshape(state, [1, self.state_size])
        # print(state)
        action = self.agent.act(state)
        bit_rate = self.ACTION_SAPCE[action][0]
        target_buffer = self.ACTION_SAPCE[action][1]
        latency_limit = self.ACTION_SAPCE[action][2]

        return bit_rate, target_buffer,latency_limit
