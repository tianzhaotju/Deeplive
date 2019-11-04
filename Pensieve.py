import numpy as np
import tensorflow as tf
import tflearn
import a3c

GAMMA = 0.99
ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6
S_INFO = 5  # bit_rate, buffer_size, now_chunk_size, bandwidth_measurement(throughput and time)
S_LEN = 50  # take how many frames in the past
A_DIM = 64
M_IN_K = 1000.0
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 6
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [500.0,850.0,1200.0,1850.0] # Kbps
BUFFER_NORM_FACTOR = 10.0
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
NN_MODEL = None


class Algorithm:
    def __init__(self):
        # fill your init vars
        n = 0
        self.BITRATE = [0, 1, 2, 3]
        self.TARGET_BUFFER = [0, 1, 2, 3]
        self.LATENCY_LIMIT = [1, 2, 3, 4]
        self.ACTION_SAPCE = []
        self.sess = tf.Session()
        self.actor = a3c.ActorNetwork(self.sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        self.critic = a3c.CriticNetwork(self.sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    # Intial
    def Initial(self,model_name):

        name = None
        name = model_name + "nn_model_ep_00000.ckpt"
        if name != None:
            self.saver.restore(self.sess, name)
        for i in self.BITRATE:
            for j in self.TARGET_BUFFER:
                for k in self.LATENCY_LIMIT:
                    action_apace = []
                    action_apace.append(i)
                    action_apace.append(j)
                    action_apace.append(k)
                    self.ACTION_SAPCE.append(action_apace)

    #Define your al
    def run(self,time,S_time_interval,S_send_data_size,S_chunk_len,S_rebuf,S_buffer_size,S_play_time_len,
            S_end_delay,S_decision_flag,S_buffer_flag,S_cdn_flag,S_skip_time,end_of_video,cdn_newest_id,download_id,cdn_has_frame,abr_init,start_avgbw):
        target_buffer = 1
        latency_limit = 4

        state = []
        length = len(S_time_interval)
        history_len = S_LEN
        for i in S_buffer_size[length - history_len:]:
            state.append(i * 0.1)
        for i in S_send_data_size[length - history_len:]:
            state.append(i * 0.00001)
        for i in S_time_interval[length - history_len:]:
            state.append(i * 10)
        for i in S_end_delay[length - history_len:]:
            state.append(i * 0.1)
        for i in S_rebuf[length - history_len:]:
            state.append(i)

        action_prob = self.actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
        action_cumsum = np.cumsum(action_prob)
        action = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
        bit_rate = self.ACTION_SAPCE[action][0]
        target_buffer = self.ACTION_SAPCE[action][1]
        latency_limit = self.ACTION_SAPCE[action][2]

        return bit_rate, target_buffer, latency_limit





def main():
    sess = tf.Session()
    actor = a3c.ActorNetwork(sess,
                             state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                             learning_rate=ACTOR_LR_RATE)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    name = "./Pensieve_models/nn_model_ep_3000.ckpt"
    saver.restore(sess, name)
    state = [np.zeros((S_INFO, S_LEN))]
    action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
    action_cumsum = np.cumsum(action_prob)
    bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
    target_buffer = 1
    print(bit_rate)

if __name__ == "__main__":
    main()

