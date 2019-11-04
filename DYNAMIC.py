import math
import numpy as np

BIT_RATE = [500.0,850.0,1200.0,1850.0]
MAX_STORE = 100
TARGET_BUFFER = 1
latency_limit = 4

class BBA():
    def __init__(self):
        self.buffer_size = 0

    def get_quality(self,segment):
        # record your params
        self.buffer_size = segment['buffer'][-1]
        bit_rate = 0
        RESEVOIR = 0.5
        CUSHION = 1.5

        if self.buffer_size < RESEVOIR:
            bit_rate = 0
        elif self.buffer_size >= RESEVOIR + CUSHION and self.buffer_size < CUSHION + CUSHION:
            bit_rate = 2
        elif self.buffer_size >= CUSHION + CUSHION:
            bit_rate = 3
        else:
            bit_rate = 1

        return bit_rate

    def get_first_quality(self,segment):
        return 0


class RBA:
    def __init__(self):
        self.buffer_size = 0
        self.p_rb = 1

    def get_quality(self, segment):
        # record your params
        bit_rate = 0
        bandwidth = self.predict_throughput(segment['throughputHistory'],0.8)
        tempBitrate = bandwidth * self.p_rb

        for i in range(len(BIT_RATE)):
            if tempBitrate >= BIT_RATE[i]:
                bit_rate = i

        return bit_rate

    def predict_throughput(self,throughputHistory,alpha):
        if alpha < 0 or alpha > 1:
            print("Invalid input!")
            alpha = 2/(len(throughputHistory)+1)
        predict = [0] * len(throughputHistory)

        for i in range(1,len(throughputHistory)):
            factor = 1 - pow(alpha, i)
            predict[i] =  (alpha * predict[i-1] + (1 - alpha) * throughputHistory[i])/factor
        return predict[-1]

    def get_first_quality(self,segment):
        return self.get_quality(segment)


class Dynamic():

    def __init__(self):
        # self.bba = Bola()
        self.bba = BBA()
        self.tput = RBA()
        self.is_buffer_based = False
        self.low_buffer_threshold = 1

    def get_quality(self, segment):
        level = segment['buffer'][-1]

        b = self.bba.get_quality(segment)
        t = self.tput.get_quality(segment)
        if self.is_buffer_based:
            if level < self.low_buffer_threshold and b < t:
                self.is_buffer_based = False
        else:
            if level > self.low_buffer_threshold and b >= t:
                self.is_buffer_based = True

        return b if self.is_buffer_based else t

    def get_first_quality(self,segment):
        if self.is_buffer_based:
            return self.bba.get_first_quality(segment)
        else:
            return self.tput.get_first_quality(segment)


class Algorithm:
    def __init__(self):
        self.dynamic = Dynamic()
        self.is_first = True
        self.next_throughput = 0
        self.next_latency = 0

    def Initial(self,model_name):
        self.last_bit_rate = 0

    def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,
            S_end_delay, S_decision_flag, S_buffer_flag, S_cdn_flag, S_skip_time, end_of_video, cdn_newest_id,
            download_id, cdn_has_frame, IntialVars, start_avgbw):
        bit_rate = 0
        target_buffer = TARGET_BUFFER
        throughputHistory = []
        if start_avgbw != -1:
            throughputHistory.append(start_avgbw)

            segment = {}
            segment['throughputHistory'] = throughputHistory
            return self.dynamic.get_first_quality(segment),target_buffer,latency_limit

        for i in range(len(S_send_data_size)-MAX_STORE,len((S_send_data_size))):
            send_data_size = S_send_data_size[i]
            time_interval = S_time_interval[i]
            bw = 0
            if time_interval != 0:
                bw = (send_data_size / time_interval) / 1000
                throughputHistory.append(bw)
        segment = {}
        segment['buffer'] = np.array(S_buffer_size[-MAX_STORE:])
        segment['time'] = np.array(S_time_interval[-MAX_STORE:])
        segment['latency'] = np.array(S_buffer_size[-MAX_STORE -1:-1]) - np.array(S_buffer_size[-MAX_STORE:])
        segment['throughputHistory'] = np.array(throughputHistory)

        bit_rate = self.dynamic.get_quality(segment=segment)
        target_buffer = TARGET_BUFFER

        return bit_rate, target_buffer,latency_limit
