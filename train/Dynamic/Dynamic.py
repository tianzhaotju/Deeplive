import math
import numpy as np

BIT_RATE = [500.0, 1200.0]
MAX_STORE = 100
TARGET_BUFFER = 2

# class Ewma():
#
#     def __init__(self, segment):
#
#         self.throughput = None
#         self.latency = None
#         self.segment_time = 1
#         self.half_life = [8000, 3000]
#         self.latency_half_life = [h / self.segment_time for h in self.half_life]
#         self.ThroughputHistory = segment['ThroughputHistory']
#         self.throughput = [0] * len(self.half_life)
#         self.weight_throughput = 0
#         self.latency = [0] * len(self.half_life)
#         self.weight_latency = 0
#
#     def push(self, time, tput, lat):
#
#         for i in range(len(self.half_life)):
#             alpha = math.pow(0.5, time / self.half_life[i])
#             self.throughput[i] = alpha * self.throughput[i] + (1 - alpha) * tput
#             alpha = math.pow(0.5, 1 / self.latency_half_life[i])
#             self.latency[i] = alpha * self.latency[i] + (1 - alpha) * lat
#
#         self.weight_throughput += time
#         self.weight_latency += 1
#
#         tput = None
#         lat = None
#         for i in range(len(self.half_life)):
#             zero_factor = 1 - math.pow(0.5, self.weight_throughput / self.half_life[i])
#             t = self.throughput[i] / zero_factor
#             tput = t if tput == None else min(tput, t)  # conservative case is min
#             zero_factor = 1 - math.pow(0.5, self.weight_latency / self.latency_half_life[i])
#             l = self.latency[i] / zero_factor
#             lat = l if lat == None else max(lat, l) # conservative case is max
#         self.throughput = tput
#         self.latency = lat
#
#         return self.throughput,self.latency


class SlidingWindow():

    def __init__(self):
        self.window_size = [MAX_STORE]
        self.throughput = None
        self.latency = None

    def get_next(self, segment):

        tput = None
        lat = None
        for ws in self.window_size:
            sample = segment['throughput'][-ws:]
            t = sum(sample) / len(sample)
            tput = t if tput == None else min(tput, t)
            sample = segment['latency'][-ws:]
            l = sum(sample) / len(sample)
            lat = l if lat == None else max(lat, l)
        self.throughput = tput
        self.latency = lat
        return self.throughput,self.latency


class Bola():

    def __init__(self):
        self.utility_offset = -math.log(BIT_RATE[0])
        self.utilities = [math.log(b) +self.utility_offset for b in BIT_RATE]
        self.segment_time = 1 #s
        self.gp = 5
        self.buffer_size = TARGET_BUFFER #s
        self.abr_osc = True
        self.Vp = (self.buffer_size - self.segment_time) / (self.utilities[-1] + self.gp)
        self.last_seek_index = 0 # TODO
        self.last_quality = 0
        self.slidingWindow = SlidingWindow()

    def quality_from_buffer(self,segment):
        level = segment['buffer']
        quality = 0
        score = None
        for q in range(len(BIT_RATE)):
            s = ((self.Vp * (self.utilities[q] + self.gp) - level) / BIT_RATE[q])
            if score == None or s > score:
                quality = q
                score = s
        return quality

    def get_quality(self, segment):
        quality = self.quality_from_buffer(segment)
        throughput, latency = self.slidingWindow.get_next(segment)
        if quality > self.last_quality:
            quality_t = self.quality_from_throughput(throughput, latency)
            if self.last_quality > quality_t and quality > quality_t:
                quality = self.last_quality
            elif quality > quality_t:
                if not self.abr_osc:
                    quality = quality_t + 1
                else:
                    quality = quality_t
        self.last_quality = quality
        return quality

    def quality_from_throughput(self, tput,lat):

        p = self.segment_time
        quality = 0
        while (quality + 1 < len(BIT_RATE) and lat + p * BIT_RATE[quality + 1] / tput <= p):
            quality += 1
        return quality

    def get_first_quality(self):
        return 0


class ThroughputRule():
    def __init__(self):
        self.segment_time = 1  # s
        self.safety_factor = 0.9
        self.slidingWindow = SlidingWindow()

    def get_quality(self, segment):
        throughput,latency = self.slidingWindow.get_next(segment)
        quality = self.quality_from_throughput(throughput * self.safety_factor,latency)
        return quality

    def quality_from_throughput(self, tput,lat):

        p = self.segment_time
        quality = 0
        while (quality + 1 < len(BIT_RATE) and lat + p * BIT_RATE[quality + 1] / tput <= p):
            quality += 1
        return quality

    def get_first_quality(self):
        return 0


class Dynamic():

    def __init__(self):
        self.bola = Bola()
        self.tput = ThroughputRule()
        self.is_bola = False
        self.low_buffer_threshold = 1

    def get_quality(self, segment):
        level = segment['buffer'][-1]

        b = self.bola.get_quality(segment)
        t = self.tput.get_quality(segment)

        if self.is_bola:
            if level < self.low_buffer_threshold and b < t:
                self.is_bola = False
        else:
            if level > self.low_buffer_threshold and b >= t:
                self.is_bola = True

        return b if self.is_bola else t

    def get_first_quality(self):
        if self.is_bola:
            return self.bola.get_first_quality()
        else:
            return self.tput.get_first_quality()


class Algorithm:
    def __init__(self):
        self.dynamic = Dynamic()
        self.is_first = True
        self.next_throughput = 0
        self.next_latency = 0

    def Initial(self):
        self.last_bit_rate = 0

    # Define your al
    def run(self, S_time_interval, S_send_data_size, S_frame_time_len, S_frame_type, S_buffer_size, S_end_delay,
            rebuf_time, cdn_has_frame, cdn_flag, buffer_flag):

        if(self.is_first):
            if S_time_interval[-MAX_STORE] != 0:
                self.is_first = False
            return self.dynamic.get_first_quality()

        segment = {}
        segment['buffer'] = np.array(S_buffer_size[-MAX_STORE:])
        segment['latency'] = np.array(S_end_delay[-MAX_STORE:])
        segment['throughput'] = np.array(S_send_data_size[-MAX_STORE:])/np.array(S_time_interval[-MAX_STORE:])

        bit_rate = self.dynamic.get_quality(segment=segment)
        target_buffer = TARGET_BUFFER

        return bit_rate, target_buffer
