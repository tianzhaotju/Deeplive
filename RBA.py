import time as tm

BIT_RATE = [500.0,850.0,1200.0,1850.0]
class Algorithm:
    def __init__(self):
        # fill your init vars
        self.buffer_size = 0
        self.p_rb = 1

    # Intial
    def Initial(self,model_name):
        IntialVars = {'throughputHistory':[]}
        return IntialVars
    def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,
            S_end_delay, S_decision_flag, S_buffer_flag, S_cdn_flag, S_skip_time, end_of_video, cdn_newest_id,
            download_id, cdn_has_frame, IntialVars,start_avgbw):
        # record your params
        target_buffer = 1
        bit_rate = 0
        throughputHistory = []
        if start_avgbw!=-1:
            throughputHistory.append(start_avgbw)
        else:
            for i in range(len(S_send_data_size)):
                send_data_size = S_send_data_size[i]
                time_interval = S_time_interval[i]
                bw = 0
                if time_interval != 0:
                    bw = (send_data_size/time_interval)/1000
                    throughputHistory.append(bw)
        bandwidth = self.predict_throughput(throughputHistory,0.8)
        tempBitrate = bandwidth * self.p_rb
        for i in range(len(BIT_RATE)):
            if tempBitrate >= BIT_RATE[i]:
                bit_rate = i
        latency_limit = 4
        return bit_rate, target_buffer,latency_limit

    def predict_throughput(self,throughputHistory,alpha):
        if alpha < 0 or alpha > 1:
            print("Invalid input!")
            alpha = 2/(len(throughputHistory)+1)
        predict = [0] * len(throughputHistory)

        for i in range(1,len(throughputHistory)):
            factor = 1 - pow(alpha, i)
            predict[i] =  (alpha * predict[i-1] + (1 - alpha) * throughputHistory[i])/factor
        return predict[-1]
