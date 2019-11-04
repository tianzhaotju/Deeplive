class Algorithm:
    def __init__(self):
        # fill your init vars
        self.buffer_size = 0

    # Intial
    def Initial(self,model_name):
        return None

    def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,
            S_end_delay, S_decision_flag, S_buffer_flag, S_cdn_flag, S_skip_time, end_of_video, cdn_newest_id,
            download_id, cdn_has_frame, IntialVars,start_avgbw):
        # record your params
        self.buffer_size = S_buffer_size[-1]
        bit_rate = 0
        RESEVOIR = 0.5
        CUSHION = 1.5

        if S_buffer_size[-1] < RESEVOIR:
            bit_rate = 0
        elif S_buffer_size[-1] >= RESEVOIR + CUSHION and S_buffer_size[-1] < CUSHION + CUSHION:
            bit_rate = 2
        elif S_buffer_size[-1] >= CUSHION + CUSHION:
            bit_rate = 3
        else:
            bit_rate = 1

        target_buffer = 3
        latency_limit = 3

        return bit_rate, target_buffer,latency_limit

