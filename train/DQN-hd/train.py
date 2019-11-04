import fixed_env as env
import load_trace as load_trace
import matplotlib.pyplot as plt
import time as time_package
import numpy as np
import ABR
from ddqn import DQNAgent

BITRATE = [0,1,2,3]
TARGET_BUFFER = [0,1,2,3]
LATENCY_LIMIT = [1,2,3,4]
ACTION_SAPCE = []
for i in BITRATE:
    for j in TARGET_BUFFER:
        for k in LATENCY_LIMIT:
            action_apace = []
            action_apace.append(i)
            action_apace.append(j)
            action_apace.append(k)
            ACTION_SAPCE.append(action_apace)

STATE_SIZE = 250
ACTION_SIZE = len(BITRATE)*len(TARGET_BUFFER)*len(LATENCY_LIMIT)
BATCH_SIZE = 32
history_len = 50
done = False
agent = DQNAgent(STATE_SIZE,ACTION_SIZE)



def train(epoch,train_trace):
    # path setting
    TRAIN_TRACES = train_trace
    video_size_file = './dataset/video_trace/sports/frame_trace_'  # video trace path setting,
    LogFile_Path = "./log/"  # log file trace path setting,
    # load the trace
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRAIN_TRACES)
    # random_seed
    random_seed = 2
    video_count = 0
    frame_time_len = 0.04
    reward_all_sum = 0
    # init the environment
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=random_seed,
                              logfile_path=LogFile_Path,
                              VIDEO_SIZE_FILE=video_size_file,
                              Debug=False)
    BIT_RATE = [500.0,850.0,1200.0,1850.0]  # kpbs
    # ABR setting
    cnt = 0
    # defalut setting
    bit_rate = 0
    last_bit_rate = 0
    target_buffer = 1
    latency_limit = 7
    # QOE setting
    reward_frame = 0
    reward_all = 0
    reward = 0
    SMOOTH_PENALTY = 0.0
    REBUF_PENALTY = 0.5
    LANTENCY_PENALTY = 0.0
    BITRATE_REWARD = 0.001
    SKIP_PENALTY = 0.0
    switch_num = 0
    rebuf_time = 0
    buffer_flag = 0
    cdn_flag = 0
    S_time_interval = [0] * 100
    S_send_data_size = [0] * 100
    S_buffer_size = [0] * 100
    S_end_delay = [0] * 100
    S_rebuf = [0] * 100


    flag = False
    n = 0
    mark = 0
    marks = 0
    while True:

        if len(agent.memory) > BATCH_SIZE and cnt % 1000 == 0:
            agent.replay(BATCH_SIZE)

        reward_frame = 0
        time, time_interval, send_data_size, chunk_len, \
        rebuf, buffer_size, play_time_len, end_delay, \
        cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, decision_flag, \
        buffer_flag, cdn_flag, skip_flag, end_of_video = net_env.get_video_frame(bit_rate, target_buffer, latency_limit)

        cnt += 1

        S_time_interval.append(time_interval)
        S_time_interval.pop(0)
        S_buffer_size.append(buffer_size)
        S_buffer_size.pop(0)
        S_send_data_size.append(send_data_size)
        S_send_data_size.pop(0)
        S_end_delay.append(end_delay)
        S_end_delay.pop(0)
        S_rebuf.append(rebuf)
        S_rebuf.pop(0)

        # # QOE setting
        # if end_delay <= 1.0:
        #     LANTENCY_PENALTY = 0.005
        # else:
        #     LANTENCY_PENALTY = 0.01

        if not cdn_flag:
            reward_frame = frame_time_len * float(BIT_RATE[
                                                      bit_rate]) * BITRATE_REWARD - REBUF_PENALTY * rebuf - LANTENCY_PENALTY * end_delay - SKIP_PENALTY * skip_frame_time_len
        else:
            reward_frame = -(REBUF_PENALTY * rebuf)
        rebuf_time += rebuf
        n+=1
        reward += reward_frame
        if decision_flag and not end_of_video:
            reward_frame = -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
            last_bit_rate = bit_rate
            reward += reward_frame
            length = len(S_buffer_size)
            if flag:
                next_state = []

                for i in S_buffer_size[length-history_len:]:
                    next_state.append(i*0.1)
                for i in S_send_data_size[length-history_len:]:
                    next_state.append(i*0.00001)
                for i in S_time_interval[length-history_len:]:
                    next_state.append(i*10)
                for i in S_end_delay[length-history_len:]:
                    next_state.append(i*0.1)
                for i in S_rebuf[length-history_len:]:
                    next_state.append(i)
                marks += 1
                if(n>=history_len-40):
                    next_state = np.reshape(next_state, [1,STATE_SIZE])
                    agent.remember(state, action, reward, next_state, done)
                    reward = 0
                else:
                    mark += 1
                n = 0

            flag = True
            state = []

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


            state = np.reshape(state, [1,STATE_SIZE])
            action = agent.act(state)
            bit_rate = ACTION_SAPCE[action][0]
            target_buffer = ACTION_SAPCE[action][1]
            latency_limit = ACTION_SAPCE[action][2]
            switch_num = 0
            rebuf_time = 0


        reward_all += reward_frame
        if end_of_video:
            agent.update_target_model()

            # Narrow the range of results
            print("video count", video_count, reward_all,mark,marks)
            reward_all_sum += reward_all / 20
            video_count += 1
            if video_count >= len(all_file_names):
                agent.save("save/"+str(epoch)+".h5")
                break
            reward_all = 0
            bit_rate = 0
            target_buffer = 1
            S_time_interval = [0] * 100
            S_send_data_size = [0] * 100
            S_buffer_size = [0] * 100
            S_end_delay = [0] * 100
            S_rebuf = [0] * 100
            rebuf_time = 0
            buffer_flag = 0
            cdn_flag = 0
            reward = 0
            flag = False
            n = 0
            mark = 0
            marks = 0

    return reward_all_sum

import csv
def main():
    epoch = 1
    out = open("QoE.csv", 'w', newline='')
    w = csv.writer(out)
    while epoch <= 100:
        train_trace = './dataset/network_trace/fixed/'
        a = train(epoch,train_trace)
        print(str(epoch) + " : " + str(a))
        w.writerow([epoch,a])
        out.flush()
        epoch +=1


if __name__ == "__main__":
    main()

