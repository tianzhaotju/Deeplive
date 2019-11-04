import LiveStreamingEnv.final_fixed_env as env
import LiveStreamingEnv.load_trace as load_trace
import matplotlib.pyplot as plt
import time as time_package
import numpy as np
import ABR
from ddqn2 import DQNAgent

ACTION_SAPCE = []
for j in range(1,19):
    ACTION_SAPCE.append(j/10)

STATE_SIZE = 200 + 40
ACTION_SIZE = 20
BATCH_SIZE = 32
history_len = 40
done = False
agent = DQNAgent(STATE_SIZE,ACTION_SIZE)


def train(epoch,train_trace):
    # path setting
    TRAIN_TRACES = train_trace
    video_size_file = './video_trace/frame_trace_'  # video trace path setting,
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
    BIT_RATE = [500.0, 1200.0]  # kpbs
    # ABR setting
    cnt = 0
    # defalut setting
    bit_rate = 0
    target_buffer = 1
    # QOE setting
    reward_frame = 0
    reward_all = 0
    reward = 0
    SMOOTH_PENALTY = 0.02
    REBUF_PENALTY = 1.5
    LANTENCY_PENALTY = 0.005
    switch_num = 0
    rebuf_time = 0
    buffer_flag = 0
    cdn_flag = 0
    S_time_interval = [0] * 1000
    S_send_data_size = [0] * 1000
    S_frame_type = [0] * 1000
    S_frame_time_len = [0] * 1000
    S_buffer_size = [0] * 1000
    S_end_delay = [0] * 1000
    S_rebuf = [0] * 1000
    S_real_quality = [0] * 1000
    cdn_has_frame = [0] * 1000



    flag = False
    n = 0
    mark = 0
    marks = 0
    while True:

        if len(agent.memory) > BATCH_SIZE and cnt % 1000 == 0:
            agent.replay(BATCH_SIZE)

        reward_frame = 0

        time, time_interval, send_data_size, frame_time_len, rebuf, buffer_size, end_delay, cdn_newest_id, downlaod_id, cdn_has_frame, decision_flag, real_quality, buffer_flag, switch, cdn_flag, end_of_video = net_env.get_video_frame(
            bit_rate, target_buffer)
        cnt += 1

        switch_num += switch
        S_time_interval.append(time_interval)
        S_time_interval.pop(0)
        S_buffer_size.append(buffer_size)
        S_buffer_size.pop(0)
        S_send_data_size.append(send_data_size)
        S_send_data_size.pop(0)
        S_frame_time_len.append(frame_time_len)
        S_frame_time_len.pop(0)
        S_end_delay.append(end_delay)
        S_end_delay.pop(0)
        S_rebuf.append(rebuf)
        S_rebuf.pop(0)
        S_real_quality.append(real_quality)
        S_real_quality.pop(0)
        if decision_flag:
            S_frame_type.append(1)
            S_frame_type.pop(0)
        else:
            S_frame_type.append(0)
            S_frame_type.pop(0)

        rebuf_time += rebuf

        n+=1
        if not cdn_flag:
            reward_frame = frame_time_len * float(
                BIT_RATE[bit_rate]) / 1000 - REBUF_PENALTY * rebuf - LANTENCY_PENALTY * end_delay
        else:
            reward_frame = -(REBUF_PENALTY * rebuf)
        reward += reward_frame
        if decision_flag and not end_of_video:
            reward_frame += -(switch_num) * SMOOTH_PENALTY * (1200 - 500) / 1000
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
                # for i in S_frame_time_len[length-history_len:]:
                #     next_state.append(i*10)
                for i in S_rebuf[length-history_len:]:
                    next_state.append(i)
                for i in S_real_quality[length-history_len:]:
                    next_state.append(i)
                marks += 1
                if(n>=history_len-10):
                    next_state = np.reshape(next_state, [1, STATE_SIZE])
                    agent.remember(state, action, reward-2, next_state, done)
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
            # for i in S_frame_time_len[length-history_len:]:
            #     state.append(i*10)
            for i in S_rebuf[length-history_len:]:
                state.append(i)
            for i in S_real_quality[length-history_len:]:
                state.append(i)

            state = np.reshape(state, [1, STATE_SIZE])

            action = agent.act(state)
            bit_rate = action[0]
            target_buffer = ACTION_SAPCE[action[1]]

            switch_num = 0
            rebuf_time = 0


        reward_all += reward_frame
        if end_of_video:
            agent.update_target_model()

            # Narrow the range of results
            print("video count", video_count, reward_all,mark,marks)
            reward_all_sum += reward_all / 100
            video_count += 1
            if video_count >= len(all_file_names):
                agent.save("save/"+str(epoch)+".h5")
                break
            reward_all = 0
            bit_rate = 0
            target_buffer = 1.5
            S_time_interval = [0] * 1000
            S_send_data_size = [0] * 1000
            S_frame_type = [0] * 1000
            S_frame_time_len = [0] * 1000
            S_buffer_size = [0] * 1000
            S_end_delay = [0] * 1000
            cdn_has_frame = [0] * 1000
            rebuf_time = 0
            buffer_flag = 0
            cdn_flag = 0
            reward = 0
            flag = False
            n = 0
            mark = 0
            marks = 0


            switch_num = 0

    return reward_all_sum


def main():
    epoch = 1
    while epoch < 1000:
        train_trace = './network_trace500/'
        a = train(epoch,train_trace)
        print(str(epoch) + " : " + str(a))
        epoch +=1


if __name__ == "__main__":
    main()

