import fixed_env as env
import load_trace as load_trace
import os
import time as tm
import csv
import tensorflow as tf


def test(user_id,ABR_NAME_,QoE_,NETWORK_TRACE_,VIDEO_TRACE_):
    #1  Algorithm Setting:  RBA, BBA, DYNAMIC, PDDQN, Pensieve
    ABR_NAME = ABR_NAME_
    #2  QoE Setting:  ar, al, hd, b, max
    QoE = QoE_
    #3  Network Dataset: high,  medium, low, fixed
    NETWORK_TRACE = NETWORK_TRACE_
    #4  Video Dataset: AsianCup_China_Uzbekistan, Fengtimo_2018_11_3, YYF_2018_08_12
    VIDEO_TRACE = VIDEO_TRACE_

    model_name = ""

    if ABR_NAME == 'BBA':
        import BBA as ABR
    if ABR_NAME == 'RBA':
        import RBA as ABR
    if ABR_NAME == 'DYNAMIC':
        import DYNAMIC as ABR
    if ABR_NAME == 'PDDQN':
        model_name = "./PDDQN_models/PDDQN_b/"
        import PDDQN_ as ABR
    if ABR_NAME == 'PDDQN-R':
        model_name = "./PDDQN_models/"+QoE+'/'
        import PDDQN_R as ABR
    if ABR_NAME == 'Pensieve':
        model_name = "./Pensieve_models/"+QoE+'/'
        import Pensieve as ABR

    SMOOTH_PENALTY = 0
    REBUF_PENALTY = 0.0
    LANTENCY_PENALTY = 0.0
    SKIP_PENALTY = 0.0
    BITRATE_REWARD = 0.0

    if QoE == 'al':
        SMOOTH_PENALTY = 0.01
        REBUF_PENALTY = 1.5
        LANTENCY_PENALTY = 0.01
        BITRATE_REWARD = 0.001
        SKIP_PENALTY = 1
    if QoE == 'ar':
        SMOOTH_PENALTY = 0.0
        REBUF_PENALTY = 3
        LANTENCY_PENALTY = 0.0
        BITRATE_REWARD = 0.001
        SKIP_PENALTY = 0.0
    if QoE == 'b':
        SMOOTH_PENALTY = 0.02
        REBUF_PENALTY = 1.5
        LANTENCY_PENALTY = 0.005
        BITRATE_REWARD = 0.001
        SKIP_PENALTY = 0.5
    if QoE == 'hd':
        SMOOTH_PENALTY = 0.0
        REBUF_PENALTY = 0.5
        LANTENCY_PENALTY = 0.0
        BITRATE_REWARD = 0.001
        SKIP_PENALTY = 0.0

    if QoE == 'max':
        SMOOTH_PENALTY = 0
        REBUF_PENALTY = 0.0
        LANTENCY_PENALTY = 0.0
        SKIP_PENALTY = 0.0
        BITRATE_REWARD = 0.001
        FILE_NAME = './'+'result/'+QoE+'_'+NETWORK_TRACE+'_'+VIDEO_TRACE+'.csv'
    else:
        FILE_NAME = './'+'result/'+ABR_NAME+'_'+QoE+'_'+NETWORK_TRACE+'_'+VIDEO_TRACE+'.csv'

    FILE_NAME = './' + 'result/Startup/' + NETWORK_TRACE +'/'+ABR_NAME+ '/QoE.csv'
    out = open(FILE_NAME, 'w', newline='')
    w = csv.writer(out)

    DEBUG = False

    LOG_FILE_PATH = './log/'

    # create result directory
    if not os.path.exists(LOG_FILE_PATH):
        os.makedirs(LOG_FILE_PATH)

    # -- End Configuration --

    network_trace_dir = './dataset/new_network_trace/' + NETWORK_TRACE + '/'
    video_trace_prefix = './dataset/video_trace/' + VIDEO_TRACE + '/frame_trace_'

    # load the trace
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(network_trace_dir)
    start_avgbw = (sum(all_cooked_bw[0][0:10])/10) *1000

    # random_seed
    random_seed = 2
    count = 0
    trace_count = 1
    FPS = 25
    frame_time_len = 0.04
    reward_all_sum = 0
    run_time = 0

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                                    all_cooked_bw=all_cooked_bw,
                                    random_seed=random_seed,
                                    logfile_path=LOG_FILE_PATH,
                                    VIDEO_SIZE_FILE=video_trace_prefix,
                                    Debug=DEBUG)

    abr = ABR.Algorithm()
    abr_init = abr.Initial(model_name)

    BIT_RATE = [500.0, 850.0, 1200.0, 1850.0]  # kpbs
    TARGET_BUFFER = [0.5,0.75,1,1.25]  # seconds
    # ABR setting
    RESEVOIR = 0.5
    CUSHION = 2

    cnt = 0
    # defalut setting
    last_bit_rate = 0
    bit_rate = 0
    target_buffer = 0
    latency_limit = 4

    # reward setting
    reward_frame = 0
    reward_all = 0

    # past_info setting
    past_frame_num = 200
    S_time_interval = [0] * past_frame_num
    S_send_data_size = [0] * past_frame_num
    S_chunk_len = [0] * past_frame_num
    S_rebuf = [0] * past_frame_num
    S_buffer_size = [0] * past_frame_num
    S_end_delay = [0] * past_frame_num
    S_chunk_size = [0] * past_frame_num
    S_play_time_len = [0] * past_frame_num
    S_decision_flag = [0] * past_frame_num
    S_buffer_flag = [0] * past_frame_num
    S_cdn_flag = [0] * past_frame_num
    S_skip_time = [0] * past_frame_num
    # params setting
    call_time_sum = 0
    reward_chunk = 0
    while True:

        reward_frame = 0

        time, time_interval, send_data_size, chunk_len, \
        rebuf, buffer_size, play_time_len, end_delay, \
        cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, decision_flag, \
        buffer_flag, cdn_flag, skip_flag, end_of_video = net_env.get_video_frame(bit_rate, target_buffer, latency_limit)
        # S_info is sequential order
        S_time_interval.pop(0)
        S_send_data_size.pop(0)
        S_chunk_len.pop(0)
        S_buffer_size.pop(0)
        S_rebuf.pop(0)
        S_end_delay.pop(0)
        S_play_time_len.pop(0)
        S_decision_flag.pop(0)
        S_buffer_flag.pop(0)
        S_cdn_flag.pop(0)
        S_skip_time.pop(0)

        S_time_interval.append(time_interval)
        S_send_data_size.append(send_data_size)
        S_chunk_len.append(chunk_len)
        S_buffer_size.append(buffer_size)
        S_rebuf.append(rebuf)
        S_end_delay.append(end_delay)
        S_play_time_len.append(play_time_len)
        S_decision_flag.append(decision_flag)
        S_buffer_flag.append(buffer_flag)
        S_cdn_flag.append(cdn_flag)
        S_skip_time.append(skip_frame_time_len)

        # QOE setting
        # if end_delay <= 1.0:
        #     LANTENCY_PENALTY = 0.005
        # else:
        #     LANTENCY_PENALTY = 0.01

        if not cdn_flag:
            reward_frame = frame_time_len * float(BIT_RATE[
                                                      bit_rate]) * BITRATE_REWARD - REBUF_PENALTY * rebuf - LANTENCY_PENALTY * end_delay - SKIP_PENALTY * skip_frame_time_len
        else:
            reward_frame = -(REBUF_PENALTY * rebuf)

        if decision_flag or end_of_video:
            reward_frame += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
            reward_chunk += reward_frame
            w.writerow([ABR_NAME,reward_chunk])
            reward_chunk = 0
            last_bit_rate = bit_rate

            # ----------------- Your Algorithm ---------------------

            cnt += 1
            timestamp_start = tm.time()
            bit_rate, target_buffer, latency_limit = abr.run(time,
                                                             S_time_interval,
                                                             S_send_data_size,
                                                             S_chunk_len,
                                                             S_rebuf,
                                                             S_buffer_size,
                                                             S_play_time_len,
                                                             S_end_delay,
                                                             S_decision_flag,
                                                             S_buffer_flag,
                                                             S_cdn_flag,
                                                             S_skip_time,
                                                             end_of_video,
                                                             cdn_newest_id,
                                                             download_id,
                                                             cdn_has_frame,
                                                             abr_init,
                                                             start_avgbw)
            start_avgbw = -1
            timestamp_end = tm.time()
            call_time_sum += timestamp_end - timestamp_start
            # -------------------- End --------------------------------
        else:
            reward_chunk += reward_frame
        if end_of_video:
            break




            # print("network traceID, network_reward, avg_running_time", trace_count, reward_all, call_time_sum / cnt)

            reward_all = reward_all/cnt
            reward_all_sum += reward_all
            run_time += call_time_sum / cnt
            if trace_count >= len(all_file_names):
                break
            trace_count += 1
            cnt = 0

            call_time_sum = 0
            last_bit_rate = 0
            reward_all = 0
            bit_rate = 0
            target_buffer = 0

            S_time_interval = [0] * past_frame_num
            S_send_data_size = [0] * past_frame_num
            S_chunk_len = [0] * past_frame_num
            S_rebuf = [0] * past_frame_num
            S_buffer_size = [0] * past_frame_num
            S_end_delay = [0] * past_frame_num
            S_chunk_size = [0] * past_frame_num
            S_play_time_len = [0] * past_frame_num
            S_decision_flag = [0] * past_frame_num
            S_buffer_flag = [0] * past_frame_num
            S_cdn_flag = [0] * past_frame_num

        reward_all += reward_frame

    return [reward_all_sum / trace_count, run_time / trace_count]


#1  Algorithm Setting:  RBA, BBA, DYNAMIC, PDDQN, Pensieve
ABR_NAME = ['RBA','BBA','DYNAMIC','PDDQN']
#2  QoE Setting:  ar, al, hd, b, max
QoE = ['al','ar','hd','b']
#3  Network Dataset: high,  medium, low, fixed
NETWORK_TRACE = ['high','medium','low','fixed']
#4  Video Dataset: AsianCup_China_Uzbekistan, Fengtimo_2018_11_3, YYF_2018_08_12
VIDEO_TRACE = ['AsianCup_China_Uzbekistan','Fengtimo_2018_11_3','YYF_2018_08_12']

k = "PDDQN"
i = NETWORK_TRACE[0]
j = QoE[3]
a = test("aaa", k, j, i, VIDEO_TRACE[0])
print(a)


# for i in NETWORK_TRACE:
#     for j in QoE:
#         FILE_NAME = './' + 'result/Network Trace/' + i + '/' + 'QoE_' + j + '/' + 'avg QoE.csv'
#         out = open(FILE_NAME, 'w', newline='')
#         w = csv.writer(out)
#         for k in ABR_NAME:
#             a = test("aaa", k, j, i, VIDEO_TRACE[0])
#             print(a)
#             b = [k]
#             b.append(a[0])
#             b.append(a[1])
#             w.writerow(b)

