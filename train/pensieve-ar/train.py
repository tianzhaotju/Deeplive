import os
import logging
import multiprocessing as mp
import fixed_env as env
import load_trace as load_trace
import matplotlib.pyplot as plt
import time as time_package
import tensorflow as tf
import numpy as np
import a3c
import csv


S_INFO = 5  # bit_rate, buffer_size, now_chunk_size, bandwidth_measurement(throughput and time)
S_LEN =  50  # take how many frames in the past
M_IN_K = 1000.0
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 8
NUM_EPOCH = 300
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 1000
VIDEO_BIT_RATE = [500.0,850.0,1200.0,1850.0] # Kbps
BUFFER_NORM_FACTOR = 10.0
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './dataset/network_trace/fixed/'
VIDEO_TRACES = './dataset/video_trace/sports/frame_trace_'
# NN_MODEL = './results/pretrain_linear_reward.ckpt'
NN_MODEL = None
BIT_RATE = [500.0,850.0,1200.0,1850.0]  # kpbs
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
A_DIM = len(ACTION_SAPCE)

out = open("QoE.csv", "w", newline="")
w = csv.writer(out)


def central_agent(net_params_queues, exp_queues):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    with tf.Session() as sess, open(LOG_FILE + '_test', 'wb') as test_log_file:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver(max_to_keep = 50)  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        epoch = 0

        # assemble experiences from agents, compute the gradients
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])
                # Note: this is synchronous version of the parallel training,
                # which is easier to understand and probe. The framework can be
                # fairly easily modified to support asynchronous training.
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0

            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []

            for i in range(NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])

            # compute aggregated gradient
            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            # assembled_actor_gradient = actor_gradient_batch[0]
            # assembled_critic_gradient = critic_gradient_batch[0]
            # for i in xrange(len(actor_gradient_batch) - 1):
            #     for j in xrange(len(assembled_actor_gradient)):
            #             assembled_actor_gradient[j] += actor_gradient_batch[i][j]
            #             assembled_critic_gradient[j] += critic_gradient_batch[i][j]
            # actor.apply_gradients(assembled_actor_gradient)
            # critic.apply_gradients(assembled_critic_gradient)
            for i in range(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

            # log training information
            epoch += 1
            avg_reward = total_reward / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len

            logging.info('Epoch: ' + str(epoch) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Avg_entropy: ' + str(avg_entropy))

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy
            })

            writer.add_summary(summary_str, epoch)
            writer.flush()
            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                logging.info("Model saved in file: " + save_path)



def agent(agent_id, all_cooked_time, all_cooked_bw,all_file_names,video_size_file, net_params_queue, exp_queue):
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw,
                                  random_seed=agent_id,
                                  VIDEO_SIZE_FILE=video_size_file,
                                  Debug=False)

    with tf.Session() as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'wb') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        bit_rate = DEFAULT_QUALITY
        target_buffer = DEFAULT_QUALITY
        latency_limit = 4
        index = 1
        action_vec = np.zeros(A_DIM)
        action_vec[index] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        video_count = 0
        reward_all_sum = 0
        reward_all = 0
        reward = 0
        switch_num = 0
        SMOOTH_PENALTY = 0.0
        REBUF_PENALTY = 3
        LANTENCY_PENALTY = 0.0
        BITRATE_REWARD = 0.001
        SKIP_PENALTY = 0.0
        epoch = 0
        n = 0
        state = np.array(s_batch[-1], copy=True)
        frame_time_len = 0.04
        last_bit_rate = DEFAULT_QUALITY
        while True:  # experience video streaming forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            time, time_interval, send_data_size, chunk_len, \
            rebuf, buffer_size, play_time_len, end_delay, \
            cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, decision_flag, \
            buffer_flag, cdn_flag, skip_flag, end_of_video = net_env.get_video_frame(bit_rate, target_buffer,
                                                                                     latency_limit)
            # # QOE setting
            # if end_delay <= 1.0:
            #     LANTENCY_PENALTY = 0.005
            # else:
            #     LANTENCY_PENALTY = 0.01

            reward_frame = 0
            epoch += 1
            if not cdn_flag:
                reward_frame = frame_time_len * float(BIT_RATE[
                                                          bit_rate]) * BITRATE_REWARD - REBUF_PENALTY * rebuf - LANTENCY_PENALTY * end_delay - SKIP_PENALTY * skip_frame_time_len
            else:
                reward_frame = -(REBUF_PENALTY * rebuf)
            reward += reward_frame

            # dequeue history record
            state = np.roll(state, -1, axis=1)
            # this should be S_INFO number of terms
            state[0, -1] = buffer_size * 0.1
            state[1, -1] = send_data_size * 0.00001
            state[2, -1] = time_interval * 10  # kilo byte / ms
            state[3, -1] = end_delay * 0.1  # 10 sec
            state[4, -1] = rebuf  # mega byte

            if decision_flag and not end_of_video:

                reward_frame = -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
                reward += reward_frame
                last_bit_rate = bit_rate
                r_batch.append(reward)

                reward = 0

                # compute action probability vector
                action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
                action_cumsum = np.cumsum(action_prob)
                temp = np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)
                index = (action_cumsum > temp).argmax()

                bit_rate = ACTION_SAPCE[index][0]
                target_buffer = ACTION_SAPCE[index][1]
                latency_limit = ACTION_SAPCE[index][2]
                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states

                entropy_record.append(a3c.compute_entropy(action_prob[0]))


                # report experience to the coordinator
                if len(r_batch) >= TRAIN_SEQ_LEN :
                    exp_queue.put([s_batch[1:],  # ignore the first chuck
                                   a_batch[1:],  # since we don't have the
                                   r_batch[1:],  # control over it
                                   end_of_video,
                                   {'entropy': entropy_record}])

                    # synchronize the network parameters from the coordinator
                    actor_net_params, critic_net_params = net_params_queue.get()
                    actor.set_network_params(actor_net_params)
                    critic.set_network_params(critic_net_params)

                    del s_batch[:]
                    del a_batch[:]
                    del r_batch[:]
                    del entropy_record[:]

                s_batch.append(state)

                action_vec = np.zeros(A_DIM)
                action_vec[index] = 1
                a_batch.append(action_vec)

            reward_all += reward_frame

            # store the state and action into batches
            if end_of_video:
                r_batch.append(reward)

                reward_all_sum += reward_all / 20
                video_count += 1
                if video_count >= len(all_file_names):
                    n += 1
                    video_count = 0
                    print(n,"agent_id ",agent_id,"reward_all_sum:",reward_all_sum)
                    w.writerow([n,reward_all_sum])
                    out.flush()
                    reward_all_sum = 0
                    net_env = env.Environment(all_cooked_time=all_cooked_time,
                                              all_cooked_bw=all_cooked_bw,
                                              random_seed=epoch,
                                              VIDEO_SIZE_FILE=video_size_file,
                                              Debug=False)
                    if n == NUM_EPOCH:
                        break

                reward_all = 0
                reward = 0
                switch_num = 0

                bit_rate = DEFAULT_QUALITY  # use the default action here
                target_buffer = DEFAULT_QUALITY

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)


def main():

    np.random.seed(RANDOM_SEED)
    assert len(ACTION_SAPCE) == A_DIM

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRAIN_TRACES)
    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,all_file_names,VIDEO_TRACES,
                                       net_params_queues[i],
                                       exp_queues[i])))

    for i in range(NUM_AGENTS):
        agents[i].start()


    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()

