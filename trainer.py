import tensorflow as tf
import pandas as pd
import numpy as np

from environment import Environment
from agent import Agent
import simulator
import visualizer
import train_assistant

tf.set_random_seed(1531)

def train_and_test(pg_agent, train_env, val_env, test_env, batch_size, episode, mkt_pv_vec):

    obs, fps = simulator.policy_simulator(train_env, pg_agent, do_action=False)
    test_obs, test_fps = simulator.policy_simulator(test_env, pg_agent, do_action=False)
    val_obs, val_fps = simulator.policy_simulator(val_env, pg_agent, do_action=False)

    # 학습
    print("Train Start!!!!!")
    print('전처리 후 데이터 수 train: {}, test: {}'.format(len(obs), len(test_obs) - 1))
    train_reward_list = []
    val_reward_list = []
    test_reward_list = []
    for i in range(episode):
        print("\ntrain episode {}/{}".format(i+1, episode))

        obs_batches, fps_batches = train_assistant.batch_shuffling(obs, fps, batch_size)
        epi_reward, epi_pv, epi_ir, nb_batch = 0, 0, 0, 0
        for _obs, _fps in zip(obs_batches, fps_batches):

            _obs, _fps = train_assistant.asset_shuffling(_obs, _fps)
            loss, pv, ir, pv_vec = pg_agent.run_batch(_obs, _fps, is_train=True)
            epi_reward += loss
            epi_pv += pv
            epi_ir += ir
            nb_batch += 1

        print("[train] avg_reward:{:9.6f} avg_PV:{:9.6f} avg_IR:{:9.6f}".format(epi_reward/nb_batch, epi_pv/nb_batch, epi_ir/nb_batch))

        val_reward, val_pv, val_ir, val_pv_vec = pg_agent.run_batch(val_obs[1:], val_fps[1:], is_train=False,
                                                                        verbose=True)
        print("[val ]      reward:{:9.6f} PV:{:9.6f}     IR:{:9.6f}".format(val_reward, val_pv, val_ir))

        test_reward, test_pv, test_ir, test_pv_vec = pg_agent.run_batch(test_obs[1:], test_fps[1:], is_train=False,
                                                                        verbose=True)
        print("[test]      reward:{:9.6f} PV:{:9.6f}     IR:{:9.6f}".format(test_reward, test_pv, test_ir))

        train_reward_list.append(epi_reward/nb_batch)
        val_reward_list.append(val_reward)
        test_reward_list.append(test_reward)

        visualizer.plot_pv(i, np.cumprod(test_pv_vec), mkt_pv_vec)
        visualizer.plot_reward(i, train_reward_list, val_reward_list, test_reward_list)

    return pg_agent


# TODO ensemble 에 맞게 수정해야함
def rolling_train_and_test(sess, train_df, test_df, batch_size, window_size, learning_rate, pre_train_episode, mkt_return):

    # train, test env 생성
    train_env = Environment(train_df, window_size)
    test_env = Environment(test_df, window_size)

    # agent 생성. 이때 train_env.obs_shape 는 test_env.obs_shape 와 같아야 한다.
    pg_agent = Agent(sess, train_env.obs_shape, lr=learning_rate)

    sess.run(tf.global_variables_initializer())

    obs, fps = simulator.policy_simulator(train_env, pg_agent, do_action=False)
    test_obs, test_fps = simulator.policy_simulator(test_env, pg_agent, do_action=False)

    # 학습
    print("Rolling Train Start!!!!!")
    print('전처리 후 데이터 수 train: {}, test: {}'.format(len(obs), len(test_obs)-1))

    train_reward_list = []
    test_reward_list = []
    for i in range(pre_train_episode):
        print("\npre train episode {}/{}".format(i + 1, pre_train_episode))

        obs_batches, fps_batches = train_assistant.batch_shuffling(obs, fps, batch_size)
        epi_reward, epi_pv, epi_ir, nb_batch = 0, 0, 0, 0
        for _obs, _fps in zip(obs_batches, fps_batches):
            _obs, _fps = train_assistant.asset_shuffling(_obs, _fps)
            loss, pv, ir, pv_vec = pg_agent.run_batch(_obs, _fps, is_train=True)
            epi_reward += loss
            epi_pv += pv
            epi_ir += ir
            nb_batch += 1

        print("[train] avg_reward:{:9.6f} avg_PV:{:9.6f} avg_IR:{:9.6f}".format(epi_reward / nb_batch,
                                                                                epi_pv / nb_batch,
                                                                                epi_ir / nb_batch))

        test_reward, test_pv, test_ir, test_pv_vec = pg_agent.run_batch(test_obs[1:], test_fps[1:], is_train=False,
                                                                        verbose=True)
        print(test_pv_vec)
        print("[test]      reward:{:9.6f} PV:{:9.6f}     IR:{:9.6f}".format(test_reward, test_pv, test_ir))

        train_reward_list.append(epi_reward / nb_batch)
        test_reward_list.append(test_reward)
        visualizer.plot_pv(i, np.cumprod(test_pv_vec), mkt_return)
        visualizer.plot_reward(i, train_reward_list, test_reward_list)

    train_obs_len = len(obs)
    test_obs_len = len(test_obs)
    obs[len(obs):len(obs)] = test_obs
    fps[len(fps):len(fps)] = test_fps
    pv_list = []
    test_env.reset()
    for i in range(test_obs_len-1):
        idx_from = train_obs_len - batch_size + 1 + i
        idx_to = idx_from + batch_size

        action = pg_agent.decide_action([obs[idx_to]])
        observation, pv, done, future_price = test_env.step(action[0])
        pv_list.append(pv)
        print("test_step: {}, PV: {}".format(i, pv))


        rolling_train_obs = obs[idx_from: idx_to]
        rolling_train_fps = fps[idx_from: idx_to]

        for j in range(20):
            rolling_train_obs, rolling_train_fps = \
                train_assistant.asset_shuffling(rolling_train_obs, rolling_train_fps)
            pg_agent.run_batch(rolling_train_obs, rolling_train_fps, is_train=True)

