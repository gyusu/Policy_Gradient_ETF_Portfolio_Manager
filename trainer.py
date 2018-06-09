import tensorflow as tf
import pandas as pd
import numpy as np

from data_manager import Data_Manager
from environment import Environment
from agent import Agent
import simulator
import visualizer
import train_assistant

def train_and_test(train_df, test_df, batch_size, window_size, learning_rate, episode):

    with tf.Session() as sess:
        # train, test env 생성
        train_env = Environment(train_df, window_size)
        test_env = Environment(test_df, window_size)

        # agent 생성. 이때 train_env.obs_shape 는 test_env.obs_shape 와 같아야 한다.
        pg_agent = Agent(sess, train_env.obs_shape, lr=learning_rate)

        sess.run(tf.global_variables_initializer())

        obs, fps = simulator.policy_simulator(train_env, pg_agent, do_action=False)
        test_obs, test_fps = simulator.policy_simulator(test_env, pg_agent, do_action=False)

        # 학습
        print("Train Start!!!!!")
        train_reward_list = []
        test_reward_list = []
        for i in range(episode):
            print("\ntrain episode {}/{}".format(i+1, episode))

            obs_batches, fps_batches = train_assistant.batch_shuffling(obs, fps, batch_size)
            epi_reward, epi_pv, epi_ir, nb_batch = 0, 0, 0, 0
            for _obs, _fps in zip(obs_batches, fps_batches):

                p = np.random.permutation(_obs.shape[1])
                _obs = _obs[:, p]
                _fps = _fps[:, p]
                loss, pv, ir, pv_vec = pg_agent.run_batch(_obs, _fps, is_train=True)
                epi_reward += loss
                epi_pv += pv
                epi_ir += ir
                nb_batch += 1

            print("[train] avg_reward:{:9.6f} avg_PV:{:9.6f} avg_IR:{:9.6f}".format(epi_reward/nb_batch, epi_pv/nb_batch, epi_ir/nb_batch))

            test_reward, test_pv, test_ir, test_pv_vec = pg_agent.run_batch(test_obs, test_fps, is_train=False, verbose=True)
            print(test_pv_vec)
            print("[test]      reward:{:9.6f} PV:{:9.6f}     IR:{:9.6f}".format(test_reward, test_pv, test_ir))

            train_reward_list.append(epi_reward/nb_batch)
            test_reward_list.append(test_reward)
            visualizer.plot_pv(i, np.cumprod(test_pv_vec))
            visualizer.plot_reward(i, train_reward_list, test_reward_list)

def rolling_train_and_test(train_df, test_df, batch_size, window_size, learning_rate, pre_train_episode):

    with tf.Session() as sess:
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

        train_reward_list = []
        test_reward_list = []
        for i in range(pre_train_episode):
            print("\npre train episode {}/{}".format(i + 1, pre_train_episode))

            obs_batches, fps_batches = train_assistant.batch_shuffling(obs, fps, batch_size)
            epi_reward, epi_pv, epi_ir, nb_batch = 0, 0, 0, 0
            for _obs, _fps in zip(obs_batches, fps_batches):
                p = np.random.permutation(_obs.shape[1])
                _obs = _obs[:, p]
                _fps = _fps[:, p]
                loss, pv, ir, pv_vec = pg_agent.run_batch(_obs, _fps, is_train=True)
                epi_reward += loss
                epi_pv += pv
                epi_ir += ir
                nb_batch += 1

            print("[train] avg_reward:{:9.6f} avg_PV:{:9.6f} avg_IR:{:9.6f}".format(epi_reward / nb_batch,
                                                                                    epi_pv / nb_batch,
                                                                                    epi_ir / nb_batch))

            test_reward, test_pv, test_ir, test_pv_vec = pg_agent.run_batch(test_obs, test_fps, is_train=False,
                                                                            verbose=True)
            print(test_pv_vec)
            print("[test]      reward:{:9.6f} PV:{:9.6f}     IR:{:9.6f}".format(test_reward, test_pv, test_ir))

            train_reward_list.append(epi_reward / nb_batch)
            test_reward_list.append(test_reward)
            visualizer.plot_pv(i, np.cumprod(test_pv_vec))
            visualizer.plot_reward(i, train_reward_list, test_reward_list)

        train_obs_len = len(obs)
        test_obs_len = len(test_obs)
        obs[len(obs):len(obs)] = test_obs
        fps[len(fps):len(fps)] = test_fps
        pv_list = []
        test_env.reset()
        for i in range(test_obs_len):
            action = pg_agent.decide_action([test_obs[i]])
            observation, pv, done, future_price = test_env.step(action[0])
            pv_list.append(pv)
            print(pv)

            idx_from = train_obs_len - test_obs_len + 2 + i
            idx_to = idx_from + test_obs_len
            rolling_train_obs = obs[idx_from: idx_to]
            rolling_train_fps = fps[idx_from: idx_to]

            pg_agent.run_batch(rolling_train_obs, rolling_train_fps, is_train=True)