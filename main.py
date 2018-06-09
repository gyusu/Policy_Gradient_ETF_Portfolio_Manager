import tensorflow as tf
import pandas as pd
import numpy as np

from data_manager import Data_Manager
from environment import Environment
from agent import Agent
import simulator
import visualizer
import train_assistant

tf.set_random_seed(1531)

visualizer.init_visualizer()

window_size = 60
batch_size = 15
episode = 200
learning_rate = 0.001

# 학습용 data
dm = Data_Manager('./gaps.db',20151113, 20180525, train_test_split=0.9)
df = dm.load_db()
train_df, test_df = dm.generate_feature_df(df)

print("train 데이터 수: {}, test 데이터 수: {}".format(len(train_df), len(test_df)))
# train_df.to_csv('train.csv')
# train_df.to_csv('test.csv')

# window_size 만큼 test_df 상단 row에 복사
test_df = pd.concat([train_df.iloc[-window_size:], test_df])
visualizer.plot_dfs([train_df, test_df], ['train', 'test'])
print("학습 데이터의 asset 개수 : ", len(train_df.columns.levels[0]))

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

        visualizer.plot_reward(i, np.cumprod(test_pv_vec))
