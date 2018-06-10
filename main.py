import pandas as pd
import numpy as np
import tensorflow as tf

np.random.seed(1531)

from data_manager import Data_Manager
import visualizer
import trainer
import ensembler
from environment import Environment
from agent import Agent

visualizer.init_visualizer()

WINDOW_SIZE = 60
BATCH_SIZE = 30
EPISODE = 8
LEARNING_RATE = 0.001
VALIDATION = 0 # train data에서 이 비율만큼 validation data로 사용
ENSEMBLE_NUM = 16
USE_TOP_N_AGENT = ENSEMBLE_NUM // 2

ROLLING_TRAIN_TEST = False

# 학습/ 테스트 data 설정
dm = Data_Manager('./gaps.db',20151113, 20171231, split_ratio=(0.6, 0.2, 0.2))
df = dm.load_db()
train_df, val_df, test_df = dm.generate_feature_df(df, WINDOW_SIZE)
print('train: {} ~ {}'.format(train_df.iloc[0].name, train_df.iloc[-1].name))
print('val  : {} ~ {}'.format(val_df.iloc[WINDOW_SIZE].name, val_df.iloc[-1].name))
print('test: {} ~ {}'.format(test_df.iloc[WINDOW_SIZE].name, test_df.iloc[-1].name))
print("데이터 수 train: {}, val: {}, test: {}".format(len(train_df), len(val_df), len(test_df)))

visualizer.plot_dfs([train_df, val_df.iloc[WINDOW_SIZE:], test_df.iloc[WINDOW_SIZE:]], ['train', 'val', 'test'])
print("학습 데이터의 asset 개수 : ", len(train_df.columns.levels[0]))

# Random Agent Test
test_env = Environment(test_df, WINDOW_SIZE)
obs = test_env.reset()
done = False
pv = 0
while not done:
    observation, pv, done, future_price = test_env.step(test_env.action_sample())
print('random agent PV: {}'.format(pv))

# Market Average Return
obs = test_env.reset()
done = False
mkt_pv_vec = []
action = np.array([1.0]*len(test_env.action_sample())) / len(test_env.action_sample())
while not done:
    observation, pv, done, future_price = test_env.step(action)
    mkt_pv_vec.append(pv)
print('Market Average Return: {}'.format(pv))


with tf.Session() as sess:
    # train, test env 생성
    train_env = Environment(train_df, WINDOW_SIZE)
    test_env = Environment(test_df, WINDOW_SIZE)
    val_env = Environment(val_df, WINDOW_SIZE)
    if ROLLING_TRAIN_TEST:
        trainer.rolling_train_and_test(sess, train_df, test_df, BATCH_SIZE, WINDOW_SIZE, LEARNING_RATE, EPISODE)
        # trainer.rolling_train_and_test_v2(train_df, test_df, BATCH_SIZE, WINDOW_SIZE, LEARNING_RATE)
    else:
        agent_list = []
        for i in range(ENSEMBLE_NUM):
            # agent 생성. 이때 train_env.obs_shape 는 test_env.obs_shape 와 같아야 한다.
            pg_agent = Agent(i, sess, train_env.obs_shape, lr=LEARNING_RATE)
            agent_list.append(pg_agent)
        sess.run(tf.global_variables_initializer())

        for pg_agent in agent_list:
            trainer.train_and_test(pg_agent, train_env, val_env, test_env, BATCH_SIZE, EPISODE, mkt_pv_vec)

        ensembler.ensemble_test(agent_list, val_env, test_env, USE_TOP_N_AGENT, mkt_pv_vec)