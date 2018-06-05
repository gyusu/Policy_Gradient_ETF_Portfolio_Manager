import tensorflow as tf
import pandas as pd

from data_manager import Data_Manager
from environment import Environment
from agent import Agent
import simulator
import visualizer
import train_assistant

visualizer.init_visualizer()

window_size = 30
batch_size = 30
episode = 100
learning_rate = 0.0005

# 학습용 data
dm = Data_Manager('./gaps.db', train_test_split=0.8)
df = dm.load_db()
train_df, test_df = dm.generate_feature_df(df)

# window_size 만큼 test_df 상단 row에 복사
test_df = pd.concat([train_df.iloc[-window_size:], test_df])
visualizer.plot_dfs([train_df, test_df], ['train', 'test'])
print("학습 데이터의 asset 개수 : ", len(train_df.columns.levels[0]))

with tf.Session() as sess:
    # train, test env 생성
    train_env = Environment(train_df)
    test_env = Environment(test_df)

    # agent 생성. 이때 train_env.obs_shape 는 test_env.obs_shape 와 같아야 한다.
    pg_agent = Agent(sess, train_env.obs_shape, lr=learning_rate)

    sess.run(tf.global_variables_initializer())

    # 학습
    print("Train Start!!!!!")
    for i in range(episode):
        print("train episode {}/{}".format(i+1, episode))
        obs, acts, rews, fps = simulator.policy_simulator(train_env, pg_agent)

        obs_batches, fps_batches = train_assistant.batch_shuffling(obs, fps, batch_size)
        epi_loss, epi_pv, epi_ir, nb_batch = 0, 1, 0, 0
        for obs, fps in zip(obs_batches, fps_batches):
            loss, pv, ir = pg_agent.train_step(obs, fps)
            epi_loss += loss
            epi_pv *= pv
            epi_ir += ir
            nb_batch += 1

        print("[episode {}] avg loss:{:.6} PV:{:.6} avg IR:{:.6}".format(i, epi_loss/nb_batch, epi_pv, epi_ir/nb_batch))

        print("test simulation")
        _, test_actions, test_rewards, _ = simulator.policy_simulator(test_env, pg_agent)
        visualizer.plot_reward(i, test_rewards)
