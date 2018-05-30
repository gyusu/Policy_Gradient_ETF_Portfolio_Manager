import tensorflow as tf

from data_manager import Data_Manager
from environment import Environment
from agent import Agent
import simulator
import visualizer


# 학습용 data
train_dm = Data_Manager('./gaps.db', min_date=20170101, max_date=20171231)
train_df = train_dm.load_db()
train_feature_df = train_dm.generate_feature_df(train_df)
visualizer.plot_df(train_df)
print("학습 데이터의 asset 개수 : ", len(train_df.columns.levels[0]))

# 테스트용 data
test_dm = Data_Manager('./gaps.db', min_date=20180101, max_date=99999999)
test_df = test_dm.load_db()
test_feature_df = test_dm.generate_feature_df(test_df)
visualizer.plot_df(test_df)
print("테스트 데이터의 asset 개수 : ", len(test_df.columns.levels[0]))


batch_size = 30
episode = 100
learning_rate = 0.0005
with tf.Session() as sess:
    # train, test env 생성
    train_env = Environment(train_feature_df)
    test_env = Environment(test_feature_df)

    # agent 생성. 이때 train_env.obs_shape 는 test_env.obs_shape 와 같아야 한다.
    pg_agent = Agent(sess, train_env.obs_shape, lr=learning_rate)

    sess.run(tf.global_variables_initializer())

    # 학습
    print("Train Start!!!!!")
    for i in range(episode):
        print("train episode {}/{}".format(i+1, episode))
        obs, acts, rews, fps = simulator.policy_simulator(train_env, pg_agent)

        for j in range(int(len(obs)/batch_size)):
            idx_from = j * batch_size
            idx_to = idx_from + batch_size
            pg_agent.train_step(obs[idx_from:idx_to+1], fps[idx_from:idx_to+1])

        simulator.policy_simulator(test_env, pg_agent)
