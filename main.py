from data_manager import Data_Manager
from environment import Environment
import visualizer
import pandas as pd
from agent import Agent
import tensorflow as tf

# 학습용 data
dm = Data_Manager('./gaps.db', min_date=20170101, max_date=20171231)
df = dm.load_db()
feature_df = dm.generate_feature_df(df)
visualizer.plot_df(df)
print("asset 개수 : ", len(df.columns.levels[0]))

# 테스트용 data
dm_test = Data_Manager('./gaps.db', min_date=20180101, max_date=99999999)
df_test = dm_test.load_db()
feature_df_test = dm_test.generate_feature_df(df_test)
visualizer.plot_df(df_test)

def policy_rollout(env, agent):
    """Run one episode."""
    obs, acts, rews, fps = [], [], [], []


    done = False
    observation = env.reset()
    while not done:
        obs.append(observation[0])
        action = pg_agent.decide_action(observation)
        observation, reward, done, future_price = env.step(action[0])

        acts.append(action[0])
        rews.append(reward)
        fps.append(future_price)

    return obs, acts, rews, fps

def policy_test(env, agent):
    # 테스트!!!!!
    print("Test Start!!!!!")

    observation = env.reset()
    done = False
    acts, rews = [], []
    i = 1
    while not done:
        action = pg_agent.decide_action(observation)
        observation, reward, done, future_price = env.step(action[0])
        print("Day {} Portfolio Value : {}".format(i, reward))
        acts.append(action[0])
        rews.append(reward)
        i += 1

def policy_train_debug(env, agent):
    # 학습 데이터로 테스트!!!!!
    print("Train-Test Start!!!!!")

    observation = env.reset()
    done = False
    acts, rews = [], []
    i = 1
    while not done:
        action = pg_agent.decide_action(observation)
        observation, reward, done, future_price = env.step(action[0])
        print("Day {} Portfolio Value : {}".format(i, reward))
        acts.append(action[0])
        rews.append(reward)
        i += 1

batch_size = 30
episode = 100

with tf.Session() as sess:
    env = Environment(feature_df)
    env_test = Environment(feature_df_test)
    pg_agent = Agent(env, sess)

    sess.run(tf.global_variables_initializer())

    # 학습!!!!!!
    print("Train Start!!!!!")
    for i in range(episode):
        print("train episode {}/{}".format(i+1, episode))
        obs, acts, rews, fps = policy_rollout(env, pg_agent)
        pg_agent.train_step(obs, fps)

        policy_train_debug(env, pg_agent)
        policy_test(env_test, pg_agent)
