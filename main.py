from data_manager import Data_Manager
from environment import Environment
import visualizer
import pandas as pd
from agent import Agent

dm = Data_Manager('./gaps.db')
df = dm.load_db()
feature_df = dm.generate_feature_df(df)
visualizer.plot_df(df)
print("asset 개수 : ", len(df.columns.levels[0]))

env = Environment(feature_df)
pg_agent = Agent(env)

observation = env.reset()

while True:

    action = pg_agent.decide_action(observation)
    observation, reward, done = env.step(action)
    print(observation.shape)
    print(reward)

    input()
    if done:
        break




