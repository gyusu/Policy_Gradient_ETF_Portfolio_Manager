import os
from data_manager import Data_Manager
from environment import Environment
import visualizer
import pandas as pd
import matplotlib.pyplot as plt

dm = Data_Manager('./gaps.db')
df = dm.load_db()

visualizer.plot_df(df)


print(len(df.columns.levels[0]))

env = Environment(df)
env.reset()

while True:
    action = env.action_sample()
    print(action)
    observation, reward, done = env.step(action)
    # print(observation)
    print(reward)
    input()
    if done:
        break




