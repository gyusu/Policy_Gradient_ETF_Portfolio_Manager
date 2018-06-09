import tensorflow as tf
import pandas as pd
import numpy as np

from data_manager import Data_Manager
from environment import Environment
from agent import Agent
import simulator
import visualizer
import train_assistant
import trainer

tf.set_random_seed(1531)

visualizer.init_visualizer()

WINDOW_SIZE = 60
BATCH_SIZE = 30
EPISODE = 200
LEARNING_RATE = 0.001
VALIDATION = False

ROLLING_TRAIN_TEST = True

# 학습/ 테스트 data 설정
dm = Data_Manager('./gaps.db',20151113, 20180525, train_test_split=0.9, validation=VALIDATION)
df = dm.load_db()
train_df, validation_df, test_df = dm.generate_feature_df(df)

print("데이터 수 train: {}, val: {}, test: {}".format(len(train_df), len(validation_df), len(test_df)))

# window_size 만큼 test_df 상단 row에 복사
if VALIDATION:
    validation_df = pd.concat([train_df.iloc[-WINDOW_SIZE:], validation_df])
    test_df = pd.concat([validation_df.iloc[-WINDOW_SIZE:], test_df])
else:
    test_df = pd.concat([train_df.iloc[-WINDOW_SIZE:], test_df])
visualizer.plot_dfs([train_df, test_df], ['train', 'test'])
print("학습 데이터의 asset 개수 : ", len(train_df.columns.levels[0]))


if ROLLING_TRAIN_TEST:
    trainer.rolling_train_and_test(train_df, test_df, BATCH_SIZE, WINDOW_SIZE, LEARNING_RATE, EPISODE)

else:
    trainer.train_and_test(train_df, test_df, BATCH_SIZE, WINDOW_SIZE, LEARNING_RATE, EPISODE)