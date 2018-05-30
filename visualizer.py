import matplotlib as mpl
import matplotlib.pyplot as plt
from data_manager import asset_name_dict

mpl.rc('font', family = 'NanumGothic')

def plot_df(df):
    asset_code_list = df.columns.levels[0]

    for i, asset_code in enumerate(asset_code_list):
        asset_df = df[asset_code]
        if i > 6:
            plt.figure(2)
        plt.plot(asset_df.index, asset_df['close']/asset_df['close'].iloc[0],
                 label=asset_name_dict[asset_code])
        plt.legend()
    plt.show()

def plot_reward(step, rewards):
    fig = plt.figure()
    plt.title("{} step".format(step))
    plt.plot(rewards)
    plt.savefig('./result/step{}.png'.format(step))
    plt.close(fig)