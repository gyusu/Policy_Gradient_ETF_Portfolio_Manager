import pandas as pd
import sqlite3
import numpy as np
import datetime as dt

# dict {종목코드: 종목명, ...}
asset_name_dict = pd.read_csv('asset_name.csv', index_col=0)
asset_name_dict = asset_name_dict.to_dict()['name']

class Data_Manager:
    def __init__(self, db_path, min_date=20160401, max_date=20180525, split_ratio=(0.8, 0.0, 0.2)):
        self._db_path = db_path
        self.asset_list = self.db_asset_list(min_date=min_date)
        self.min_date = min_date
        self.max_date = max_date
        self.split_ratio = split_ratio

    def db_asset_list(self, min_date=0):
        """
        db에 저장된 모든 종목코드를 가져와 반환하는 메소드
        min_date가 설정된 경우에는 최소 min_date 시점 이후 데이터가 있는 종목만 반환
        :return: 종목코드 list
        """
        with sqlite3.connect(self._db_path) as con:
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")

            asset_list = np.array(cur.fetchall())
            asset_list = asset_list.flatten()
            asset_list = [asset for asset in asset_list if asset.startswith('A')]  # use normal stock only
            asset_list.remove('A130730') # KOSEF 단기자금은 변동성 없는 현금성 자산이므로 국내채권과 자산의 역할이 비슷하여 제외함

            if min_date != 0:
                for asset in asset_list:
                    cur.execute("SELECT date FROM {} ORDER BY date ASC LIMIT 1".format(asset))
                    if cur.fetchall()[0][0] > min_date:
                        asset_list.remove(asset)

            return asset_list

    def load_db(self):

        df_list = []
        with sqlite3.connect(self._db_path) as con:
            cur = con.cursor()
            for asset in self.asset_list:
                cur.execute("SELECT * FROM {} WHERE date >= {} AND date <= {}".format(
                            asset, self.min_date, self.max_date))
                df = pd.DataFrame(cur.fetchall(), columns=(
                    'date', 'open', 'high', 'low', 'close', 'volume'))

                df.index = pd.to_datetime(df.date, format='%Y%m%d')
                del df['date']

                df.name = asset
                df_list.append(df)

            # concatenate all df to one df with 2-level index
            df = pd.concat(df_list, axis=1, keys=[df.name for df in df_list], names=['Asset', 'Feature'])

        return df

    def generate_feature_df(self, chart_df, window_size):
        feature_df = chart_df

        for asset in self.asset_list:

            # open_lastclose_ratio : (open - close(-1))/close(-1)
            feature_df.loc[:, (asset, 'open_lastclose_ratio')] = np.zeros(len(feature_df))
            feature_df.ix[1:, (asset, 'open_lastclose_ratio')] = \
                (feature_df.ix[1:, (asset, 'open')].values - feature_df.ix[:-1, (asset, 'close')].values) / \
                feature_df.ix[:-1, (asset, 'close')].values

            # high_close_ratio : (high - close) / close
            feature_df.loc[:, (asset, 'high_close_ratio')] = \
                (feature_df.loc[:, (asset, 'high')].values - feature_df.loc[:, (asset, 'close')].values) / \
                feature_df.loc[:, (asset, 'close')].values

            # low_close_ratio : (low - close) / close
            feature_df.loc[:, (asset, 'low_close_ratio')] = \
                (feature_df.loc[:, (asset, 'low')].values - feature_df.loc[:, (asset, 'close')].values) / \
                feature_df.loc[:, (asset, 'close')].values

            # (close - close(-1)) / (close(-1)
            feature_df.loc[:, (asset, 'close_lastclose_ratio')] = np.zeros(len(feature_df))
            feature_df.ix[1:, (asset, 'close_lastclose_ratio')] = \
                (feature_df.ix[1:, (asset, 'close')].values - feature_df.ix[:-1, (asset, 'close')].values) / \
                feature_df.ix[:-1, (asset, 'close')].values

            # (volume - volume(-1)) / (volume(-1))
            feature_df.loc[:, (asset, 'volume_lastvolume_ratio')] = np.zeros(len(feature_df))
            feature_df.loc[1:, (asset, 'volume_lastvolume_ratio')] = \
                (feature_df.ix[1:, (asset, 'volume')].values - feature_df.ix[:-1, (asset, 'volume')].values) / \
                feature_df.ix[:-1, (asset, 'volume')]\
                    .replace(to_replace=0, method='ffill').replace(to_replace=0, method='bfill').values

        df = feature_df.sort_index(axis=1)

        df_train = df.iloc[:int(len(df) * self.split_ratio[0])]
        df_validation = df.iloc[int(len(df) * self.split_ratio[0])-window_size:
                                int(len(df) * (self.split_ratio[0] + self.split_ratio[1]))]
        df_test = df.iloc[int(len(df) * (self.split_ratio[0] + self.split_ratio[1])) - window_size:]

        return df_train, df_validation, df_test
