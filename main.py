import os
from data_manager import Data_Manager
import pandas as pd

dm = Data_Manager('./gaps.db')
df = dm.load_db()
print(df)

for code in df.columns.levels[1]:
    print(code)




