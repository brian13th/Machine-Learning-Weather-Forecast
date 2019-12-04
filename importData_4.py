import pandas as pd
import numpy as np


# pycharm output
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
np.set_printoptions(threshold=np.inf)

def features(data):
    data = data.values
    x = np.append(data[1:], [1])
    y = np.append(data[2:], [1, 1])
    z = np.append(data[3:], [1, 1, 1])
    return x, y, z

# import data from csv file
filename1 = 'lamia_2018.csv'

# df1 = pd.read_csv(filename1).set_index('YEARMODA')
df = pd.read_csv(filename1)

# drop irrelevant features
df = df.drop(columns=['  ', '  .1', '  .2', '  .3', '  .4', '  .5'])
df.columns = ['STN', 'WBAN', 'YEARMODA', 'TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'MXSPD', 'GUST', 'MAX', 'MIN', 'PRCP', 'SNDP', 'FRSHTT']
df = df.drop(columns=['STN','WBAN', 'VISIB', 'GUST', 'PRCP', 'SNDP', 'FRSHTT', 'YEARMODA'])
df['MAX'] = df['MAX'].str.replace('*', '')
df['MAX'] = df['MAX'].str.replace(' ', '')
df['MIN'] = df['MIN'].str.replace('*', '')
df['MIN'] = df['MIN'].str.replace(' ', '')
df['MAX'] = df.MAX.astype(float)
df['MIN'] = df.MIN.astype(float)
# df.info()
# df.info
# df.describe()

# interpolation for the values that are 9999,9
for index, row in df.iterrows():
    if row['DEWP'] == 9999.9:
        row['DEWP'] = round((df.loc[index - 1, ['DEWP']] + df.loc[index + 1, ['DEWP']])/2.0)

for data in df.columns:
    df[data + '2'], df[data + '3'], df[data + '4'] = features(df[data])

# create X, y values for training
X = df.drop(df.index[[-1, -2, -3, -4]])
y = df['TEMP'].drop(df.index[[0, 1, 2, 3]])
X, target_X = X[:302], X[302:]
y, target_y = y[:302], y[302:]