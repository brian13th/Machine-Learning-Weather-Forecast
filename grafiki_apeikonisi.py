import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a = '\u00b0'
# pycharm output
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 20)
np.set_printoptions(threshold=np.inf)

# import data from csv file
filename1 = 'lamia_2018.csv'

# df1 = pd.read_csv(filename1).set_index('YEARMODA')
df = pd.read_csv(filename1)


# drop irrelevant features
df = df.drop(columns=['  ', '  .1', '  .2', '  .3', '  .4', '  .5'])
df.columns = ['STN', 'WBAN', 'YEARMODA', 'TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'MXSPD', 'GUST', 'MAX', 'MIN', 'PRCP', 'SNDP', 'FRSHTT']

df['YEARMODA'] = pd.to_datetime(df['YEARMODA'], format='%Y%m%d')
df = df.set_index('YEARMODA')
df = df.drop(columns=['STN','WBAN', 'VISIB', 'GUST', 'PRCP', 'SNDP', 'FRSHTT'])
df['MAX'] = df['MAX'].str.replace('*', '')
df['MAX'] = df['MAX'].str.replace(' ', '')
df['MIN'] = df['MIN'].str.replace('*', '')
df['MIN'] = df['MIN'].str.replace(' ', '')
df['MAX'] = df.MAX.astype(float)
df['MIN'] = df.MIN.astype(float)

# df.info()
# df.info
# df.describe()
# df['TEMP'].rolling(25).mean().plot() FOR PLOTING AND SMOOTHING THE PLOT
#  df['TEMP'].unique() when just want a single value
########### df[['TEMP', 'MAX', 'MIN']].plot() very nice plot for different values!!!!!~~~~~!!!df[['TEMP', 'MAX', 'MIN']].plot(figsize=(8,5), legend=True)
# df.describe()
# df.corr() or df.corrwith(df['TEMP']) for correlations
### df.plot(subplots=True, layout=(10,8))
####df[['TEMP', 'MAX', 'MIN']].rolling(25).mean().plot(title=f'Ετήσια κατανομή θερμοκρασίας\n({a}F)')
# plt.figure()
df[['WDSP', 'MXSPD']].rolling(25).mean().plot(title=f'Ετήσια κατανομή ταχύτητας ανέμου\n(knots)')
plt.xlabel('Ημέρα')
plt.ylabel('Μέση ταχύτητα ανέμου')
plt.legend()
plt.show()
