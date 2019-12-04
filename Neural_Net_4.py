import matplotlib.pyplot as plt
import importData_4 as imD4
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.metrics import explained_variance_score as expl
from sklearn.metrics import median_absolute_error as mdae
from sklearn.metrics import mean_squared_log_error as msle

from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import time
from datetime import datetime

a = '\u00b0'
### regular expression for tensorboard ([^\s]+)
#parameters for finding the best set of layers
# NAME = f"dense_layers_{layer}_{str(time.time())}"
NAME = f"nn_real_5_500_{str(datetime.fromtimestamp(time.time()))}"
print(NAME)
tensorboard = TensorBoard(log_dir=f"logs/{NAME}",
                          histogram_freq=50,
                          write_graph=True)
# dense_layers = [5, 10, 20, 40, 80]

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

# seed = 5
# np.random.seed(seed)

X = imD4.X.values
y = imD4.y.values.reshape(-1,1)
target_X = imD4.target_X.values
target_y = imD4.target_y.values.reshape(-1,1)

# Normalize with withn range [0,1]
scalerMm = MinMaxScaler(feature_range=(0,1))
scalerMm_X = scalerMm.fit(X)
scalerMm_y = scalerMm.fit(y)
X = scalerMm_X.transform(X)
y = scalerMm_y.transform(y)
target_X = scalerMm_X.transform(target_X)
target_y = scalerMm_y.transform(target_y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#build network
model = Sequential()
model.add(Dense(10, input_dim=32, kernel_initializer='uniform', activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
### optimizer = 'Nadam', 'rmsprop'
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[soft_acc])

#fit network
# callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=10)]
#ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1
callbacks = [tensorboard]
H = model.fit(X_train, y_train, batch_size=50, epochs=100, validation_data=(X_test, y_test), verbose=0)

# # model.summary()
# # print the metrics of mean squared error
# # plt.plot(H.history['loss'], label='train loss')
# plt.plot(H.history['val_loss'], label='val loss', color='orange')
# plt.title('Training Error')
# plt.xlabel('nb_epochs')
# plt.legend()
# plt.show()
# # #
# # plt.plot(H.history['soft_acc'], label='train accuracy')
# plt.plot(H.history['val_soft_acc'], label='val accuracy', color='orange')
# plt.title('Training Accuracy')
# plt.xlabel('nb_epochs')
# plt.legend()
# plt.show()

# # make predictions
# predictions = model.predict(target_X)
# # # plot the predictions
# plt.plot(scalerMm_X.inverse_transform(target_y),'-o', label='real')
# plt.plot(scalerMm_X.inverse_transform(predictions), '-o', label='prediction')
# plt.title('Temperature Predictions')
# plt.xlabel('nb_days')
# plt.ylabel(f'Temperature ({a}C)')
# plt.legend()
# # gi.plt.plot(scalerMm_X.inverse_transform(target),'-b', scalerMm_X.inverse_transform(temp_p), '-g')
# plt.show()
# print(f"MSE: {mse(scalerMm_X.inverse_transform(target_y), scalerMm_X.inverse_transform(predictions))}")
# print(f"MAE: {mae(scalerMm_X.inverse_transform(target_y), scalerMm_X.inverse_transform(predictions))}")
# print(f"R2: {r2(scalerMm_X.inverse_transform(target_y), scalerMm_X.inverse_transform(predictions))}")
# print(f"EXPL: {expl(scalerMm_X.inverse_transform(target_y), scalerMm_X.inverse_transform(predictions))}")
# print(f"MDAE: {mdae(scalerMm_X.inverse_transform(target_y), scalerMm_X.inverse_transform(predictions))}")
# print(f"MSLE: {msle(scalerMm_X.inverse_transform(target_y), scalerMm_X.inverse_transform(predictions))}")