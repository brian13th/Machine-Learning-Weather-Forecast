import matplotlib.pyplot as plt
import importData_4 as imD4
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
import pandas as pd
import time
from datetime import datetime

a = '\u00b0'
# seed = 5
# np.random.seed(seed)

NAME = f"lstm_real_5_500_{str(datetime.fromtimestamp(time.time()))}"
print(NAME)
tensorboard = TensorBoard(log_dir=f"logs/{NAME}",
                          histogram_freq=50,
                          write_graph=True)

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

X = imD4.X.values
y = imD4.y.values.reshape(-1,1)
target_X = imD4.target_X.values
target_y = imD4.target_y.values.reshape(-1,1)

# Normalize with withn range [0,1]
scalerMm = MinMaxScaler(feature_range=(0,1))
scalerMm_X = scalerMm.fit(X)
scalerMm_y = scalerMm.fit(y)
X = scalerMm_X.transform(X).reshape(302,1,32)
y = scalerMm_y.transform(y)
target_X = scalerMm_X.transform(target_X).reshape(20,1,32)
target_y = scalerMm_y.transform(target_y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# for epoch in [10, 20, 30, 50, 70, 90, 100]:
# design network
model = Sequential()
model.add(LSTM(10, input_shape=(1,32),return_sequences=True, kernel_initializer='uniform', activation='relu'))
model.add(LSTM(10, input_shape=(1,32), activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[soft_acc])

# fit network
callbacks = [tensorboard]
H = model.fit(X_train, y_train, batch_size=50, epochs=500, validation_data=(X_test, y_test), verbose=0)

# # plt.plot(H.history['loss'], label='train loss')
# plt.plot(H.history['val_loss'], label='val loss', color='orange')
# plt.title('Training Error')
# plt.xlabel('nb_epochs')
# plt.legend()
# plt.show()

# # plt.plot(H.history['soft_acc'], label='train accuracy')
# plt.plot(H.history['val_soft_acc'], label='val accuracy', color='orange')
# plt.title('Training Accuracy')
# plt.xlabel('nb_epochs')
# plt.legend()
# plt.show()

# make predictions
predictions = model.predict(target_X)
# plot the predictions
plt.plot(scalerMm_X.inverse_transform(target_y),'-o', label='real')
plt.plot(scalerMm_X.inverse_transform(predictions), '-o', label='prediction')
plt.title('Temperature Predictions')
plt.xlabel('nb_days')
plt.ylabel(f'Temperature ({a}C)')
plt.legend()
# gi.plt.plot(scalerMm_X.inverse_transform(target),'-b', scalerMm_X.inverse_transform(temp_p), '-g')
plt.show()
print(f"MSE: {mse(scalerMm_X.inverse_transform(target_y), scalerMm_X.inverse_transform(predictions))}")
print(f"MAE: {mae(scalerMm_X.inverse_transform(target_y), scalerMm_X.inverse_transform(predictions))}")


# OLD
# # model.summary()
# # for layer in model.layers:
# #     print(layer.name, layer.inbound_nodes, layer.outbound_nodes)
# #print the metrics of mean squared error
# plt.plot(H.history['loss'], label='train')
# plt.plot(H.history['val_loss'], label='test')
# plt.legend()
# plt.show()
#
# plt.plot(H.history['soft_acc'], label='train accuracy')
# plt.plot(H.history['val_soft_acc'], label='test accuracy')
# plt.legend()
# plt.show()
#
# make predictions
predictions = model.predict(target_X)
# # plot the predictions
plt.plot(scalerMm_X.inverse_transform(target_y), '-bo', scalerMm_X.inverse_transform(predictions), '-g+')
# gi.plt.plot(scalerMm_X.inverse_transform(target),'-b', scalerMm_X.inverse_transform(temp_p), '-g')
plt.show()