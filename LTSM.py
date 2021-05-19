##https://teddylee777.github.io/tensorflow/

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import requests
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Data input
#https://au.finance.yahoo.com/quote/VRT.AX?p=VRT.AX&.tsrc=fin-srch
df = pd.read_csv('csv/VRT.AX.csv')

#df.describe()

print(min(df['Date']), max(df['Date']))
# '2016-03-03' '2021-03-03'

## Plot
# plt.figure(figsize=(16, 9))
# sns.lineplot(y=df['Close'], x=df['Date'])
# plt.xlabel('time')
# plt.ylabel('price')

scaler = MinMaxScaler()
scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df_scaled = scaler.fit_transform(df[scale_cols])

df_scaled = pd.DataFrame(df_scaled, columns= ['Open_S', 'High_S', 'Low_S', 'Close_S', 'Volume_S'])
df_scaled = pd.concat([df['Date'], df_scaled], axis=1)


# Use only 200 days for test data
TEST_SIZE = 200
train = df_scaled[:-TEST_SIZE]
test = df_scaled[-TEST_SIZE:]

def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

feature_cols = ['Open_S', 'High_S', 'Low_S', 'Volume_S']
label_cols = ['Close_S']

train_feature = train[feature_cols]
train_label = train[label_cols]

test_feature = test[feature_cols]
test_label = test[label_cols]

# train dataset
train_feature, train_label = make_dataset(train_feature, train_label, 20)

# train, validation set 생성
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)
# (836, 20, 4) (210, 20, 4) (836, 1) (210, 1)

# test dataset (실제 예측 해볼 데이터)
test_feature, test_label = make_dataset(test_feature, test_label, 20)
test_feature.shape, test_label.shape
# ((180, 20, 4), (180, 1))

#Keras를 활용한 LSTM 모델 생성
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(16,
               input_shape=(train_feature.shape[1], train_feature.shape[2]),
               activation='relu',
               return_sequences=False)
          )
model.add(Dense(1))

# Model traning
model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=5)

model_path = 'model'
filename = os.path.join(model_path, 'tmp_checkpoint.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(x_train, y_train,
                                    epochs=200,
                                    batch_size=16,
                                    validation_data=(x_valid, y_valid),
                                    callbacks=[early_stop, checkpoint])

# 예측
model.load_weights(filename)
pred = model.predict(test_feature)

pred.shape

# Plot
plt.figure(figsize=(12, 9))
plt.plot(test_label, label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()