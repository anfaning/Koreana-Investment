### Keras and Tensorflow >2.0
import matplotlib.pyplot as plt

### Data Collection
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import random as rn

# Setting the seed for numpy-generated random numbers
np.random.seed(37)

# Setting the seed for python random numbers
rn.seed(1254)

from sklearn.preprocessing import MinMaxScaler

yf.pdr_override()
ticker = "TGR.AX"

df = yf.download(ticker, period= "5y")
# df = yf.download(ticker, start= "2016-05-31", end= "2021-06-01")

#df.describe()

df1=df.reset_index()['Open']

scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# print(X_train.shape), print(y_train.shape)
# print(X_test.shape), print(ytest.shape)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

### Create the Stacked LSTM model

import tensorflow as tf
#tf.__version__
tf.random.set_seed(2)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM


d = 0.2
model=Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=(100,1)))
model.add(Dropout(d))
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(d))
model.add(LSTM(128))
model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
model.add(Dense(1, kernel_initializer="uniform", activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.summary()

start = datetime.now()
start_time = start.strftime("%H:%M:%S")

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)

end = datetime.now()
end_time = end.strftime("%H:%M:%S")


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))

### Plotting
# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

#len(test_data)

x_input=test_data[-100:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# demonstrate prediction for next 30 days
lst_output = []
n_steps = 100
i = 0
while (i < 30):

    if (len(temp_input) > 100):
        # print(temp_input)
        x_input = np.array(temp_input[1:])
        print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # print(temp_input)
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i + 1


day_new=np.arange(1,101)
day_pred=np.arange(101,131)

plt.plot(day_new,scaler.inverse_transform(df1[-100:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
plt.show()

# df3=df1.tolist()
# df3.extend(lst_output)
# df3=scaler.inverse_transform(df3[-180:]).tolist()
# plt.plot(df3)
# plt.show()

print("Start time =", start_time, "End time =", end_time)
df_output = pd.DataFrame(scaler.inverse_transform(lst_output), columns= ["Prediction"])

# Add future weekdays to df_output and create to csv
df_date = pd.DataFrame([])
td = datetime.today()
while (len(df_date) < 30):
    if (td.weekday() in [5, 6]):
            td = td + timedelta(days=1)
    else:
        td = td + timedelta(days=1)
        df_date = df_date.append(pd.DataFrame([td.strftime("%d-%m")]))


df_date.reset_index(drop=True, inplace=True)
df_output.insert(0, "Date", df_date)

## Download to CSV
date = datetime.now().strftime("%d%m")
df_output.to_csv(f'output/{ticker}_Prd_{date}.csv', index = False, header=True)








