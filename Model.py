import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt  
from sklearn import metrics  
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model, Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, LSTM, Dropout, Activation

df = pd.read_csv('stock data.csv')
print(np.max(df.iloc[0:1234,1]), np.min(df.iloc[0:1234,1]))
train_set = df.iloc[0:1234,1:2].values
test_set = df.iloc[1234:1254,1:2].values
scaler = MinMaxScaler(feature_range = (0,1))
train_set_scaled = scaler.fit_transform(train_set)

X_train = []  
Y_train = [] 
for i in range(60, 1234):  
    X_train.append(train_set_scaled[i-60:i, 0])
    Y_train.append(train_set_scaled[i, 0])
    
X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=60, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.3))

model.add(LSTM(units=60, return_sequences=True))  
model.add(Dropout(0.3))

model.add(LSTM(units=60, return_sequences=True))  
model.add(Dropout(0.2))

model.add(LSTM(units=60))  
model.add(Dropout(0.2))

model.add(Dense(units = 1))  

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, Y_train, epochs = 100, batch_size = 32)

tot = pd.concat((df.iloc[0:1234,:]['open'], df.iloc[1234:1254,:]['open']), axis=0)
test_ip = tot[len(tot) - len(df.iloc[1234:1254,:]) - 60:].values  

test_ip = scaler.transform(test_ip.reshape(-1,1))  

X_test = []  
for i in range(60, 80):  
    X_test.append(test_ip[i-60:i, 0])

X_test = np.array(X_test)  
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 
predict = model.predict(X_test) 
predict = scaler.inverse_transform(predict) 

plt.figure(figsize=(10,6))  
plt.plot(df.iloc[1234:1254,1:2].values, color='blue', label='Actual Stock Price')  
plt.plot(predict , color='red', label='Predicted Stock Price')  
plt.title('Stock Price Prediction')  
plt.xlabel('Date')  
plt.ylabel('Stock Price')  
plt.legend()  
plt.show()

Actual = np.array(df.iloc[1234:1254,1]) 
Predicted = predict  
print('Mean Absolute Error:', metrics.mean_absolute_error(Actual, Predicted))  
print('Mean Squared Error:', metrics.mean_squared_error(Actual, Predicted))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Actual, Predicted))) 
print('Mean Value of Stock Prices:', np.mean(df.iloc[0:1254,1]))