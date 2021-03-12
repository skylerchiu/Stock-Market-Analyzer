
"""The Stock 'Em Robot

Original file is located at
    https://colab.research.google.com/drive/19cOocT_W4YoNiqa9ZBC2TMKdjY-YG6vq
"""

#Uses recurrent neural networks and tensorflow to predict a future

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from pandas_datareader.data import get_quote_yahoo
import yfinance as yf
from pandas_datareader import data as pdr



from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#@title  { display-mode: "form" }
get_quote_yahoo('AAPL')

#@title
get_quote_yahoo('AMC')

#@title
get_quote_yahoo('LSPD.TO')

#@title
get_quote_yahoo('TSLA')

#@title
get_quote_yahoo('NOK')

#@title
get_quote_yahoo('HIVE.V')

#Select company name
company = 'AAPL'

print(company)
intraday_data = yf.download(tickers=company, #period: The number of days/month of data required. The valid frequencies are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
 period="1d",
 interval="30m") #interval: The frequency of data. The valid intervals are 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo ,
pd.set_option("max_rows", None)
intraday_data.head(18)

intraday_data_graph = yf.download(tickers=company, #period: The number of days/month of data required. The valid frequencies are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
 period="1d",
 interval="1m", auto_adjust=True)
intraday_data_graph.shape
plt.figure(figsize = (16,8))
plt.title('Intraday Graph', fontsize = 24)
plt.plot(intraday_data_graph['Close'])
plt.xlabel('Time', fontsize = 18)
plt.ylabel('Closing Price (USD)', fontsize = 18)
plt.show()

#Select date to represent on graph
data = web.DataReader(company, data_source ='yahoo', start = '2012-01-17', end = '2022-01-01')

data.shape
plt.figure(figsize = (16,8))
plt.title('Close Price History', fontsize = 24)
plt.plot(data['Close'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Closing Price (USD)', fontsize = 18)
plt.show()

'''
#Candle stick graph
import plotly.graph_objects as go



df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['AAPL.Open'],
                high=df['AAPL.High'],
                low=df['AAPL.Low'],
                close=df['AAPL.Close'])])

fig.show()'''

#Prepares and scales data
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
  x_train.append(scaled_data[x-prediction_days:x, 0])
  y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

#Builds the Model
model = Sequential() #Calls Squential method

model.add(LSTM(units = 50, return_sequences = True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences = True))
model.add (Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #Prediction of next closing price

model.compile (optimizer = 'adam',loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size = 150) #Epcohs are how many times you go through your training set. Base =25 batch size =32

#Test the model accuracy on existing data
test_start = dt.datetime(2021,2,25) #For more stable stocks, choose a start date further in the past. For volitile stocks, select a start date a week or two prior
test_end = dt.datetime.now() 

test_data = web.DataReader(company,'yahoo',test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data["Close"], test_data['Close']), axis = 0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#Makes preditions on data

x_test = []

for x in range(prediction_days, len(model_inputs)):
  x_test.append(model_inputs[x-prediction_days:x, 0])

x_test=np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices) #reverses scaling

plt.plot(actual_prices, color = "black", label = f"Actual")
plt.plot(predicted_prices, color = "green", label = f"Prediction")
plt.title(f"{company} Share Price")
plt.xlabel('Number of Days Past Start Date')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()

#Predicting one day into the future
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print ("The projected price for ",company, " for tomorrow is approximately:")
print(prediction)