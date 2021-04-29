#@title Run This Line To Activate Program
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from pandas_datareader.data import get_quote_yahoo
import yfinance as yf
from pandas_datareader import data as pdr
# %matplotlib inline
import mplfinance as mpf
from mpl_finance import candlestick_ohlc 



from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# if error shows yfinance not installed, run: pip install yfinance
# if error shows mplfinance not installed,  run: pip install mplfinance
# if error shows mpl_finance not installed, run: pip install mpl_finance

pip install yfinance

pip install mpl_finance

pip install mplfinance

"""# Personal Tickers"""

#@title Personal Portfolio Daily Summary{ vertical-output: true }
##Run to see data

stocks = ['SPY','AAPL', 'AC.TO', 'TSLA', "LSPD.TO", 'MSFT', 'TD.TO', 'HIVE.V']

date = dt.datetime.now()

print("Daily Summary for " +str(date))
print()
print()

for x in stocks:
  print(x)
  print("-------------------------------")
  print("Current Price: $", get_quote_yahoo(x).price[0], get_quote_yahoo(x).financialCurrency[0])
  print("Intraday change:", round(get_quote_yahoo(x).regularMarketChange[0], 2), " , ", round(get_quote_yahoo(x).regularMarketChangePercent[0], 2), "%")
  print(" ")
  print("Open: $", get_quote_yahoo(x).regularMarketOpen[0])
  print("Daily High: $", get_quote_yahoo(x).regularMarketDayHigh[0])
  print("Daily Low: $", get_quote_yahoo(x).	regularMarketDayLow[0])
  print("Volume:", get_quote_yahoo(x).regularMarketVolume[0])
  print(" ")
  print(" ")
  print(" ")

#@title  SPY
get_quote_yahoo('SPY')

#@title  AAPL
get_quote_yahoo('AAPL')

#@title TSLA
get_quote_yahoo('TSLA')

#@title LSPD.TO
get_quote_yahoo('LSPD.TO')

#@title AC.TO
get_quote_yahoo('AC.TO')

#@title MSFT
get_quote_yahoo('MSFT')

#@title  TD.TO
get_quote_yahoo('TD.TO')

#@title HIVE.V
get_quote_yahoo('HIVE.V')

"""# Custom Stock Data and Visualizers"""

#@title Stock Data for the Last 10 Minutes
company = input("Enter the stock symbol : ") 
print(company)
intraday_data = yf.download(tickers=company, #period: The number of days/month of data required. The valid frequencies are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
 period="1d",
 interval="1m") #interval: The frequency of data. The valid intervals are 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo ,
pd.set_option("max_rows", None)
intraday_data.tail(10)

# Commented out IPython magic to ensure Python compatibility.
#@title Intraday Chart
# %matplotlib inline
import mplfinance as mpf


company = input("Enter the stock symbol: ")

intraday_data = yf.download(tickers=company, #period: The number of days/month of data required. The valid frequencies are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
 period="1d",
 interval="1m")


graph_title = company + " Intraday Graph"
Graph_Type = 'candle' #@param ["line", "candle"] {allow-input: true}

mpf.plot(intraday_data,type = Graph_Type, figratio = (2,1),
         title = graph_title , mav = (10), volume = True, 
         style = 'charles')

#@title Specific Time Period Chart

company = input("Enter the stock symbol: ")
data = web.DataReader(company, data_source ='yahoo', start = '2012-01-17', end =dt.datetime.now() )
Graph_Type = 'candle' #@param ["line", "candle"] {allow-input: true}
start_date = '2020-10-10' #@param {type:"date"}
end = '2021-04-17' #@param {type:"date"}
graph_title = company + " " +start_date + " - Present"
mpf.plot(data[start_date : end],type = Graph_Type, figratio = (7,3),
         title = graph_title , mav = (2), volume = True, 
         style = 'charles')

#@title Closing Price History
#Select date to represent on graph
company = input("Enter the stock symbol: ")
data = web.DataReader(company, data_source ='yahoo', start = '2007-01-01', end =dt.datetime.now() )

data.shape
plt.figure(figsize = (16,8))
plt.title(company +' Close Price History', fontsize = 24)
plt.plot(data['Close'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Closing Price (USD)', fontsize = 18)

plt.show()

"""# Indicators and Analysis Charts"""

#@title Pivot point and Resistance indicator (Year to Date)

start_resistance = dt.datetime(2021,1,1)
now = dt.datetime.now()
stock = input("Enter the stock symbol: ")

df = pdr.get_data_yahoo(stock, start_resistance, now)

df["Close"].plot(Label = "CLosing Price")

pivots=[]
dates = [] #stores dates of pivots
counter = 0 #counts number of days since last pivot
lastPivot = 0 #Stores last Pivot value

Range = [0,0,0,0,0,0,0,0,0,0]
dateRange = [0,0,0,0,0,0,0,0,0,0]

for i in df.index:
  currentMax = max(Range, default = 0)
  value = round(df["Close"][i],2)

  Range = Range[1:9]
  Range.append(value)
  dateRange = dateRange[1:9]
  dateRange.append(i)

  if currentMax == max(Range, default = 0):
    counter += 1
  else:
    counter = 0
  if counter == 5:
    lastPivot = currentMax
    dateloc = Range.index(lastPivot)
    lastDate = dateRange[dateloc]

    pivots.append(lastPivot)
    dates.append(lastDate)

pivot_high_1=df['High'][-21:-1].max()
pivot_low_1=df['High'][-21:-1].min()

print()
numer_of_days_the_resistance_should_hold = 30
timeD = dt.timedelta(days = numer_of_days_the_resistance_should_hold) #determines how many days to draw pivot point for

for index in range(len(pivots)):
  plt.plot_date([dates[index], dates[index]+ timeD],
    [pivots[index], pivots[index]], linestyle = "-", linewidth = 2, marker = ",")
  


print("Notable Pivot Points: ")
for index in range(len(pivots)):
  print("$" + str(pivots[index])+" on " + str(dates[index]))
print()

rounded_pivot_high_1 = round(pivot_high_1, 2)
rounded_pivot_low_1 = round(pivot_low_1, 2)
print("Major Support at: $" + str(rounded_pivot_low_1))
print("Major Resistance at: $" + str(rounded_pivot_high_1))
print()


plt.title(stock + " Pivot Point Indicator")
plt.ylabel("Price")
plt.axhline(y = pivot_low_1, color = 'g', linewidth = 5, linestyle = '-', alpha=0.2, label = "Support Line")
plt.axhline(y = pivot_high_1, color = 'r', linewidth = 5, linestyle = '-', alpha=0.2, label = "Resistance Line")
plt.rcParams['figure.dpi'] = 150
plt.legend()
plt.show()
print()

#@title Simple Moving Averages and Bollinger Bands Indicator
#import relevant libraries
import yfinance as yf
import datetime as dt
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import datetime as datetime
import numpy as np
from mpl_finance import candlestick_ohlc
yf.pdr_override() #activate yahoo finance workaround

smasUsed = [10,20,50] #choose simply moving average over these day ranges
#@markdown Starting Date:
year = 2021# @param {type:"integer"}
month = 1 #@param {type:"integer"}
day = 1#@param {type:"integer"}
start = dt.datetime(year,month,day) - dt.timedelta(days = max(smasUsed)) #Sets starting point of dataframe
now = dt.datetime.now() #Sets end point of dataframe to present day
stock = input("Enter the stock symbol: ")

prices = pdr.get_data_yahoo (stock, start, now)
fig, ax1 = plt.subplots() 

#Calculate moving average
for x in smasUsed:
  sma = x
  prices['SMA_' + str(sma)] = prices.iloc[:,4].rolling(window=sma).mean() #calculates sma and creates col, storing under SMA_*sma value*

#Calculate Bollinger Bands. The Bollinger bands will be 2 standard deviations above, and 2 standard deviations below the SMA. The price is likely to revert into the Bollinger Bands if it were to exit
BBperiod = 15 #choose moving average period
stdev = 2 #Choose standard deviations
prices['SMA' + str(BBperiod)] = prices.iloc[:,4].rolling(window=BBperiod).mean() #calculates standard deviation
prices['STDEV'] = prices.iloc[:,4].rolling(window = BBperiod).std()
prices['LowerBand'] = prices['SMA' +str(BBperiod)] - (stdev*prices['STDEV'])
prices['UpperBand'] = prices['SMA' +str(BBperiod)] + (stdev*prices['STDEV'])
prices["Date"] = mdates.date2num(prices.index) #converts time stamp to number

#Calculate 10.4.4 stochastic.

Period = 10
K = 4
D = 4

prices["RolHigh"] = prices["High"].rolling(window=Period).max() #Finds High of period
prices["RolLow"] = prices["Low"].rolling(window = Period).min() #Finds Low of period
prices["stok"] = ((prices["Adj Close"] - prices["RolLow"])/(prices["RolHigh"] - prices["RolLow"]))*100
prices["K"] = prices["stok"].rolling(window=K).mean()
prices["D"] = prices["K"].rolling(window=D).mean()
prices["GD"] = prices["High"]
ohlc = []

prices = prices.iloc[max(smasUsed):]

greenDotDate = []
greenDot = []
lastK = 0
lastD = 0
lastLow = 0
lastClose = 0
lastLowBB=0

for i in prices.index:
  append_me = prices["Date"][i], prices["Open"][i], prices["High"][i], prices["Low"][i], prices["Adj Close"][i], prices["Volume"][i]
  ohlc.append(append_me)

    #Check for Green Dot
  if prices['K'][i]>prices['D'][i] and lastK<lastD and lastK <60:
    plt.plot(prices["Date"][i],prices["High"][i]+1, marker="o", ms=4, ls="", color='g') #plot green dot

    greenDotDate.append(i) #store green dot date
    greenDot.append(prices["High"][i])  #store green dot value

    #Check for Lower Bollinger Band Bounce
  if ((lastLow<lastLowBB) or (prices['Low'][i]<prices['LowerBand'][i])) and (prices['Adj Close'][i]>lastClose and prices['Adj Close'][i]>prices['LowerBand'][i]) and lastK <60:  
    plt.plot(prices["Date"][i],prices["Low"][i]-1, marker="o", ms=4, ls="", color='b') #plot blue dot
 
  #store values
  lastK=prices['K'][i]
  lastD=prices['D'][i]
  lastLow=prices['Low'][i]
  lastClose=prices['Adj Close'][i]
  lastLowBB=prices['LowerBand'][i]




for x in smasUsed:
  sma = x
  prices['SMA_' + str(sma)].plot(label = str(x) +' day moving average')
prices['UpperBand'].plot(label = 'Upper Bollinger Band', color = 'lightgray')
prices['LowerBand'].plot(label = 'Lower Bollinger Band', color = 'lightgray')

#plot candlesticks
candlestick_ohlc(ax1, ohlc, width = 0.5, colorup = 'g', colordown='r', alpha=0.75)

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.yaxis.set_major_locator(mticker.MaxNLocator(8))
plt.tick_params(axis = 'x', rotation = 45)


pivots=[]
dates = [] #stores dates of pivots
counter = 0 #counts number of days since last pivot
lastPivot = 0 #Stores last Pivot value

Range = [0,0,0,0,0,0,0,0,0,0]
dateRange = [0,0,0,0,0,0,0,0,0,0]

for i in prices.index:
  currentMax = max(Range, default = 0)
  value = round(prices["High"][i],2)

  Range = Range[1:9]
  Range.append(value)
  dateRange = dateRange[1:9]
  dateRange.append(i)

  if currentMax == max(Range, default = 0):
    counter += 1
  else:
    counter = 0
  if counter == 5:
    lastPivot = currentMax
    dateloc = Range.index(lastPivot)
    lastDate = dateRange[dateloc]

    pivots.append(currentMax)
    dates.append(lastDate)

print()
numer_of_days_the_resistanct_should_hold = 30
timeD = dt.timedelta(days = numer_of_days_the_resistanct_should_hold) #determines how many days to draw pivot point for

for index in range(len(pivots)):
  plt.plot_date([dates[index]-(timeD*.075), dates[index]+ timeD],
    [pivots[index], pivots[index]], linestyle = "--", linewidth = 1, marker = ",")
  plt.annotate(str(pivots[index]), (mdates.date2num(dates[index]), pivots[index]), xytext = (-10,7),
                   textcoords = 'offset points', fontsize = 7, arrowprops = dict(arrowstyle = '-|>'))
plt.xlabel('Date')
plt.ylabel("Price")
plt.title(stock)
plt.ylim(prices["Low"].min(), prices["High"].max()*1.05)
plt.rcParams['figure.dpi'] = 200
plt.autoscale()
plt.tight_layout()
plt.legend( prop={'size': 6})
plt.show()
