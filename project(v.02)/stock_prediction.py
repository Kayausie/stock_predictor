# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn import preprocessing
import tensorflow as tf
import datetime
from collections import deque
from sklearn.metrics import mean_absolute_error
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

#------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
#------------------------------------------------------------------------------
# DATA_SOURCE = "yahoo"
COMPANY = 'AMZN'

PREDICTION_DAYS = 60 # Number of days to look back to base the prediction

TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2023-08-01'       # End date to read

# data = web.DataReader(COMPANY, DATA_SOURCE, TRAIN_START, TRAIN_END) # Read data using yahoo


import yfinance as yf
#TASKC.2: Writing a function to process the data
#First of all, we need to define a function to shuffle the dataset, this function will be used later on in our main process_data function. 
#Our function will take in two arrays a and b, and shuffle them but still keeps the corrensponding relationship between the elements. doing this will allow
#us to mix up the order of the training data, which is important for training a good model.
def shuffle_in_unison(a, b):
    # in the first line, we are going to create a random state using numpy's random get_state function
    # this state will be used to ensure that we shuffle both arrays in the same way
    state = np.random.get_state()
    #shuffle the first array
    np.random.shuffle(a)
    #set the random state back for the second array
    np.random.set_state(state)
    #this shuffle will be the same as the first one, as they shared the same random state
    np.random.shuffle(b)
# Now we will move to our main function process_data, this function will take in a stock ticker, start and end date for the data we want to load, furthermore 
#also params such as shuffle, store, split_by_date, test_ratio
def process_data(ticker,startDate, endDate, test_ratio, scale=True, split_by_date=True,shuffle=True,store=True,
                 feature_columns=['Close', 'Volume', 'Open', 'High', 'Low'], nan_strategy="drop"
                 , n_steps=50, lookup_step=1):
    # first, we will initialize a list to hold all the sequences with n_steps length, accompanied with the target value
    sequence_data = []
    # initialize a deque to hold the past maximumly n_steps of data
    sequences = deque(maxlen=n_steps)
    # create a dictionary to store the result
    result = {}
    # construct the X's and y's
    X, y = [], []
    #Check if startDate and endDate are provided, otherwise
    #it will raise an exception
    if startDate is None or endDate is None:
        raise Exception("Please provide a start date and end date in the format YYYY-MM-DD")
    #nan_strategy (how to deal with NaN values): drop, ffill, bfill. If the param passed is not one of these 3, raise an error
    # I will explain these 3 strategies more in detail later on
    if nan_strategy not in ["drop", "ffill", "bfill"]:
        raise ValueError("nan_strategy must be one of: 'drop', 'ffill', 'bfill'")
    #Load data from Yahoo Finance: if ticker is a string, load data from Yahoo Finance, otherwise if a DataFrame, use it directly, or raise an error if neither
    if isinstance(ticker, str):
        #download the data using yfinance library
        data = yf.download(ticker, startDate, endDate)
    elif isinstance(ticker, pd.DataFrame):
        # use loaded data directly otherwise
        data = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    #Deal with NaN values first: drop(drop entire row if there is nan value), or forward fill, or backward fill
    if nan_strategy == "drop":
        # Drop any row that has a NaN in ANY of the selected feature columns
        data.dropna(inplace=True)
    elif nan_strategy in ("ffill", "bfill"):
        # Forward or backward fill only the feature columns
        # For example, if we have a dataframe like this:
        # index       A  B  C
        # 0           1  2  3
        # 1           4  NaN 6 , we can see that column B has a NaN value in row 1
        # if we do a forward fill, the NaN will be filled with the value from the previous row, which is 2
        # so the dataframe will become:
        # index       A  B  C
        # 0           1  2  3
        # 1           4  2  6 ( value changed from NaN to 2)
        # if we do a backward fill, the NaN will be filled with the value from the next row, which is 7
        # so the dataframe will become:
        # index   A  B  C
        # 0       1  7  3
        # 1       4  7  6 (value changed from NaN to 7)
        method = nan_strategy
        # fill NaN values only in the feature columns
        data[feature_columns] = data[feature_columns].fillna(method=method)
    #save a copy of the original dataframe
    result['df'] = data.copy()
    #store the data in a csv file if store param i set to True, default is True
    if store:
        data.to_csv("data.csv", index=True) # Save a copy of the original data
    # add date as a column
    if "date" not in data.columns:
        # in Yfinance, the date is the index of the dataframe, so we need to reset the index to make it a column
        data["date"] = data.index
    #scale the data (prices) from 0 to 1 if scale param is set to True, default is True
    if scale:
        # initialize the scaler
        column_scaler = {}
        # scale the data (prices) from 0 to 1 for all feature columns
        for column in feature_columns:
            # in this part, we are using MinMaxScaler from sklearn to scale the data
            # Using scaler will help us scale down the data to a range of 0 to 1, which will help the model to converge faster
            # for example, if we have a column with values ranging from 100 to 1000, the model will have a hard time learning the patterns in the data
            # because the values are too large, but if we scale it down to a range of 0 to 1, the model will have an easier time learning the patterns
            # Imagine we have a dataset: 1000, 2000, 3000, 4000, 5000
            # but with provided MinMaxScaler from 0 to 1, it will be transformed to: 0, 0.25, 0.5, 0.75, 1
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            #scale the column and save the scaler
            data[column] = scaler.fit_transform(data[column])
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

     # add the target column (future) by shifting by `lookup_step`
     # for example, our lookup step is 2, and our open column in chronological order is: 4, 2, 3, 1, 2
     # then our future column will be: 3, 1, 2, NaN, NaN (shifted by 2), as we have shifted the whole column up by 2, and append it to be another new column
     # Apparentlym the last 2 rows will be NaN as there is no data for them, so you will see a line which we will drop these NaN values later on
    data['Future'] = data['Open'].shift(-lookup_step)
    # Before that, we need to save the last `lookup_step` sequence for future testing
    last_sequence = np.array(data[feature_columns].tail(lookup_step))
    
    # drop NaNs
    data.dropna(inplace=True)
    # After this step, we have finished some basic data preprocessing steps,
    # now we will deal with splitting the data to training and testing sets
    # Now, we will loop through the data and create sequences of n_steps length
    # right here, we use zip to combine the feature columns and the future column together, for example one of the record will look like:
    # [131, 123, 182,192,800980, "2020-01-01"], 132. The first part is the feature columns with one value of date, and the second part is the target value
    for entry, target in zip(data[feature_columns + ["date"]].values, data['future'].values):
        #save the sequence with length of 60 first to the sequences deque
        sequences.append(entry)
        if len(sequences) == n_steps:
            # if the length of the sequences is equal to n_steps, we will append the sequence and the target to the sequence_data list
            # Using deque will allow us to do both FIFO or LIFO, so if we append a new item, the oldest item will be removed automatically
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # this is the last sequence we will use to predict the future stock price that we will store it in the result dictionary
    result['last_sequence'] = last_sequence
    
    # append all the sequences and targets to the X and y lists
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # if split_by_date param is set to True, we will split the data by date, ;
    # otherwise we will split it randomly
    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        # Right here, we split the data by calculating the number of training samples based on the test_ratio param
        # Doing so, will enable all training data to be before the testing data in chronological order
        train_samples = int((1 - test_ratio) * len(X))
        # Let's say train samples is 800, then the following line will get the first 800 samples as training data
        result["X_train"] = X[:train_samples]
        # similarly, the following line will get the first 800 target values as training target values
        result["y_train"] = y[:train_samples]
        # the following line will get the remaining samples as testing data
        result["X_test"]  = X[train_samples:]
        # similarly as above 
        result["y_test"]  = y[train_samples:]
        # if you choose shuffle to be true, we'll shuffle the training and testing data
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:    
        # split the dataset randomly, and divide it as per the test_ratio param
        # for example, if test_ratio is 0.2, then 20% of the data will be used as testing data, and the rest 80% as training data
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_ratio, shuffle=shuffle)

    # get the list of test set dates
    # right here, result[X_test] is a 3D array, which we have converted earlier
    # it is as a format of : [number of samples, n_steps, number of features + 1 (date) 
    # for example: ([123,234,123,432,654], [234,123,432,654,876], ... ,[345,456,567,678,789]) ( 5 records)
    # this is an array of 5 samples ( record), each sample has 5 timesteps, and each timestep has 1 feature
    # for the following line, we will get all samples (:), the last timestep (-1), and the last column (-1) which is the date
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    # result["test_df"].index.duplicated will return a boolean array, which is True for duplicated dates, and False for unique dates
    # Then, the "~" operator will invert the boolean array, so that we get True for unique dates, and False for duplicated dates
    # so after that, result["test_df"] will only keep the rows with unique dates, as these values are True in the boolean array
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    # finally, we return the result dictionary
    return result
def plot_financial_data(ticker,type="days", startDate="2023-07-02", endDate="2024-07-02", tradeDays=365):
    if(type not in ["days, range"]):
        raise ValueError("type must be one of: 'days', 'range'")
    if isinstance(ticker, str):
        #download the data using yfinance library
        data = yf.download(ticker, period = f"{tradeDays}d")
    elif isinstance(ticker, pd.DataFrame):
        # use loaded data directly otherwise
        data = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    if(type == "range"):
        try:
            datetime.date.fromisoformat(startDate)
            datetime.date.fromisoformat(endDate)
        except ValueError:
            raise ValueError("Please provide a start date and end date in the format YYYY-MM-DD")
def plot_candlestick_data(ticker,type="days", startDate="2023-07-02", endDate="2024-07-02", tradeDays=365):
    if isinstance(ticker, str):
        ticker = yf.Ticker(ticker)
    if(type not in ["days", "range"]):
        raise ValueError("type must be one of: 'days', 'range'")
    if(type == "range"):
        try:
            datetime.date.fromisoformat(startDate)
            datetime.date.fromisoformat(endDate)
        except ValueError:
            raise ValueError("Please provide a start date and end date in the format YYYY-MM-DD")
    else:
        # validate tradeDays, if the number typed is not a positive integer, raise an error. Otherwise, default to 365 days
        if(not isinstance(tradeDays, int) or tradeDays <= 0):
            raise ValueError("tradeDays must be a positive integer")
    if isinstance(ticker, yf.Ticker):
        #download the data using yfinance library, either with a range of dates or a number of days
        if(type == "range"):
            data = ticker.history( start=startDate, end=endDate)
        else:
            data = ticker.history( period = f"{tradeDays}d")
    elif isinstance(ticker, pd.DataFrame):
        # use loaded data directly otherwise
        data = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    print(data)
    data["Open"] = data["Open"].astype(float)
    #Using mplfinance to plot the data
    # fplt.plot(
    #         data,
    #         type='candle',
    #         title='365 trading days ',
    #         ylabel='Price ($)',
    #         style='yahoo',
    #     )
    #Using plotly to plot the data
    candlestick = go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close']
                            )
    fig = go.Figure(data=[candlestick])
    fig.show()
def plot_boxplot_data(ticker,type="days", startDate="2023-07-02", endDate="2024-07-02", tradeDays=365, day_range=5):
    if isinstance(ticker, str):
        ticker = yf.Ticker(ticker)
    if(type not in ["days", "range"]):
        raise ValueError("type must be one of: 'days', 'range'")
    if(type == "range"):
        try:
            datetime.date.fromisoformat(startDate)
            datetime.date.fromisoformat(endDate)
        except ValueError:
            raise ValueError("Please provide a start date and end date in the format YYYY-MM-DD")
    else:
        # validate tradeDays, if the number typed is not a positive integer, raise an error. Otherwise, default to 365 days
        if(not isinstance(tradeDays, int) or tradeDays <= 0):
            raise ValueError("tradeDays must be a positive integer")
    if isinstance(ticker, yf.Ticker):
        #download the data using yfinance library, either with a range of dates or a number of days
        if(type == "range"):
            data = ticker.history( start=startDate, end=endDate)
        else:
            data = ticker.history( period = f"{tradeDays}d")
    elif isinstance(ticker, pd.DataFrame):
        # use loaded data directly otherwise
        data = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    data["dates"]=data.index
    data["Open"] = data["Open"].astype(float)
    print(data)
    #create dataset for box plot of day_range
    data_shown = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])
    #Todo: create dataset for day_range
    sequences = deque(maxlen=day_range)
    for entry in data.values:
        sequences.append(entry)
        if len(sequences) == day_range:
            temp_rec = {}
            temp_rec["Date"] = sequences[0][7]
            temp_rec["Open"] = sequences[0][0]
            temp_rec["Close"] = sequences[4][3]
            temp_rec["High"] = max([rec[1] for rec in sequences])
            temp_rec["Low"] = min([rec[2] for rec in sequences])
            data_shown = pd.concat([data_shown, pd.DataFrame([temp_rec])], ignore_index=True)
            temp_rec.clear()
            sequences.clear()
    print(data_shown)
    #Using plotly to plot the data
    candlestick = go.Candlestick(
                            x=data_shown['Date'],
                            open=data_shown['Open'],
                            high=data_shown['High'],
                            low=data_shown['Low'],
                            close=data_shown['Close']
                            )
    fig = go.Figure(data=[candlestick])
    medians = (data_shown['High'] + data_shown['Low']) / 2
     
    half_span = pd.Timedelta(days= (day_range * 0.7/ 2.15)if day_range%2==1  else (day_range*0.7/2.1))
# Loop over each candle, add a short horizontal line at the median
    for xi, median in zip(data_shown["Date"], medians):
        fig.add_shape(
            type="line",
            x0=xi - half_span,   # start a bit before the candle center
            x1=xi + half_span,   # end a bit after the candle center
            y0=median,
            y1=median,
            line=dict(color="firebrick", width=2)
        )
    fig.show()
data = pd.DataFrame({

    "Date": ["2026-01-01","2026-01-06","2026-01-11"],
    "q1":   [25,  30,  28],
    "q3":   [75,  78,  80],
    "median":[50,  55,  53],
    "whislo":[10,  12,  11],
    "whishi":[95, 100,  98],
})
# Get the data for the stock AAPL
data = process_data(COMPANY, TRAIN_START, TRAIN_END, test_ratio=0.2, n_steps=60, lookup_step=1)
PRICE_VALUE = "Close"

scaler = MinMaxScaler(feature_range=(0, 1)) 
# Note that, by default, feature_range=(0, 1). Thus, if you want a different 
# feature_range (min,max) then you'll need to specify it here
scaled_data = scaler.fit_transform(data['df'][PRICE_VALUE].values.reshape(-1, 1)) 
#------------------------------------------------------------------------------
# Prepare Data
## To do:
# 1) Check if data has been prepared before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Use a different price value eg. mid-point of Open & Close
# 3) Change the Prediction days
#------------------------------------------------------------------------------
x_train = []
y_train = []

# Number of days to look back to base the prediction

scaled_data = scaled_data[:,0] # Turn the 2D array back to a 1D array
# Prepare the data
for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x-PREDICTION_DAYS:x])
    y_train.append(scaled_data[x])

# Convert them into an array
x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# and q = PREDICTION_DAYS; while y_train is a 1D array(p)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# We now reshape x_train into a 3D array(p, q, 1); Note that x_train 
# is an array of p inputs with each input being a 2D array 

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# This is our first hidden layer which also spcifies an input layer. 
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.
    
# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=25, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'

# test_data = web.DataReader(COMPANY, DATA_SOURCE, TEST_START, TEST_END)

test_data = yf.download(COMPANY,TEST_START,TEST_END)


# The above bug is the reason for the following line of code
# test_data = test_data[1:]

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((data['df'][PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the fisrt
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

model_inputs = scaler.transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above 
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period 
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------


real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

mae = mean_absolute_error(actual_prices, predicted_prices)
print("Mean Absolute Error:", mae)
#------------------------------------------------------------------------------
#metrics
buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
# Sell profit: predicted price lower than current price
sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
# test_df = pd.DataFrame()
# test_df["actual"] = actual_prices
# test_df["predicted"] = predicted_prices

# # Add buy/sell profit columns
# test_df["buy_profit"] = [buy_profit(c, p, t) for c, p, t in zip(test_df["actual"], test_df["predicted"], test_df["actual"])]
# test_df["sell_profit"] = [sell_profit(c, p, t) for c, p, t in zip(test_df["actual"], test_df["predicted"], test_df["actual"])]
# accuracy_score = (len(test_df[test_df['buy_profit'] > 0]) + len(test_df[test_df['sell_profit'] > 0])) / len(test_df)
# print("Accuracy score:", accuracy_score)
# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??