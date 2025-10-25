import datetime
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, SimpleRNN
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from parameters import *
from tensorflow.keras.losses import Huber
from stock_prediction import process_data
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau
# test for the DEEP Learning model, parameters:
#number of layers
#size of each layer
#layer name(LSTM, GRU, RNU, etc)
#return a deep learning model that can be used to predict stock prices

# explore different DL networks and experiment with different hyperparameter configurations
COMPANY = 'AMZN'
PREDICTION_DAYS = 60
TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2023-08-01'       # End date to read
model_name = "mymodel"
list_config = [ 
    {"name":"LSTM_1x128", "cell":LSTM,      "n_layers":2, "units":128,        "dropout":0.2,
     "epochs":100, "batch_size":64, "optimizer":"adam", "bidirectional":False},
    {"name":"GRU_1x128",  "cell":GRU,       "n_layers":2, "units":128,        "dropout":0.2,
     "epochs":20, "batch_size":64, "optimizer":"adam","bidirectional":False},
    {"name":"RNN_1x128",  "cell":SimpleRNN, "n_layers":2, "units":128,        "dropout":0.2,
     "epochs":20, "batch_size":64, "optimizer":"adam","bidirectional":    False},

    {"name":"LSTM_2x128", "cell":LSTM,      "n_layers":3, "units":128,    "dropout":0.2,
     "epochs":40, "batch_size":64, "optimizer":"adam","bidirectional":False},

   
    {"name":"GRU_2x64",   "cell":GRU,       "n_layers":3, "units":64,      "dropout":0.2,
     "epochs":40, "batch_size":64, "optimizer":"adam","bidirectional":True},

   
    {"name":"LSTM_2x128_do0.3", "cell":LSTM, "n_layers":3, "units":128, "dropout":0.3,
     "epochs":60, "batch_size":64, "optimizer":"adam","bidirectional":True},

   
    {"name":"LSTM_2x128_bs128", "cell":LSTM, "n_layers":2, "units":256, "dropout":0.4,
     "epochs":20, "batch_size":128, "optimizer":"adam","bidirectional":False}]
# Task C.4: in this task, we will now implement our 
# model creation and training function, which will cover the initialization of:
# - the model layers
# - units per layer
# - dropout rate
# - cell type (LSTM, GRU, RNN)
# - bidirectionality
# - optimizer type
# - loss function
# - sequence length of the window
# - number of features
# first of all, we need to create a function that will translate a number to a list, the function
# will check if the input is a list, if not, it will create a list of the same value repeated n_layers times
def to_list(x, n_layers):
        
        # condition to check if x is a list
        if not isinstance(x, list):
            # return the list of x repeated n_layers times, for example, if x=0.2 and n_layers=3, it will return [0.2, 0.2, 0.2]
            return [x]*n_layers
        return x
# this is our function to create the model
def create_model(sequence_length, n_features, dropout=0.2, units=256, cell=LSTM, n_layers=2,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False, output_units=1):
    # a codition to check if dropout, units, bidirectional are lists, if they are, check if their length is equal to n_layers
    # the extended approach here is to allow the user to input either a single value for dropout, units, bidirectional
    if isinstance(dropout, list) or isinstance(units, list) or isinstance(bidirectional, list) :
        if len(dropout) != n_layers or len(units) != n_layers or len(bidirectional) != n_layers:
            # if not, raise an error, otherwise, continue the code outside this if statement
            raise ValueError("Length of dropout/units/bidirectional list must be equal to n_layers")
    # convert dropout, units, bidirectional to lists if they are not
    dropout = to_list(dropout, n_layers)
    units = to_list(units, n_layers)
    bidirectional = to_list(bidirectional, n_layers)
    # initialize the model: we need to create an empty sequential model first to stack our layers on
    model = Sequential()
    # loop through the number of layers and add the layers to the model
    for i in range(n_layers):
        if i == 0:
            # for the first layer, we need to specify the input shape, including the sequence length and number of features
            # under here we need to check if the layer is bidirectional or not
            # Bidirectional is a wrapper that takes a layer as input and makes it bidirectional. It means that 
            # the layer will have two sets of weights, one for the forward direction and one for the backward direction. In 
            # simple terms, it means that the layer will be able to learn from both past and future data points. We can specify if 
            # the layer is bidirectional or not by using a boolean value
            # if the layer is bidirectional, we need to wrap the layer with Bidirectional, specifying the input shape
            # In here we chose return_sequences=True because we need to stack more layers on top of this layer. So the 
            # output of this layer will be a sequence whose size of (B, L, U)( B: the number of windows, L: length of each window, U:
            # hidden features generated by DL models) of the same length as the input sequence
            if bidirectional[i]:
                model.add(Bidirectional(cell(units[i], return_sequences=True), input_shape=(sequence_length, n_features)))
            else:
            # if the layer is not bidirectional, we can just add the layer directly (LSTM by default, but we can choose other models, which 
            # can be RNN, GRU, etc.), specifying the input shape
                model.add(cell(units[i], return_sequences=True, input_shape=( sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer, for the last layer, we need to set return_sequences=False because we don't need to stack more layers on top of this layer
            # so the output of this layer will be a single value (B, U) instead
            # The structure is quite the same as the first layer, but we don't need to specify the input shape here
            if bidirectional[i]:
                model.add(Bidirectional(cell(units[i], return_sequences=False)))
            else:
                model.add(cell(units[i], return_sequences=False))
        else:
            # hidden layers, or middle layers, 
            # for the hidden layers, the structure is quite the same as the first layer, 
            # but we don't need to specify the input shape here
            if bidirectional[i]:
                model.add(Bidirectional(cell(units[i], return_sequences=True)))
            else:
                model.add(cell(units[i], return_sequences=True))
        # add dropout after each layer
        # Dropout here is a regularization technique that helps 
        # to prevent overfitting. It works by randomly setting a fraction of input units to 0 at each update during training 
        # time, which helps to prevent the model from becoming too reliant on any one feature.
        # The dropout rate is a hyperparameter that can be tuned to find the optimal value for a given dataset and model architecture
        model.add(Dropout(dropout[i]))
    # Finally, we need to add the output layer, which is a dense layer with a single unit and linear activation function
    # because we are doing regression here, predicting a single continuous value (the stock price)
    model.add(Dense(output_units))
    # compile the model with the specified loss function and optimizer
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model
def test_experiment(config):
    data = process_data(COMPANY, TRAIN_START, TRAIN_END, test_ratio=0.2, n_steps=60, lookup_step=1)
    model = create_model(PREDICTION_DAYS, len(FEATURE_COLUMNS), 
                         dropout=config["dropout"], 
                         units=config["units"], 
                         cell=config["cell"], 
                         n_layers=config["n_layers"], 
                         optimizer=config["optimizer"], 
                         bidirectional=config["bidirectional"])
    model.compile(
        loss="mse",          # or "huber" as string
        optimizer=config["optimizer"],      # or your chosen optimizer
        metrics=["mae"]        # track mean absolute error
    )
    checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=False, save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    history = model.fit(data["X_train"], data["y_train"],
                    batch_size=config["batch_size"],
                    epochs=config["epochs"],
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, 
                               tensorboard,
                               ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=5, min_lr=1e-5, verbose=1)
                               ],
                    verbose=1)
if __name__ == "__main__":
    # you can change the index to test different configurations
    test_experiment(list_config[6])