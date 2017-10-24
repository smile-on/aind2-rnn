import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    ''' separate array like series into set of sliding window data subsets.
    Horizont sliced implementation.
    ''' 
    a = np.asarray(series)
    n = len(a)
    step = 1
    n_strides = int((n-window_size)/step)
    # np slice [start: stop: step]
    strides = [series[i: i+n-window_size: step][:, None] for i in range(0, window_size, step)]
    # stacking window_size number of potentialy long strides 
    X = np.hstack(strides)
    # numpy slice [start: stop: step]
    y = series[window_size: n: step][:, None]
    return X,y

def window_transform_series_index(series, window_size):
    ''' separate array like series into set of sliding window data subsets.
    A cross index based implementation.
    ''' 
    a = np.asarray(series)
    n = len(a)
    step = 1
    n_strides = int((n-window_size)/step)
    index = np.arange(window_size)[None, :] + step*np.arange(n_strides)[:, None]
    X = a[index]
    # numpy slice [start: stop: step]
    y = series[window_size: n: step][:, None]
    return X,y

def window_transform_series_horizont(series, window_size):
    ''' separate array like series into set of sliding window data subsets.
    Horizont sliced implementation.
    ''' 
    a = np.asarray(series)
    n = len(a)
    step = 1
    n_strides = int((n-window_size)/step)
    # np slice [start: stop: step]
    strides = [series[i: i+n-window_size: step][:, None] for i in range(0, window_size, step)]
    # stacking window_size number of potentialy long strides 
    X = np.hstack(strides)
    # numpy slice [start: stop: step]
    y = series[window_size: n: step][:, None]
    return X,y

def window_transform_series_vertical(series, window_size):
    ''' separate array like series into set of sliding window data subsets.
    Vertical sliced implementation.
    ''' 
    a = np.asarray(series)
    n = len(a)
    step = 1
    n_strides = int((n-window_size)/step)
    # each slice is a window rooling over n_strides [start: stop: step]
    strides = [series[i: i+window_size: step] for i in range(0, n_strides)]
    # stacking potentialy big number of window_size strides 
    X = np.vstack(strides)
    # numpy slice [start: stop: step]
    y = series[window_size: n: step][:, None]
    return X,y

    
# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    pass


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
