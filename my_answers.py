from typing import List
from io import StringIO
import string
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size, step_size = 1):
    ''' separate array like series into set of sliding window data subsets.
    Horizont sliced implementation.
    ''' 
    a = np.asarray(series)
    n = len(a)
    n_strides = int((n-window_size)/step_size)
    # np slice [start: stop: step]
    strides = [series[i: i+n-window_size: step_size][:, None] for i in range(0, window_size, step_size)]
    # stacking window_size number of potentialy long strides 
    X = np.hstack(strides)
    # numpy slice [start: stop: step]
    y = series[window_size: n: step_size][:, None]
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

# Note Numpy has standard tool for rooling window slicing np.lib.stride_tricks.as_strided()
# see https://gist.github.com/codehacken/708f19ae746784cef6e68b037af65788

    
# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1), activation='linear', recurrent_activation='linear')) 
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text: str) -> str:
    '''
    Removes all non "English letters" characters from text.
    All ascii lowercase and the punctuation !,.:;? and spaces should be included.
    :param text: String in ascii codepage. 
    '''
    udacity_punctuations = '!,.:;?' # <> string.punctuation
    udacity_whitespaces = ' ' # <> string.whitespace
    english_letters = string.ascii_lowercase + udacity_punctuations + udacity_whitespaces
    clean = None
    with StringIO(text) as in_txt:
        with StringIO() as out_txt:
            # copy text only
            c = '?'
            while c:
                c = in_txt.read(1) # char by char
                if c in english_letters:
                    out_txt.write(c)
            clean = out_txt.getvalue()
    return clean

# based on template offered by udacity
def cleaned_text_udacity(text):
    not_letters = ['!', ',', '.', ':', ';', '?', '(', ')', '{', '}'
                   , '#', '$', '%', '&', '*', '~', '-', '+', '/', '\\', '@'
                   , '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
                   , '\xa0', '©', 'ã', 'à', 'â', 'è', 'é']
    for c in not_letters:
        text = text.replace(c, ' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text: str, window_size: int, step_size: int) -> (List,List):
    '''
    transforms the input text into a set of input/output pairs of window-size for use with our RNN model.
    Note: the return items should be lists - not numpy arrays.
    :return inputs,outputs: inputs and outputs of RNN data sets as python list.
    '''
    # containers for input/output pairs
    inputs = []
    outputs = []
    starts = [s for s in range(0, len(text) - window_size, step_size)]
    for start in starts:
        end = start + window_size
        inputs.append(text[start: end])
        outputs.append(text[end])
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size: int, num_chars: int) ->Sequential:
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars), 
                   activation='sigmoid', recurrent_activation='tanh'))
    model.add(Dense(num_chars, activation='linear'))
    model.add(Activation('softmax'))
    return model


def scale_one(ds, v_min=None, v_max=None):
    ''' returns dataset scaled to [0..+1]'''
    if(v_min==None):
        v_min = np.nanmin(ds)
    if(v_max==None):
        v_max = np.nanmax(ds)
    scale = v_max - v_min
    return (ds - v_min) / scale


def scale_two(ds, v_min=None, v_max=None):
    ''' returns dataset scaled to [-1..+1]'''
    return scale_one(ds, v_min, v_max) * 2 -1


def descale_two(ds, v_min, v_max):
    '''reverse scaling of scale_two'''
    ds1 = (ds + 1 ) / 2
    return descale_one(ds1, v_min, v_max)


def descale_one(ds, v_min, v_max):
    '''reverse scaling of scale_one'''
    scale = v_max - v_min
    return (ds * scale) + v_min

