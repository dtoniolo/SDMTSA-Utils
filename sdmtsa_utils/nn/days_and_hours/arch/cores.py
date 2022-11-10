import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU


def M(input_shape, output_length, **gru_kwargs):
    core = tf.keras.Sequential()
    core.add(GRU(units=3*64, activation='relu', input_shape=input_shape,
                 **gru_kwargs))
    core.add(Dense(2*64, activation='relu'))
    core.add(Dense(64, activation='relu'))
    core.add(Dense(output_length, activation='relu'))
    return core


def M2(input_shape, output_length, **gru_kwargs):
    core = tf.keras.Sequential()
    core.add(GRU(units=4*64, activation='relu', input_shape=input_shape,
                 **gru_kwargs))
    core.add(Dense(3*64, activation='relu'))
    core.add(Dense(2*64, activation='relu'))
    core.add(Dense(64, activation='relu'))
    core.add(Dense(output_length, activation='relu'))
    return core
