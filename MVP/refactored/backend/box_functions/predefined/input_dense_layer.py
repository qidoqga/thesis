import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import math


def invoke(*args):
    if args and all(isinstance(a, (int, float)) for a in args):
        input_dim = len(args)
    else:
        input_dim = {inputs}
    model = Sequential([
        tf.keras.Input(shape=(input_dim,)),
        Dense({neurons}, activation='{activation}')
    ])
    return model


meta = {
    "name": "Dense Input Layer",
    "min_args": 1,
    "max_args": math.inf
}
