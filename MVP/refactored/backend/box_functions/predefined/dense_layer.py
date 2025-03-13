import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import math

def invoke(*args):
    if len(args) != 1:
        raise ValueError('Dense Layer expects exactly one argument: the model.')
    model = args[0]
    model.add(Dense({neurons}, activation='{activation}'))
    return model

meta = {
    "name": "Dense Layer",
    "min_args": 1,
    "max_args": 1
}
