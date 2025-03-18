import math
import torch.nn as nn


def invoke(vocab_size, model_dim):
    layer = nn.Embedding(vocab_size, model_dim)
    return layer


meta = {
    "name": "Test Embedding Layer",
    "min_args": 2,
    "max_args": 2
}
