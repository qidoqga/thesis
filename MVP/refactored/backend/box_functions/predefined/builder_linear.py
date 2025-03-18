import torch.nn as nn
from torchinfo import summary
import torch
import math


def invoke(builder, model_dim, vocab_size):
    builder.set_linear_layer(nn.Linear(model_dim, vocab_size))
    return builder


meta = {
    "name": "Builder Linear",
    "min_args": 3,
    "max_args": 3
}
