import torch.nn as nn
from torchinfo import summary
import torch
import math


def invoke(builder, model_dim, vocab_size):
    builder.set_embedding_layer(nn.Embedding(vocab_size, model_dim))
    return builder


meta = {
    "name": "Builder Embedding",
    "min_args": 3,
    "max_args": 3
}
