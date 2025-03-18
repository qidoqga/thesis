import torch.nn as nn


def invoke(vocab_size, model_dim):
    layer = nn.Linear(model_dim, vocab_size)
    return layer


meta = {
    "name": "Test Linear Layer",
    "min_args": 2,
    "max_args": 2
}
