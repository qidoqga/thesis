import math
import torch.nn as nn


def invoke(vocab_size, model_dim):
    layer = nn.Embedding(vocab_size, model_dim)
    return layer


meta = {
    "name": "Embedding Layer 1",
    "min_args": 2,
    "max_args": 2
}
