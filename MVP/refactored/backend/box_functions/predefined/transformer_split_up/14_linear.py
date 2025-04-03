import torch.nn as nn


def invoke(builder):
    model_dim = builder.get_model_dim()
    vocab_size = builder.get_vocab_size()
    builder.set_linear_layer(nn.Linear(model_dim, vocab_size))
    return builder


meta = {
    "name": "14 Linear",
    "min_args": 1,
    "max_args": 1
}
